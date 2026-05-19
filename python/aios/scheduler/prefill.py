from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch

from ..core import Batch, Req, SamplingParams
from .common import PendingReq

if TYPE_CHECKING:
    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager


@dataclass
class PrefillManager:
    """Holds pending requests and admits one varlen prefill batch."""

    cache_manager: "CacheManager"
    table_manager: "TableManager"
    decode_manager: "DecodeManager"
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, pending: PendingReq) -> None:
        self.pending_list.append(pending)

    @property
    def runnable(self) -> bool:
        return bool(self.pending_list)

    def _can_admit(
        self,
        pending: PendingReq,
        max_running: int,
        scheduled_count: int = 0,
        scheduled_reserved: int = 0,
    ) -> bool:
        # Capacity: one free table slot + one free running-set slot.
        if self.table_manager.available_size - scheduled_count <= 0:
            return False
        if len(self.decode_manager.running_reqs) + scheduled_count >= max_running:
            return False
        needed = pending.input_len + pending.output_len
        reserved = self.decode_manager.inflight_tokens + scheduled_reserved
        free_pages = len(self.cache_manager._free_slots)
        return (needed + reserved) <= free_pages

    def schedule_next_batch(
        self, max_running: int, max_prefill_tokens: int | None = None
    ) -> Batch | None:
        if not self.pending_list:
            return None

        token_budget = max_prefill_tokens if max_prefill_tokens is not None else float("inf")
        selected: List[PendingReq] = []
        scheduled_tokens = 0
        scheduled_reserved = 0
        for pending in self.pending_list:
            prompt_len = pending.input_len
            if selected and scheduled_tokens + prompt_len > token_budget:
                break
            if prompt_len > token_budget and selected:
                break
            if not self._can_admit(
                pending, max_running, len(selected), scheduled_reserved
            ):
                break
            selected.append(pending)
            scheduled_tokens += prompt_len
            scheduled_reserved += prompt_len + pending.output_len
            if scheduled_tokens >= token_budget:
                break

        if not selected:
            return None
        self.pending_list = self.pending_list[len(selected) :]

        reqs: List[Req] = []
        for pending in selected:
            # Allocate table slot and write prompt tokens. KV pages are allocated
            # later in Scheduler._prepare_batch, matching mini-sglang's flow.
            table_idx = self.table_manager.allocate()
            prompt_len = pending.input_len
            self.table_manager.token_pool[table_idx, :prompt_len] = pending.input_ids.to(
                torch.int32
            )

            req = Req(
                input_ids=pending.input_ids,
                cached_len=0,
                output_len=pending.output_len,
                uid=pending.uid,
                sampling_params=pending.sampling_params,
                table_idx=table_idx,
            )
            reqs.append(req)

        return Batch(reqs=reqs, phase="prefill")
