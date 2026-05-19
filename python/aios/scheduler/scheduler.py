from __future__ import annotations

from typing import List, Tuple, TypeAlias

import torch

from ..attention import BaseAttentionBackend
from ..core import Batch, Req, SamplingParams
from .cache import CacheManager
from .common import PendingReq
from .decode import DecodeManager
from .prefill import PrefillManager
from .table import TableManager


Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


class Scheduler:
    """Continuous-batching scheduler.

    Composes PrefillManager + DecodeManager, aligned with mini-sglang's top-level
    Scheduler:
      - prefill_manager.schedule_next_batch(prefill_budget) is tried first.
      - decode_manager.schedule_next_batch() runs otherwise over the full running set.
      - On completion, resources are freed immediately via _free_req_resources so that
        a new pending request can reuse the slot / pages in the next iteration.

    Deliberate simplifications vs mini-sglang (documented in the lesson docs):
      1. No chunked prefill (long-prompt splitting deferred).
      2. No prefix caching (CacheManager.cache_req stays commented; direct page free).
      3. Single CUDA stream, no overlap_loop.
      4. Single-process; requests are pushed via add_request, no IPC receive_msg.
    """

    def __init__(
        self,
        table_manager: TableManager,
        cache_manager: CacheManager,
        eos_token_id: int,
        device: torch.device,
        max_running_reqs: int,
        attn_backend: BaseAttentionBackend,
        prefill_token_budget: int | None = None,
    ) -> None:
        self.table_manager = table_manager
        self.cache_manager = cache_manager
        self.eos_token_id = eos_token_id
        self.device = device
        self.max_running = max_running_reqs
        self.attn_backend = attn_backend
        self.prefill_token_budget = prefill_token_budget

        self.decode_manager = DecodeManager(page_size=1)
        self.prefill_manager = PrefillManager(
            cache_manager=cache_manager,
            table_manager=table_manager,
            decode_manager=self.decode_manager,
        )

        self.finished: List[Req] = []
        self._next_uid = 0

    # --------------------------------------------------------------- admission

    def add_request(
        self, input_ids: torch.Tensor, sampling_params: SamplingParams
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self.prefill_manager.add_one_req(
            PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
        )
        return uid

    # -------------------------------------------------------------- scheduling

    def schedule_next_batch(self) -> Batch | None:
        # Prefill-first policy (matches mini-sglang default).
        batch = (
            self.prefill_manager.schedule_next_batch(
                self.max_running, self.prefill_token_budget
            )
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> Batch:
        self.cache_manager.allocate_paged(batch.reqs, self.table_manager.page_table)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        batch.input_ids = self.table_manager.token_pool[input_mapping].long()
        batch.out_loc = self.table_manager.page_table[input_mapping]
        self.attn_backend.prepare_metadata(batch)
        return batch

    # ---------------------------------------------------------- post-processing

    def process_batch_output(
        self, batch: Batch, next_tokens: torch.Tensor
    ) -> None:
        tokens = next_tokens.view(-1).tolist()
        for req, tok in zip(batch.reqs, tokens):
            finished = self._advance(req, tok)
            if finished:
                if batch.is_decode:
                    self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished.append(req)
        if batch.is_prefill:
            unfinished_reqs = [req for req in batch.reqs if req not in self.finished]
            self.decode_manager.filter_reqs(unfinished_reqs)

    def debug_state(self, batch: Batch | None = None) -> str:
        return (
            f"Batch={batch} "
            f"Pending={self.prefill_manager.pending_list} "
            f"Running={self.decode_manager.running_reqs} "
            f"Finished={self.finished}"
        )

    def _advance(self, req: Req, tok: int) -> bool:
        req.complete_one()
        self.table_manager.token_pool[req.table_idx, req.device_len - 1] = tok
        req.generated.append(tok)
        hit_eos = (not req.sampling_params.ignore_eos) and (tok == self.eos_token_id)
        return hit_eos or not req.can_decode

    def _free_req_resources(self, req: Req) -> None:
        used_pages = self.table_manager.page_table[req.table_idx, : req.cached_len]
        self.table_manager.free(req.table_idx)
        self.cache_manager._free(used_pages)

    # -------------------------------------------------------------- inspection

    @property
    def has_work(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def collect_results(self, tokenizer) -> list[dict]:
        results = [
            {
                "uid": req.uid,
                "token_ids": req.generated,
                "text": tokenizer.decode(req.generated, skip_special_tokens=True),
            }
            for req in self.finished
        ]
        results.sort(key=lambda r: r["uid"])
        return results


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(req.extend_len for req in batch.reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)
