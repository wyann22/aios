from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, List

import torch

from ..core import Batch, Req, SamplingParams
from ..engine.sample import Sampler
from .cache import CacheManager
from .table import TableManager


@dataclass
class _ReqState:
    """Scheduler-private per-request state."""

    req: Req
    sampler: Sampler
    finished: bool = False


@dataclass
class ScheduledBatch:
    """Scheduler -> Engine execution unit."""

    batch: Batch
    samplers: List[Sampler]  # aligned with batch.reqs
    state_indices: List[int]  # indices into Scheduler._states


class Scheduler:
    """Static batch scheduler: prefill-first, then decode loop.

    State machine per req (mini-sglang aligned):
      - cached_len:  # tokens whose KV has been written to the paged cache
      - device_len:  cached_len + # tokens in this extend (forward's input)
      - Pages for the extend range [cached_len, device_len) are allocated at
        schedule time (before forward), never after.
      - After forward, complete_one() advances: cached_len = device_len,
        device_len += 1. The sampled token is written at token_pool[device_len-1].
    """

    def __init__(
        self,
        table_manager: TableManager,
        cache_manager: CacheManager,
        eos_token_id: int,
        device: torch.device,
    ) -> None:
        self.table_manager = table_manager
        self.cache_manager = cache_manager
        self.eos_token_id = eos_token_id
        self.device = device
        self._states: List[_ReqState] = []

    def init_requests(
        self,
        all_input_ids: List[torch.Tensor],
        params_list: List[SamplingParams],
    ) -> None:
        """Allocate table slot + prompt pages, write prompt tokens to token_pool."""
        for uid, (ids, sp) in enumerate(zip(all_input_ids, params_list)):
            table_idx = self.table_manager.allocate()
            prompt_len = len(ids)

            pages = self.cache_manager.allocate(prompt_len)
            self.table_manager.page_table[table_idx, :prompt_len] = pages
            self.table_manager.token_pool[table_idx, :prompt_len] = ids.to(torch.int32)

            req = Req(
                input_ids=ids,
                cached_len=0,
                output_len=sp.max_tokens,
                uid=uid,
                sampling_params=sp,
                table_idx=table_idx,
            )
            self._states.append(_ReqState(req=req, sampler=Sampler(sp)))

    def iter_prefill_batches(self) -> Generator[ScheduledBatch, None, None]:
        """Group requests by prompt length, yield one prefill ScheduledBatch per group."""
        by_len: dict[int, list[int]] = defaultdict(list)
        for idx, state in enumerate(self._states):
            by_len[len(state.req.input_ids)].append(idx)

        for prompt_len, indices in by_len.items():
            reqs = [self._states[i].req for i in indices]
            samplers = [self._states[i].sampler for i in indices]
            B = len(reqs)

            input_ids = torch.stack([r.input_ids for r in reqs]).to(self.device).long()
            positions = torch.arange(prompt_len, device=self.device).unsqueeze(0).expand(B, -1)
            # Prompt pages were allocated in init_requests.
            out_loc = torch.stack([
                self.table_manager.page_table[r.table_idx, :prompt_len] for r in reqs
            ])

            batch = Batch(
                reqs=reqs,
                phase="prefill",
                input_ids=input_ids,
                positions=positions,
                out_loc=out_loc,
                page_table=self.table_manager.page_table,
            )
            yield ScheduledBatch(batch=batch, samplers=samplers, state_indices=indices)

    def schedule_decode_batch(self) -> ScheduledBatch | None:
        """Build a decode batch of all active reqs, allocating one page per req."""
        active = [(i, s) for i, s in enumerate(self._states) if not s.finished]
        if not active:
            return None

        indices = [i for i, _ in active]
        reqs = [s.req for _, s in active]
        samplers = [s.sampler for _, s in active]
        B = len(reqs)

        table_idxs = torch.tensor(
            [r.table_idx for r in reqs], device=self.device, dtype=torch.long
        )
        positions_1d = torch.tensor(
            [r.cached_len for r in reqs], device=self.device, dtype=torch.long
        )

        # Allocate one page per req (this step's KV write slot, at position cached_len).
        new_pages = self.cache_manager.allocate(B)
        self.table_manager.page_table[table_idxs, positions_1d] = new_pages

        # Input token = token_pool[cached_len] (written by previous step).
        input_ids = self.table_manager.token_pool[table_idxs, positions_1d].long().unsqueeze(1)
        positions = positions_1d.unsqueeze(1)  # (B, 1)
        out_loc = new_pages.unsqueeze(1)  # (B, 1)

        batch = Batch(
            reqs=reqs,
            phase="decode",
            input_ids=input_ids,
            positions=positions,
            out_loc=out_loc,
            page_table=self.table_manager.page_table,
        )
        return ScheduledBatch(batch=batch, samplers=samplers, state_indices=indices)

    def process_batch_output(
        self, scheduled: ScheduledBatch, next_tokens: torch.Tensor
    ) -> None:
        """Advance state, write sampled token to token_pool, check termination."""
        next_tokens = next_tokens.view(-1)

        for i, state_idx in enumerate(scheduled.state_indices):
            state = self._states[state_idx]
            req = state.req
            tok = next_tokens[i].item()

            req.complete_one()
            self.table_manager.token_pool[req.table_idx, req.device_len - 1] = tok

            hit_eos = (not req.sampling_params.ignore_eos) and (tok == self.eos_token_id)
            if hit_eos or not req.can_decode():
                state.finished = True

    @property
    def has_active_reqs(self) -> bool:
        return any(not s.finished for s in self._states)

    def finalize_results(self, tokenizer) -> list[dict]:
        """Extract output tokens, free resources, return results sorted by uid."""
        results: list[dict] = []
        for state in self._states:
            req = state.req
            prompt_len = len(req.input_ids)
            gen_ids = self.table_manager.token_pool[
                req.table_idx, prompt_len : req.device_len
            ].cpu().tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": gen_ids})

            # Only [0, cached_len) pages were actually allocated/written.
            used_pages = self.table_manager.page_table[req.table_idx, : req.cached_len]
            self.cache_manager._free(used_pages)
            self.table_manager.free(req.table_idx)

        return [r for _, r in sorted(
            zip(self._states, results), key=lambda x: x[0].req.uid
        )]
