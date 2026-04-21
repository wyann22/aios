from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..kvcache import MHAKVCache
    from ..scheduler.scheduler import ScheduledBatch


class Engine:
    """Execution layer: batched forward + per-request sampling (lesson 6)."""

    def __init__(self, model, mha_kv_cache: MHAKVCache) -> None:
        self.model = model
        self.mha_kv_cache = mha_kv_cache

    def run_batch(self, scheduled: ScheduledBatch) -> torch.Tensor:
        """Run model forward on a batch, sample next tokens.

        Returns: (B,) tensor of next token ids.
        """
        batch = scheduled.batch
        logits = self.model.forward(
            batch.input_ids,
            paged_kv_cache=self.mha_kv_cache,
            batch=batch,
        )
        # logits: (B, seq_len, vocab) -> take last position
        last_logits = logits[:, -1, :]  # (B, vocab)

        # Per-request sampling (supports different sampling_params)
        next_tokens = []
        for i, sampler in enumerate(scheduled.samplers):
            tok = sampler.sample(last_logits[i : i + 1])  # (1, 1)
            next_tokens.append(tok.view(-1)[0])
        return torch.stack(next_tokens)  # (B,)
