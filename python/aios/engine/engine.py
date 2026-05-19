from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..core import get_global_ctx
from .sample import Sampler

if TYPE_CHECKING:
    from ..core import Batch
    from ..kvcache import MHAKVCache


class Engine:
    """Execution layer: batched forward + per-request sampling (lesson 6)."""

    def __init__(self, model, mha_kv_cache: MHAKVCache) -> None:
        self.model = model
        self.mha_kv_cache = mha_kv_cache

    def run_batch(self, batch: Batch) -> torch.Tensor:
        """Run model forward on a batch, sample next tokens.

        Returns: (B,) tensor of next token ids.
        """
        ctx = get_global_ctx()
        with ctx.forward_batch(batch):
            logits = self.model.forward()
        last_logits = logits[: batch.size]  # (B, vocab)

        # Per-request sampling (supports different sampling_params)
        next_tokens = []
        for i, req in enumerate(batch.reqs):
            sampler = Sampler(req.sampling_params)
            tok = sampler.sample(last_logits[i : i + 1])  # (1, 1)
            next_tokens.append(tok.view(-1)[0])
        return torch.stack(next_tokens)  # (B,)
