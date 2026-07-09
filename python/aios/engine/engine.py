from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..core import get_global_ctx
from .sample import Sampler

if TYPE_CHECKING:
    from ..core import Batch
    from .graph import GraphRunner
    from ..kvcache import MHAKVCache


class Engine:
    """Execution layer: batched forward + per-request sampling (lesson 6)."""

    def __init__(
        self,
        model,
        mha_kv_cache: MHAKVCache,
        graph_runner: GraphRunner | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.model = model
        self.mha_kv_cache = mha_kv_cache
        self.graph_runner = graph_runner
        self.stream = stream

    def run_batch(self, batch: Batch) -> torch.Tensor:
        """Run model forward on a batch, sample next tokens.

        Returns: (B,) tensor of next token ids.
        """
        if self.stream is not None:
            assert torch.cuda.current_stream() == self.stream
        ctx = get_global_ctx()
        with ctx.forward_batch(batch):
            if self.graph_runner is not None and self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()
        last_logits = logits[: batch.size]  # (B, vocab)

        # Per-request sampling (supports different sampling_params)
        next_tokens = []
        for i, req in enumerate(batch.reqs):
            sampler = Sampler(req.sampling_params)
            tok = sampler.sample(last_logits[i : i + 1])  # (1, 1)
            next_tokens.append(tok.view(-1)[0])
        return torch.stack(next_tokens)  # (B,)
