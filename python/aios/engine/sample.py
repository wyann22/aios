from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..core import SamplingParams


@dataclass
class Sampler:
    """Independent sampler module (aligned with mini-sglang's engine/sample.py)."""

    sampling_params: SamplingParams

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits. logits shape: (batch, vocab_size)"""
        if self.sampling_params.is_greedy:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / self.sampling_params.temperature
        if self.sampling_params.top_k > 0:
            topk_vals = torch.topk(
                logits, min(self.sampling_params.top_k, logits.size(-1))
            ).values
            logits = logits.masked_fill(
                logits < topk_vals[..., -1:], float("-inf")
            )
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
