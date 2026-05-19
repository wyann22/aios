from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core import SamplingParams


@dataclass
class PendingReq:
    """A request waiting to be admitted to prefill."""

    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
