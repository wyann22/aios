from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseOP


class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
