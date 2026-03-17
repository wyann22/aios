from __future__ import annotations

import torch

from .base import BaseOP

    
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight
