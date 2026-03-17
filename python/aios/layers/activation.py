from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU activation on gate, then element-wise multiply with up."""
    return F.silu(gate) * up


__all__ = ["silu_and_mul"]
