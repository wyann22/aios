from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    from flashinfer import silu_and_mul as flashinfer_silu_and_mul

    return flashinfer_silu_and_mul(x, out=out)


__all__ = ["silu_and_mul"]
