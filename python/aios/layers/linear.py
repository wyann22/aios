from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .base import BaseOP


class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearQKVMerged(Linear):
    """One packed projection for Q, K, and V.

    The packed layout is ``[Q | K | V]`` on the output dimension.  It matches
    mini-sglang's ``LinearQKVMerged`` layout, while remaining replicated on one
    GPU until the tensor-parallel lesson.
    """

    def __init__(
        self,
        hidden_size: int,
        q_size: int,
        kv_size: int,
        has_bias: bool = False,
    ) -> None:
        super().__init__(hidden_size, q_size + 2 * kv_size, has_bias)
        self.q_size = q_size
        self.kv_size = kv_size


class LinearColParallelMerged(Linear):
    """One packed column projection for independent output branches.

    For Qwen3 SwiGLU this is ``[gate | up]``.  The name deliberately follows
    mini-sglang so that the later tensor-parallel lesson can shard it without
    changing the model-level contract.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        has_bias: bool = False,
    ) -> None:
        self.output_sizes = tuple(output_sizes)
        super().__init__(input_size, sum(self.output_sizes), has_bias)
