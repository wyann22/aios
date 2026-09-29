from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from aios.distributed import DistributedCommunicator, div_even, get_tp_info

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    def __init__(
        self,
        full_input_size: int,
        full_output_size: int,
        local_input_size: int,
        local_output_size: int,
        has_bias: bool,
    ) -> None:
        self.full_input_size = full_input_size
        self.full_output_size = full_output_size
        self.local_input_size = local_input_size
        self.local_output_size = local_output_size
        self.weight = torch.empty(local_output_size, local_input_size)
        self.bias = torch.empty(local_output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """A full linear layer replicated on every tensor-parallel rank."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        super().__init__(
            input_size, output_size, input_size, output_size, has_bias
        )


class LinearColParallelMerged(_LinearTPImpl):
    """Shard each packed output branch across TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        has_bias: bool = False,
    ) -> None:
        tp_size = get_tp_info().size
        self.output_sizes = tuple(output_sizes)
        self.local_output_sizes = tuple(
            div_even(size, tp_size) for size in self.output_sizes
        )
        super().__init__(
            input_size,
            sum(self.output_sizes),
            input_size,
            sum(self.local_output_sizes),
            has_bias,
        )


class LinearQKVMerged(_LinearTPImpl):
    """Packed QKV projection with rank-local query and KV heads."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool = False,
    ) -> None:
        tp_size = get_tp_info().size
        self.local_num_qo_heads = div_even(num_qo_heads, tp_size)
        self.local_num_kv_heads = div_even(
            num_kv_heads, tp_size, allow_replicate=True
        )
        self.q_size = self.local_num_qo_heads * head_dim
        self.kv_size = self.local_num_kv_heads * head_dim
        super().__init__(
            hidden_size,
            (num_qo_heads + 2 * num_kv_heads) * head_dim,
            hidden_size,
            self.q_size + 2 * self.kv_size,
            has_bias,
        )


class LinearRowParallel(_LinearTPImpl):
    """Shard the input dimension, then sum partial outputs with AllReduce."""

    def __init__(
        self, input_size: int, output_size: int, has_bias: bool = False
    ) -> None:
        tp_info = get_tp_info()
        self._tp_rank = tp_info.rank
        self._tp_size = tp_info.size
        self._comm = DistributedCommunicator()
        super().__init__(
            input_size,
            output_size,
            div_even(input_size, self._tp_size),
            output_size,
            has_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bias must be added once, not once per rank before the sum.
        bias = self.bias if self._tp_rank == 0 else None
        y = F.linear(x, self.weight, bias)
        return self._comm.all_reduce(y)


class LinearOProj(LinearRowParallel):
    pass


# Keep the previous lesson's public name source-compatible.
Linear = LinearReplicated
