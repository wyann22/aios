from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from .info import DistributedInfo, get_tp_info, reset_tp_info, set_tp_info


def div_even(a: int, b: int, *, allow_replicate: bool = False) -> int:
    """Divide evenly, optionally replicating KV heads when TP is wider."""
    if allow_replicate and b > a:
        if b % a != 0:
            raise ValueError(f"TP size {b} must be divisible by KV heads {a}")
        return 1
    if a % b != 0:
        raise ValueError(f"{a} must be divisible by tensor parallel size {b}")
    return a // b


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def initialize_distributed(
    tensor_parallel_size: int,
    *,
    timeout_seconds: float = 120.0,
) -> DistributedInfo:
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if tensor_parallel_size != world_size:
        if tensor_parallel_size > 1:
            raise RuntimeError(
                "tensor_parallel_size must match torchrun WORLD_SIZE; launch with "
                f"`torchrun --standalone --nproc-per-node={tensor_parallel_size} ...`"
            )
        if world_size != 1:
            raise RuntimeError(
                f"torchrun started WORLD_SIZE={world_size}, but tensor_parallel_size=1"
            )

    if tensor_parallel_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
    set_tp_info(rank=rank, size=tensor_parallel_size, local_rank=local_rank)
    return get_tp_info()


class DistributedCommunicator:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if get_tp_info().size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tp_info().size
        if tp_size == 1:
            return x
        output_shape = list(x.shape)
        output_shape[0] *= tp_size
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(output, x.contiguous())
        return output

    def broadcast(self, x: torch.Tensor, src: int = 0) -> torch.Tensor:
        if get_tp_info().size > 1:
            dist.broadcast(x, src=src)
        return x


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    reset_tp_info()
