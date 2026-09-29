from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import safetensors
import torch

from aios.distributed import div_ceil, get_tp_info
from aios.layers import BaseOP

# HF checkpoints keep these projections separate.  Fused inference operators
# consume the same tensors packed along their output dimension.
packed_modules_mapping: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _checkpoint_index(files: Iterable[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in files:
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in index:
                    raise RuntimeError(f"Duplicate safetensors key: {name}")
                index[name] = path
    return index


def _read_tensor(index: dict[str, str], name: str) -> torch.Tensor:
    try:
        path = index[name]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required tensor: {name}") from exc
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _packed_source_names(target_name: str) -> tuple[str, ...] | None:
    for packed_name, source_names in packed_modules_mapping.items():
        marker = f".{packed_name}."
        if marker in target_name:
            return tuple(
                target_name.replace(marker, f".{source_name}.")
                for source_name in source_names
            )
    return None


def _shard_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    if any(part in name for part in (".q_proj.", ".gate_proj.", ".up_proj.")):
        # Column-parallel weights own disjoint output channels (dim 0).
        return tensor.chunk(world_size, dim=0)[rank].contiguous()
    if any(part in name for part in (".k_proj.", ".v_proj.")):
        if world_size > num_kv_heads:
            if world_size % num_kv_heads != 0:
                raise ValueError(
                    f"TP size {world_size} cannot replicate {num_kv_heads} KV heads"
                )
            # Keep a KV head intact; several query ranks reuse it for GQA.
            head_dim = tensor.shape[0] // num_kv_heads
            head_idx = rank * num_kv_heads // world_size
            return tensor.narrow(0, head_idx * head_dim, head_dim).contiguous()
        # Normal GQA case: each rank owns a disjoint group of KV heads.
        return tensor.chunk(world_size, dim=0)[rank].contiguous()
    if any(part in name for part in (".o_proj.", ".down_proj.")):
        # Row-parallel weights split input channels (dim 1).
        return tensor.chunk(world_size, dim=1)[rank].contiguous()
    if "embed_tokens.weight" in name or "lm_head.weight" in name:
        # Equal-size vocabulary shards simplify lookup and AllGather; pad the tail.
        rows_per_rank = div_ceil(tensor.shape[0], world_size)
        start = rank * rows_per_rank
        end = min(start + rows_per_rank, tensor.shape[0])
        shard = tensor[start:end].contiguous()
        if shard.shape[0] < rows_per_rank:
            shard = torch.cat(
                [
                    shard,
                    torch.zeros(
                        rows_per_rank - shard.shape[0],
                        tensor.shape[1],
                        dtype=tensor.dtype,
                    ),
                ],
                dim=0,
            )
        return shard
    return tensor


def load_weights(
    model: BaseOP,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    num_kv_heads: int,
) -> None:
    """Load an HF safetensors checkpoint and pack fused inference weights.

    Directly matching tensors retain their HF names.  QKV and gate/up tensors
    are concatenated on dim 0 before ``BaseOP.load_state_dict`` assigns them to
    the fused modules.  This keeps packing at the loader boundary rather than
    leaking checkpoint-format knowledge into model layers.
    """
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    index = _checkpoint_index(files)
    tp_info = get_tp_info()
    fused_state_dict: dict[str, torch.Tensor] = {}
    for target_name in model.state_dict():
        source_names = _packed_source_names(target_name)
        if source_names is None:
            tensor = _shard_tensor(
                target_name,
                _read_tensor(index, target_name),
                rank=tp_info.rank,
                world_size=tp_info.size,
                num_kv_heads=num_kv_heads,
            )
        else:
            tensor = torch.cat(
                [
                    _shard_tensor(
                        name,
                        _read_tensor(index, name),
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=num_kv_heads,
                    )
                    for name in source_names
                ],
                dim=0,
            )
        fused_state_dict[target_name] = tensor.to(device=device, dtype=dtype)

    model.load_state_dict(fused_state_dict)
