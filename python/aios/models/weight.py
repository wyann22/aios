from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import safetensors
import torch

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
            return tuple(target_name.replace(marker, f".{source_name}.") for source_name in source_names)
    return None


def load_weights(model: BaseOP, model_path: str, device: torch.device, dtype: torch.dtype) -> None:
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
    fused_state_dict: dict[str, torch.Tensor] = {}
    for target_name in model.state_dict():
        source_names = _packed_source_names(target_name)
        if source_names is None:
            tensor = _read_tensor(index, target_name)
        else:
            tensor = torch.cat([_read_tensor(index, name) for name in source_names], dim=0)
        fused_state_dict[target_name] = tensor.to(device=device, dtype=dtype)

    model.load_state_dict(fused_state_dict)
