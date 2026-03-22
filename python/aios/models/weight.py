from __future__ import annotations

import glob
import os

import safetensors
import torch

from aios.layers import BaseOP


def load_weights(model: BaseOP, model_path: str, device: torch.device, dtype: torch.dtype):
    """Load safetensors weights directly into a BaseOP model.

    No QKV fusion, no tensor parallelism sharding.
    Safetensors keys match BaseOP state_dict keys exactly.
    """
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    state_dict = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    state_dict = {k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
