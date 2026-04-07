from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """Per-request runtime state (mini-sglang style, simplified for lessons)."""

    input_ids: torch.Tensor
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: Any | None = None
    block_table: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len <= self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def can_decode(self) -> bool:
        return self.remain_len > 0
