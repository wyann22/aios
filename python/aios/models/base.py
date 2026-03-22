from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aios.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
