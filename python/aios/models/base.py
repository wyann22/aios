from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from aios.layers import BaseOP


class BaseLLMModel(ABC, BaseOP):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        forward = cls.__dict__.get("forward")
        if forward is None or getattr(forward, "_is_timed_forward", False):
            return
        @wraps(forward)

        def timed_forward(self, *args, **kwargs):
            if not hasattr(self, "_timed_forward_count"):
                self._timed_forward_count = 0
            self._timed_forward_count += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = perf_counter()
            result = forward(self, *args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (perf_counter() - start) * 1000
            if self._timed_forward_count % 32 == 0:
                print(f"{type(self).__name__}.forward took {elapsed_ms:.2f} ms")
            return result

        timed_forward._is_timed_forward = True
        cls.forward = timed_forward

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, kv_cache: Any | None = None) -> torch.Tensor: ...
