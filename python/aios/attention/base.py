from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from aios.core import Batch
    from aios.kvcache import MHAKVCache


@dataclass
class BaseAttentionMetadata(ABC):
    """Per-batch attention metadata, aligned with mini-sglang's interface."""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttentionBackend(ABC):
    """Attention backend interface.

    The model layer owns QKV projection, norm, RoPE, and output projection.
    Backends own KV cache writes, metadata construction, and prefill/decode
    kernel dispatch.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> BaseAttentionMetadata: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridAttentionBackend(BaseAttentionBackend):
    """Dispatch to separate prefill/decode backends by batch phase."""

    def __init__(
        self,
        prefill_backend: BaseAttentionBackend,
        decode_backend: BaseAttentionBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, paged_kv_cache, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> BaseAttentionMetadata:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
