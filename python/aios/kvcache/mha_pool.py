from __future__ import annotations

import torch
from .base import BaseKVCache, KVCacheLayout

class MHAKVCache(BaseKVCache):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
        page_size: int = 1,
    ):
        local_kv_heads = num_kv_heads # no tp for now
        self.page_size = page_size
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, page_size, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4, 5)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, page_size, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        
        # we will shape it as (2, num_layers, num_pages, local_kv_heads, page_size, head_dim)
        self._kv_buffer = kv_buffer
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        Store k, v tensors.
        k/v shape: (num_tokens, num_kv_heads, head_dim) (where num_tokens is sum of all sequences extended length in this batch)
        out_loc shape: (num_tokens,)
        out_loc is the global flat page/slot index.
        For page_size = 1, out_loc is exactly the page index. (Actually if page_size=1, out_loc = page_idx).
        For page_size > 1, out_loc = page_idx * page_size + slot_in_page_idx
        """
        if self.page_size == 1:
            # Simple indexing for page_size=1
            self._k_buffer[layer_id, out_loc, :, 0, :] = k
            self._v_buffer[layer_id, out_loc, :, 0, :] = v
        else:
            page_idx = out_loc // self.page_size
            slot_idx = out_loc % self.page_size
            self._k_buffer[layer_id, page_idx, :, slot_idx, :] = k
            self._v_buffer[layer_id, page_idx, :, slot_idx, :] = v

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
