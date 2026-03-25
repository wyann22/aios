from __future__ import annotations

import torch


class DynamicKVCache:
    """A simple per-layer KV cache that grows with ``torch.cat``.

    Each layer stores:
    - key cache:   (batch, num_kv_heads, seq_len_so_far, head_dim)
    - value cache: (batch, num_kv_heads, seq_len_so_far, head_dim)

    This matches the core idea behind Hugging Face's dynamic cache, but keeps
    the implementation intentionally small and easy to read.
    """

    def __init__(self, num_layers: int):
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self._seq_len = 0

    def get_seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_keys = self.key_cache[layer_idx]
        cached_values = self.value_cache[layer_idx]

        if cached_keys is None:
            full_keys = key_states
            full_values = value_states
        else:
            full_keys = torch.cat([cached_keys, key_states], dim=2)
            full_values = torch.cat([cached_values, value_states], dim=2)

        self.key_cache[layer_idx] = full_keys
        self.value_cache[layer_idx] = full_values

        # All layers grow by the same amount every decoding step, so tracking
        # the first layer is enough to know the current cached sequence length.
        if layer_idx == 0:
            self._seq_len = full_keys.shape[2]

        return full_keys, full_values

    def clear(self) -> None:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = None
            self.value_cache[layer_idx] = None
        self._seq_len = 0
