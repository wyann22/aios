from __future__ import annotations

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 1000000.0):
        super().__init__()
        # Force CPU creation even inside `with torch.device("meta")` context,
        # since cos/sin caches are precomputed buffers (not learned parameters).
        with torch.device("cpu"):
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            t = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cache = emb.cos()  # (max_pos, head_dim)
            self._sin_cache = emb.sin()  # (max_pos, head_dim)

    def forward(self, position_ids: torch.Tensor):
        # position_ids: (batch, seq_len)
        cos = self._cos_cache[position_ids]  # (batch, seq_len, head_dim)
        sin = self._sin_cache[position_ids]  # (batch, seq_len, head_dim)
        return cos.unsqueeze(1), sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

    def set_device(self, device: torch.device):
        self._cos_cache = self._cos_cache.to(device)
        self._sin_cache = self._sin_cache.to(device)
