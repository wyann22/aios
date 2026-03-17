from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .base import BaseOP, _concat_prefix


class Embedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class LMHead(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tie_word_embeddings: bool = False,
        tied_embedding: Embedding | None = None,
    ):
        self._tie_word_embeddings = tie_word_embeddings
        self._tied_embedding = tied_embedding
        if not tie_word_embeddings:
            self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._tied_embedding.weight if self._tie_word_embeddings else self.weight
        return F.linear(x, w)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self._tie_word_embeddings:
            # Pop lm_head.weight if present (tied to embedding)
            key = _concat_prefix(prefix, "weight")
            if key in state_dict:
                state_dict.pop(key)
        else:
            super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self._tie_word_embeddings:
            return result if result is not None else {}
        return super().state_dict(prefix=prefix, result=result)
