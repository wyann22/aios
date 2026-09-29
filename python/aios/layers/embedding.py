from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from aios.core import get_global_ctx
from aios.distributed import DistributedCommunicator, div_ceil, get_tp_info

from .base import BaseOP, _concat_prefix


class VocabParallelEmbedding(BaseOP):
    """Split vocabulary rows across TP ranks and sum masked lookups."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        tp_info = get_tp_info()
        self._tp_rank = tp_info.rank
        self._tp_size = tp_info.size
        self._num_embeddings = num_embeddings
        self._num_embeddings_per_rank = div_ceil(num_embeddings, self._tp_size)
        self._vocab_start = self._num_embeddings_per_rank * self._tp_rank
        self._vocab_end = min(
            self._vocab_start + self._num_embeddings_per_rank, num_embeddings
        )
        self._comm = DistributedCommunicator()
        self.weight = torch.empty(self._num_embeddings_per_rank, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._tp_size == 1:
            return F.embedding(input_ids, self.weight)

        local_mask = (input_ids >= self._vocab_start) & (input_ids < self._vocab_end)
        local_ids = (input_ids - self._vocab_start).masked_fill(~local_mask, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(~local_mask.unsqueeze(-1), 0)
        return self._comm.all_reduce(output)


class ParallelLMHead(VocabParallelEmbedding):
    """Compute local vocabulary logits and gather them in rank order."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self._num_embeddings_per_rank) if bias else None
        self._tied_embedding = tied_embedding
        if (tied_embedding is not None) != tie_word_embeddings:
            raise ValueError("tied_embedding and tie_word_embeddings disagree")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()

        module = self._tied_embedding or self
        local_logits = F.linear(x, module.weight, self.bias)
        if self._tp_size == 1:
            return local_logits[:, : self._num_embeddings]

        input_shape = local_logits.shape
        gathered = self._comm.all_gather(local_logits)
        gathered = gathered.view((self._tp_size,) + input_shape)
        gathered = gathered.permute(1, 0, 2).contiguous()
        return gathered.reshape(input_shape[0], -1)[:, : self._num_embeddings]

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self._tied_embedding is None:
            super().load_state_dict(
                state_dict, prefix=prefix, _internal=_internal
            )
            return
        state_dict.pop(_concat_prefix(prefix, "weight"), None)
        state_dict.pop(_concat_prefix(prefix, "bias"), None)

    def state_dict(self, *, prefix: str = "") -> Dict[str, torch.Tensor]:
        if self._tied_embedding is not None:
            return {}
        return super().state_dict(prefix=prefix)


# Previous lesson aliases.
Embedding = VocabParallelEmbedding
LMHead = ParallelLMHead
