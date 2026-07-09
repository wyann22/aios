from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from aios.core import Batch, Req, get_global_ctx

if TYPE_CHECKING:
    from aios.attention import BaseAttentionBackend
    from aios.models import BaseLLMModel


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
    ) -> "GraphCaptureBuffer":
        return cls(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        buffer_slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[buffer_slice]
        batch.out_loc = self.out_loc[buffer_slice]
        batch.positions = self.positions[buffer_slice]

    def copy_from(self, batch: Batch) -> None:
        buffer_slice = slice(batch.padded_size)
        self.input_ids[buffer_slice].copy_(batch.input_ids)
        self.out_loc[buffer_slice].copy_(batch.out_loc)
        self.positions[buffer_slice].copy_(batch.positions)


def determine_cuda_graph_bs(
    cuda_graph_bs: list[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> list[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs
    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        cuda_graph_max_bs = 256 if free_memory_gb > 80 else 160
    if cuda_graph_max_bs < 1:
        return []
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


def mem_gb(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


class GraphRunner:
    """CUDA graph capture/replay for decode batches.

    The runner owns static input buffers and one graph per padded batch size.
    Schedulers still build normal ``Batch`` objects; ``pad_batch`` only appends
    dummy requests so decode shapes match a captured bucket.
    """

    def __init__(
        self,
        *,
        model: BaseLLMModel,
        attn_backend: BaseAttentionBackend,
        stream: torch.cuda.Stream,
        device: torch.device,
        vocab_size: int,
        max_seq_len: int,
        dummy_req: Req,
        free_memory: int,
        cuda_graph_bs: list[int] | None = None,
        cuda_graph_max_bs: int | None = None,
    ) -> None:
        self.attn_backend = attn_backend
        self.device = device
        self.dummy_req = dummy_req
        self.stream = stream
        self.graph_bs_list = sorted(
            determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory)
        )
        self.max_graph_bs = max(self.graph_bs_list) if self.graph_bs_list else 0
        self.graph_map: dict[int, torch.cuda.CUDAGraph] = {}
        self.buffer: GraphCaptureBuffer | None = None
        self._capture_graphs(model, vocab_size, max_seq_len)

    def _capture_graphs(
        self,
        model: BaseLLMModel,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        if self.max_graph_bs == 0:
            return

        self.attn_backend.init_capture_graph(max_seq_len, self.graph_bs_list)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pool = None
        for bs in sorted(self.graph_bs_list, reverse=True):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)

            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()
            self.graph_map[bs] = graph

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def pad_batch(self, batch: Batch) -> None:
        if self.can_use_cuda_graph(batch):
            padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
            batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        else:
            batch.padded_reqs = batch.reqs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        assert self.buffer is not None
        self.buffer.copy_from(batch)
        self.attn_backend.prepare_for_replay(batch)
        self.graph_map[batch.padded_size].replay()
        return self.buffer.logits[: batch.size]

    def destroy_cuda_graphs(self) -> None:
        self.graph_map.clear()
        self.buffer = None
        gc.collect()
