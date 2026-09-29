from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..attention import FlashInferBackend
from ..core import Context, Req, SamplingParams, clear_global_ctx, set_global_ctx
from ..distributed import (
    DistributedCommunicator,
    destroy_distributed,
    div_even,
    get_tp_info,
)
from ..kvcache import MHAKVCache
from ..models import ModelConfig, create_model, load_weights
from .graph import GraphRunner, get_free_memory
from .sample import BatchSamplingArgs, Sampler

if TYPE_CHECKING:
    from ..core import Batch


class Engine:
    """Own GPU execution resources, matching mini-sglang's Engine boundary."""

    def __init__(
        self,
        *,
        model_path: str,
        model_config: ModelConfig,
        dtype: torch.dtype,
        device: torch.device,
        max_running_reqs: int,
        memory_ratio: float,
        enable_cuda_graph: bool,
        cuda_graph_bs: list[int] | None,
        cuda_graph_max_bs: int | None,
    ) -> None:
        self.device = device
        torch.cuda.set_device(device)
        self.stream = torch.cuda.Stream(device=device)
        torch.cuda.set_stream(self.stream)
        self.dtype = dtype
        self.model_config = model_config
        self.max_running_reqs = max_running_reqs
        self.tp_info = get_tp_info()
        self._comm = DistributedCommunicator()

        initial_free_memory = self._sync_get_free_memory()
        with torch.device("meta"):
            self.model = create_model(model_path, model_config)
        load_weights(
            self.model,
            model_path,
            device,
            dtype,
            num_kv_heads=model_config.num_kv_heads,
        )
        self.model.model._rotary_emb.set_device(device)

        self.num_pages = self._determine_num_pages(
            initial_free_memory, model_config, memory_ratio
        )
        self.max_seq_len = min(model_config.max_position_embeddings, self.num_pages)
        self.aligned_max_seq_len = _align_up_32(self.max_seq_len)

        self.ctx = Context(page_size=1)
        self.ctx.kv_cache = self.kv_cache = MHAKVCache(
            num_kv_heads=model_config.num_kv_heads,
            num_layers=model_config.num_layers,
            head_dim=model_config.head_dim,
            num_pages=self.num_pages + 1,
            page_size=self.ctx.page_size,
            dtype=dtype,
            device=device,
        )
        self.ctx.page_table = self.page_table = torch.zeros(
            (max_running_reqs + 1, self.aligned_max_seq_len),
            dtype=torch.int32,
            device=device,
        )
        set_global_ctx(self.ctx)
        self.ctx.attn_backend = self.attn_backend = FlashInferBackend(model_config)

        self.sampler = Sampler(device, model_config.vocab_size)

        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32),
            table_idx=max_running_reqs,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=SamplingParams(ignore_eos=True, max_tokens=1),
        )
        self.dummy_page = self.num_pages
        self.reset_page_table()

        self.graph_runner: GraphRunner | None = None
        if enable_cuda_graph:
            self.graph_runner = GraphRunner(
                stream=self.stream,
                device=device,
                model=self.model,
                attn_backend=self.attn_backend,
                cuda_graph_bs=cuda_graph_bs,
                cuda_graph_max_bs=cuda_graph_max_bs,
                free_memory=get_free_memory(device),
                max_seq_len=self.aligned_max_seq_len,
                vocab_size=model_config.vocab_size,
                dummy_req=self.dummy_req,
            )

    def reset_page_table(self) -> None:
        self.page_table[: self.max_running_reqs].zero_()
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)

    def _sync_get_free_memory(self) -> int:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        if self.tp_info.size > 1:
            free_tensor = torch.tensor(
                free_memory, dtype=torch.int64, device=self.device
            )
            torch.distributed.all_reduce(
                free_tensor, op=torch.distributed.ReduceOp.MIN
            )
            free_memory = int(free_tensor.item())
        return free_memory

    def _determine_num_pages(
        self,
        initial_free_memory: int,
        config: ModelConfig,
        memory_ratio: float,
    ) -> int:
        free_after_model = self._sync_get_free_memory()
        model_memory = initial_free_memory - free_after_model
        cache_per_page = (
            2
            * config.head_dim
            * div_even(
                config.num_kv_heads,
                self.tp_info.size,
                allow_replicate=True,
            )
            * self.dtype.itemsize
            * config.num_layers
        )
        available_memory = int(memory_ratio * initial_free_memory) - model_memory
        num_pages = available_memory // cache_per_page
        assert num_pages > 1, (
            "Not enough GPU memory for KV cache: "
            f"available={available_memory}, bytes_per_page={cache_per_page}"
        )
        return num_pages

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs
    ) -> torch.Tensor:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if (
                self.graph_runner is not None
                and self.graph_runner.can_use_cuda_graph(batch)
            ):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        if self.tp_info.is_primary:
            next_tokens = self.sampler.sample(
                logits[: batch.size], args
            ).to(torch.int32)
        else:
            next_tokens = torch.empty(
                batch.size, dtype=torch.int32, device=self.device
            )
        return self._comm.broadcast(next_tokens, src=0)

    def shutdown(self) -> None:
        if self.graph_runner is not None:
            self.graph_runner.destroy_cuda_graphs()
            self.graph_runner = None
        clear_global_ctx()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32
