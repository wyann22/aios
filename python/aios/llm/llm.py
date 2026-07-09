from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..core import Context, Req, SamplingParams, clear_global_ctx, set_global_ctx
from ..models import ModelConfig, create_model, load_weights
from ..engine.engine import Engine
from ..engine.graph import GraphRunner, get_free_memory
from ..kvcache import MHAKVCache
from ..scheduler import CacheManager
from ..scheduler.scheduler import Scheduler
from ..scheduler.table import TableManager


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


class LLM:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        self.device = _normalize_cuda_device(kwargs.get("device", "cuda"))
        assert self.device.type == "cuda", "AIOS only supports CUDA execution"
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.dtype = dtype
        self.max_running_reqs = int(kwargs.get("max_running_reqs", 16))
        self.enable_cuda_graph = bool(
            kwargs.get("enable_cuda_graph", kwargs.get("cuda_graph", False))
        )

        model_path = _resolve_model_path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = ModelConfig.from_hf(hf_config)
        self._num_layers = config.num_layers
        self._vocab_size = config.vocab_size

        with torch.device("meta"):
            self.model = create_model(model_path, config)

        load_weights(self.model, model_path, self.device, self.dtype)
        self.model.model._rotary_emb.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.num_pages = self._determine_num_pages(config, kwargs.get("memory_ratio", 0.9))
        self.max_seq_len = min(config.max_position_embeddings, self.num_pages)
        self.aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.mha_kv_cache = MHAKVCache(
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            head_dim=config.head_dim,
            num_pages=self.num_pages + 1,
            page_size=1,
            dtype=self.dtype,
            device=self.device,
        )
        self.cache_manager = CacheManager(self.device, self.num_pages)
        self.ctx = Context(page_size=1)
        self.ctx.kv_cache = self.mha_kv_cache
        self.ctx.attn_backend = self.model.attn_backend
        self.page_table = torch.zeros(
            (self.max_running_reqs + 1, self.aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        self.dummy_table_idx = self.max_running_reqs
        self.dummy_page = self.num_pages
        self._reset_page_table()
        self.ctx.page_table = self.page_table
        set_global_ctx(self.ctx)
        self.graph_runner = self._init_graph_runner(config, kwargs)

    def _init_graph_runner(
        self, config: ModelConfig, kwargs: dict
    ) -> GraphRunner | None:
        if not self.enable_cuda_graph:
            return None

        cuda_graph_bs = kwargs.get("cuda_graph_bs")
        if isinstance(cuda_graph_bs, str):
            cuda_graph_bs = [int(item) for item in cuda_graph_bs.split(",") if item]
        if cuda_graph_bs is not None:
            cuda_graph_bs = [bs for bs in cuda_graph_bs if bs <= self.max_running_reqs]
        cuda_graph_max_bs = kwargs.get("cuda_graph_max_bs", self.max_running_reqs)
        if cuda_graph_max_bs is not None:
            cuda_graph_max_bs = min(int(cuda_graph_max_bs), self.max_running_reqs)
        free_memory = get_free_memory(self.device)

        dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32),
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=SamplingParams(ignore_eos=True, max_tokens=1),
            table_idx=self.dummy_table_idx,
        )
        return GraphRunner(
            model=self.model,
            attn_backend=self.model.attn_backend,
            stream=self.stream,
            device=self.device,
            vocab_size=config.vocab_size,
            max_seq_len=self.aligned_max_seq_len,
            dummy_req=dummy_req,
            free_memory=free_memory,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
        )

    def close(self) -> None:
        if self.graph_runner is not None:
            self.graph_runner.destroy_cuda_graphs()
            self.graph_runner = None
        clear_global_ctx()

    def _reset_page_table(self) -> None:
        self.page_table[: self.max_running_reqs].zero_()
        self.page_table[self.dummy_table_idx].fill_(self.dummy_page)

    def _determine_num_pages(self, config: ModelConfig, memory_ratio: float) -> int:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info(self.device)[0]
        cache_per_page = (
            2 * config.head_dim * config.num_kv_heads * 1 * self.dtype.itemsize * config.num_layers
        )
        available_memory = int(memory_ratio * free_memory)
        num_pages = available_memory // cache_per_page
        assert num_pages > 1, f"Not enough GPU memory for KV cache (free={free_memory}, per_page={cache_per_page})"
        return num_pages

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams] | None = None,
        max_running_reqs: int | None = None,
        prefill_token_budget: int | None = None,
        debug_scheduler: bool = False,
    ) -> List[dict]:
        """Continuous-batching generation with flat varlen prefill (lesson 8)."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            params_list = sampling_params

        all_input_ids: List[torch.Tensor] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            else:
                ids = torch.tensor(prompt)
            all_input_ids.append(ids)

        if max_running_reqs is None:
            max_running_reqs = min(len(prompts), self.max_running_reqs)
        max_running_reqs = max(1, min(max_running_reqs, len(prompts), self.max_running_reqs))
        
        max_total_len = max(
            len(ids) + sp.max_tokens for ids, sp in zip(all_input_ids, params_list)
        )
        if max_total_len > self.max_seq_len:
            raise ValueError(
                f"Requested sequence length {max_total_len} exceeds max_seq_len={self.max_seq_len}"
            )
        self._reset_page_table()
        table_manager = TableManager(max_running_reqs, self.page_table)

        scheduler = Scheduler(
            table_manager=table_manager,
            cache_manager=self.cache_manager,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            max_running_reqs=max_running_reqs,
            attn_backend=self.model.attn_backend,
            prefill_token_budget=prefill_token_budget,
            graph_runner=self.graph_runner,
        )
        engine = Engine(
            model=self.model,
            mha_kv_cache=self.mha_kv_cache,
            graph_runner=self.graph_runner,
            stream=self.stream,
        )

        for ids, sp in zip(all_input_ids, params_list):
            scheduler.add_request(ids, sp)

        iter_idx = 0
        with torch.cuda.stream(self.stream):
            while scheduler.has_work:
                batch = scheduler.schedule_next_batch()
                if batch is None:
                    break
                next_tokens = engine.run_batch(batch)
                scheduler.process_batch_output(batch, next_tokens)
                if debug_scheduler:
                    print(f"[{iter_idx}] {scheduler.debug_state(batch)}")
                iter_idx += 1

        return scheduler.collect_results(self.tokenizer)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _normalize_cuda_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device
