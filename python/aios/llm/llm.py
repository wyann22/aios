from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..core import SamplingParams
from ..distributed import initialize_distributed
from ..engine.engine import Engine
from ..models import ModelConfig
from ..scheduler import CacheManager
from ..scheduler.scheduler import Scheduler
from ..scheduler.table import TableManager


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


class LLM:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        self.tensor_parallel_size = int(kwargs.get("tensor_parallel_size", 1))
        self.tp_info = initialize_distributed(
            self.tensor_parallel_size,
            timeout_seconds=float(kwargs.get("distributed_timeout", 120.0)),
        )
        if self.tensor_parallel_size > 1:
            self.device = torch.device(f"cuda:{self.tp_info.local_rank}")
        else:
            self.device = _normalize_cuda_device(kwargs.get("device", "cuda"))
        assert self.device.type == "cuda", "AIOS only supports CUDA execution"
        self.dtype = dtype
        self.max_running_reqs = int(kwargs.get("max_running_reqs", 16))
        self.enable_cuda_graph = bool(
            kwargs.get("enable_cuda_graph", kwargs.get("cuda_graph", False))
        )

        model_path = _resolve_model_path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = ModelConfig.from_hf(hf_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        cuda_graph_bs = kwargs.get("cuda_graph_bs")
        if isinstance(cuda_graph_bs, str):
            cuda_graph_bs = [int(item) for item in cuda_graph_bs.split(",") if item]
        if cuda_graph_bs is not None:
            cuda_graph_bs = [bs for bs in cuda_graph_bs if bs <= self.max_running_reqs]
        cuda_graph_max_bs = kwargs.get("cuda_graph_max_bs", self.max_running_reqs)
        if cuda_graph_max_bs is not None:
            cuda_graph_max_bs = min(int(cuda_graph_max_bs), self.max_running_reqs)

        self.engine = Engine(
            model_path=model_path,
            model_config=config,
            dtype=dtype,
            device=self.device,
            max_running_reqs=self.max_running_reqs,
            memory_ratio=float(kwargs.get("memory_ratio", 0.9)),
            enable_cuda_graph=self.enable_cuda_graph,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
        )
        self.stream = self.engine.stream
        self.model = self.engine.model
        self.attn_backend = self.engine.attn_backend
        self.page_table = self.engine.page_table
        self.max_seq_len = self.engine.max_seq_len
        self.graph_runner = self.engine.graph_runner
        self.cache_manager = CacheManager(
            self.engine.num_pages, self.engine.ctx.page_size, self.page_table
        )

    @property
    def is_primary(self) -> bool:
        return self.tp_info.is_primary

    def close(self) -> None:
        self.engine.shutdown()
        self.graph_runner = None

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
        max_running_reqs = max(
            1, min(max_running_reqs, len(prompts), self.max_running_reqs)
        )

        max_total_len = max(
            len(ids) + sp.max_tokens for ids, sp in zip(all_input_ids, params_list)
        )
        if max_total_len > self.max_seq_len:
            raise ValueError(
                f"Requested sequence length {max_total_len} exceeds max_seq_len={self.max_seq_len}"
            )
        self.engine.reset_page_table()
        table_manager = TableManager(max_running_reqs, self.page_table)

        scheduler = Scheduler(
            table_manager=table_manager,
            cache_manager=self.cache_manager,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            attn_backend=self.attn_backend,
            sampler=self.engine.sampler,
            prefill_token_budget=prefill_token_budget,
            graph_runner=self.graph_runner,
        )
        for ids, sp in zip(all_input_ids, params_list):
            scheduler.add_request(ids, sp)

        iter_idx = 0
        with torch.cuda.stream(self.stream):
            while scheduler.has_work:
                forward_input = scheduler.schedule_next_batch()
                if forward_input is None:
                    break
                batch = forward_input.batch
                batch.input_ids = table_manager.token_pool[forward_input.input_tuple]
                next_tokens = self.engine.forward_batch(
                    batch, forward_input.sample_args
                )
                scheduler.process_batch_output(forward_input, next_tokens)
                if debug_scheduler:
                    print(f"[{iter_idx}] {scheduler.debug_state(batch)}")
                iter_idx += 1

        return scheduler.collect_results(self.tokenizer)


def _normalize_cuda_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device
