from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..core import Context, SamplingParams, set_global_ctx
from ..models import ModelConfig, create_model, load_weights
from ..engine.engine import Engine
from ..kvcache import MHAKVCache, KVCacheLayout
from ..scheduler import CacheManager
from ..scheduler.scheduler import Scheduler
from ..scheduler.table import TableManager


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


class LLM:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        self.device = torch.device(kwargs.get("device", "cuda"))
        self.dtype = dtype

        model_path = _resolve_model_path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = ModelConfig.from_hf(hf_config)
        self._num_layers = config.num_layers

        with torch.device("meta"):
            self.model = create_model(model_path, config)

        load_weights(self.model, model_path, self.device, self.dtype)
        self.model.model._rotary_emb.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.num_pages = self._determine_num_pages(config, kwargs.get("memory_ratio", 0.9))
        self.mha_kv_cache = MHAKVCache(
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            head_dim=config.head_dim,
            num_pages=self.num_pages,
            dtype=self.dtype,
            kv_layout=KVCacheLayout.LayerFirst,
            device=self.device,
            page_size=1,
        )
        self.cache_manager = CacheManager(self.device, self.num_pages)
        self.ctx = Context(page_size=1)
        self.ctx.kv_cache = self.mha_kv_cache
        self.ctx.attn_backend = self.model.attn_backend
        set_global_ctx(self.ctx)

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
            max_running_reqs = len(prompts)
        max_running_reqs = max(1, min(max_running_reqs, len(prompts)))
        
        max_total_len = max(
            len(ids) + sp.max_tokens for ids, sp in zip(all_input_ids, params_list)
        )
        page_table = torch.zeros(
            (max_running_reqs, max_total_len), dtype=torch.int32, device=self.device
        )
        self.ctx.page_table = page_table
        table_manager = TableManager(max_running_reqs, page_table)

        scheduler = Scheduler(
            table_manager=table_manager,
            cache_manager=self.cache_manager,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            max_running_reqs=max_running_reqs,
            attn_backend=self.model.attn_backend,
            prefill_token_budget=prefill_token_budget,
        )
        engine = Engine(model=self.model, mha_kv_cache=self.mha_kv_cache)

        for ids, sp in zip(all_input_ids, params_list):
            scheduler.add_request(ids, sp)

        iter_idx = 0
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
