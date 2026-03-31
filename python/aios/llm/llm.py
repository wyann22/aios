from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..models import ModelConfig, create_model, load_weights
from ..engine import DynamicKVCache, Sampler
from ..kvcache import MHAKVCache, KVCacheLayout
from ..scheduler import CacheManager
from ..core import SamplingParams


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace hub ID to a local path if needed."""
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


        # Move rotary embedding cache to device
        self.model.model._rotary_emb.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Paged KV Cache Memory Pool
        self.num_pages = 2048 # ~128MB per token/layer pool
        self.mha_kv_cache = MHAKVCache(
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            head_dim=config.head_dim,
            num_pages=self.num_pages,
            dtype=self.dtype,
            kv_layout=KVCacheLayout.LayerFirst,
            device=self.device,
            page_size=1
        )
        self.cache_manager = CacheManager(self.device, self.num_pages)


    @torch.no_grad()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams] | None = None,
        use_kv_cache: bool = True,
        use_paged_kv_cache: bool = False,
    ) -> List[dict]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Normalize to per-request sampling params list
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            params_list = sampling_params

        results = []
        for prompt, sp in zip(prompts, params_list):
            sampler = Sampler(sp)

            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            else:
                input_ids = torch.tensor([prompt], device=self.device)

            generated = input_ids.clone()

            kv_cache = None
            paged_block_table = None
            paged_seq_len = 0
            if use_paged_kv_cache:
                paged_block_table = self.cache_manager.allocate(input_ids.shape[1])
            elif use_kv_cache:
                kv_cache = DynamicKVCache(self._num_layers)

            model_input = input_ids

            for step in range(sp.max_tokens):
                if use_paged_kv_cache:
                    logits = self.model.forward(
                        model_input,
                        paged_kv_cache=self.mha_kv_cache,
                        paged_block_table=paged_block_table,
                        paged_seq_len=paged_seq_len,
                    )
                else:
                    logits = self.model.forward(model_input, kv_cache=kv_cache)

                next_logits = logits[:, -1, :]
                next_token = sampler.sample(next_logits)
                generated = torch.cat([generated, next_token], dim=-1)

                if not sp.ignore_eos and next_token.item() == self.tokenizer.eos_token_id:
                    break

                if use_paged_kv_cache:
                    paged_seq_len = generated.shape[1] - 1
                    new_block = self.cache_manager.allocate(1)
                    paged_block_table = torch.cat([paged_block_table, new_block])

                model_input = next_token if (kv_cache is not None or use_paged_kv_cache) else generated

            if paged_block_table is not None:
                self.cache_manager._free(paged_block_table)

            new_token_ids = generated[0][input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": new_token_ids})

        return results
