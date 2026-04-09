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
from ..core import Req, SamplingParams


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
        trace_paged_kv: bool = False,
    ) -> List[dict]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Normalize to per-request sampling params list
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            params_list = sampling_params

        def _short_tensor(tensor: torch.Tensor, max_items: int = 12) -> str:
            values = tensor.detach().cpu().tolist()
            if len(values) <= max_items:
                return str(values)
            head = values[: max_items // 2]
            tail = values[-(max_items - len(head)) :]
            return f"{head} ... {tail} (len={len(values)})"

        if use_paged_kv_cache and trace_paged_kv:
            print(
                "[PagedKV] "
                "init "
                f"free_pages={len(self.cache_manager._free_slots)}"
            )

        results = []
        for req_id, (prompt, sp) in enumerate(zip(prompts, params_list)):
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
            req: Req | None = None
            if use_paged_kv_cache:
                block_table = self.cache_manager.allocate(input_ids.shape[1])
                req = Req(
                    input_ids=input_ids[0],
                    cached_len=0,
                    output_len=sp.max_tokens,
                    uid=req_id,
                    sampling_params=sp,
                    block_table=block_table,
                    trace_paged_kv=trace_paged_kv,
                )
                if trace_paged_kv:
                    print(
                        "[PagedKV] "
                        f"allocate_prompt req={req.uid}, "
                        f"block_table={_short_tensor(req.block_table)}, "
                        f"free_pages={len(self.cache_manager._free_slots)}"
                    )
            elif use_kv_cache:
                kv_cache = DynamicKVCache(self._num_layers)

            model_input = input_ids

            for step in range(sp.max_tokens):
                if use_paged_kv_cache:
                    assert req is not None
                    logits = self.model.forward(
                        model_input,
                        paged_kv_cache=self.mha_kv_cache,
                        req=req,
                    )
                else:
                    logits = self.model.forward(model_input, kv_cache=kv_cache)

                next_logits = logits[:, -1, :]
                next_token = sampler.sample(next_logits)
                generated = torch.cat([generated, next_token], dim=-1)

                hit_eos = (not sp.ignore_eos) and next_token.item() == self.tokenizer.eos_token_id

                if hit_eos:
                    break

                if use_paged_kv_cache:
                    assert req is not None and req.block_table is not None
                    req.complete_one()
                    new_block = self.cache_manager.allocate(1)
                    req.block_table = torch.cat([req.block_table, new_block])
                    if trace_paged_kv:
                        print(
                            "[PagedKV] "
                            f"allocate_decode req={req.uid}, "
                            f"block_table={_short_tensor(req.block_table)}, "
                            f"free_pages={len(self.cache_manager._free_slots)}"
                        )

                model_input = next_token if (kv_cache is not None or use_paged_kv_cache) else generated

            if req is not None and req.block_table is not None:
                free_indices = req.block_table
                self.cache_manager._free(free_indices)
                if trace_paged_kv:
                    print(
                        "[PagedKV] "
                        f"free req={req.uid}, "
                        f"block_table={_short_tensor(free_indices)}, "
                        f"free_pages={len(self.cache_manager._free_slots)}"
                    )

            new_token_ids = generated[0][input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": new_token_ids})

        return results
