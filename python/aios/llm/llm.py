from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..models import ModelConfig, create_model, load_weights
from ..engine import DynamicKVCache, Sampler
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

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams] | None = None,
        use_kv_cache: bool = True,
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

            kv_cache = DynamicKVCache(self._num_layers) if use_kv_cache else None
            model_input = input_ids

            for _ in range(sp.max_tokens):
                logits = self.model.forward(model_input, kv_cache=kv_cache)
                next_logits = logits[:, -1, :]
                next_token = sampler.sample(next_logits)
                generated = torch.cat([generated, next_token], dim=-1)

                if not sp.ignore_eos and next_token.item() == self.tokenizer.eos_token_id:
                    break

                model_input = next_token if use_kv_cache else generated

            new_token_ids = generated[0][input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": new_token_ids})

        return results
