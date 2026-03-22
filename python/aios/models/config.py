from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, config) -> ModelConfig:
        """Load from a transformers config object (aligned with mini-sglang)."""
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 1000000.0),
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        )

    @classmethod
    def from_json(cls, model_path: str) -> ModelConfig:
        """Load from config.json file (fallback)."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            num_layers=data["num_hidden_layers"],
            num_qo_heads=data["num_attention_heads"],
            num_kv_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            intermediate_size=data["intermediate_size"],
            hidden_act=data.get("hidden_act", "silu"),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 1000000.0),
            max_position_embeddings=data.get("max_position_embeddings", 32768),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )
