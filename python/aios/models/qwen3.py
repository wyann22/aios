from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from aios.layers import (
    BaseOP,
    Embedding,
    Linear,
    LMHead,
    OPList,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    silu_and_mul,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self._scale = config.head_dim ** -0.5
        self._layer_idx = layer_idx

        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        kv_cache: Any | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self._layer_idx, k, v)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj.forward(attn_output)


class Qwen3MLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)
        # Dispatch activation by config (like mini-sglang GatedMLP)
        match config.hidden_act:
            case "silu":
                self._act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self._act_fn(self.gate_proj.forward(x), self.up_proj.forward(x)))


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: Any | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([Qwen3DecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, input_ids: torch.Tensor, kv_cache: Any | None = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens.forward(input_ids)

        past_len = kv_cache.get_seq_len() if kv_cache is not None else 0

        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
        ).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self._rotary_emb.forward(position_ids)

        total_kv_len = past_len + seq_len
        q_positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(1)
        k_positions = torch.arange(total_kv_len, device=input_ids.device).unsqueeze(0)
        causal_mask = torch.where(
            k_positions > q_positions,
            torch.tensor(float("-inf"), device=input_ids.device, dtype=hidden_states.dtype),
            torch.tensor(0.0, device=input_ids.device, dtype=hidden_states.dtype),
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers.op_list:
            hidden_states = layer.forward(
                hidden_states,
                causal_mask,
                position_embeddings,
                kv_cache=kv_cache,
            )
        return self.norm.forward(hidden_states)


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self, input_ids: torch.Tensor, kv_cache: Any | None = None) -> torch.Tensor:
        hidden_states = self.model.forward(input_ids, kv_cache=kv_cache)
        return self.lm_head.forward(hidden_states)
