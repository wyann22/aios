from __future__ import annotations

from typing import TYPE_CHECKING

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
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj.forward(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj.forward(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj.forward(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(hidden_states, position_embeddings, attention_mask)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        hidden_states = self.embed_tokens.forward(input_ids)

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self._rotary_emb.forward(position_ids)

        causal_mask = torch.full(
            (S, S), float("-inf"), device=input_ids.device, dtype=hidden_states.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers.op_list:
            hidden_states = layer.forward(hidden_states, causal_mask, position_embeddings)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.forward(input_ids)
        return self.lm_head.forward(hidden_states)
