from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from aios.attention import BaseAttentionBackend, FlashInferAttentionBackend
from aios.core import get_global_ctx
from aios.layers import (
    BaseOP,
    Embedding,
    Linear,
    LinearColParallelMerged,
    LinearQKVMerged,
    LMHead,
    OPList,
    RMSNorm,
    RMSNormFused,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    silu_and_mul,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from aios.core import Batch
    from .config import ModelConfig
    from aios.kvcache import MHAKVCache


class Qwen3Attention(BaseOP):
    def __init__(
        self, config: ModelConfig, layer_idx: int, attn_backend: BaseAttentionBackend
    ):
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self._scale = config.head_dim ** -0.5
        self._layer_idx = layer_idx

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = LinearQKVMerged(
            config.hidden_size, self.q_size, self.kv_size
        )
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self._attn_backend = attn_backend

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_cache: MHAKVCache,
        batch: Batch,
    ) -> torch.Tensor:
        total_tokens, _ = hidden_states.shape
        qkv = self.qkv_proj.forward(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self._attn_backend.forward(
            q, k, v, paged_kv_cache, self._layer_idx, batch
        )
        return self.o_proj.forward(attn_output.reshape(q.size(0), -1))


class Qwen3MLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size, config.intermediate_size]
        )
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)
        match config.hidden_act:
            case "silu":
                self._act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        return self.down_proj.forward(self._act_fn(gate_up))


class Qwen3DecoderLayer(BaseOP):
    def __init__(
        self, config: ModelConfig, layer_idx: int, attn_backend: BaseAttentionBackend
    ):
        self.self_attn = Qwen3Attention(config, layer_idx, attn_backend)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_cache: MHAKVCache,
        batch: Batch,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.self_attn.forward(
            hidden_states, position_embeddings, paged_kv_cache, batch
        )
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.attn_backend = FlashInferAttentionBackend(config)
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [
                Qwen3DecoderLayer(config, i, self.attn_backend)
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self._rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        input_ids = ctx.batch.input_ids
        paged_kv_cache = ctx.kv_cache
        batch = ctx.batch
        hidden_states = self.embed_tokens.forward(input_ids)
        position_embeddings = self._rotary_emb.forward(batch.positions)

        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            hidden_states, residual = layer.forward(
                hidden_states, position_embeddings, paged_kv_cache, batch, residual
            )
        return self.norm.forward(hidden_states, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.attn_backend = self.model.attn_backend
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(
        self,
    ) -> torch.Tensor:
        hidden_states = self.model.forward()
        return self.lm_head.forward(hidden_states)
