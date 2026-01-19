"""
Lesson 1: LLM Basics - Code Module

This package contains implementations of core Transformer components:
- attention.py: Scaled dot-product and multi-head attention
- positional_encoding.py: Sinusoidal and RoPE positional encodings
- transformer_block.py: Complete Transformer block with FFN
"""

from .attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    MultiHeadAttention,
    count_parameters,
)

from .positional_encoding import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
)

from .transformer_block import (
    RMSNorm,
    LayerNorm,
    FeedForward,
    TransformerBlock,
    SimpleTransformer,
)

__all__ = [
    # Attention
    "scaled_dot_product_attention",
    "create_causal_mask",
    "MultiHeadAttention",
    "count_parameters",
    # Positional Encoding
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    # Transformer Block
    "RMSNorm",
    "LayerNorm",
    "FeedForward",
    "TransformerBlock",
    "SimpleTransformer",
]
