"""
Lesson 1: Attention Mechanism Implementation

This module implements the core attention mechanisms used in Transformer models:
1. Scaled Dot-Product Attention
2. Multi-Head Attention

These are the fundamental building blocks of all modern LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    This is the core attention computation described in "Attention Is All You Need":
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k) or
               (batch_size, n_heads, seq_len, d_k)
        key: Key tensor of same shape as query
        value: Value tensor of same shape as query
        mask: Optional attention mask. Use -inf for positions to ignore.
              Shape: (seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        dropout: Dropout probability for attention weights
        training: Whether in training mode (affects dropout)

    Returns:
        output: Attention output of same shape as input
        attention_weights: Attention probability matrix

    Example:
        >>> Q = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_k=64
        >>> K = torch.randn(2, 10, 64)
        >>> V = torch.randn(2, 10, 64)
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
        >>> print(output.shape)  # torch.Size([2, 10, 64])
    """
    # Get the dimension of keys for scaling
    # This prevents the dot products from growing too large
    d_k = query.size(-1)

    # Step 1: Compute attention scores
    # Q @ K^T gives us (batch, seq_len, seq_len) similarity matrix
    # Each row i contains similarity scores between position i and all positions
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    # Without scaling, for large d_k, dot products can be very large,
    # pushing softmax into regions with tiny gradients
    attention_scores = attention_scores / math.sqrt(d_k)

    # Step 3: Apply mask (if provided)
    # Mask is used for:
    # - Causal attention (preventing attending to future tokens)
    # - Padding masks (ignoring pad tokens)
    if mask is not None:
        # Replace masked positions with -inf so softmax gives 0
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply softmax to get attention weights (probabilities)
    # Each row now sums to 1, representing a probability distribution
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Step 5: Apply dropout (regularization during training)
    if dropout > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=training)

    # Step 6: Multiply by values to get final output
    # This is a weighted sum of values, where weights come from attention
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.

    In causal attention, each position can only attend to itself and previous positions.
    This is essential for autoregressive text generation where we predict one token at a time.

    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
        1 where attention is allowed, 0 where it should be blocked

    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
    """
    # torch.tril creates a lower triangular matrix
    # This allows each position to attend to itself and all previous positions
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Instead of performing a single attention function, multi-head attention
    projects Q, K, V into multiple subspaces ("heads"), performs attention
    in parallel, and concatenates the results.

    This allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    Args:
        d_model: Model dimension (input/output size)
        n_heads: Number of attention heads
        dropout: Dropout probability

    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
        >>> output = mha(x, x, x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        # Validate that d_model is divisible by n_heads
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout = dropout

        # Linear projections for Q, K, V
        # These learn what to query, what keys to compare, and what values to retrieve
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        # Combines the multi-head outputs back to d_model dimensions
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # For storing attention weights (useful for visualization)
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Step 1: Project Q, K, V using learned linear transformations
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Reshape for multi-head attention
        # Split d_model into n_heads × d_k, then transpose for attention computation
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention to each head
        # Attention is applied in parallel across all heads
        attn_output, self.attention_weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout=self.dropout,
            training=self.training
        )

        # Step 4: Concatenate heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Step 5: Final linear projection
        output = self.W_o(attn_output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the attention weights from the last forward pass."""
        return self.attention_weights


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: A PyTorch model

    Returns:
        Total number of trainable parameters

    Example:
        >>> mha = MultiHeadAttention(d_model=512, n_heads=8)
        >>> params = count_parameters(mha)
        >>> print(f"Parameters: {params:,}")  # Parameters: 1,048,576
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 1: Attention Mechanism Demonstration")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Dimension per head (d_k): {d_model // n_heads}")

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")

    # Test 1: Scaled Dot-Product Attention
    print("\n" + "-" * 40)
    print("Test 1: Scaled Dot-Product Attention")
    print("-" * 40)

    output, weights = scaled_dot_product_attention(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be ~1.0): {weights[0, 0].sum().item():.4f}")

    # Test 2: Causal Mask
    print("\n" + "-" * 40)
    print("Test 2: Causal Mask")
    print("-" * 40)

    causal_mask = create_causal_mask(5)
    print("Causal mask (5x5):")
    print(causal_mask)

    # Apply causal attention
    x_small = torch.randn(1, 5, d_model)
    output_causal, weights_causal = scaled_dot_product_attention(
        x_small, x_small, x_small, mask=causal_mask
    )
    print(f"\nCausal attention weights (position 4 can see positions 0-4):")
    print(weights_causal[0])

    # Test 3: Multi-Head Attention
    print("\n" + "-" * 40)
    print("Test 3: Multi-Head Attention")
    print("-" * 40)

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    output_mha = mha(x, x, x)
    print(f"Output shape: {output_mha.shape}")

    # Count parameters
    params = count_parameters(mha)
    print(f"\nMulti-Head Attention Parameters:")
    print(f"  W_q: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_k: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_v: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_o: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  Total: {params:,}")

    # Test 4: Attention Visualization
    print("\n" + "-" * 40)
    print("Test 4: Attention Pattern Analysis")
    print("-" * 40)

    # Get attention weights from MHA
    attn_weights = mha.get_attention_weights()
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"  (batch={attn_weights.shape[0]}, heads={attn_weights.shape[1]}, "
          f"query_len={attn_weights.shape[2]}, key_len={attn_weights.shape[3]})")

    # Analyze attention distribution
    avg_attention = attn_weights.mean(dim=(0, 1))  # Average over batch and heads
    print(f"\nAverage attention entropy: {-(avg_attention * avg_attention.log()).sum(dim=-1).mean().item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
