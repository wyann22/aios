"""
Lesson 1: Transformer Block Implementation

This module implements a complete Transformer decoder block, the fundamental
building block of modern LLMs like GPT and Llama.

Components:
1. RMSNorm (used in Llama) vs LayerNorm (used in GPT)
2. Feed-Forward Network (FFN) with SwiGLU activation
3. Complete Transformer Block with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from attention import MultiHeadAttention, create_causal_mask


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that only normalizes by the
    root mean square (no mean centering). Used in Llama for efficiency.

    Formula:
        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability

    Example:
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (for comparison with RMSNorm).

    LayerNorm normalizes across the feature dimension, centering around
    the mean and scaling by standard deviation.

    Formula:
        LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability

    Example:
        >>> norm = LayerNorm(512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma
        self.bias = nn.Parameter(torch.zeros(d_model))   # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    The FFN applies two linear transformations with a non-linear activation
    in between. This is applied to each position independently.

    Original Transformer:
        FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Llama uses SwiGLU (Swish-Gated Linear Unit):
        FFN(x) = (Swish(x @ W1) * (x @ W3)) @ W2

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (typically 4 * d_model, or 8/3 * d_model for SwiGLU)
        dropout: Dropout probability
        use_swiglu: Whether to use SwiGLU activation (Llama-style)

    Example:
        >>> ffn = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = True
    ):
        super().__init__()

        # Default hidden dimension
        if d_ff is None:
            # Llama uses 8/3 * d_model (rounded to multiple of 256)
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256

        self.use_swiglu = use_swiglu

        if use_swiglu:
            # SwiGLU: requires 3 weight matrices
            self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
            self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
            self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        else:
            # Standard FFN: 2 weight matrices
            self.w1 = nn.Linear(d_model, d_ff, bias=True)
            self.w2 = nn.Linear(d_ff, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: Swish(x @ W1) * (x @ W3)
            # Swish(x) = x * sigmoid(x), also called SiLU
            return self.dropout(
                self.w2(F.silu(self.w1(x)) * self.w3(x))
            )
        else:
            # Standard: ReLU(x @ W1) @ W2
            return self.dropout(
                self.w2(F.relu(self.w1(x)))
            )


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.

    This implements the core building block of decoder-only LLMs.
    Architecture (Pre-Norm, as used in Llama):

        x ─────┬────> RMSNorm ──> Masked Multi-Head Attention ──┐
               │                                                 │
               └────────────────────── Add <────────────────────┘
                                        │
               ┌────────────────────────┤
               │                        │
               │         ┌──> RMSNorm ──┴──> Feed-Forward ──┐
               │         │                                   │
               └─────────┴────────────── Add <──────────────┘
                                          │
                                       output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        use_swiglu: Use SwiGLU activation in FFN

    Example:
        >>> block = TransformerBlock(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = True
    ):
        super().__init__()

        # Layer normalization (pre-norm architecture)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_swiglu=use_swiglu
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask (causal mask for auto-regressive)

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        # Pre-norm: normalize before attention
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, mask=mask)
        x = x + self.dropout(attn_out)  # Residual connection

        # Feed-forward with residual connection
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)  # Residual connection

        return x


class SimpleTransformer(nn.Module):
    """
    A simple Transformer model for demonstration.

    This combines:
    - Token embedding
    - Positional encoding (simple learned embeddings)
    - Multiple Transformer blocks
    - Output projection to vocabulary

    This is a simplified version to understand the full architecture.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer blocks
        max_seq_len: Maximum sequence length
        dropout: Dropout probability

    Example:
        >>> model = SimpleTransformer(vocab_size=32000, d_model=512, n_heads=8, n_layers=6)
        >>> tokens = torch.randint(0, 32000, (2, 100))
        >>> logits = model(tokens)
        >>> print(logits.shape)  # torch.Size([2, 100, 32000])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding (learned, for simplicity)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(d_model)

        # Output projection (to vocabulary)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        # This is a common technique to reduce parameters and improve performance
        self.output.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape

        # Create position indices
        positions = torch.arange(seq_len, device=tokens.device)

        # Get embeddings
        tok_emb = self.token_embedding(tokens)      # (batch, seq_len, d_model)
        pos_emb = self.position_embedding(positions)  # (seq_len, d_model)

        # Combine token and position embeddings
        x = tok_emb + pos_emb

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len, device=tokens.device)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output(x)

        return logits

    def count_parameters(self) -> dict:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts for each component
        """
        counts = {
            "token_embedding": sum(p.numel() for p in self.token_embedding.parameters()),
            "position_embedding": sum(p.numel() for p in self.position_embedding.parameters()),
            "blocks": sum(p.numel() for p in self.blocks.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
            "output": 0,  # Weight tied with embedding
        }
        counts["total"] = sum(counts.values())
        return counts


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 1: Transformer Block Demonstration")
    print("=" * 60)

    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_layers = 6
    vocab_size = 32000

    # Test 1: RMSNorm vs LayerNorm
    print("\n" + "-" * 40)
    print("Test 1: RMSNorm vs LayerNorm")
    print("-" * 40)

    x = torch.randn(batch_size, seq_len, d_model)
    rms_norm = RMSNorm(d_model)
    layer_norm = LayerNorm(d_model)

    rms_out = rms_norm(x)
    ln_out = layer_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"RMSNorm output shape: {rms_out.shape}")
    print(f"LayerNorm output shape: {ln_out.shape}")

    print(f"\nRMSNorm - mean: {rms_out.mean().item():.6f}, std: {rms_out.std().item():.4f}")
    print(f"LayerNorm - mean: {ln_out.mean().item():.6f}, std: {ln_out.std().item():.4f}")

    # Test 2: Feed-Forward Network
    print("\n" + "-" * 40)
    print("Test 2: Feed-Forward Network (SwiGLU)")
    print("-" * 40)

    ffn = FeedForward(d_model=d_model, use_swiglu=True)
    ffn_out = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {ffn_out.shape}")

    ffn_params = sum(p.numel() for p in ffn.parameters())
    print(f"FFN parameters: {ffn_params:,}")

    # Test 3: Transformer Block
    print("\n" + "-" * 40)
    print("Test 3: Transformer Block")
    print("-" * 40)

    block = TransformerBlock(d_model=d_model, n_heads=n_heads)
    mask = create_causal_mask(seq_len)
    block_out = block(x, mask=mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {block_out.shape}")

    block_params = sum(p.numel() for p in block.parameters())
    print(f"Block parameters: {block_params:,}")

    # Test 4: Complete Transformer
    print("\n" + "-" * 40)
    print("Test 4: Complete Simple Transformer")
    print("-" * 40)

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=2048
    )

    # Create sample input (token IDs)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {tokens.shape}")

    # Forward pass
    logits = model(tokens)
    print(f"Output logits shape: {logits.shape}")

    # Parameter breakdown
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Test 5: Gradient flow
    print("\n" + "-" * 40)
    print("Test 5: Gradient Flow Check")
    print("-" * 40)

    # Simple training step to verify gradients flow
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target.view(-1)
    )
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient flows to embeddings: {model.token_embedding.weight.grad is not None}")
    print(f"Gradient flows to first block: {model.blocks[0].attention.W_q.weight.grad is not None}")

    # Check gradient magnitudes
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    print("\nSample gradient norms:")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
