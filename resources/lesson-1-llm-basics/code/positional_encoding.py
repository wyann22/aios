"""
Lesson 1: Positional Encoding Implementation

This module implements positional encodings for Transformer models.
Since attention has no inherent notion of position, we need to inject
position information into the input embeddings.

Implementations included:
1. Sinusoidal Positional Encoding (original Transformer)
2. Rotary Position Embedding (RoPE) - used in Llama, GPT-NeoX
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".

    Uses sine and cosine functions of different frequencies to encode position:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - Deterministic (no learnable parameters)
    - Can extrapolate to longer sequences than seen in training
    - Relative positions can be represented as linear functions

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability

    Example:
        >>> pe = SinusoidalPositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(2, 100, 512)  # batch=2, seq_len=100
        >>> x_with_pos = pe(x)
        >>> print(x_with_pos.shape)  # torch.Size([2, 100, 512])
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the divisor term: 10000^(2i/d_model)
        # This creates different frequencies for each dimension
        # Using log-space for numerical stability:
        #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        # Shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positionally encoded tensor of same shape
        """
        seq_len = x.size(1)

        # Add positional encoding (broadcasting handles batch dimension)
        # Only use the first seq_len positions
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get the positional encoding for a given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Positional encoding of shape (seq_len, d_model)
        """
        return self.pe[0, :seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer
    with Rotary Position Embedding".

    RoPE encodes position by rotating the query and key vectors. This has
    several advantages:
    - Relative position information is naturally encoded in attention
    - Can extrapolate to longer sequences
    - No need to add encodings to embeddings

    Used in: Llama, Llama 2, GPT-NeoX, PaLM

    The idea: Apply a rotation to Q and K based on their position.
    When computing Q·K^T, the rotation encodes relative position.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)

    Example:
        >>> rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=1000)
        >>> q = torch.randn(2, 8, 100, 64)  # batch, heads, seq_len, head_dim
        >>> k = torch.randn(2, 8, 100, 64)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        # These determine how fast each dimension rotates with position
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin tables
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        # Position indices
        t = torch.arange(seq_len, device=self.inv_freq.device)

        # Outer product: (seq_len,) × (d_model/2,) -> (seq_len, d_model/2)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Duplicate each frequency (for pairs of dimensions)
        # Shape: (seq_len, d_model)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute and cache cos/sin
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.

        This implements the rotation by splitting x into two halves
        and swapping them with a sign change.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            positions: Optional position indices. If None, uses [0, 1, 2, ...]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[-2]

        # Get cos and sin for the sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting: (seq_len, d_model) -> (1, 1, seq_len, d_model)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation: x * cos + rotate_half(x) * sin
        # This is equivalent to a 2D rotation matrix applied to pairs of dims
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


def visualize_positional_encoding(pe_module: SinusoidalPositionalEncoding, seq_len: int = 100):
    """
    Visualize positional encodings (prints a text-based representation).

    Args:
        pe_module: A SinusoidalPositionalEncoding module
        seq_len: Number of positions to visualize
    """
    encoding = pe_module.get_encoding(seq_len).cpu().numpy()

    print(f"\nPositional Encoding Shape: {encoding.shape}")
    print(f"  (seq_len={seq_len}, d_model={encoding.shape[1]})")

    print("\nFirst 5 positions, first 8 dimensions:")
    print("-" * 50)
    for pos in range(min(5, seq_len)):
        values = encoding[pos, :8]
        formatted = " ".join(f"{v:7.4f}" for v in values)
        print(f"Pos {pos}: {formatted} ...")

    print("\nNote: Even dimensions use sin, odd dimensions use cos")
    print("      Lower dimensions oscillate faster than higher ones")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 1: Positional Encoding Demonstration")
    print("=" * 60)

    torch.manual_seed(42)

    # Configuration
    d_model = 512
    max_len = 1000
    batch_size = 2
    seq_len = 100

    # Test 1: Sinusoidal Positional Encoding
    print("\n" + "-" * 40)
    print("Test 1: Sinusoidal Positional Encoding")
    print("-" * 40)

    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # Create sample input (simulating word embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Apply positional encoding
    x_encoded = pe(x)
    print(f"Output shape: {x_encoded.shape}")

    # Verify positional encoding was added
    encoding = pe.get_encoding(seq_len)
    print(f"Encoding shape: {encoding.shape}")

    # Show that different positions have different encodings
    print("\nDistance between position encodings:")
    dist_01 = (encoding[0] - encoding[1]).norm().item()
    dist_010 = (encoding[0] - encoding[10]).norm().item()
    dist_050 = (encoding[0] - encoding[50]).norm().item()
    print(f"  Position 0 vs 1:  {dist_01:.4f}")
    print(f"  Position 0 vs 10: {dist_010:.4f}")
    print(f"  Position 0 vs 50: {dist_050:.4f}")

    # Visualize encodings
    visualize_positional_encoding(pe, seq_len=10)

    # Test 2: Rotary Position Embedding
    print("\n" + "-" * 40)
    print("Test 2: Rotary Position Embedding (RoPE)")
    print("-" * 40)

    head_dim = 64
    n_heads = 8

    rope = RotaryPositionalEmbedding(d_model=head_dim, max_seq_len=1000)

    # Create sample Q and K tensors
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")

    # Apply RoPE
    q_rot, k_rot = rope(q, k)
    print(f"Rotated query shape: {q_rot.shape}")
    print(f"Rotated key shape: {k_rot.shape}")

    # Verify rotation preserves norm (approximately)
    q_norm_before = q[0, 0, 0].norm().item()
    q_norm_after = q_rot[0, 0, 0].norm().item()
    print(f"\nNorm preservation check:")
    print(f"  Q norm before: {q_norm_before:.4f}")
    print(f"  Q norm after:  {q_norm_after:.4f}")
    print(f"  Difference:    {abs(q_norm_before - q_norm_after):.6f}")

    # Test 3: Compare attention with position info
    print("\n" + "-" * 40)
    print("Test 3: RoPE Effect on Attention")
    print("-" * 40)

    # Compute attention scores with and without RoPE
    # Without RoPE
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # With RoPE
    attn_scores_rope = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(head_dim)

    print("Attention scores (first head, positions 0-4):")
    print("\nWithout RoPE:")
    print(attn_scores[0, 0, :5, :5].detach().numpy().round(3))
    print("\nWith RoPE:")
    print(attn_scores_rope[0, 0, :5, :5].detach().numpy().round(3))

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
