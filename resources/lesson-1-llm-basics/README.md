# Lesson 1: LLM Basics

## Overview

This lesson introduces the fundamental concepts behind Large Language Models (LLMs). You will learn about the Transformer architecture, understand how attention mechanisms work, and explore what model parameters mean in the context of LLMs.

### Learning Objectives

By the end of this lesson, you will be able to:
- Explain the core components of the Transformer architecture
- Understand how self-attention and multi-head attention work
- Describe what model parameters (weights and biases) represent
- Calculate the approximate number of parameters in a Transformer model
- Implement a basic attention mechanism from scratch

## Prerequisites

- Basic Python programming knowledge
- Basic linear algebra (matrix multiplication, vectors)
- Familiarity with neural network concepts (optional but helpful)

## Concepts

### 1. The Transformer Architecture

The Transformer is a neural network architecture introduced in the landmark paper "Attention Is All You Need" (Vaswani et al., 2017). Unlike previous architectures like RNNs and LSTMs that process sequences sequentially, Transformers process all positions in parallel using attention mechanisms.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐           ┌─────────────────┐         │
│  │     ENCODER     │           │     DECODER     │         │
│  │                 │           │                 │         │
│  │  ┌───────────┐  │           │  ┌───────────┐  │         │
│  │  │Multi-Head │  │           │  │  Masked   │  │         │
│  │  │ Attention │  │           │  │ Multi-Head│  │         │
│  │  └─────┬─────┘  │           │  │ Attention │  │         │
│  │        │        │           │  └─────┬─────┘  │         │
│  │  ┌─────▼─────┐  │           │        │        │         │
│  │  │ Add & Norm│  │           │  ┌─────▼─────┐  │         │
│  │  └─────┬─────┘  │           │  │ Add & Norm│  │         │
│  │        │        │           │  └─────┬─────┘  │         │
│  │  ┌─────▼─────┐  │           │        │        │         │
│  │  │Feed Forward│ │  ──────►  │  ┌─────▼─────┐  │         │
│  │  │  Network  │  │           │  │Cross-Attn │  │         │
│  │  └─────┬─────┘  │           │  └─────┬─────┘  │         │
│  │        │        │           │        │        │         │
│  │  ┌─────▼─────┐  │           │  ┌─────▼─────┐  │         │
│  │  │ Add & Norm│  │           │  │Feed Forward│ │         │
│  │  └───────────┘  │           │  └───────────┘  │         │
│  │                 │           │                 │         │
│  │     × N layers  │           │     × N layers  │         │
│  └─────────────────┘           └─────────────────┘         │
│                                                             │
│  ┌─────────────────┐           ┌─────────────────┐         │
│  │   Input         │           │   Output        │         │
│  │   Embedding     │           │   Embedding     │         │
│  │       +         │           │       +         │         │
│  │   Positional    │           │   Positional    │         │
│  │   Encoding      │           │   Encoding      │         │
│  └─────────────────┘           └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Components:

1. **Encoder**: Processes the input sequence and creates representations
2. **Decoder**: Generates output tokens using encoder representations
3. **Multi-Head Attention**: Allows the model to focus on different parts of the input
4. **Feed-Forward Network**: Applies non-linear transformations
5. **Add & Norm**: Residual connections with layer normalization

> **Note**: Modern LLMs like GPT and Llama use a **decoder-only** architecture, which simplifies the original encoder-decoder design for text generation tasks.

### 2. Attention Mechanism

Attention is the core innovation of Transformers. It allows the model to weigh the importance of different parts of the input when processing each position.

#### Self-Attention

Self-attention computes relationships between all positions in a sequence. For each position, it asks: "How relevant is every other position to me?"

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION MECHANISM                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "The cat sat on the mat"                           │
│                                                             │
│  When processing "sat":                                     │
│                                                             │
│  The   cat   sat   on    the   mat                         │
│   │     │     │     │     │     │                          │
│   ▼     ▼     ▼     ▼     ▼     ▼                          │
│  0.1   0.4   1.0   0.1   0.1   0.3   ◄── Attention weights │
│                                                             │
│  "sat" attends most to itself and "cat" (the subject)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Query, Key, Value (QKV)

The attention mechanism uses three learned projections:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

```
┌─────────────────────────────────────────────────────────────┐
│                     QKV COMPUTATION                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Embedding (X)                                        │
│         │                                                   │
│         ├──────► × W_Q ──────► Q (Query)                   │
│         │                                                   │
│         ├──────► × W_K ──────► K (Key)                     │
│         │                                                   │
│         └──────► × W_V ──────► V (Value)                   │
│                                                             │
│  W_Q, W_K, W_V are learned weight matrices                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Scaled Dot-Product Attention

The attention formula:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

Where:
- `Q × K^T`: Dot product measures similarity between queries and keys
- `√d_k`: Scaling factor (d_k = dimension of keys) prevents large values
- `softmax`: Converts scores to probabilities (weights sum to 1)
- `× V`: Weighted sum of values based on attention weights

```
┌─────────────────────────────────────────────────────────────┐
│              SCALED DOT-PRODUCT ATTENTION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      Q          K^T                                         │
│   ┌─────┐    ┌─────┐                                       │
│   │     │    │     │      MatMul                           │
│   │     │ ×  │     │  ──────────►  Attention Scores        │
│   │     │    │     │                    │                  │
│   └─────┘    └─────┘                    │                  │
│                                         ▼                  │
│                                  Scale (÷ √d_k)            │
│                                         │                  │
│                                         ▼                  │
│                                   Softmax                  │
│                                         │                  │
│                                         ▼                  │
│                           Attention Weights × V            │
│                                         │                  │
│                                         ▼                  │
│                                     Output                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Multi-Head Attention

Instead of performing a single attention computation, multi-head attention runs multiple attention operations in parallel ("heads"), each learning different relationships.

```
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-HEAD ATTENTION                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input                                                      │
│    │                                                        │
│    ├────► Head 1: Attention(Q₁, K₁, V₁) ──┐                │
│    │                                       │                │
│    ├────► Head 2: Attention(Q₂, K₂, V₂) ──┼──► Concat ──► Linear │
│    │                                       │                │
│    ├────► Head 3: Attention(Q₃, K₃, V₃) ──┤                │
│    │              ...                      │                │
│    └────► Head h: Attention(Qₕ, Kₕ, Vₕ) ──┘                │
│                                                             │
│  Each head can focus on different aspects:                  │
│  - Head 1: Subject-verb relationships                       │
│  - Head 2: Adjective-noun relationships                     │
│  - Head 3: Long-range dependencies                          │
│  - etc.                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Positional Encoding

Since attention processes all positions in parallel, the model has no inherent sense of order. Positional encodings add position information to the input embeddings.

#### Sinusoidal Positional Encoding (Original Transformer)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos`: Position in the sequence
- `i`: Dimension index
- `d_model`: Embedding dimension

```
┌─────────────────────────────────────────────────────────────┐
│                  POSITIONAL ENCODING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]        │
│  Position 1:  [sin(1/f), cos(1/f), sin(1/f²), cos(1/f²),.] │
│  Position 2:  [sin(2/f), cos(2/f), sin(2/f²), cos(2/f²),.] │
│     ...                                                     │
│                                                             │
│  Token Embedding + Positional Encoding = Final Embedding   │
│                                                             │
│  Modern approaches: RoPE (Rotary Position Embedding)       │
│                     ALiBi (Attention with Linear Biases)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Model Parameters

Parameters are the learned values in a neural network. In LLMs, parameters consist of **weights** and **biases**.

#### Weights

Weights determine the strength of connections between neurons. They are matrices that transform inputs:

```python
output = input @ weight + bias
```

#### Biases

Biases are offset values that shift the activation, allowing the model to fit data better:

```
┌─────────────────────────────────────────────────────────────┐
│                    WEIGHTS AND BIASES                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Linear Layer: y = Wx + b                                   │
│                                                             │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐             │
│  │  Input  │  ×   │ Weight  │  +   │  Bias   │  =  Output  │
│  │   (x)   │      │   (W)   │      │   (b)   │             │
│  │ [1024]  │      │[1024×   │      │ [4096]  │    [4096]   │
│  │         │      │  4096]  │      │         │             │
│  └─────────┘      └─────────┘      └─────────┘             │
│                                                             │
│  Parameters in this layer: 1024 × 4096 + 4096 = 4,198,400  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Parameter Count in Transformers

For a Transformer with:
- `d_model`: Hidden dimension (e.g., 4096)
- `n_layers`: Number of layers (e.g., 32)
- `n_heads`: Number of attention heads (e.g., 32)
- `vocab_size`: Vocabulary size (e.g., 32000)
- `d_ff`: Feed-forward dimension (typically 4 × d_model)

Main parameter sources:
1. **Embedding layer**: `vocab_size × d_model`
2. **Per layer**:
   - QKV projections: `3 × d_model × d_model`
   - Output projection: `d_model × d_model`
   - Feed-forward: `2 × d_model × d_ff`
3. **Output layer**: `d_model × vocab_size`

```
┌─────────────────────────────────────────────────────────────┐
│           LLAMA 2 7B PARAMETER BREAKDOWN                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Configuration:                                             │
│  - d_model = 4096                                           │
│  - n_layers = 32                                            │
│  - n_heads = 32                                             │
│  - vocab_size = 32000                                       │
│  - d_ff = 11008                                             │
│                                                             │
│  Embeddings:      32000 × 4096        =    131,072,000     │
│  Per Layer:                                                 │
│    - Attention:   4 × 4096 × 4096     =     67,108,864     │
│    - FFN:         2 × 4096 × 11008    =     90,177,536     │
│    - LayerNorms:  2 × 4096            =          8,192     │
│                                                             │
│  Total per layer:                     =    157,294,592     │
│  All 32 layers:   32 × 157,294,592    =  5,033,426,944     │
│                                                             │
│  Final LayerNorm: 4096                =          4,096     │
│  Output Layer:    4096 × 32000        =    131,072,000     │
│                                                             │
│  ─────────────────────────────────────────────────────     │
│  TOTAL (approx):                      ≈   6.7 Billion      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. Decoder-Only Architecture (Modern LLMs)

Most modern LLMs (GPT, Llama, Claude) use a decoder-only architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                 DECODER-ONLY ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   DECODER BLOCK                      │   │
│  │                                                      │   │
│  │  Input ──► RMSNorm ──► Masked Multi-Head Attention  │   │
│  │              │                    │                  │   │
│  │              └────── Add ◄────────┘                  │   │
│  │                       │                              │   │
│  │                       ▼                              │   │
│  │              RMSNorm ──► Feed-Forward Network       │   │
│  │                │                    │                │   │
│  │                └────── Add ◄────────┘                │   │
│  │                         │                            │   │
│  │                      Output                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          × N layers                         │
│                                                             │
│  Key difference: Uses CAUSAL (masked) attention            │
│  - Each token can only attend to previous tokens           │
│  - Enables autoregressive text generation                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

See the `code/` directory for implementation examples:

- `attention.py`: Basic attention mechanism implementation
- `transformer_block.py`: Complete Transformer block
- `positional_encoding.py`: Sinusoidal positional encoding

### Quick Start

```python
import torch
from code.attention import scaled_dot_product_attention, MultiHeadAttention

# Example: Scaled dot-product attention
batch_size, seq_len, d_model = 2, 10, 512
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # [2, 10, 512]
print(f"Attention weights shape: {attention_weights.shape}")  # [2, 10, 10]
```

## Exercises

1. **Implement scaled dot-product attention from scratch**
   - Compute Q×K^T, apply scaling, softmax, and multiply by V
   - Verify your implementation matches PyTorch's built-in functions

2. **Visualize attention patterns**
   - Create a simple sentence and compute attention weights
   - Plot the attention matrix as a heatmap
   - Analyze which words attend to which

3. **Calculate parameter counts**
   - Given a model configuration, calculate the total parameters
   - Compare your calculation with Llama 2 7B specifications

4. **Implement causal masking**
   - Modify attention to prevent tokens from attending to future positions
   - This is essential for autoregressive generation

## Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper (Vaswani et al., 2017)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA paper (Touvron et al., 2023)

### Tutorials and Articles
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation by Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Video series

### Documentation
- [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
