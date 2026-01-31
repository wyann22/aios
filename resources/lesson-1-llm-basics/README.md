# Lesson 1: LLM Basics (Tokenizer, Decoder-only Transformer, Attention, Parameters)

This lesson builds the core mental model you’ll use throughout the course:

- **Tokenizer**: text → token IDs (numbers)
- **Decoder-only Transformer**: token IDs → logits over vocabulary
- **Generation**: repeatedly choose the next token ID and decode back to text

Flow (encode → model → decode → autoregressive loop):

```
          ┌───────────────────────┐
Prompt ──►│ Tokenizer (ENCODE)    │──► token_ids[0:T]
  text    └───────────────────────┘
                    │
                    ▼
          ┌───────────────────────┐
          │ Decoder-only          │──► logits[T, vocab]
          │ Transformer           │
          └───────────────────────┘
                    │
                    ▼
          ┌───────────────────────┐
          │ Sampling / Greedy     │──► next_token_id
          └───────────────────────┘
                    │
                    ├────────────── append to input ──────────────┐
                    ▼                                              │
          ┌───────────────────────┐                                │
          │ Tokenizer (DECODE)    │──► text delta (new tokens)      │
          └───────────────────────┘                                │
                    │                                              │
                    └──────────── repeat until <eos>/max_len ◄─────┘
```

You’ll also run small, readable scripts that implement:

- a **toy tokenizer** (subword pieces; from scratch, no external tokenizer library)
- **scaled dot-product attention** and a **minimal Transformer block** in PyTorch

---

## Concepts

### 1) Tokenization: text → tokens → IDs

LLMs do not read characters directly. They read **token IDs**.

Typical tokenization flow:

```
Text ──► (Tokenizer) ──► Tokens (strings) ──► Token IDs (ints)
```

Why tokenization matters:

- **Context window** is measured in **tokens**, not characters.
- **Latency / throughput** is roughly “work per generated token”.
- Token boundaries affect what the model can represent easily (names, code, languages).

#### An intuitive tokenizer explanation (no jargon)

Think of a tokenizer as a **reversible dictionary-based compressor**:

- **Vocabulary**: a fixed list of allowed “pieces” (tokens).
- **Encode**: split text into pieces and map each piece to an integer (**token ID**).
- **Decode**: map token IDs back to pieces and join them back into text.

Why not character-level or word-level?

- **Character-level**: sequences become very long (more tokens), attention gets much more expensive.
- **Word-level**: vocabulary explodes (names, typos, new words, code identifiers), causing OOV problems.

The practical compromise is **subword pieces**: common fragments become single pieces, and rare words are
composed from multiple pieces. This keeps both vocab size and sequence length manageable.

Also, real tokenizers define **special tokens** (e.g. `<bos>`, `<eos>`, `<pad>`, chat templates).

Key takeaway you’ll reuse later: **context length, KV cache size, and throughput are measured in tokens**.

---

### 2) Transformer anatomy (high level)

One Transformer layer typically looks like:

```
x ──► Norm ──► Self-Attention ──► +Residual ──► Norm ──► FFN ──► +Residual
```

Key components:

- **Embedding**: maps token IDs → vectors (shape: `[seq_len, d_model]`)
- **Self-attention**: mixes information across positions using Q/K/V
- **FFN / MLP**: per-token non-linear transform (two linear layers with activation)
- **Residual connections**: stabilize optimization and preserve information
- **Normalization**: LayerNorm or RMSNorm (LLMs often use RMSNorm)

---

### 3) Attention: Q, K, V and causal masking

Given hidden states \(X\) (one vector per token position), the layer produces:

- \(Q = XW_Q\)
- \(K = XW_K\)
- \(V = XW_V\)

Then attention weights (per token) are:

\[
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

For autoregressive generation, we apply a **causal mask** so position *t* cannot
attend to future positions \((t+1, t+2, ...)\).

Compute cost:

- \(QK^T\) is \(O(n^2)\) in sequence length \(n\) (this becomes a major bottleneck).

---

### 4) Where do parameters come from?

If your model width is \(d\):

- Attention projections are roughly \(3d^2\) (Q,K,V) + \(d^2\) (output) = \(4d^2\)
- FFN is roughly \(d \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d\) = \(2dd_{\text{ff}}\)

Rule of thumb: for typical LLMs \(d_{\text{ff}} \approx 4d\), so FFN dominates.

---

## Implementation (Hands-on)

Install dependencies (recommended to use a venv):

```bash
pip install -r resources/lesson-1-llm-basics/requirements.txt
```

### Step 1: Toy tokenizer (subword pieces) from scratch

Run:

```bash
python resources/lesson-1-llm-basics/step1_tokenizer_basics.py
```

What to look for:

- How a tokenizer maps text ↔ token pieces ↔ token IDs
- How frequent patterns become single “pieces”
- How encode/decode works
- How changing vocab size affects tokenization granularity

---

### Step 2: Attention + minimal Transformer block (PyTorch)

Run:

```bash
python resources/lesson-1-llm-basics/step2_attention_and_transformer.py
```

What to look for:

- Shapes for \(Q, K, V\), attention logits, attention weights
- The causal mask behavior
- Parameter count estimates vs actual PyTorch parameter counts

---

## Exercises

- **Exercise 1 (Tokenizer)**: Extend the toy tokenizer to support a small set of **special tokens**
  like `<bos>`, `<eos>`, `<pad>`. Verify decode is stable.

- **Exercise 2 (Attention)**: Add **top-k attention** as a debugging tool:
  keep only the largest-k attention scores per query position before softmax (for interpretability).

- **Exercise 3 (Parameters)**: For the Transformer block in Step 2, compute:
  - total parameters from formulas
  - total parameters from `sum(p.numel() for p in model.parameters())`
  Compare and explain any mismatch (bias terms, LayerNorm weights, etc.).

- **Exercise 4 (Complexity)**: Increase sequence length in Step 2 and time the forward pass.
  Observe the \(O(n^2)\) scaling from attention.

---

## Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)