# Lesson 4: Prefill/Decode Split — Understanding KV Cache

## Objectives

By the end of this lesson you will:

1. Understand the **two phases** of autoregressive LLM inference: **prefill** and **decode**
2. Know *why* naive generation without a KV cache is O(n^2) in total compute
3. Implement a simple **dynamic KV cache** using `list[tuple[Tensor, Tensor]]` per layer (the same strategy used by HuggingFace `DynamicCache`)
4. Observe a **~5x speedup** (from ~5 tok/s to ~25-30 tok/s) on Qwen3-8B

---

## Cache in Computer Science (Quick Introduction)

A **cache** is a smaller, faster storage layer that keeps data likely to be reused,
so we do not recompute or refetch the same thing repeatedly.

Classic examples:
- **CPU cache (L1/L2/L3):** stores recently used memory lines to avoid expensive DRAM access.
- **Web/CDN cache:** stores HTTP responses close to users to avoid repeated backend work.
- **Database cache:** stores hot query results or pages to reduce disk and query cost.

The same first-principles idea applies to LLM inference:
- If old data will be reused, keep it.
- Recompute only what is new.
- Trade some memory for large compute/time savings.

For autoregressive decoding, old token **K/V states** are reused every step, so KV cache is
the direct caching strategy for attention.

---

## Why Do We Need a KV Cache?

### The core problem: redundant computation

Recall from Lesson 1 that attention computes:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

During autoregressive generation, the model produces **one token at a time**. At each step it needs attention over **all previous tokens** to decide what comes next.

**Without a KV cache**, the naive approach is:

```
Step 1:  Process tokens [0]              → generate token 1         (1 token)
Step 2:  Process tokens [0, 1]           → generate token 2         (2 tokens)
Step 3:  Process tokens [0, 1, 2]        → generate token 3         (3 tokens)
...
Step n:  Process tokens [0, 1, ..., n-1] → generate token n         (n tokens)

Total work = 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n^2)
```

For generating 1000 tokens, that means ~500,000 units of redundant computation. The model re-reads the same prompt tokens and re-computes the same K and V projections over and over.

### The insight: K and V don't change for past tokens

Once a token has been processed, its Key and Value vectors are **fixed** (they depend only on the token's position and the model weights). Only the **Query** vector for the *new* token is needed at each step.

This is the key insight behind the KV cache: **cache the K and V tensors from previous steps, and only compute Q, K, V for the new token.**

---

## The Two Phases of LLM Inference

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: PREFILL (prompt processing)                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Input: "What is the capital of France?"                   │    │
│  │         [tok0, tok1, tok2, tok3, tok4, tok5, tok6]         │    │
│  │                                                            │    │
│  │  Process ALL prompt tokens in parallel                     │    │
│  │  → Compute Q, K, V for every prompt position               │    │
│  │  → Save K, V into cache (per layer)                        │    │
│  │  → Return logits for last position → first generated token │    │
│  │                                                            │    │
│  │  Compute: O(T^2) for T prompt tokens (one-time cost)       │    │
│  │  This phase is COMPUTE-BOUND (large matrix multiplies)     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  PHASE 2: DECODE (token-by-token generation)                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Step 1: Input = [new_tok_7]                               │    │
│  │          Compute Q7, K7, V7 for this ONE token             │    │
│  │          Append K7,V7 to cache → cache has [K0..K7, V0..V7]│    │
│  │          Attend Q7 against ALL cached K,V                  │    │
│  │          → next token                                      │    │
│  │                                                            │    │
│  │  Step 2: Input = [new_tok_8]                               │    │
│  │          Compute Q8, K8, V8 for this ONE token             │    │
│  │          Append K8,V8 to cache → cache has [K0..K8, V0..V8]│    │
│  │          Attend Q8 against ALL cached K,V                  │    │
│  │          → next token                                      │    │
│  │                                                            │    │
│  │  Compute per step: O(n) where n = total sequence length    │    │
│  │  This phase is MEMORY-BOUND (loading cached K,V from HBM)  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prefill vs Decode: Key Differences

```
┌───────────────────┬──────────────────────┬──────────────────────────┐
│     Aspect        │      Prefill         │        Decode            │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ Tokens processed  │ All prompt tokens    │ One token per step       │
│                   │ at once              │                          │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ Q, K, V computed  │ For all T positions  │ For 1 position           │
│                   │                      │                          │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ KV cache action   │ Populate (write)     │ Read + append            │
│                   │                      │                          │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ Attention shape   │ (T, T) — full        │ (1, n) — one query row   │
│                   │ matrix               │ against all keys         │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ Bottleneck        │ Compute-bound        │ Memory-bandwidth-bound   │
│                   │ (large GEMMs)        │ (loading KV from HBM)    │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ GPU utilization   │ High (good FLOP/s)   │ Low (small matmuls)      │
│                   │                      │                          │
└───────────────────┴──────────────────────┴──────────────────────────┘
```

---

## Data Flow: With vs Without KV Cache

### WITHOUT KV Cache (Lesson 3 approach)

```
Step 1: [tok0]                    → forward(4 tokens)  → tok1    recompute: 0
Step 2: [tok0, tok1]              → forward(5 tokens)  → tok2    recompute: tok0
Step 3: [tok0, tok1, tok2]        → forward(6 tokens)  → tok3    recompute: tok0,1
...
Step N: [tok0, ..., tokN-1]       → forward(N+3 tokens) → tokN   recompute: tok0..N-2

Total forward calls touching T+N tokens each → O((T+N)^2) total compute
```

### WITH KV Cache

```
PREFILL:
  [tok0, tok1, tok2, tok3]  → forward(4 tokens, populate cache) → tok4

DECODE:
  Step 1: [tok4]  → forward(1 token, read cache of 4, append) → tok5
  Step 2: [tok5]  → forward(1 token, read cache of 5, append) → tok6
  Step 3: [tok6]  → forward(1 token, read cache of 6, append) → tok7
  ...

Total = T^2 (prefill) + N * (T+N) (decode) ≈ O(T^2 + N*T) for N >> T
Much less than O((T+N)^2) without cache!
```

---

## Simple Dynamic KV Cache Implementation

The simplest KV cache is a Python list of tuples — one `(K, V)` pair per layer:

```python
# KV cache: list of (key_cache, value_cache) tuples, one per layer
# key_cache shape:   (batch, num_kv_heads, seq_len_so_far, head_dim)
# value_cache shape: (batch, num_kv_heads, seq_len_so_far, head_dim)

past_key_values: list[tuple[Tensor, Tensor]] = []
```

During attention, the cache is updated with `torch.cat`:

```python
# In the attention forward:
if past_key_value is not None:
    # Concatenate new K,V with cached K,V along the sequence dimension
    k = torch.cat([past_key_value[0], k], dim=2)  # dim=2 is seq_len
    v = torch.cat([past_key_value[1], v], dim=2)

# The new (k, v) now contains ALL keys/values from position 0 to current
present_key_value = (k, v)
```

This is exactly how HuggingFace's `DynamicCache` works internally.

---

## The Problem with `torch.cat`

While `torch.cat` gives us correctness, it has a serious performance issue:

```
Step 1: cache = [K0, V0]                    → cat copies 1 entry
Step 2: cache = [K0, K1, V0, V1]            → cat copies 2 entries
Step 3: cache = [K0, K1, K2, V0, V1, V2]    → cat copies 3 entries
...
Step n: cache = [K0..Kn, V0..Vn]            → cat copies n entries

Total copies = 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n^2)
```

Each `torch.cat` allocates a **new tensor** and copies the entire old cache plus the new entry. For a 36-layer model generating 1000 tokens:

```
Allocations: 36 layers × 2 (K,V) × 1000 steps = 72,000 allocations
Copy volume: 36 × 2 × (1 + 2 + ... + 1000) ≈ 36 million tensor copies
```

This is why we'll replace `torch.cat` with pre-allocated caches in Lesson 5. But for now, `torch.cat` is **correct** and gives us a solid ~5x speedup over no cache at all.

---

## KV Cache Memory Budget

For Qwen3-8B with bfloat16:

```
Per token, per layer:
  K: num_kv_heads × head_dim × 2 bytes = 8 × 128 × 2 = 2,048 bytes
  V: num_kv_heads × head_dim × 2 bytes = 8 × 128 × 2 = 2,048 bytes
  Total: 4,096 bytes per token per layer

For 36 layers:
  4,096 × 36 = 147,456 bytes ≈ 144 KB per token

For 1000 tokens:
  144 KB × 1000 = 144 MB

For max context (40,960 tokens):
  144 KB × 40,960 ≈ 5.76 GB — this is significant GPU memory!
```

This is why KV cache memory management is so important (and why Lessons 5-6 exist).

---

## Running the Demo

```bash
pip install -r resources/lesson-4-kv-cache/requirements.txt

# Run with a model path or HuggingFace model id
python resources/lesson-4-kv-cache/run_lesson4.py --model Qwen/Qwen3-0.6B
```

The script will:
1. Invoke `benchmark/bench.py` with lesson-4 defaults (`smaller input`, `larger output`)
2. Run benchmark **with KV cache**
3. Run benchmark **without KV cache**
4. Print a side-by-side throughput comparison and speedup

Example output:

```
[KV_CACHE] Total: 232tok, Time: 6.78s, Throughput: 34.19tok/s
[NO_CACHE] Total: 232tok, Time: 7.02s, Throughput: 33.07tok/s
Speedup: 1.03x
```

---

## Exercises

### Exercise 1: Benchmark varying prompt lengths

Modify the script to test different prompt lengths (10, 50, 200, 500 tokens). How does the prefill time scale? How does it affect decode speed?

### Exercise 2: Measure cache memory

After generating N tokens, print `torch.cuda.memory_allocated()`. Plot memory vs number of generated tokens. Does it match the formula above?

### Exercise 3: Inspect cache shapes

Add a debug print inside the attention forward that shows the shapes of the cached K and V tensors at each decode step. Verify they grow by exactly 1 in the sequence dimension each step.

### Exercise 4: Count allocations

Use `torch.cuda.memory_stats()` to count the number of allocation calls with and without KV cache. Compare to the theoretical O(n^2) allocation count for `torch.cat`.

---

## What's Next?

The `torch.cat` approach works but creates **memory fragmentation** (thousands of allocations and copies). In **Lesson 5**, we'll replace this with a **pre-allocated KV cache** — a single contiguous buffer that we write into using position indices. This eliminates fragmentation and further improves performance.

---

## Additional Resources

- [The KV Cache Explained (Jay Mody)](https://jaykmody.com/blog/gpt-kv-cache/)
- [Efficient Memory Management for LLM Serving (vLLM paper)](https://arxiv.org/abs/2309.06180)
- [HuggingFace DynamicCache source code](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
