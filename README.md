# AIOS: Build an LLM Inference Engine from Scratch

Have you ever wondered how ChatGPT generates responses? How a model with billions of parameters actually runs on your GPU? Or why inference optimization matters so much?

AIOS is a hands-on learning project where you build an LLM inference engine from scratch. By the end, you won't just *use* LLMs—you'll understand how inference works under the hood.

## Who This Is For

- **Software engineers** curious about AI/ML systems
- **ML practitioners** who want to understand inference beyond training
- **System engineers** interested in GPU programming and optimization
- **Students** looking for practical, implementable knowledge

## What You'll Build

By completing this project, you will:

- Build a production-grade LLM inference engine from a simple HuggingFace model
- Implement every major optimization: KV cache, paged attention, continuous batching, FlashAttention, CUDA graphs, tensor parallelism
- Go from ~5 tok/s to ~1200+ tok/s — a **240x improvement**
- Serve an OpenAI-compatible API

## The Key Mental Model: LLM Inference Engine as an "Operating System"

Traditional computing has an OS that bridges applications and hardware. AI computing has an inference engine that bridges LLMs and GPUs:

| Operating System | Inference Engine |
|-----------------|------------------|
| Process scheduling | Request batching & scheduling |
| Memory management (virtual memory, paging) | KV cache management (paged attention) |
| I/O scheduling | Prefill/decode scheduling |
| Device drivers | Kernel/operator optimization |
| Multi-core parallelism | Multi-GPU tensor parallelism |

## Performance Progression

Each lesson adds one major optimization. Here's the throughput progression:

| Lesson | Throughput (32 req) | Key Optimization | Multiplier |
|--------|-------------------|------------------|------------|
| 3 | ~5 tok/s | Baseline (no KV cache) | 1x |
| 4 | ~25 tok/s | KV cache reuse | 5x |
| 5 | ~30 tok/s | Pre-allocated cache | 1.2x |
| 6 | ~30 tok/s | Paged cache (memory efficiency) | — |
| 7 | ~400 tok/s | Batching | 13x |
| 8 | ~600 tok/s | Continuous batching | 1.5x |
| 9 | ~900 tok/s | FlashAttention | 1.5x |
| 10 | ~1000 tok/s | Fused layers | 1.1x |
| 11 | ~1200 tok/s | CUDA graphs | 1.2x |
| 12 | ~1200 tok/s | Sampling (quality) | — |
| 13 | ~1200 tok/s + prefix | Prefix caching | prefill savings |
| 14 | ~2000 tok/s (2 GPU) | Tensor parallelism | 1.7x |
| 15 | Production API | Serving layer | — |

## Course Roadmap

### Foundation (Lessons 0–2)

- **[Lesson 0: Introduction](resources/lesson-0-introduction/README.md)** — Traditional software vs LLMs, why inference matters, course objective
- **[Lesson 1: LLM Basics](resources/lesson-1-llm-basics/README.md)** — Tokenizer, Transformer architecture, attention, positional encoding, parameters
- **[Lesson 2: Running Qwen3 with PyTorch](resources/lesson-2-run-qwen3/README.md)** — Loading weights, tokenization, forward pass, generation with HuggingFace

### Building the Engine (Lessons 3–8)

- **[Lesson 3: Remove HuggingFace Dependencies](resources/lesson-3-remove-hf-deps/README.md)** — Own your model: pure nn.Module, direct safetensors loading, manual generation loop
- **[Lesson 4: Prefill/Decode Split](resources/lesson-4-kv-cache/README.md)** — Understanding KV cache: O(n²) → O(n) compute, 5x speedup
- **[Lesson 5: Pre-allocated KV Cache](resources/lesson-5-preallocated-kv-cache/README.md)** — Stop memory fragmentation: contiguous cache, position-indexed writes, fused RMSNorm
- **[Lesson 6: Paged KV Cache](resources/lesson-6-paged-kv-cache/README.md)** — Virtual memory for LLM: block allocator, block tables, slot mapping
- **[Lesson 7: Batching](resources/lesson-7-batching/README.md)** — Variable-length batching with cu_seqlens, the Context pattern, 13x throughput
- **[Lesson 8: The Scheduler](resources/lesson-8-scheduler/README.md)** — Continuous batching, prefill-first scheduling, preemption, engine loop

### Optimization (Lessons 9–12)

- **[Lesson 9: FlashAttention](resources/lesson-9-flash-attention/README.md)** — O(N) memory attention, Triton KV cache kernel, 1.5x speedup
- **[Lesson 10: Fused Layers](resources/lesson-10-fused-layers/README.md)** — QKV fusion, gate+up fusion, smart weight loading, packed_modules_mapping
- **[Lesson 11: CUDA Graphs](resources/lesson-11-cuda-graphs/README.md)** — Capture and replay decode, eliminate CPU launch overhead, 1.2x speedup
- **[Lesson 12: Sampling](resources/lesson-12-sampling/README.md)** — Gumbel-max trick, per-request temperature, top-k/top-p filtering

### Scaling and Serving (Lessons 13–15)

- **[Lesson 13: Prefix Caching](resources/lesson-13-prefix-caching/README.md)** — Hash-chain caching, shared system prompts, ~50x prefill reduction
- **[Lesson 14: Tensor Parallelism](resources/lesson-14-tensor-parallelism/README.md)** — Column/row parallel, NCCL AllReduce, multi-GPU coordination
- **[Lesson 15: API Server & Benchmarking](resources/lesson-15-api-server/README.md)** — OpenAI-compatible API, SSE streaming, throughput/latency benchmarks

## Engine Architecture (Final State)

```
aios/
├── config.py                    # Engine configuration
├── sampling_params.py           # Per-request sampling parameters
├── llm.py                       # User-facing API
├── engine/
│   ├── llm_engine.py            # Orchestrator (model + scheduler + tokenizer)
│   ├── scheduler.py             # Prefill-first continuous batching scheduler
│   ├── model_runner.py          # GPU execution, KV cache, CUDA graphs, TP
│   ├── sequence.py              # Per-request state machine
│   └── block_manager.py         # Paged KV cache + prefix caching
├── models/
│   └── qwen3.py                 # Inference-only Qwen3 (no HF model deps)
├── layers/
│   ├── attention.py             # FlashAttention + Triton KV write
│   ├── linear.py                # TP-aware linear layers
│   ├── layernorm.py             # RMSNorm with fused residual add
│   ├── rotary_embedding.py      # Precomputed RoPE
│   ├── activation.py            # SiluAndMul (fused gate*up)
│   ├── embed_head.py            # Vocab-parallel embedding + LM head
│   └── sampler.py               # Gumbel-max sampling
└── utils/
    ├── loader.py                # Weight loader (safetensors → fused modules)
    └── context.py               # ThreadLocal context for attention metadata
```

## Model Support

Supports all Qwen3 sizes. Exercises are model-size agnostic — config is loaded from `config.json`:

| Model | Parameters | Recommended Use |
|-------|-----------|-----------------|
| Qwen3-0.6B | 0.6B | Development, fast iteration |
| Qwen3-1.7B | 1.7B | Development, basic testing |
| Qwen3-4B | 4B | Testing, light benchmarking |
| Qwen3-8B | 8B | Benchmarking |
| Qwen3-14B | 14B | Benchmarking (needs TP=2) |
| Qwen3-32B | 32B | Benchmarking (needs TP=4) |

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended: A100/H100 for full benchmarks, any GPU for learning)

### Install Dependencies

```bash
pip install torch transformers safetensors flash-attn triton xxhash numpy tqdm
```

### Run the Engine

```bash
# Simple generation
python generate.py --model /path/to/Qwen3-0.6B

# Benchmark throughput
python benchmark.py --model /path/to/Qwen3-0.6B --num-prompts 32

# Using the Python API
python -c "
from aios import LLM, SamplingParams
llm = LLM('/path/to/Qwen3-0.6B')
outputs = llm.generate(['Hello world'], SamplingParams(max_tokens=64))
print(outputs[0]['text'])
"
```

### Follow the Course

Start from the beginning:

1. **[Lesson 0: Introduction](resources/lesson-0-introduction/README.md)**
2. **[Lesson 1: LLM Basics](resources/lesson-1-llm-basics/README.md)**
3. Continue through Lessons 2–15

Each lesson includes:
- **README.md** — Concepts, diagrams, step-by-step guide
- **run_lessonN.py** — Standalone demo script
- **requirements.txt** — Dependencies for this lesson

## References

This project is inspired by:
- **[nano-vllm](https://github.com/some/nano-vllm)** (~1,200 lines) — Minimal vLLM clone with clean architecture
- **[mini-sglang](https://github.com/some/mini-sglang)** (~6,400 lines) — Feature-complete SGLang reference

## License

Educational project. See individual files for details.
