# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

AIOS is a hands-on course for building an LLM inference engine from scratch (16 lessons, 0–15). Students progressively implement a Qwen3-based inference framework, going from HuggingFace usage to a production-grade engine (~240x speedup). The course language is **Chinese** (中文) for README/slides, English for code comments. The target model is **Qwen3** (all sizes from 0.6B to 32B).

## Project Structure

```
aios/                                    # Final inference engine (target of Lessons 3–15)
├── config.py                            # EngineConfig dataclass
├── sampling_params.py                   # Per-request sampling params
├── llm.py                               # User-facing API (LLM class)
├── engine/
│   ├── llm_engine.py                    # Orchestrator
│   ├── scheduler.py                     # Prefill-first continuous batching
│   ├── model_runner.py                  # GPU execution, KV cache, CUDA graphs
│   ├── sequence.py                      # Per-request state machine
│   └── block_manager.py                # Paged KV cache + prefix caching
├── models/
│   └── qwen3.py                         # Inference-only Qwen3 (flat tokens, fused QKV, FlashAttn)
├── layers/
│   ├── attention.py                     # FlashAttention + Triton KV write kernel
│   ├── linear.py                        # TP-aware (Column/Row/QKV/Merged parallel)
│   ├── layernorm.py                     # RMSNorm with fused residual add
│   ├── rotary_embedding.py              # Precomputed RoPE with LRU cache
│   ├── activation.py                    # SiluAndMul (fused gate*up)
│   ├── embed_head.py                    # VocabParallelEmbedding + ParallelLMHead
│   └── sampler.py                       # Gumbel-max sampling
└── utils/
    ├── loader.py                        # Safetensors weight loader (packed_modules_mapping)
    └── context.py                       # ThreadLocal for attention metadata

resources/                               # Course lesson materials
├── lesson-0-introduction/               # Intro, download Qwen3 code, manual weight loading
├── lesson-1-llm-basics/                 # HF usage, manual Qwen3 implementation (855 lines)
│   ├── qwen3_model/                     # HF source: modeling_qwen3.py (553 lines) + config
│   ├── run_hg_qwen3.py                  # HF pipeline usage
│   └── run_manual_qwen3.py              # Complete manual Qwen3 with weight loading + generation
├── lesson-2-run-qwen3/                  # CURRENT: Patch HF→torch-only (7 step scripts)
├── lesson-4-kv-cache/                   # KV cache (Lesson 3 was merged into Lesson 2)
├── lesson-6-paged-kv-cache/
├── lesson-7-batching/
├── lesson-8-scheduler/
├── lesson-9-flash-attention/
├── lesson-10-fused-layers/
├── lesson-11-cuda-graphs/
├── lesson-12-sampling/
├── lesson-13-prefix-caching/
└── lession-2-run-qwen3-src/             # (typo intentional) Original HF source + hg_config.json

benchmark.py                             # Throughput benchmark script
generate.py                              # Simple generation script
```

## Current Work State

### Completed Lessons
- **Lesson 0**: Done (4 step scripts, README, slides)
- **Lesson 1**: Done (run_hg_qwen3.py, run_manual_qwen3.py, README, slides)

### In Progress: Lesson 2 — Patch HF modeling_qwen3.py to Torch-Only
**Approach**: Take the HF `modeling_qwen3.py` (553 lines, 20+ HF dependencies) and progressively patch it into a standalone torch-only model. 12 logical steps grouped into 7 script files.

**Deliverables** (in `resources/lesson-2-run-qwen3/`):
| File | Content |
|------|---------|
| `modeling_qwen3_original.py` | Unmodified HF source (reference copy) |
| `step01_remove_unused_classes.py` | Steps 1-3: delete unused classes, replace ACT2FN→SiLU, remove kernel decorators |
| `step02_replace_base_classes.py` | Steps 4-5: nn.Module base, tuple returns, remove decorators |
| `step03_local_config.py` | Step 6: local Qwen3Config dataclass with `from_json()` |
| `step04_simplify_rope_attention.py` | Steps 7-9: simple RoPE, eager-only attention, inline causal mask |
| `step05_remove_cache.py` | Step 10: remove KV cache entirely (full recompute per step) |
| `step06_weight_loading.py` | Step 11: safetensors weight loading |
| `step07_generation.py` | Step 12: generation loop + end-to-end demo (no KV cache) |
| `run_lesson2.py` | Combines everything, runs end-to-end |

**Key design decision**: NO KV cache in Lesson 2. Each generation step recomputes the full sequence. KV cache is Lesson 4's topic.

**HF dependencies being removed** (20+ items in 5 difficulty tiers):
- Trivial: task-specific classes (QA, SeqCls, TokenCls)
- Easy: `ACT2FN`→`nn.SiLU()`, kernel decorators (delete)
- Medium: `PreTrainedModel`/`GradientCheckpointingLayer`→`nn.Module`, output dataclasses→tuples
- Hard: `Qwen3Config`→local dataclass, `ROPE_INIT_FUNCTIONS`→inline, attention dispatch→eager only
- Complex: `Cache`/`DynamicCache`→removed entirely, `create_causal_mask`→inline

### Pending Lessons (3–15)
Lesson materials for 3-15 mostly have placeholder READMEs. The `aios/` engine code is already scaffolded. Course roadmap is in main `README.md`.

## Key Reference Files

- **HF modeling_qwen3.py**: `resources/lesson-1-llm-basics/qwen3_model/modeling_qwen3.py` (553 lines, THE source file for Lesson 2 patches)
- **HF config.json**: `resources/lession-2-run-qwen3-src/qwen3_model/hg_config.json` (Qwen3-8B defaults)
- **Manual Qwen3 reference**: `resources/lesson-1-llm-basics/run_manual_qwen3.py` (855 lines, complete manual implementation with Chinese comments)
- **Final engine model**: `aios/models/qwen3.py` (flat token repr, fused QKV, FlashAttn — Lesson 3+ target)
- **Active plan file**: `/home/yan.wang/.claude/plans/snuggly-exploring-diffie.md` (12-step patch plan)

## Qwen3 Architecture Details

- **Normalization**: RMSNorm (pre-norm in each sublayer)
- **Activation**: SiLU (Swish) in SwiGLU MLP: `down(silu(gate(x)) * up(x))`
- **Attention**: Grouped Query Attention (GQA), QK-Norm (RMSNorm on Q/K heads)
- **Position encoding**: RoPE (Rotary Position Embedding), theta=1000000
- **Config** (Qwen3-8B): hidden=4096, intermediate=14336, layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936
- **Weight format**: safetensors, keys prefixed with `model.` (e.g., `model.layers.0.self_attn.q_proj.weight`)

## Development Conventions

- Course READMEs and slides in **Chinese** (中文)
- Code comments in English
- Each step script is **self-contained** (full model code + test/demo at bottom)
- Use `if __name__ == "__main__"` for runnable demos
- Weight paths via argparse `--model /path/to/Qwen3-0.6B`
- Tokenizer loaded from HF (`AutoTokenizer.from_pretrained`)
- Each script has a `# === STEP N: <title> ===` header block explaining what changed

## mini-sglang Alignment Requirement

For Lessons 3-15 and the `python/aios/` inference engine, the default requirement is to follow the mini-sglang reference implementation as closely as possible.

- Before changing scheduler, engine, attention backend, KV cache, batch metadata, request lifecycle, sampling, CUDA graph, or related inference-engine code, inspect the corresponding mini-sglang reference files under `.claude/skills/course/references/mini-sglang/`.
- Match mini-sglang's class names, method names, field names, field semantics, data ownership, call order, and lifecycle whenever feasible.
- Do not introduce teaching simplifications, renamed concepts, moved responsibilities, or alternate data flow when mini-sglang already has a clear pattern, unless the user explicitly approves the deviation.
- If a deviation is unavoidable, state it clearly before implementation, explain why exact alignment is not feasible, and document the difference in the lesson README/TECHNICAL docs.
- When the user asks whether the implementation matches mini-sglang, perform a code-level comparison against the reference and fix all avoidable differences.
- In particular, attention metadata, `Batch` fields, `padded_reqs`, page table usage, KV cache indexing, prefill/decode split, FlashInfer wrapper usage, and CUDA graph padding should preserve mini-sglang semantics and call timing.

## Environment

- Python 3.10+, PyTorch 2.0+, CUDA GPU
- Key packages: `torch`, `safetensors`, `transformers` (tokenizer only), `flash-attn`, `triton`
- Model paths on this machine are typically under `/data4/` or similar

## Testing Conventions

- GPU / CUDA / FlashInfer end-to-end tests must be run outside the sandbox; the sandbox may hide CUDA devices and produce false failures such as `RuntimeError: No CUDA GPUs are available`.
- Prefer an idle GPU with `CUDA_VISIBLE_DEVICES=<id>`; GPU 1 is often free on this machine.
- For FlashInfer JIT tests, set CUDA explicitly to avoid a malformed `nvcc` path:
  `CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH`.
- Use a fresh or known-good FlashInfer cache directory when debugging JIT issues, for example:
  `FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-e2e`.
- Minimal end-to-end benchmark example:
  `CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-e2e PYTHONPATH=python python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 2 --max-input-len 40 --max-output-len 8 --max-running-reqs 2`


## /make-slide
When the user types "/make-slide", read `.claude/skills/make-slide/SKILL.md` and follow the presentation creation workflow. Browse themes at https://make-slide.vercel.app
