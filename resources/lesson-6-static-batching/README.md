# Lesson 6: Static Batching + Scheduler/Engine Split

Lesson 6 merges the previous Lesson 6/7 split into one runnable milestone.
Starting from Lesson 5 paged KV cache, this lesson upgrades the runtime from
"multi-request but serial forward" to "real batched prefill/decode forward"
with a mini-sglang-style `Scheduler` (dispatch) + `Engine` (execute) split.

## What This Lesson Adds

- A real static-batch paged KV path in `LLM.generate(..., use_static_batch=True)`.
- Mini-sglang-style request data plane:
  - `Req` and `Batch` in `python/aios/core.py`
  - `TableManager` (LIFO slot pool + `page_table/token_pool`) in `python/aios/scheduler/`
- Mini-sglang-style control/compute split for lesson runtime:
  - `Scheduler` in `python/aios/scheduler/scheduler.py` (batch preparation + request state transitions)
  - `Engine` in `python/aios/engine/engine.py` (model forward + sampling)
- Batched paged attention path (`reqs: list[Req]`) in `python/aios/models/qwen3.py`.
- Benchmark switch `--static-batch` in `benchmark/bench.py`.
- Lesson suite runner `run_lesson6.py` now compares baseline vs static batch and prints GPU metrics.

## Current Data Flow

- `LLM._generate_static_batch_paged(...)` orchestrates.
- `Scheduler.init_requests(...)` allocates slots/pages and initializes `Req` states.
- `Scheduler.iter_prefill_batches()` builds prefill batches by prompt length.
- `Engine.run_batch(...)` runs forward + sampling for each batch.
- `Scheduler.process_batch_output(...)` updates request states and allocates decode pages.
- `Scheduler.schedule_decode_batch()` repeats decode until done.
- `Scheduler.finalize_results()` releases pages and request slots.

## Data Structures (Role + Fields)

### `Req` (`python/aios/core.py`)

Role: per-request runtime state used by scheduler/model execution.

| Field | Type | Description |
|---|---|---|
| `input_ids` | `torch.Tensor` (CPU) | Host-side token sequence (prompt + generated tokens). |
| `table_idx` | `int` | Row index in `TableManager.page_table/token_pool`. |
| `cached_len` | `int` | Number of tokens already committed into KV cache. |
| `output_len` | `int` | Max generation length budget for this request. |
| `uid` | `int` | Request ID. |
| `sampling_params` | `SamplingParams` | Per-request sampling config. |
| `block_table` | `torch.Tensor \| None` | Lesson compatibility mapping field. |
| `trace_paged_kv` | `bool` | Enable per-request paged-KV debug logs. |

Derived runtime fields/properties:

| Field/Property | Description |
|---|---|
| `device_len` | Current effective sequence length on device path. |
| `max_device_len` | `len(prompt) + output_len`. |
| `remain_len` | Remaining decode capacity. |
| `extend_len` | Length to extend this round. |
| `can_decode` | Whether decode can continue. |

### `Batch` (`python/aios/core.py`)

Role: one forward batch descriptor (`prefill` or `decode`).

| Field | Type | Description |
|---|---|---|
| `reqs` | `list[Req]` | Requests in current batch. |
| `phase` | `"prefill" \| "decode"` | Batch phase. |
| `input_ids` | `torch.Tensor` | Reserved parity field with mini-sglang. |
| `positions` | `torch.Tensor` | Reserved parity field with mini-sglang. |
| `out_loc` | `torch.Tensor` | Reserved parity field with mini-sglang. |
| `padded_reqs` | `list[Req]` | Reserved parity field with mini-sglang. |
| `attn_metadata` | `Any` | Reserved parity field with mini-sglang. |

### `ScheduledBatch` (`python/aios/scheduler/scheduler.py`)

Role: execution package produced by `Scheduler` and consumed by `Engine`.

| Field | Type | Description |
|---|---|---|
| `batch` | `Batch` | Batch metadata (`reqs` + `phase`). |
| `input_ids` | `torch.Tensor` | Real model input tensor for this forward. |
| `samplers` | `list[Sampler]` | Request-aligned samplers for next-token sampling. |
| `state_indices` | `list[int]` | Mapping back to scheduler state slots for output write-back. |

### `_ReqState` (`python/aios/scheduler/scheduler.py`, internal)

Role: scheduler-private request state for progression.

| Field | Description |
|---|---|
| `req_id` | Request ID. |
| `input_ids` | Initial prompt tensor on device. |
| `generated` | Full running sequence (prompt + generated). |
| `sampler` | Per-request sampler. |
| `sampling_params` | Per-request sampling params. |
| `req` | Backing `Req` object. |
| `generated_steps` | Number of generated tokens. |
| `finished` | Whether request is finished. |
| `model_input` | Next decode input token (typically shape `(1,1)`). |

### `TableManager` (`python/aios/scheduler/table.py`)

Role: centralized manager for request slots and table state.

| Field | Description |
|---|---|
| `_free_slots` | Free request slot pool (LIFO). |
| `page_table` | Mapping `[req_idx, pos] -> physical_page`. |
| `token_pool` | Storage `[req_idx, pos] -> token_id` (currently mostly mirrored state). |
| `max_seq_len` | Max sequence length per request in table. |

Key methods:
- `allocate()/free()`: request slot lifecycle.
- `write_pages()`: update page mapping.
- `write_tokens()`: update token storage.

### `Engine` (`python/aios/engine/engine.py`)

Role: execution layer only (forward + sample).

| Field/Method | Description |
|---|---|
| `model` | Underlying model instance. |
| `paged_kv_cache` | Paged KV cache pool. |
| `run_batch(batch, input_ids, samplers)` | Execute one batch and return next-token tensor. |

## What TableManager Does Right Now

- Active responsibilities:
  - Manages request slots (`allocate/free`).
  - Owns `page_table` for logical-position to physical-page mapping.
  - Owns `token_pool` (currently mainly a mirrored state container in this lesson path).
- Not fully parity yet:
  - Unlike mini-sglang production flow, `token_pool` is not yet the primary input/output routing path.
  - `Req.block_table` is still kept for lesson compatibility.

## Remaining Gaps vs mini-sglang

- Scheduler and Engine are split, but still in-process and called from `LLM.generate` (not multi-process/message-driven workers).
- No chunked prefill and no token-budgeted `PrefillAdder` policy.
- No prefix cache lifecycle (`match/lock/unlock/cache/evict`).
- No production attention backend metadata pipeline (`fa/fi/trtllm`) or CUDA graph replay.
- `Req.block_table` is kept for lesson compatibility, while mini-sglang mainly relies on `table_idx -> page_table`.

## Diagrams

![Static Batch Architecture](./static_batch_architecture.svg)
![Static Batch Prefill](./static_batch_prefill.svg)
![Static Batch Decode](./static_batch_decode.svg)
![Static Batch Scheduler Policy](./static_batch_scheduler_policy.svg)

## Scheduler Policy (Current Lesson 6)

- Figure convention: each request is a horizontal bar, bar length maps to `prompt_len`; same prefill batch uses the same color; in decode, EOS-hit requests are removed round by round.
- `prefill-first`: consume all prefill groups before entering decode loop.
- Prefill grouping: requests are grouped by prompt length for batched prefill forward.
- Decode batching: each step gathers all requests where `unfinished && model_input != None` into `(B_active, 1)`.
- No mixed prefill+decode in one scheduling step in this lesson path.
- No chunked prefill and no token-budgeted clipping (still different from mini-sglang full scheduler).

## Quick Run

Regenerate diagram SVGs from DOT files:

```bash
dot -Tsvg resources/lesson-6-static-batching/static_batch_architecture.dot \
  -o resources/lesson-6-static-batching/static_batch_architecture.svg
dot -Tsvg resources/lesson-6-static-batching/static_batch_prefill.dot \
  -o resources/lesson-6-static-batching/static_batch_prefill.svg
dot -Tsvg resources/lesson-6-static-batching/static_batch_decode.dot \
  -o resources/lesson-6-static-batching/static_batch_decode.svg
dot -Tsvg resources/lesson-6-static-batching/static_batch_scheduler_policy.dot \
  -o resources/lesson-6-static-batching/static_batch_scheduler_policy.svg
```

```bash
python resources/lesson-6-static-batching/run_lesson6.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --suite historical \
  --collect-gpu-metrics
```

Run one custom case:

```bash
python resources/lesson-6-static-batching/run_lesson6.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --suite single \
  --num-seqs 32 \
  --max-input-len 128 \
  --max-output-len 256 \
  --collect-gpu-metrics
```

Default `historical` suite cases:
- A: `num_seqs=8, max_input_len=64, max_output_len=256`
- B: `num_seqs=16, max_input_len=128, max_output_len=256`
- C: `num_seqs=24, max_input_len=128, max_output_len=256`
- D: `num_seqs=32, max_input_len=128, max_output_len=256`

Or run benchmark directly:

```bash
python benchmark/bench.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --num-seqs 16 \
  --max-input-len 64 \
  --max-output-len 256 \
  --paged-kv-cache \
  --static-batch
```

For full Chinese explanation and code walkthrough, see `README_CN.md`.

## Benchmark Records (2026-04-14, Example Snapshot)

These numbers are an example snapshot from lesson iteration.
Use `run_lesson6.py` to regenerate latest results on your machine.

Model: `/data4/home/yan.wang/huggingface/Qwen3-0.6B`  
Device: `CUDA_VISIBLE_DEVICES=0`

### Case A (`num-seqs=2`, `max-input-len=32`, `max-output-len=128`)

- Dynamic KV (no batching):
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 2 --max-input-len 32 --max-output-len 128`
  - `[KV_CACHE] Total: 196tok, Time: 5.30s, Throughput: 36.96tok/s`
- Static batch:
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 2 --max-input-len 32 --max-output-len 128 --paged-kv-cache --static-batch`
  - `[STATIC_BATCH] Total: 196tok, Time: 4.00s, Throughput: 49.05tok/s`
- Speedup: `1.33x`

### Case B (`num-seqs=8`, `max-input-len=64`, `max-output-len=128`)

- Dynamic KV (no batching):
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 8 --max-input-len 64 --max-output-len 128`
  - `[KV_CACHE] Total: 900tok, Time: 32.57s, Throughput: 27.63tok/s`
- Static batch:
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 8 --max-input-len 64 --max-output-len 128 --paged-kv-cache --static-batch`
  - `[STATIC_BATCH] Total: 900tok, Time: 8.35s, Throughput: 107.81tok/s`
- Speedup: `3.90x`

### Case C (`num-seqs=12`, `max-input-len=64`, `max-output-len=128`) + GPU Metrics

- Dynamic KV (no batching):
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 12 --max-input-len 64 --max-output-len 128`
  - `[KV_CACHE] Total: 1096tok, Time: 28.74s, Throughput: 38.13tok/s`
  - `AVG_GPU_UTIL=25.08%`, `AVG_MEM_USED_MIB=10270.45`
- Static batch:
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 12 --max-input-len 64 --max-output-len 128 --paged-kv-cache --static-batch`
  - `[STATIC_BATCH] Total: 1096tok, Time: 9.10s, Throughput: 120.44tok/s`
  - `AVG_GPU_UTIL=19.23%`, `AVG_MEM_USED_MIB=10117.61`
- Speedup: `3.16x`

### Case D (`num-seqs=16`, `max-input-len=64`, `max-output-len=128`)

- Dynamic KV (no batching):
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 16 --max-input-len 64 --max-output-len 128`
  - `[KV_CACHE] Total: 1495tok, Time: 38.62s, Throughput: 38.71tok/s`
- Static batch:
  - `CUDA_VISIBLE_DEVICES=0 python benchmark/bench.py --model /data4/home/yan.wang/huggingface/Qwen3-0.6B --num-seqs 16 --max-input-len 64 --max-output-len 128 --paged-kv-cache --static-batch`
  - `[STATIC_BATCH] Total: 1495tok, Time: 11.39s, Throughput: 131.23tok/s`
- Speedup: `3.39x`
