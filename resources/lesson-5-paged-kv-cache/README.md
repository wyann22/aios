# Lesson 5: Paged / Blocked KV Cache

## Concept
In previous lessons we introduced `torch.cat` (dynamic tensor growth, leading to copying overhead) and learned about contiguous allocation mapping. Pre-allocating max max context lengths for every request statically is computationally wasteful (causing large OOM at scale). 

Modern LLM backends (vLLM, mini-sglang) use a concept introduced as **PagedAttention**. This borrows from the Operating System principle of virtual memory and paging.

Instead of contiguous allocation, KV records are stored in static sized `Blocks` (Pages). The inference engine runs a `BlockManager`. When a request asks for 1 token of generation, it checks if it hasn't filled its current Block. If it has, the manager allocates `1` new block, returning a physical pointer.

The generation attention loop uses an array representing `block_table[logical_token_index] = physical_block_index`. 

## Implementation Details

In `aios`:
- A pool `MHAKVCache` allocates the overall memory layout as `[num_pages, num_layers, num_kv_heads, page_size, head_dim]`.
- A `CacheManager` with a `NaiveCacheManager` manages free slots using integer IDs (`0...num_pages-1`).
- Request state is maintained by `Req` (`cached_len`, `device_len`, `block_table`) in the paged path.
- `Qwen3Attention` writes via `MHAKVCache.store_kv(...)` and gathers history by `req.block_table`.

## Run Exercise

```bash
python resources/lesson-5-paged-kv-cache/run_lesson5.py
```
This script benchmarks the `DynamicKVCache` logic side-by-side with the paged path (`block_table + MHAKVCache`) to visualize how chunked writes and indexing operate seamlessly with basic autoregressive inference.
