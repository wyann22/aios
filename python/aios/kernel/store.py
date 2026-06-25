from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_cache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    k_ptr,
    v_ptr,
    k_token_stride,
    v_token_stride,
    cache_token_stride,
    width: tl.constexpr,
    block_size: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    index = tl.load(indices_ptr + token_idx).to(tl.int64)
    mask = offsets < width
    k_values = tl.load(k_ptr + token_idx * k_token_stride + offsets, mask=mask)
    v_values = tl.load(v_ptr + token_idx * v_token_stride + offsets, mask=mask)
    cache_offsets = index * cache_token_stride + offsets
    tl.store(k_cache_ptr + cache_offsets, k_values, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, v_values, mask=mask)


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Scatter a flat batch of K/V vectors into a contiguous CUDA cache."""
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("KV cache must have shape (num_slots, num_heads, head_dim)")
    if k.shape[1:] != k_cache.shape[1:] or v.shape[1:] != v_cache.shape[1:]:
        raise ValueError("K/V input shape does not match the cache layout")
    if indices.numel() != k.shape[0] or k.shape != v.shape:
        raise ValueError("KV write indices and K/V tensors must have matching token counts")
    assert all(tensor.is_cuda for tensor in (k_cache, v_cache, indices, k, v))

    width = k_cache.shape[1] * k_cache.shape[2]
    block_size = triton.next_power_of_2(width)
    _store_cache_kernel[(k.shape[0],)](
        k_cache,
        v_cache,
        indices.contiguous(),
        k,
        v,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        width=width,
        block_size=block_size,
    )
