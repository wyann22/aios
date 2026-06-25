"""Lesson 9 benchmark: unfused operators versus packed/fused operators.

Usage:
    CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH PYTHONPATH=python \
    python resources/lesson-9-fused-layers/run_lesson9.py --e2e \
      --model /data4/home/yan.wang/huggingface/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from aios.kernel import store_cache
from aios.layers import silu_and_mul


def _time_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _report(name: str, baseline_ms: float, fused_ms: float) -> None:
    speedup = baseline_ms / fused_ms if fused_ms else float("inf")
    print(
        f"{name:<20} baseline={baseline_ms:8.3f} ms  "
        f"fused={fused_ms:8.3f} ms  speedup={speedup:5.2f}x"
    )


def benchmark_layers(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_size = args.hidden_size
    kv_size = hidden_size // 2
    intermediate_size = hidden_size * 3
    x = torch.randn(args.tokens, hidden_size, device=device, dtype=dtype)

    q_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    k_weight = torch.randn(kv_size, hidden_size, device=device, dtype=dtype)
    v_weight = torch.randn(kv_size, hidden_size, device=device, dtype=dtype)
    qkv_weight = torch.cat([q_weight, k_weight, v_weight])
    _report(
        "QKV projection",
        _time_cuda(
            lambda: (F.linear(x, q_weight), F.linear(x, k_weight), F.linear(x, v_weight)),
            args.warmup,
            args.iterations,
        ),
        _time_cuda(lambda: F.linear(x, qkv_weight), args.warmup, args.iterations),
    )

    gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    up_weight = torch.randn_like(gate_weight)
    gate_up_weight = torch.cat([gate_weight, up_weight])
    _report(
        "SwiGLU projection",
        _time_cuda(
            lambda: F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight),
            args.warmup,
            args.iterations,
        ),
        _time_cuda(
            lambda: silu_and_mul(F.linear(x, gate_up_weight)),
            args.warmup,
            args.iterations,
        ),
    )

    k = torch.randn(args.tokens, 8, 64, device=device, dtype=dtype)
    v = torch.randn_like(k)
    slots = torch.arange(args.tokens, device=device, dtype=torch.int32)
    k_cache = torch.empty(args.tokens * 2, 8, 64, device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    _report(
        "KV cache store",
        _time_cuda(
            lambda: (
                k_cache.__setitem__(slots, k),
                v_cache.__setitem__(slots, v),
            ),
            args.warmup,
            args.iterations,
        ),
        _time_cuda(
            lambda: store_cache(k_cache, v_cache, slots, k, v),
            args.warmup,
            args.iterations,
        ),
    )


def run_e2e(model_path: str) -> None:
    from aios.core import SamplingParams
    from aios.llm import LLM

    llm = LLM(model_path, memory_ratio=0.2)
    result = llm.generate(
        [[151644, 872, 198]],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
    )
    print(f"E2E generated token ids: {result[0]['token_ids']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 9 fused-layer benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--e2e", action="store_true")
    args = parser.parse_args()
    benchmark_layers(args)
    if args.e2e:
        run_e2e(args.model)


if __name__ == "__main__":
    main()
