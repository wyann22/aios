"""Lesson 10 runner: eager decode versus CUDA graph replay.

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-lesson10 PYTHONPATH=python \
    python resources/lesson-10-cuda-graphs/run_lesson10.py \
      --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
      --cuda-visible-devices 1 --suite all
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from random import randint, seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lesson 10 CUDA graph runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--suite", choices=("bench", "e2e", "all"), default="bench")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--min-prompt-len", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--cuda-graph-bs", type=str, default="1,2,4")
    parser.add_argument("--memory-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


def build_workload(args: argparse.Namespace):
    from aios.core import SamplingParams

    seed(args.seed)
    prompts = [
        [randint(0, 10000) for _ in range(randint(args.min_prompt_len, args.max_prompt_len))]
        for _ in range(args.num_seqs)
    ]
    params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
        for _ in range(args.num_seqs)
    ]
    return prompts, params


def run_generation(llm, name: str, prompts, params, max_running: int) -> tuple[float, float]:
    total_output_tokens = sum(sp.max_tokens for sp in params)
    start = time.time()
    llm.generate(prompts, params, max_running_reqs=max_running)
    elapsed = time.time() - start
    tps = total_output_tokens / elapsed
    print(f"[{name}] output_tokens={total_output_tokens} elapsed={elapsed:.2f}s tps={tps:.2f}")
    return elapsed, tps


def destroy_llm(llm) -> None:
    import torch

    if hasattr(llm, "close"):
        llm.close()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def run_e2e(args: argparse.Namespace) -> None:
    from aios import LLM, SamplingParams

    graph_bs = [int(item) for item in args.cuda_graph_bs.split(",") if item]
    llm = LLM(
        args.model,
        memory_ratio=args.memory_ratio,
        max_running_reqs=args.max_running,
        enable_cuda_graph=True,
        cuda_graph_bs=graph_bs,
    )
    result = llm.generate(
        [[151644, 872, 198]],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
        max_running_reqs=1,
    )
    print(f"[E2E_CUDA_GRAPH] token_ids={result[0]['token_ids']}")
    destroy_llm(llm)


def run_bench(args: argparse.Namespace) -> None:
    import torch

    from aios import LLM

    prompts, params = build_workload(args)
    graph_bs = [int(item) for item in args.cuda_graph_bs.split(",") if item]
    print(
        "Workload: "
        f"num_seqs={args.num_seqs} prompt_len={args.min_prompt_len}..{args.max_prompt_len} "
        f"max_tokens={args.max_tokens} max_running={args.max_running} graph_bs={graph_bs}"
    )

    eager = LLM(
        args.model,
        memory_ratio=args.memory_ratio,
        max_running_reqs=args.max_running,
        enable_cuda_graph=False,
    )
    eager.generate(prompts[:1], params[0], max_running_reqs=1)
    eager_elapsed, eager_tps = run_generation(
        eager, "EAGER_DECODE", prompts, params, args.max_running
    )
    destroy_llm(eager)

    torch.cuda.synchronize()
    capture_start = time.time()
    graph = LLM(
        args.model,
        memory_ratio=args.memory_ratio,
        max_running_reqs=args.max_running,
        enable_cuda_graph=True,
        cuda_graph_bs=graph_bs,
    )
    torch.cuda.synchronize()
    capture_elapsed = time.time() - capture_start
    graph.generate(prompts[:1], params[0], max_running_reqs=1)
    graph_elapsed, graph_tps = run_generation(
        graph, "CUDA_GRAPH", prompts, params, args.max_running
    )
    destroy_llm(graph)

    print(
        "Summary: "
        f"capture_time={capture_elapsed:.2f}s "
        f"speedup={eager_elapsed / graph_elapsed:.2f}x "
        f"eager_tps={eager_tps:.2f} graph_tps={graph_tps:.2f}"
    )


def main() -> None:
    args = parse_args()
    configure_env(args)
    if args.suite in ("e2e", "all"):
        run_e2e(args)
    if args.suite in ("bench", "all"):
        run_bench(args)


if __name__ == "__main__":
    main()
