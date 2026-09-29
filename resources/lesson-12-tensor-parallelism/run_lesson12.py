"""Lesson 12 runner: single-GPU baseline versus tensor parallel inference.

Examples:
    python resources/lesson-12-tensor-parallelism/run_lesson12.py --suite check

    torchrun --standalone --nproc-per-node=2 \
      resources/lesson-12-tensor-parallelism/run_lesson12.py \
      --suite e2e --tp-size 2 --model /path/to/Qwen3-0.6B

    python resources/lesson-12-tensor-parallelism/run_lesson12.py \
      --suite compare --tp-size 2 --model /path/to/Qwen3-0.6B \
      --cuda-visible-devices 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lesson 12 tensor parallel runner")
    parser.add_argument("--suite", choices=("check", "e2e", "bench", "compare"), default="check")
    parser.add_argument("--model", default="/data4/home/yan.wang/huggingface/Qwen3-0.6B")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--memory-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


def run_shape_checks() -> None:
    from aios.distributed.info import reset_tp_info, set_tp_info
    from aios.layers import (
        LinearColParallelMerged,
        LinearOProj,
        LinearQKVMerged,
        LinearRowParallel,
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from aios.models.weight import _shard_tensor

    try:
        for rank in (0, 1):
            set_tp_info(rank=rank, size=2, local_rank=rank)
            assert LinearQKVMerged(16, 2, 8, 4).weight.shape == (16, 16)
            assert LinearColParallelMerged(16, [12, 12]).weight.shape == (12, 16)
            assert LinearRowParallel(12, 16).weight.shape == (16, 6)
            assert LinearOProj(16, 16).weight.shape == (16, 8)
            assert VocabParallelEmbedding(17, 8).weight.shape == (9, 8)
            assert ParallelLMHead(17, 8).weight.shape == (9, 8)

        full = torch.arange(8 * 6).reshape(8, 6)
        col_shards = [
            _shard_tensor(
                "model.q_proj.weight",
                full,
                rank=rank,
                world_size=2,
                num_kv_heads=2,
            )
            for rank in (0, 1)
        ]
        row_shards = [
            _shard_tensor(
                "model.o_proj.weight",
                full,
                rank=rank,
                world_size=2,
                num_kv_heads=2,
            )
            for rank in (0, 1)
        ]
        assert torch.equal(torch.cat(col_shards, dim=0), full)
        assert torch.equal(torch.cat(row_shards, dim=1), full)

        kv_weight = torch.arange(8 * 2 * 4).reshape(16, 4)
        replicated_kv = [
            _shard_tensor(
                "model.k_proj.weight",
                kv_weight,
                rank=rank,
                world_size=16,
                num_kv_heads=8,
            )
            for rank in (0, 1, 2)
        ]
        assert torch.equal(replicated_kv[0], replicated_kv[1])
        assert not torch.equal(replicated_kv[1], replicated_kv[2])

        torch.manual_seed(0)
        x = torch.randn(3, 6)
        weight = torch.randn(8, 6)
        full_output = torch.nn.functional.linear(x, weight)
        column_output = torch.cat(
            [torch.nn.functional.linear(x, shard) for shard in weight.chunk(2, dim=0)],
            dim=-1,
        )
        assert torch.allclose(column_output, full_output)

        input_shards = x.chunk(2, dim=-1)
        row_output = sum(
            torch.nn.functional.linear(x_shard, weight_shard.float())
            for x_shard, weight_shard in zip(input_shards, weight.chunk(2, dim=1))
        )
        assert torch.allclose(row_output, full_output, atol=1e-5)
    finally:
        reset_tp_info()
    print("[CHECK] TP layer shapes and checkpoint sharding passed")


def make_prompts(num_seqs: int, input_len: int) -> list[str]:
    texts = [
        "Who are you and what can you help me with? ",
        "Explain tensor parallelism in one concise paragraph. ",
        "请用一句话介绍大语言模型推理。",
        "Write a short greeting for a new student. ",
    ]
    prompts: list[str] = []
    for index in range(num_seqs):
        text = texts[index % len(texts)]
        repeats = max(1, input_len // 8)
        prompts.append((text * repeats).strip())
    return prompts


def run_inference(args: argparse.Namespace, *, benchmark: bool) -> dict:
    from aios import LLM, SamplingParams

    if args.warmup_runs < 0 or args.repeats < 1:
        raise ValueError("warmup-runs must be >= 0 and repeats must be >= 1")
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp_size,
        max_running_reqs=args.max_running,
        memory_ratio=args.memory_ratio,
        enable_cuda_graph=args.cuda_graph,
    )
    prompts = make_prompts(args.num_seqs, args.input_len)
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
    max_running_reqs = min(args.max_running, args.num_seqs)
    for _ in range(args.warmup_runs):
        llm.generate(prompts, params, max_running_reqs=max_running_reqs)
    torch.cuda.synchronize(llm.device)

    elapsed_runs: list[float] = []
    outputs = []
    output_runs: list[list[list[int]]] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(
            prompts,
            params,
            max_running_reqs=max_running_reqs,
        )
        torch.cuda.synchronize(llm.device)
        elapsed_runs.append(time.perf_counter() - start)
        run_tokens = [item["token_ids"] for item in outputs]
        output_runs.append(run_tokens)

    elapsed = sum(elapsed_runs)
    generated = [item["token_ids"] for item in outputs]
    total_tokens = args.num_seqs * args.max_tokens * args.repeats
    result = {
        "tp_size": args.tp_size,
        "elapsed_s": elapsed / args.repeats,
        "elapsed_runs_s": elapsed_runs,
        "tokens_per_s": total_tokens / elapsed,
        "outputs": generated,
        "stable_outputs": all(run == output_runs[0] for run in output_runs),
    }
    if not result["stable_outputs"]:
        result["output_runs"] = output_runs
    if llm.is_primary:
        label = "BENCH" if benchmark else "E2E"
        print(
            f"[{label}] TP={args.tp_size}, avg_time={result['elapsed_s']:.3f}s, "
            f"throughput={result['tokens_per_s']:.2f} tok/s"
        )
        print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))
    llm.close()
    return result


def _parse_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise RuntimeError(f"Child process did not emit RESULT_JSON:\n{stdout}")


def _run_child(command: list[str], env: dict[str, str]) -> dict:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Child exited with code {completed.returncode}: {' '.join(command)}"
        )
    return _parse_result(completed.stdout)


def run_comparison(args: argparse.Namespace) -> None:
    if args.tp_size < 2:
        raise ValueError("--suite compare requires --tp-size >= 2")
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    repo_python = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (repo_python, env.get("PYTHONPATH", "")) if item
    )
    common = [
        script,
        "--suite",
        "bench",
        "--model",
        args.model,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--max-tokens",
        str(args.max_tokens),
        "--max-running",
        str(args.max_running),
        "--memory-ratio",
        str(args.memory_ratio),
        "--warmup-runs",
        str(args.warmup_runs),
        "--repeats",
        str(args.repeats),
    ]
    if args.cuda_graph:
        common.append("--cuda-graph")

    single = _run_child([sys.executable, *common, "--tp-size", "1"], env)
    parallel = _run_child(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.tp_size}",
            *common,
            "--tp-size",
            str(args.tp_size),
        ],
        env,
    )
    if not single["stable_outputs"] or not parallel["stable_outputs"]:
        raise AssertionError("Repeated greedy runs produced different tokens")
    if single["outputs"] != parallel["outputs"]:
        raise AssertionError("TP output tokens differ from the single-GPU baseline")
    speedup = parallel["tokens_per_s"] / single["tokens_per_s"]
    print("\n[COMPARE] greedy outputs match exactly")
    print(f"[COMPARE] TP={args.tp_size} speedup: {speedup:.2f}x")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.suite == "check":
        run_shape_checks()
    elif args.suite == "compare":
        run_comparison(args)
    else:
        run_inference(args, benchmark=args.suite == "bench")


if __name__ == "__main__":
    main()
