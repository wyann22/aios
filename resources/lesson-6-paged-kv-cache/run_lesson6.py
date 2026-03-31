"""
Lesson 6 runner that invokes benchmark/bench.py testing paged memory cache allocation.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

RESULT_RE = re.compile(
    r"\[(KV_CACHE|NO_CACHE|PAGED_CACHE)\]\s+Total:\s+(\d+)tok,\s+Time:\s+([0-9.]+)s,\s+Throughput:\s+([0-9.]+)tok/s"
)

def run_bench(
    bench_py: Path,
    *,
    model: str,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    no_kv_cache: bool = False,
    paged_kv_cache: bool = False,
    cwd: Path,
) -> tuple[str, float, float]:
    cmd = [
        '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=0', sys.executable,
        str(bench_py),
        "--model",
        model,
        "--num-seqs",
        str(num_seqs),
        "--max-input-len",
        str(max_input_len),
        "--max-output-len",
        str(max_output_len),
    ]
    if no_kv_cache:
        cmd.append("--no-kv-cache")
    if paged_kv_cache:
        cmd.append("--paged-kv-cache")

    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
    if proc.stderr.strip():
        print(proc.stderr.strip())

    match = RESULT_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"Failed to parse benchmark output. output: {proc.stdout}")
    mode, _total, elapsed, throughput = match.groups()
    return mode, float(elapsed), float(throughput)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lesson-6 benchmark via benchmark/bench.py")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("--num-seqs", type=int, default=16, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=256)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    bench_py = repo_root / "benchmark" / "bench.py"

    print("Running baseline dynamic KV cache:")
    _, cache_time, cache_tps = run_bench(
        bench_py,
        model=args.model,
        num_seqs=args.num_seqs,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        paged_kv_cache=False,
        cwd=repo_root,
    )

    print("\nRunning Paged KV cache:")
    _, paged_time, paged_tps = run_bench(
        bench_py,
        model=args.model,
        num_seqs=args.num_seqs,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        paged_kv_cache=True,
        cwd=repo_root,
    )

    print("\n=== Lesson 6 Performance Comparison ===")
    print(f"Dynamic KV cache:    {cache_tps:.2f} tok/s ({cache_time:.2f}s)")
    print(f"Paged KV cache:      {paged_tps:.2f} tok/s ({paged_time:.2f}s)")
    if cache_tps > 0:
        print(f"Speedup vs Dynamic:  {paged_tps / cache_tps:.2f}x")

if __name__ == "__main__":
    main()
