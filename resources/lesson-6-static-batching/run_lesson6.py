"""
Lesson 6 runner: compare dynamic KV cache baseline vs static batching.

Default mode runs the historical comparison suite used in lesson docs:
  - Case A: num_seqs=8,  max_input_len=64,  max_output_len=256
  - Case B: num_seqs=16, max_input_len=128, max_output_len=256
  - Case C: num_seqs=24, max_input_len=128, max_output_len=256
  - Case D: num_seqs=32, max_input_len=128, max_output_len=256
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RESULT_RE = re.compile(
    r"\[(KV_CACHE|NO_CACHE|PAGED_CACHE|STATIC_BATCH)\]\s+Total:\s+(\d+)tok,\s+Time:\s+([0-9.]+)s,\s+Throughput:\s+([0-9.]+)tok/s"
)


@dataclass(frozen=True)
class CaseConfig:
    name: str
    num_seqs: int
    max_input_len: int
    max_output_len: int


@dataclass(frozen=True)
class GPUMetrics:
    peak_mem_used_mib: float
    samples: int


@dataclass(frozen=True)
class BenchResult:
    mode: str
    total_tokens: int
    elapsed_s: float
    throughput: float
    command: str
    gpu_metrics: GPUMetrics | None


class BenchCommandError(RuntimeError):
    def __init__(self, command: str, returncode: int, stdout: str, stderr: str):
        details = [
            f"Benchmark command failed with exit code {returncode}.",
            f"Command: {command}",
        ]
        if stdout.strip():
            details.append(f"stdout:\n{stdout.strip()}")
        if stderr.strip():
            details.append(f"stderr:\n{stderr.strip()}")
        super().__init__("\n".join(details))
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


HISTORICAL_CASES: list[CaseConfig] = [
    CaseConfig(name="A", num_seqs=8, max_input_len=128, max_output_len=256),
    CaseConfig(name="B", num_seqs=16, max_input_len=128, max_output_len=256),
    CaseConfig(name="C", num_seqs=24, max_input_len=128, max_output_len=256),
    CaseConfig(name="D", num_seqs=32, max_input_len=128, max_output_len=256),
]


def _parse_gpu_metrics(path: Path) -> GPUMetrics | None:
    peak_mem = 0.0
    samples = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            try:
                mem_used = float(value)
            except ValueError:
                continue
            peak_mem = max(peak_mem, mem_used)
            samples += 1
    if samples == 0:
        return None
    return GPUMetrics(
        peak_mem_used_mib=peak_mem,
        samples=samples,
    )


def _first_gpu_id(cuda_visible_devices: str | None) -> str:
    if not cuda_visible_devices:
        return "0"
    return cuda_visible_devices.split(",")[0].strip()


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_bench(
    bench_py: Path,
    *,
    model: str,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    cuda_visible_devices: str | None = None,
    paged_kv_cache: bool = False,
    static_batch: bool = False,
    collect_gpu_metrics: bool = True,
    cwd: Path,
) -> BenchResult:
    cmd = [
        sys.executable,
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
    if paged_kv_cache:
        cmd.append("--paged-kv-cache")
    if static_batch:
        cmd.append("--static-batch")

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    command_str = _format_command(cmd)
    print(f"$ {command_str}", flush=True)

    monitor_proc: subprocess.Popen[str] | None = None
    monitor_fp = None
    gpu_metrics_file = None
    if collect_gpu_metrics:
        gpu_metrics_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix="lesson6_gpu_", suffix=".csv", delete=False
        )
        gpu_metrics_file.close()
        try:
            monitor_fp = open(gpu_metrics_file.name, "w", encoding="utf-8")
            monitor_proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id",
                    _first_gpu_id(cuda_visible_devices),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=monitor_fp,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            monitor_proc = None
            if monitor_fp is not None:
                monitor_fp.close()
                monitor_fp = None

    try:
        proc = subprocess.run(
            cmd, check=False, cwd=cwd, capture_output=True, text=True, env=env
        )
    finally:
        if monitor_proc is not None:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()
        if monitor_fp is not None:
            monitor_fp.close()

    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)

    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr, flush=True)

    if proc.returncode != 0:
        raise BenchCommandError(command_str, proc.returncode, proc.stdout, proc.stderr)

    match = RESULT_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"Failed to parse benchmark output.\nstdout:\n{proc.stdout}")
    mode, total, elapsed, throughput = match.groups()

    gpu_metrics = None
    if gpu_metrics_file is not None:
        try:
            gpu_metrics = _parse_gpu_metrics(Path(gpu_metrics_file.name))
        finally:
            Path(gpu_metrics_file.name).unlink(missing_ok=True)

    return BenchResult(
        mode=mode,
        total_tokens=int(total),
        elapsed_s=float(elapsed),
        throughput=float(throughput),
        command=command_str,
        gpu_metrics=gpu_metrics,
    )


def _print_case_result(case: CaseConfig, baseline: BenchResult, static: BenchResult) -> None:
    speedup = static.throughput / baseline.throughput if baseline.throughput > 0 else 0.0
    print(
        f"\n=== Case {case.name}: num_seqs={case.num_seqs}, max_in={case.max_input_len}, max_out={case.max_output_len} ===",
        flush=True,
    )
    print(
        f"{baseline.mode:>12}: {baseline.throughput:8.2f} tok/s ({baseline.elapsed_s:6.2f}s, total={baseline.total_tokens})",
        flush=True,
    )
    if baseline.gpu_metrics is not None:
        gm = baseline.gpu_metrics
        print(
            f"{'':>12}  peak_mem={gm.peak_mem_used_mib:8.2f} MiB  samples={gm.samples}",
            flush=True,
        )
    print(
        f"{static.mode:>12}: {static.throughput:8.2f} tok/s ({static.elapsed_s:6.2f}s, total={static.total_tokens})",
        flush=True,
    )
    if static.gpu_metrics is not None:
        gm = static.gpu_metrics
        print(
            f"{'':>12}  peak_mem={gm.peak_mem_used_mib:8.2f} MiB  samples={gm.samples}",
            flush=True,
        )
    print(f"{'SPEEDUP':>12}: {speedup:8.2f}x", flush=True)


def run_case(
    bench_py: Path,
    *,
    model: str,
    case: CaseConfig,
    cuda_visible_devices: str | None,
    collect_gpu_metrics: bool,
    cwd: Path,
) -> tuple[BenchResult, BenchResult]:
    print(
        f"\n[Case {case.name}] Running dynamic KV baseline (no batching)...",
        flush=True,
    )
    baseline = run_bench(
        bench_py,
        model=model,
        num_seqs=case.num_seqs,
        max_input_len=case.max_input_len,
        max_output_len=case.max_output_len,
        cuda_visible_devices=cuda_visible_devices,
        paged_kv_cache=False,
        static_batch=False,
        collect_gpu_metrics=collect_gpu_metrics,
        cwd=cwd,
    )

    print(
        f"\n[Case {case.name}] Running paged KV + static batching...",
        flush=True,
    )
    static = run_bench(
        bench_py,
        model=model,
        num_seqs=case.num_seqs,
        max_input_len=case.max_input_len,
        max_output_len=case.max_output_len,
        cuda_visible_devices=cuda_visible_devices,
        paged_kv_cache=True,
        static_batch=True,
        collect_gpu_metrics=collect_gpu_metrics,
        cwd=cwd,
    )
    return baseline, static


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lesson-6 static batching benchmark suite")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["historical", "single"],
        default="historical",
        help="historical: run all lesson doc cases; single: run one custom case",
    )
    parser.add_argument("--num-seqs", type=int, default=32, help="Only used in --suite single")
    parser.add_argument("--max-input-len", type=int, default=128, help="Only used in --suite single")
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=256,
        help="Output length limit; overrides historical case max_output_len when --suite historical",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES (e.g. 0 or 1)",
    )
    parser.add_argument(
        "--collect-gpu-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect average GPU utilization and memory usage by sampling nvidia-smi",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    bench_py = repo_root / "benchmark" / "bench.py"

    if args.suite == "single":
        cases = [
            CaseConfig(
                name="S",
                num_seqs=args.num_seqs,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
            )
        ]
    else:
        cases = [
            CaseConfig(
                name=case.name,
                num_seqs=case.num_seqs,
                max_input_len=case.max_input_len,
                max_output_len=args.max_output_len,
            )
            for case in HISTORICAL_CASES
        ]

    print(
        f"Running lesson-6 suite '{args.suite}' with {len(cases)} case(s).",
        flush=True,
    )

    summary_rows: list[tuple[CaseConfig, BenchResult, BenchResult]] = []
    for case in cases:
        baseline, static = run_case(
            bench_py,
            model=args.model,
            case=case,
            cuda_visible_devices=args.cuda_visible_devices,
            collect_gpu_metrics=args.collect_gpu_metrics,
            cwd=repo_root,
        )
        _print_case_result(case, baseline, static)
        summary_rows.append((case, baseline, static))

    def _fmt_peak_gpu_mem(r: BenchResult) -> str:
        if r.gpu_metrics is None:
            return "N/A"
        return f"{r.gpu_metrics.peak_mem_used_mib:.0f}"

    print("\n=== Lesson 6 Summary ===", flush=True)
    print(
        "Case  num_seqs  baseline(tok/s)  static(tok/s)  speedup  "
        "base_peak_mem(MiB)  static_peak_mem(MiB)",
        flush=True,
    )
    for case, baseline, static in summary_rows:
        speedup = static.throughput / baseline.throughput if baseline.throughput > 0 else 0.0
        print(
            f"{case.name:>4}  {case.num_seqs:>8}  {baseline.throughput:>15.2f}  "
            f"{static.throughput:>13.2f}  {speedup:>7.2f}x  "
            f"{_fmt_peak_gpu_mem(baseline):>18}  {_fmt_peak_gpu_mem(static):>20}",
            flush=True,
        )


if __name__ == "__main__":
    main()
