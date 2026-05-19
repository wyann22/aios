"""
Lesson 8 runner: flat-token varlen prefill.

Usage:
    python resources/lesson-8-flat-varlen-prefill/run_lesson8.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import time
from random import randint, seed
from typing import List

from aios.core import SamplingParams
from aios.llm import LLM


def _build_workload(num_seqs: int, min_prompt_len: int, max_prompt_len: int, max_tokens: int):
    prompts: List[List[int]] = [
        [randint(0, 10000) for _ in range(randint(min_prompt_len, max_prompt_len))]
        for _ in range(num_seqs)
    ]
    params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)
        for _ in range(num_seqs)
    ]
    return prompts, params


def _run_case(
    llm: LLM,
    name: str,
    prompts: List[List[int]],
    params: List[SamplingParams],
    max_running: int,
    prefill_token_budget: int | None,
):
    total_output_tokens = sum(sp.max_tokens for sp in params)
    total_prompt_tokens = sum(len(p) for p in prompts)
    t0 = time.time()
    llm.generate(
        prompts,
        params,
        max_running_reqs=max_running,
        prefill_token_budget=prefill_token_budget,
    )
    elapsed = time.time() - t0
    print(
        f"[{name}] elapsed={elapsed:.2f}s "
        f"output_tps={total_output_tokens / elapsed:.2f} "
        f"prefill_tps={total_prompt_tokens / elapsed:.2f}"
    )
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Lesson 8 flat varlen prefill runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--min-prompt-len", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-scheduler", action="store_true")
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    seed(args.seed)
    prompts, params = _build_workload(
        args.num_seqs, args.min_prompt_len, args.max_prompt_len, args.max_tokens
    )
    total_prompt_tokens = sum(len(p) for p in prompts)
    total_output_tokens = sum(sp.max_tokens for sp in params)

    print(
        f"Workload: num_seqs={args.num_seqs}, prompt_len="
        f"{args.min_prompt_len}..{args.max_prompt_len}, "
        f"prompt_tokens={total_prompt_tokens}, output_tokens={total_output_tokens}, "
        f"max_running={args.max_running}"
    )

    llm = LLM(args.model)
    llm.generate(
        [[randint(0, 10000) for _ in range(8)]],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2),
    )

    # Lesson 7-compatible mode: one prefill request per iteration by using the
    # smallest prompt as the effective token budget.
    min_len = min(len(p) for p in prompts)
    lesson7_time = _run_case(
        llm, "LESSON7_COMPAT", prompts, params, args.max_running, min_len
    )
    lesson8_time = _run_case(
        llm, "LESSON8_VARLEN", prompts, params, args.max_running, None
    )
    print(f"speedup={lesson7_time / lesson8_time:.2f}x")

    if args.debug_scheduler:
        llm.generate(
            prompts[: min(4, len(prompts))],
            params[: min(4, len(params))],
            max_running_reqs=min(args.max_running, 4),
            debug_scheduler=True,
        )


if __name__ == "__main__":
    main()
