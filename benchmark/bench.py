# Adapted from: mini-sglang/benchmark/offline/bench.py

import argparse
import time
from random import randint, seed

from aios.core import SamplingParams
from aios.llm import LLM


def main():
    parser = argparse.ArgumentParser(description="AIOS offline benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable lesson-4 KV cache path")
    args = parser.parse_args()

    seed(0)

    llm = LLM(args.model)

    input_low = min(32, args.max_input_len)
    output_low = min(64, args.max_output_len)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(input_low, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(output_low, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    # warm up
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1), use_kv_cache=not args.no_kv_cache)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_kv_cache=not args.no_kv_cache)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    mode = "NO_CACHE" if args.no_kv_cache else "KV_CACHE"
    print(f"[{mode}] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
