# Adapted from: mini-sglang/benchmark/offline/bench.py

import argparse
import time
from random import randint, seed

from aios.core import SamplingParams
from aios.llm import LLM


def main():
    parser = argparse.ArgumentParser(description="AIOS offline benchmark (continuous batching)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument(
        "--max-running-reqs",
        type=int,
        default=None,
        help="Cap concurrently running reqs",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="Enable decode CUDA graph replay")
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help="Largest decode batch size to capture when --cuda-graph is enabled",
    )
    args = parser.parse_args()

    seed(0)

    llm = LLM(
        args.model,
        max_running_reqs=args.max_running_reqs or args.num_seqs,
        enable_cuda_graph=args.cuda_graph,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )

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

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, max_running_reqs=args.max_running_reqs)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"[CONTINUOUS_BATCH] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
