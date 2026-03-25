"""
CLI entry point for AIOS inference engine.

Usage:
    python -m aios --model /path/to/Qwen3-0.6B --prompt "Who are you?"
    python -m aios --model /path/to/Qwen3-0.6B --prompt "Hello" --temperature 0.8 --max-tokens 128
"""

import argparse
import time

from aios import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="AIOS LLM Inference Engine")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory or HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Who are you?",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (-1 = disabled)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g. cuda, cuda:0, cuda:1)")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable lesson-4 dynamic KV cache path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    t0 = time.perf_counter()
    llm = LLM(args.model, device=args.device)
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.1f}s\n")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    results = llm.generate([args.prompt], sampling_params, use_kv_cache=not args.no_kv_cache)
    for r in results:
        print(r["text"])


if __name__ == "__main__":
    main()
