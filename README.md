# AIOS: Introduction to LLM Inference (Learning Project)

Have you ever wondered how ChatGPT generates responses? How a model with billions of parameters actually runs on your GPU? Or why inference optimization matters so much?

AIOS is a hands-on learning project where you build an LLM inference framework from scratch. By the end, you won’t just *use* LLMs—you’ll understand how inference works under the hood.

## Who This Is For

- **Software engineers** curious about AI/ML systems
- **ML practitioners** who want to understand inference beyond training
- **System engineers** interested in GPU programming and optimization
- **Students** looking for practical, implementable knowledge

## What You’ll Build

By completing this project, you will:

- Implement a working LLM inference engine using PyTorch
- Load and run modern LLMs (e.g., Llama/Qwen-class models) end-to-end
- Understand every major Transformer component (attention, FFN, layer norms, embeddings)
- Apply real-world inference optimizations (KV-cache, batching, TP/PP, quantization, etc.)

## The Key Mental Model: LLM Inference Engine as an “Operating System”

Traditional computing has an OS that bridges applications and hardware. AI computing has an inference engine that bridges LLMs and GPUs/NPUs:

- **Operating system**: process scheduling, memory management, I/O scheduling, device drivers
- **Inference engine**: request batching & scheduling, GPU memory management, KV-cache management, kernel/operator optimization, multi-device parallelism

This project’s goal is to help you build that “bridge” layer, step by step.

## Why Inference Matters

Training happens rarely; inference happens constantly. Every chatbot response, code completion, and AI feature runs inference—so small efficiency gains translate directly into large cost and latency improvements.

Inference is challenging because it’s:

- **Memory heavy** (weights + activations + KV-cache)
- **Compute heavy** (large GEMMs + attention)
- **Latency sensitive** (time-to-first-token and tokens/sec matter for UX)

## Course Roadmap (High Level)

- **Lesson 0**: Introduction (traditional software vs LLMs, why inference matters, course objective)
- **Lesson 1**: LLM basics (Tokenizer, Transformer architecture, attention, positional encoding, parameters)
- **Lesson 2**: Running an LLM with PyTorch (loading weights, tokenization, forward pass, generation)
- **Next**: KV-cache, batching/scheduling, quantization, multi-GPU parallelism, serving, profiling

For the detailed Lesson 0 content, see `resources/lesson-0-introduction/README.md`.

## Prerequisites

- **Python**: comfortable with functions, classes, and basic data structures
- **Linear algebra**: basic matrix/vector operations

Helpful (not required):

- PyTorch basics
- Neural network fundamentals
- GPU/CUDA concepts

## Getting Started

Start here:

- **Lesson 0 (Introduction)**: `resources/lesson-0-introduction/README.md`

Then continue to:

- **Lesson 1 (LLM Basics)**: `resources/lesson-1-llm-basics/README.md`

