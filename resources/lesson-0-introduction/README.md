# Lesson 0: Introduction to LLM Inference

## Welcome to the AIOS LLM Inference Learning Project

Have you ever wondered how ChatGPT generates responses? How a model with billions of parameters actually runs on your GPU? Or why inference optimization matters so much?

This course takes you on a hands-on journey to build an LLM inference framework from scratch. By the end, you won't just *use* LLMs—you'll understand exactly how they work under the hood.

### Who Is This Course For?

- **Software Engineers** curious about AI/ML systems
- **ML Practitioners** who want to understand inference beyond training
- **System Engineers** interested in GPU programming and optimization
- **Students** looking for practical, implementable knowledge

### What You'll Build

By completing this course, you will:
- Implement a working LLM inference engine using PyTorch
- Load and run the Llama/Qwen model from scratch
- Understand every layer of the Transformer architecture
- Master optimization techniques like kv-cache, TP/PP and so on for real-world performance

---

## LLMs vs Traditional Software: A Paradigm Shift

Before diving into implementation, let's understand what makes LLMs fundamentally different from traditional software. This comparison will give you the mental framework needed throughout this course.

### The Traditional Software Paradigm

Traditional software follows explicit rules defined by programmers:

```
┌─────────────────────────────────────────────────────────────┐
│                  TRADITIONAL SOFTWARE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► Rules (Code) ──► Output                         │
│                                                             │
│   Example: Spell Checker                                    │
│                                                             │
│   "helo" ──► Dictionary Lookup ──► "hello"                  │
│              + Edit Distance                                 │
│              + Defined Rules                                 │
│                                                             │
│   Characteristics:                                          │
│   ✓ Deterministic (same input = same output)               │
│   ✓ Explainable (follow the code path)                     │
│   ✓ Predictable resource usage                             │
│   ✓ Easy to debug and test                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

In traditional software:
- **Behavior is explicit**: Every possible action is coded by a programmer
- **Logic is visible**: You can trace exactly why a decision was made
- **Testing is straightforward**: Define inputs, verify outputs
- **Memory is structured**: Data fits in defined schemas

### The LLM Paradigm

LLMs learn patterns from data rather than following explicit rules:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM-BASED SOFTWARE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► Learned Weights ──► Probability ──► Output      │
│             (Parameters)        Distribution                 │
│                                                             │
│   Example: Language Understanding                           │
│                                                             │
│   "The bank was steep" ──► Neural Network ──► "riverbank"   │
│                            (Context Analysis)                │
│                                                             │
│   Characteristics:                                          │
│   ✗ Non-deterministic (temperature, sampling)              │
│   ✗ Black box (weights encode patterns)                    │
│   ✓ Generalizes to unseen inputs                           │
│   ✓ Handles ambiguity naturally                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

In LLM-based software:
- **Behavior is learned**: Patterns emerge from training data
- **Logic is implicit**: Encoded in billions of weight values
- **Testing is probabilistic**: Same input may produce different outputs
- **Memory is distributed**: Knowledge spread across parameters

### Side-by-Side Comparison

```
┌────────────────────┬─────────────────────┬─────────────────────┐
│     Aspect         │ Traditional Software│        LLMs         │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Decision Making    │ If-else, rules      │ Matrix multiplies   │
│                    │                     │ + learned weights   │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Knowledge Storage  │ Databases, files    │ Model parameters    │
│                    │                     │ (weights/biases)    │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Handling New Cases │ Add new code/rules  │ Already generalizes │
│                    │                     │ (if in training)    │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Execution Flow     │ Control flow (loops,│ Forward pass        │
│                    │ branches, calls)    │ through layers      │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Debugging          │ Breakpoints, logs   │ Attention analysis, │
│                    │                     │ probing, ablation   │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Resource Usage     │ CPU-bound           │ GPU/Memory-bound    │
│                    │ O(n) typical        │ O(n²) attention     │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Modification       │ Edit source code    │ Fine-tune, prompt,  │
│                    │                     │ or retrain          │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Failures           │ Crashes, errors     │ Hallucinations,     │
│                    │                     │ wrong answers       │
└────────────────────┴─────────────────────┴─────────────────────┘
```

### A Concrete Example: Sentiment Analysis

Let's see how the same task is approached differently:

#### Traditional Approach
```python
# Rule-based sentiment analysis
def analyze_sentiment(text):
    positive_words = {"good", "great", "excellent", "happy", "love"}
    negative_words = {"bad", "terrible", "awful", "sad", "hate"}

    words = text.lower().split()
    score = 0

    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1

    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    return "neutral"

# Limitation: "This movie is not bad" → incorrectly negative
```

**Limitations**:
- Can't handle negation ("not bad" = positive)
- Can't understand context ("This kills!" in gaming = positive)
- Requires manual dictionary maintenance
- Fails on new words, slang, or domain-specific language

#### LLM Approach
```python
# LLM-based sentiment analysis (simplified)
def analyze_sentiment(text, model, tokenizer):
    prompt = f"Analyze the sentiment of: '{text}'\n\nSentiment:"

    tokens = tokenizer.encode(prompt)
    logits = model.forward(tokens)  # Neural network forward pass

    # Model outputs probability distribution over next tokens
    # "positive", "negative", "neutral" have highest probabilities
    return decode_sentiment(logits)

# Handles: "This movie is not bad" → positive
# Handles: "This kills!" (gaming context) → positive
```

**Advantages**:
- Understands context and nuance
- Handles negation naturally
- Generalizes to new expressions
- No manual rule maintenance

### Understanding the "Black Box"

When people call LLMs a "black box," they mean this:

```
┌─────────────────────────────────────────────────────────────┐
│                    INSIDE AN LLM                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Traditional Program:                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ if word in dictionary:                              │   │
│  │     return correct_spelling(word)  ← You can read   │   │
│  │ else:                                     this!     │   │
│  │     return find_closest_match(word)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  LLM:                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ weights = [                                         │   │
│  │   0.0234, -0.1567, 0.8921, 0.0012, -0.4521, ...    │   │
│  │   (7 billion more numbers)                          │   │
│  │ ]                                                   │   │
│  │                                                     │   │
│  │ output = input @ weights[layer1]                    │   │
│  │ output = activation(output)    ← What does this     │   │
│  │ output = output @ weights[layer2]   number mean?    │   │
│  │ ... (32 more layers)                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  The "knowledge" is distributed across billions of          │
│  numbers, not readable code.                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This is why understanding LLM internals matters—and why this course exists.

---

## LLM Is Still Software: Input → Process → Output

Despite all the differences, it's crucial to understand: **LLM is fundamentally still software**. Like any program, it takes input, processes it, and produces output. The difference lies in *what* the input/output are and *how* the processing happens.

```
┌─────────────────────────────────────────────────────────────┐
│           THE UNIVERSAL SOFTWARE MODEL                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ALL software follows: Input ──► Process ──► Output        │
│                                                             │
│   Traditional Software:                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Input: Structured data (JSON, SQL query, HTTP)     │  │
│   │  Process: Execute code logic (if/else, loops)       │  │
│   │  Output: Structured data (JSON, HTML, database)     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   LLM Software:                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Input: Token sequence (natural language → numbers) │  │
│   │  Process: Matrix operations through neural layers   │  │
│   │  Output: Probability distribution → next token      │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Input/Output Comparison

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│     Aspect         │ Traditional Software │         LLM          │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Input Format       │ Structured:          │ Sequential:          │
│                    │ - API parameters     │ - Token IDs          │
│                    │ - Database queries   │ - Embeddings         │
│                    │ - Form data          │ - Attention masks    │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Input Size         │ Variable, usually    │ Fixed context window │
│                    │ small (KB)           │ (4K-128K tokens)     │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Output Format      │ Structured:          │ Probability dist:    │
│                    │ - JSON/XML           │ - Logits over vocab  │
│                    │ - Rendered HTML      │ - Sampled token ID   │
│                    │ - Binary data        │ - Then decoded text  │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Output Generation  │ Complete response    │ Autoregressive:      │
│                    │ at once              │ one token at a time  │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Determinism        │ Same input =         │ Same input ≠         │
│                    │ Same output          │ Same output          │
│                    │                      │ (unless temp=0)      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Processing Model   │ Request-Response     │ Streaming generation │
│                    │ (single pass)        │ (iterative decode)   │
└────────────────────┴──────────────────────┴──────────────────────┘
```

### Key Similarity

Both are **deterministic systems** at the hardware level:
- Traditional software: CPU executes instructions sequentially
- LLM: GPU executes matrix multiplications in parallel

The "randomness" in LLMs comes from **sampling strategies** (temperature, top-p), not from the computation itself. With temperature=0, an LLM is perfectly deterministic.

---

## Compute and Memory: A Fundamental Difference

The most profound difference between LLMs and traditional software lies in their **resource consumption patterns**. This difference fundamentally shapes the hardware they run on.

### Traditional Software: Compute-Light, Logic-Heavy

```
┌─────────────────────────────────────────────────────────────┐
│            TRADITIONAL SOFTWARE RESOURCE PROFILE             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Typical Web Server Request:                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Parse HTTP request          ~1,000 CPU cycles  │    │
│  │  2. Database query              ~10,000 cycles     │    │
│  │  3. Business logic              ~5,000 cycles      │    │
│  │  4. Render response             ~2,000 cycles      │    │
│  │  ─────────────────────────────────────────────     │    │
│  │  Total: ~20,000 CPU cycles per request             │    │
│  │  At 3 GHz: ~7 microseconds                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Memory Access Pattern:                                     │
│  • Random access, cache-friendly                           │
│  • Working set: MB range                                   │
│  • Bandwidth: 10-50 GB/s sufficient                        │
│                                                             │
│  Bottleneck: I/O (disk, network), not compute              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### LLM Inference: Compute-Heavy, Bandwidth-Hungry

```
┌─────────────────────────────────────────────────────────────┐
│              LLM INFERENCE RESOURCE PROFILE                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Single Token Generation (Llama 7B):                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Per layer (32 layers total):                      │    │
│  │  • Attention QKV:     4096 × 4096 × 3 = 50M ops   │    │
│  │  • Attention output:  4096 × 4096     = 17M ops   │    │
│  │  • FFN up:            4096 × 11008    = 45M ops   │    │
│  │  • FFN down:          11008 × 4096    = 45M ops   │    │
│  │  ─────────────────────────────────────────────     │    │
│  │  Per layer: ~160 million operations               │    │
│  │  All layers: 32 × 160M = 5 billion operations     │    │
│  │                                                    │    │
│  │  Total: ~14 TFLOPS per token                      │    │
│  │  At 100 tokens/sec: 1.4 PFLOPS sustained          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Memory Access Pattern:                                     │
│  • Sequential, massive reads                               │
│  • Working set: 14+ GB (model weights)                     │
│  • Bandwidth: 1-3 TB/s required                            │
│                                                             │
│  Bottleneck: Memory bandwidth AND compute                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Quantitative Comparison

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│     Metric         │ Traditional Software │    LLM Inference     │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Compute per        │ Thousands of         │ Billions of          │
│ Request            │ CPU cycles           │ FLOPs                │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Memory Footprint   │ MB range             │ GB to TB range       │
│                    │ (application + data) │ (model weights)      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Memory Bandwidth   │ 10-50 GB/s           │ 1-3 TB/s             │
│ Required           │                      │                      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Compute Pattern    │ Irregular, branchy   │ Regular, dense       │
│                    │ (control flow)       │ (matrix multiply)    │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Parallelism        │ Task-level           │ Data-level (SIMD)    │
│                    │ (threads, processes) │ (thousands of cores) │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Latency Tolerance  │ Microseconds         │ Milliseconds         │
│                    │                      │                      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Power Consumption  │ 50-200W (server)     │ 300-700W (per GPU)   │
│                    │                      │                      │
└────────────────────┴──────────────────────┴──────────────────────┘
```

### Why This Matters

This difference in resource requirements leads to completely different optimization strategies:

| Traditional Software | LLM Inference |
|---------------------|---------------|
| Cache optimization | KV-cache reuse |
| Query optimization | Batching strategies |
| Connection pooling | Continuous batching |
| CDN / Load balancing | Tensor parallelism |
| Async I/O | Speculative decoding |

---

## Hardware Platforms: CPU vs GPU/NPU

The resource profile difference explains why LLMs need specialized hardware.

### CPU: Optimized for Traditional Software

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Core 0    Core 1    Core 2    ...    Core N       │   │
│  │  ┌─────┐  ┌─────┐   ┌─────┐         ┌─────┐       │   │
│  │  │ ALU │  │ ALU │   │ ALU │         │ ALU │       │   │
│  │  │ FPU │  │ FPU │   │ FPU │         │ FPU │       │   │
│  │  │ L1$ │  │ L1$ │   │ L1$ │         │ L1$ │       │   │
│  │  │ L2$ │  │ L2$ │   │ L2$ │         │ L2$ │       │   │
│  │  └─────┘  └─────┘   └─────┘         └─────┘       │   │
│  │           Shared L3 Cache (30-100 MB)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Characteristics:                                           │
│  • Few powerful cores (8-64)                               │
│  • Large caches (optimize for locality)                    │
│  • Branch prediction (handle control flow)                 │
│  • Out-of-order execution (hide latency)                   │
│  • Memory bandwidth: 50-200 GB/s                           │
│                                                             │
│  Great for: Irregular workloads, low-latency, branchy code │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### GPU: Optimized for Parallel Computation

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SM 0      SM 1      SM 2    ...    SM N           │   │
│  │  ┌─────┐  ┌─────┐   ┌─────┐       ┌─────┐         │   │
│  │  │█████│  │█████│   │█████│       │█████│         │   │
│  │  │█████│  │█████│   │█████│       │█████│  Each   │   │
│  │  │█████│  │█████│   │█████│       │█████│  block  │   │
│  │  │█████│  │█████│   │█████│       │█████│  = 128  │   │
│  │  └─────┘  └─────┘   └─────┘       └─────┘  cores  │   │
│  │                                                    │   │
│  │  Total: 10,000+ simple cores                       │   │
│  └─────────────────────────────────────────────────────┘   │
│  │                   HBM Memory                       │   │
│  │             (80 GB @ 3 TB/s bandwidth)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Characteristics:                                           │
│  • Thousands of simple cores (10,000+)                     │
│  • Small caches (optimize for throughput)                  │
│  • SIMD execution (same instruction, many data)            │
│  • High memory bandwidth: 1-3 TB/s                         │
│  • Tensor cores for matrix operations                      │
│                                                             │
│  Great for: Regular workloads, massive parallelism, GEMM   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Comparison

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│     Feature        │      CPU             │      GPU/NPU         │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Core Count         │ 8-64 cores           │ 10,000+ cores        │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Core Complexity    │ High (OoO, branch    │ Low (simple ALU)     │
│                    │ pred, speculation)   │                      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Clock Speed        │ 3-5 GHz              │ 1-2 GHz              │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Memory Bandwidth   │ 50-200 GB/s          │ 1-3 TB/s             │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Memory Capacity    │ 128GB-2TB (DDR)      │ 24-80GB (HBM)        │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Peak FLOPS (FP16)  │ 1-5 TFLOPS           │ 300-1000 TFLOPS      │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Power              │ 100-300W             │ 300-700W             │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Programming Model  │ Sequential threads   │ CUDA/OpenCL kernels  │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Best For           │ Latency-sensitive,   │ Throughput-oriented, │
│                    │ branchy code         │ parallel workloads   │
└────────────────────┴──────────────────────┴──────────────────────┘
```

---

## The Bridge: LLM Inference Engine as "Operating System"

Here's the key insight of this course:

```
┌─────────────────────────────────────────────────────────────┐
│         THE SOFTWARE STACK ANALOGY                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   TRADITIONAL COMPUTING              AI COMPUTING            │
│                                                             │
│   ┌─────────────────┐              ┌─────────────────┐      │
│   │  Applications   │              │   LLM Models    │      │
│   │  (Chrome, Word) │              │ (Llama, Qwen)   │      │
│   └────────┬────────┘              └────────┬────────┘      │
│            │                                │               │
│            ▼                                ▼               │
│   ┌─────────────────┐              ┌─────────────────┐      │
│   │ Operating System│              │ Inference Engine│      │
│   │ (Linux, Windows)│              │ (vLLM, TGI,     │      │
│   │                 │              │  TensorRT-LLM)  │      │
│   │ • Process mgmt  │              │ • Memory mgmt   │      │
│   │ • Memory mgmt   │              │ • KV-cache mgmt │      │
│   │ • I/O scheduling│              │ • Batch sched   │      │
│   │ • Device drivers│              │ • Kernel optim  │      │
│   └────────┬────────┘              └────────┬────────┘      │
│            │                                │               │
│            ▼                                ▼               │
│   ┌─────────────────┐              ┌─────────────────┐      │
│   │     CPU         │              │    GPU / NPU    │      │
│   │   Hardware      │              │    Hardware     │      │
│   └─────────────────┘              └─────────────────┘      │
│                                                             │
│   The OS is the BRIDGE between software and hardware.       │
│   The inference engine is the BRIDGE between LLM and GPU.   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why We Need This "OS" Layer

Just as an operating system abstracts hardware complexity for applications:

| Operating System Role | Inference Engine Role |
|----------------------|----------------------|
| Memory allocation | GPU memory management |
| Process scheduling | Request batching & scheduling |
| Virtual memory | KV-cache paging (PagedAttention) |
| Device drivers | CUDA kernel optimization |
| File system caching | Prefix caching |
| Multi-process isolation | Multi-tenant serving |

### Without an Inference Engine

```python
# Naive approach: direct model execution
model = load_model("llama-7b")  # 14GB GPU memory

for request in requests:
    output = model.generate(request)  # One at a time
    # GPU utilization: 10-20%
    # Throughput: 10 tokens/sec
    # Memory: Wasted on unused KV-cache
```

### With an Inference Engine

```python
# Optimized approach: inference engine manages everything
engine = InferenceEngine(
    model="llama-7b",
    tensor_parallel=2,           # Split across GPUs
    max_batch_size=64,           # Batch requests
    kv_cache_policy="paged",     # Efficient memory
)

# Engine handles:
# - Continuous batching (add/remove requests dynamically)
# - KV-cache management (PagedAttention)
# - Memory optimization (quantization, offloading)
# - Parallelism (TP, PP across devices)
#
# GPU utilization: 80%+
# Throughput: 1000+ tokens/sec
# Memory: Efficiently shared
```

---

## Course Goal: Build an "Operating System" for LLM

This brings us to the core mission of this course:

```
┌─────────────────────────────────────────────────────────────┐
│                   COURSE OBJECTIVE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Build an LLM inference engine FROM SCRATCH                │
│                                                             │
│   Just as understanding OS internals makes you a better     │
│   systems programmer, understanding inference engines       │
│   makes you a better AI infrastructure engineer.            │
│                                                             │
│   What you'll implement:                                    │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Layer 1: Model Loading & Execution                 │  │
│   │  • Load Llama/Qwen weights                         │  │
│   │  • Implement forward pass                          │  │
│   │  • Tokenization & generation                       │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │  Layer 2: Memory Management                         │  │
│   │  • KV-cache implementation                         │  │
│   │  • PagedAttention                                  │  │
│   │  • Memory-efficient attention                      │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │  Layer 3: Scheduling & Batching                     │  │
│   │  • Continuous batching                             │  │
│   │  • Request scheduling                              │  │
│   │  • Preemption strategies                           │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │  Layer 4: Parallelism & Optimization                │  │
│   │  • Tensor Parallelism (TP)                         │  │
│   │  • Pipeline Parallelism (PP)                       │  │
│   │  • Quantization (INT8, INT4)                       │  │
│   │  • Speculative decoding                            │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   By the end, you'll have built your own "AIOS" -          │
│   an AI Operating System for LLM inference.                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Build From Scratch?

1. **Deep Understanding**: Using vLLM is easy; understanding *why* it works makes you invaluable
2. **Debugging Skills**: When production breaks, you need to understand every layer
3. **Innovation Ability**: The next breakthrough might come from you
4. **Career Growth**: AI infra engineers who understand internals are in high demand

---

## Why LLM Inference Matters

### Training vs Inference

```
┌─────────────────────────────────────────────────────────────┐
│                TRAINING vs INFERENCE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRAINING (Learning)                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data ──► Model ──► Prediction ──► Loss ──► Update   │  │
│  │           │                              │            │  │
│  │           └────────────────◄─────────────┘            │  │
│  │                      Backpropagation                  │  │
│  │                                                       │  │
│  │  • Requires massive compute (1000s of GPUs)          │  │
│  │  • Months of time                                     │  │
│  │  • Cost: $millions to $100+ million                   │  │
│  │  • Done once by AI labs                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  INFERENCE (Using)                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Input ──► Trained Model ──► Output                   │  │
│  │                                                       │  │
│  │  • Runs on your hardware                              │  │
│  │  • Milliseconds to seconds per response               │  │
│  │  • Cost: electricity + hardware                       │  │
│  │  • Happens billions of times daily                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  This course focuses on INFERENCE                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Focus on Inference?

1. **Practical Impact**: Every ChatGPT response, every Copilot suggestion, every AI feature runs inference
2. **Accessible Hardware**: You can run inference on a single GPU; training requires clusters
3. **Optimization Matters**: 10x inference speedup = 10x cost reduction in production
4. **Growing Demand**: The gap between AI researchers and inference engineers is widening

### What Makes Inference Challenging?

```
┌─────────────────────────────────────────────────────────────┐
│              INFERENCE CHALLENGES                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. MEMORY                                                  │
│     ┌────────────────────────────────────────────────┐     │
│     │  Llama 2 7B = 7 billion parameters             │     │
│     │  × 2 bytes (float16) = 14 GB just for weights │     │
│     │  + Activations + KV Cache = 20-40 GB          │     │
│     └────────────────────────────────────────────────┘     │
│                                                             │
│  2. COMPUTE                                                 │
│     ┌────────────────────────────────────────────────┐     │
│     │  Each token: billions of multiply-adds         │     │
│     │  Attention: O(n²) with sequence length        │     │
│     │  Autoregressive: generate one token at a time │     │
│     └────────────────────────────────────────────────┘     │
│                                                             │
│  3. LATENCY                                                 │
│     ┌────────────────────────────────────────────────┐     │
│     │  Users expect real-time responses              │     │
│     │  Time-to-first-token critical for UX          │     │
│     │  Throughput vs latency tradeoffs              │     │
│     └────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Course Roadmap

This course is structured as a progressive journey:

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNING PATH                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Lesson 0: Introduction (You are here!)                     │
│     ├── LLMs vs Traditional Software                        │
│     ├── Why Inference Matters                               │
│     └── Course Overview                                     │
│                                                             │
│  Lesson 1: LLM Basics                                       │
│     ├── Transformer Architecture                            │
│     ├── Attention Mechanism                                 │
│     ├── Positional Encoding                                 │
│     └── Understanding Parameters                            │
│                                                             │
│  Lesson 2: Running Llama 2 with PyTorch                     │
│     ├── Loading Model Weights                               │
│     ├── Tokenization                                        │
│     ├── Forward Pass Implementation                         │
│     └── Text Generation                                     │
│                                                             │
│  [Future Lessons]                                           │
│     ├── KV Cache Optimization                               │
│     ├── Batching Strategies                                 │
│     ├── Quantization                                        │
│     ├── Memory Management                                   │
│     └── Advanced Optimizations                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Makes This Course Different

| Traditional ML Courses | This Course |
|----------------------|-------------|
| Focus on training | Focus on inference |
| Use high-level APIs | Build from scratch |
| Theory-heavy | Implementation-focused |
| Generic examples | Real LLM (Llama/Qwen) |
| Assume ML background | Start from fundamentals |

---

## Prerequisites

### Required Knowledge
- **Python Programming**: Comfortable with classes, functions, and basic data structures
- **Basic Linear Algebra**: Matrix multiplication, vectors, transpose operations

### Helpful But Not Required
- PyTorch basics (we'll teach what you need)
- Neural network fundamentals
- GPU programming concepts

### Setup Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ system RAM

---

## Getting Started

Ready to dive in? Here's your first exercise:

### Exercise 0.1: Traditional vs LLM Thinking

Consider this problem: **"Translate 'Hello, how are you?' to French"**

1. How would you solve this with traditional programming?
   - What dictionaries/rules would you need?
   - How would you handle grammar?
   - What about idioms and context?

2. How does an LLM approach this?
   - What patterns might it have learned?
   - Why doesn't it need explicit grammar rules?

Think about these questions before moving to Lesson 1. The answers will become clear as you build your understanding.

---

## Next Steps

Proceed to [Lesson 1: LLM Basics](../lesson-1-llm-basics/README.md) to learn about the Transformer architecture and attention mechanism.

---

## Additional Resources

### Recommended Reading
- [What Is ChatGPT Doing... and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) - Stephen Wolfram's intuitive explanation
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Visual guide to language models

### Videos
- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) - Beautiful visual explanation
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Building a GPT from scratch

### Papers (For the Curious)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The Transformer paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 paper

---

*Welcome aboard. Let's build something amazing.*
