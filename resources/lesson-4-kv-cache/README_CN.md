# 第 4 课：Prefill/Decode 拆分，理解 KV Cache

## 学习目标

完成本课后，你将能够：

1. 理解自回归 LLM 推理的**两个阶段**：**prefill** 和 **decode**
2. 理解为什么没有 KV cache 的朴素生成方式，总计算量会是 O(n^2)
3. 实现一个简单的**动态 KV cache**：每层使用 `list[tuple[Tensor, Tensor]]` 保存缓存（这也是 HuggingFace `DynamicCache` 的基本策略）
4. 在 Qwen3-8B 上观察到大约 **5 倍加速**（从约 5 tok/s 提升到约 25-30 tok/s）

---

## 计算机科学中的 Cache（快速介绍）

**Cache（缓存）**是一层更小、更快的存储，用来保存那些很可能会被重复使用的数据，
这样我们就不必一遍又一遍地重新计算或重新读取相同内容。

经典例子：
- **CPU cache（L1/L2/L3）**：保存最近使用的内存数据，避免昂贵的 DRAM 访问。
- **Web/CDN cache**：把 HTTP 响应缓存在离用户更近的位置，减少重复的后端工作。
- **数据库缓存**：缓存热点查询结果或热点页面，降低磁盘访问和查询开销。

同样的第一性原理也适用于 LLM 推理：
- 旧数据如果还会复用，就把它留下。
- 只重新计算新增的部分。
- 用一部分内存换取显著的计算和时间收益。

在自回归解码中，历史 token 的 **K/V 状态**会在每一步被反复使用，所以 KV cache 就是注意力计算中最直接的缓存策略。

---

## 为什么需要 KV Cache？

### 核心问题：重复计算

回忆第 1 课，注意力计算公式为：

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

在自回归生成过程中，模型是**一次生成一个 token**。每一步都需要对**全部历史 token**做注意力，才能决定下一个 token 是什么。

**没有 KV cache** 时，朴素做法如下：

```
Step 1:  处理 tokens [0]              → 生成 token 1         （1 个 token）
Step 2:  处理 tokens [0, 1]           → 生成 token 2         （2 个 token）
Step 3:  处理 tokens [0, 1, 2]        → 生成 token 3         （3 个 token）
...
Step n:  处理 tokens [0, 1, ..., n-1] → 生成 token n         （n 个 token）

总工作量 = 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n^2)
```

如果生成 1000 个 token，就意味着大约 500,000 个单位的重复计算。模型会反复读取相同的 prompt token，并一遍又一遍地重新计算它们的 K 和 V 投影。

### 关键洞察：历史 token 的 K 和 V 不会变化

一旦某个 token 被处理过，它对应的 Key 和 Value 向量就是**固定的**（它们只依赖该 token 的位置以及模型权重）。每一步真正需要新算的，其实只是**新 token**对应的 **Query** 向量。

这就是 KV cache 背后的核心思想：**把前面步骤算过的 K 和 V 缓存起来，每一步只为新 token 计算 Q、K、V。**

---

## LLM 推理的两个阶段

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM 推理流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1：PREFILL（处理 prompt）                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  输入："What is the capital of France?"                    │    │
│  │       [tok0, tok1, tok2, tok3, tok4, tok5, tok6]          │    │
│  │                                                            │    │
│  │  并行处理所有 prompt token                                 │    │
│  │  → 为每个位置计算 Q、K、V                                  │    │
│  │  → 将 K、V 写入 cache（按层保存）                          │    │
│  │  → 返回最后一个位置的 logits → 生成第一个输出 token        │    │
│  │                                                            │    │
│  │  计算量：对 T 个 prompt token 是 O(T^2)（一次性成本）      │    │
│  │  这一阶段是计算瓶颈型（大矩阵乘法）                        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  阶段 2：DECODE（逐 token 生成）                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Step 1: 输入 = [new_tok_7]                                │    │
│  │          只为这 1 个 token 计算 Q7、K7、V7                 │    │
│  │          将 K7,V7 追加到 cache → cache 变为 [K0..K7,V0..V7]│    │
│  │          用 Q7 对 cache 中全部 K,V 做注意力                │    │
│  │          → 得到下一个 token                                │    │
│  │                                                            │    │
│  │  Step 2: 输入 = [new_tok_8]                                │    │
│  │          只为这 1 个 token 计算 Q8、K8、V8                 │    │
│  │          将 K8,V8 追加到 cache → cache 变为 [K0..K8,V0..V8]│    │
│  │          用 Q8 对 cache 中全部 K,V 做注意力                │    │
│  │          → 得到下一个 token                                │    │
│  │                                                            │    │
│  │  每一步计算量：O(n)，其中 n 是当前总序列长度               │    │
│  │  这一阶段是内存带宽瓶颈型（从 HBM 中读取缓存的 K,V）       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prefill 与 Decode 的关键差异

```
┌───────────────────┬──────────────────────┬──────────────────────────┐
│       维度        │       Prefill        │          Decode          │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ 处理的 token 数量 │ 一次处理全部 prompt  │ 每一步只处理 1 个 token │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ 计算 Q/K/V 的范围 │ 全部 T 个位置        │ 仅当前 1 个位置         │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ KV cache 动作     │ 填充（写入）         │ 读取 + 追加             │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ 注意力矩阵形状    │ (T, T) 完整矩阵      │ (1, n) 单行 query       │
│                   │                      │ 对全部 key 做注意力     │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ 主要瓶颈          │ 算力瓶颈             │ 内存带宽瓶颈            │
│                   │ （大规模 GEMM）      │ （从 HBM 读 KV）        │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ GPU 利用率        │ 高（FLOP/s 好）      │ 低（矩阵乘法很小）      │
└───────────────────┴──────────────────────┴──────────────────────────┘
```

---

## 数据流：有无 KV Cache 的区别

### 不使用 KV Cache（第 3 课的方式）

```
Step 1: [tok0]                    → forward(4 tokens)   → tok1    重算：0
Step 2: [tok0, tok1]              → forward(5 tokens)   → tok2    重算：tok0
Step 3: [tok0, tok1, tok2]        → forward(6 tokens)   → tok3    重算：tok0,1
...
Step N: [tok0, ..., tokN-1]       → forward(N+3 tokens) → tokN    重算：tok0..N-2

每次 forward 都会重新触碰 T+N 规模的 token，总计算量 → O((T+N)^2)
```

### 使用 KV Cache

```
PREFILL:
  [tok0, tok1, tok2, tok3]  → forward(4 tokens, 填充 cache) → tok4

DECODE:
  Step 1: [tok4]  → forward(1 token, 读取长度为 4 的 cache，并追加) → tok5
  Step 2: [tok5]  → forward(1 token, 读取长度为 5 的 cache，并追加) → tok6
  Step 3: [tok6]  → forward(1 token, 读取长度为 6 的 cache，并追加) → tok7
  ...

总量 = T^2（prefill） + N * (T+N)（decode）≈ O(T^2 + N*T)，当 N >> T 时远小于无 cache 的 O((T+N)^2)
```

---

## 简单的动态 KV Cache 实现

最简单的 KV cache 形式，就是一个 Python 列表，里面每层存一个 `(K, V)` 元组：

```python
# KV cache: 每层保存一个 (key_cache, value_cache) 元组
# key_cache shape:   (batch, num_kv_heads, seq_len_so_far, head_dim)
# value_cache shape: (batch, num_kv_heads, seq_len_so_far, head_dim)

past_key_values: list[tuple[Tensor, Tensor]] = []
```

在 attention 中，使用 `torch.cat` 更新 cache：

```python
# In the attention forward:
if past_key_value is not None:
    # 把新 K,V 沿着 sequence 维度拼接到历史缓存后面
    k = torch.cat([past_key_value[0], k], dim=2)  # dim=2 是 seq_len
    v = torch.cat([past_key_value[1], v], dim=2)

# 新的 (k, v) 就包含了从位置 0 到当前位置的全部 key/value
present_key_value = (k, v)
```

这和 HuggingFace 内部 `DynamicCache` 的工作方式本质上是一样的。

---

## `torch.cat` 的问题

虽然 `torch.cat` 能保证结果正确，但它有一个明显的性能问题：

```
Step 1: cache = [K0, V0]                    → cat 拷贝 1 份
Step 2: cache = [K0, K1, V0, V1]            → cat 拷贝 2 份
Step 3: cache = [K0, K1, K2, V0, V1, V2]    → cat 拷贝 3 份
...
Step n: cache = [K0..Kn, V0..Vn]            → cat 拷贝 n 份

总拷贝量 = 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n^2)
```

每次 `torch.cat` 都会分配一个**新的 tensor**，把整个旧 cache 连同新的条目一起重新复制一遍。对于一个 36 层模型，生成 1000 个 token：

```
分配次数：36 层 × 2（K,V）× 1000 步 = 72,000 次分配
拷贝体量：36 × 2 × (1 + 2 + ... + 1000) ≈ 3600 万份 tensor 拷贝
```

这也是为什么我们会在第 5 课里把 `torch.cat` 替换成预分配 cache。当前这一课里，`torch.cat` 虽然不够高效，但它是**正确的**，并且已经能带来相对“完全不做 cache”大约 **5 倍左右**的加速。

---

## KV Cache 的内存预算

对于使用 bfloat16 的 Qwen3-8B：

```
每层、每个 token：
  K: num_kv_heads × head_dim × 2 bytes = 8 × 128 × 2 = 2,048 bytes
  V: num_kv_heads × head_dim × 2 bytes = 8 × 128 × 2 = 2,048 bytes
  合计：每层每个 token 4,096 bytes

36 层总计：
  4,096 × 36 = 147,456 bytes ≈ 144 KB / token

1000 个 token：
  144 KB × 1000 = 144 MB

最大上下文（40,960 tokens）：
  144 KB × 40,960 ≈ 5.76 GB
```

这也是为什么 KV cache 的内存管理如此重要，也是第 5 课和第 6 课存在的原因。

---

## 运行演示

```bash
pip install -r resources/lesson-4-kv-cache/requirements.txt

# 可传入本地模型路径，也可传入 HuggingFace model id
python resources/lesson-4-kv-cache/run_lesson4.py --model Qwen/Qwen3-0.6B
```

脚本会做以下几件事：
1. 以第 4 课的默认参数调用 `benchmark/bench.py`（`更短输入`、`更长输出`）
2. 运行**开启 KV cache**的 benchmark
3. 运行**关闭 KV cache**的 benchmark
4. 并排打印吞吐对比和加速比

示例输出：

```
[KV_CACHE] Total: 232tok, Time: 6.78s, Throughput: 34.19tok/s
[NO_CACHE] Total: 232tok, Time: 7.02s, Throughput: 33.07tok/s
Speedup: 1.03x
```

---

## 练习

### 练习 1：测试不同 prompt 长度下的 benchmark

修改脚本，测试不同的 prompt 长度（10、50、200、500 token）。观察 prefill 时间如何缩放？它又会如何影响 decode 速度？

### 练习 2：测量 cache 内存

生成 N 个 token 后，打印 `torch.cuda.memory_allocated()`。把“显存占用”与“已生成 token 数量”的关系画出来。结果是否与上面的公式一致？

### 练习 3：检查 cache 的 shape

在 attention forward 内加入调试打印，显示每个 decode 步里缓存的 K 和 V 的 shape。验证 sequence 维度是否每一步都恰好增加 1。

### 练习 4：统计分配次数

使用 `torch.cuda.memory_stats()` 统计有无 KV cache 时的内存分配次数。再与 `torch.cat` 理论上的 O(n^2) 分配数量进行比较。

---

## 接下来是什么？

`torch.cat` 方案虽然可用，但会带来**内存碎片化**（成千上万次分配与复制）。在**第 5 课**中，我们会把它替换为**预分配 KV cache**：先申请一整块连续 buffer，再通过位置索引写入。这样可以消除碎片化，并进一步提升性能。

---

## 延伸阅读

- [The KV Cache Explained (Jay Mody)](https://jaykmody.com/blog/gpt-kv-cache/)
- [Efficient Memory Management for LLM Serving (vLLM paper)](https://arxiv.org/abs/2309.06180)
- [HuggingFace DynamicCache source code](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
