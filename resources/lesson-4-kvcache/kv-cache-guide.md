# KV Cache 教学指南

> 基于 mini-sglang 实现的 KV Cache 机制全解析，为 aios 后续课程做准备。

---

## 1. 为什么需要 KV Cache

### 当前 aios 的问题

在 `llm.py` 的生成循环中，每生成一个 token 都要对**整个序列**做一次完整的 forward：

```python
for _ in range(sp.max_tokens):
    logits = self.model.forward(generated)   # generated 每轮都变长
    next_token = sampler.sample(logits[:, -1, :])
    generated = torch.cat([generated, next_token], dim=-1)
```

假设输入 100 个 token，要生成 100 个 token，总计算量为：

```
第1步: attention 计算 101 个 token
第2步: attention 计算 102 个 token
...
第100步: attention 计算 200 个 token
总计: 101 + 102 + ... + 200 = 15,050 次 token attention
```

但实际上，第 N 步只有**第 N 个 token 的 Q** 是新的，之前所有 token 的 K、V 完全相同。重复计算是巨大的浪费。

### KV Cache 的核心思想

> **缓存已计算的 K、V，每步只计算新 token 的 Q、K、V，用新 Q 对所有缓存的 K、V 做 attention。**

优化后：

```
第1步 (prefill): 计算 100 个 token 的 K、V，缓存；计算 attention 得到下一个 token
第2步 (decode):  只计算 1 个新 token 的 Q、K、V，对 101 个 K、V 做 attention
第3步 (decode):  只计算 1 个新 token 的 Q、K、V，对 102 个 K、V 做 attention
...
```

计算量从 O(n²) 降到 O(n)。这就是为什么 KV Cache 能带来 **~5x** 的吞吐提升。

---

## 2. 两个阶段：Prefill 与 Decode

KV Cache 将推理过程分为两个截然不同的阶段：

### Prefill（预填充）

- **输入**：整个 prompt（如 100 个 token）
- **计算**：一次完整的 forward，所有 token 并行计算
- **输出**：第一个生成的 token + 缓存的 K、V
- **特点**：compute-bound（计算密集），GPU 利用率高

### Decode（解码）

- **输入**：上一步生成的 1 个 token
- **计算**：只算这 1 个 token 的 Q、K、V，与缓存的 K、V 做 attention
- **输出**：下一个 token + 更新缓存（追加新的 K、V）
- **特点**：memory-bound（访存密集），需要读取大量缓存

```
┌─────────────────────────────────────────────────┐
│                  Prefill Phase                   │
│  Input: [t0, t1, t2, ..., t99]  (100 tokens)    │
│  → 计算所有 K, V 并缓存                          │
│  → 输出 t100                                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                  Decode Phase                    │
│  Step 1: input=[t100] → Q100 × [K0..K100] → t101│
│  Step 2: input=[t101] → Q101 × [K0..K101] → t102│
│  Step 3: input=[t102] → Q102 × [K0..K102] → t103│
│  ...                                             │
└─────────────────────────────────────────────────┘
```

---

## 3. mini-sglang 的 KV Cache 架构

mini-sglang 使用**分页 KV Cache（Paged KV Cache）**，灵感来自操作系统的虚拟内存分页。

### 3.1 整体架构

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  CacheManager │────▶│  PageTable    │────▶│  MHAKVCache Pool │
│  (分配/回收)   │     │  (虚→物映射)   │     │  (GPU 显存)       │
└──────────────┘     └───────────────┘     └──────────────────┘
       │                                            ▲
       │                                            │
       ▼                                            │
┌──────────────┐                           ┌────────┴─────────┐
│ PrefixCache  │                           │  Attention Layer  │
│ (前缀复用)    │                           │  store_kv / read  │
└──────────────┘                           └──────────────────┘
```

### 3.2 KV Cache Pool（物理内存）

```python
# mha_pool.py
class MHAKVCache(BaseKVCachePool):
    def __init__(self, num_layers, num_pages, page_size, num_kv_heads, head_dim, dtype, device):
        # 核心数据结构：一块预分配的 GPU 显存
        self._kv_buffer = torch.zeros(
            (2, num_layers, num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype, device=device,
        )
        # 2 = K 和 V
        # num_layers = transformer 层数
        # num_pages = 总页数（由可用显存决定）
        # page_size = 每页包含的 token 数
```

内存布局示意（以 page_size=4 为例）：

```
KV Buffer (K部分, 单层):
┌────────┬────────┬────────┬────────┬─────┐
│ Page 0 │ Page 1 │ Page 2 │ Page 3 │ ... │
│ t0-t3  │ t4-t7  │ t8-t11 │ t12-15 │     │
└────────┴────────┴────────┴────────┴─────┘

每个 Page 内部:
┌───────────────────────────────────────┐
│ Token 0: [kv_heads × head_dim] floats │
│ Token 1: [kv_heads × head_dim] floats │
│ Token 2: [kv_heads × head_dim] floats │
│ Token 3: [kv_heads × head_dim] floats │
└───────────────────────────────────────┘
```

### 3.3 Page Table（虚拟→物理映射）

Page Table 将每个请求的 token 序号映射到 KV Cache Pool 中的物理位置：

```python
# engine.py
page_table = torch.zeros(
    (max_running_req + 1, aligned_max_seq_len),
    dtype=torch.int32, device=device,
)
# page_table[req_idx, token_pos] = cache_location
```

示意：

```
Request 0 (seq_len=6, 占用 Page 0 和 Page 1):
page_table[0] = [0, 1, 2, 3, 4, 5, ...]
                 ↑  ↑  ↑  ↑  ↑  ↑
              Page0内位置    Page1内位置

Request 1 (seq_len=3, 占用 Page 2):
page_table[1] = [8, 9, 10, ...]
                 ↑  ↑   ↑
              Page2 内位置
```

### 3.4 CacheManager（分配与回收）

```python
# cache.py
class CacheManager:
    def __init__(self, num_pages, page_size, page_table):
        # 空闲页列表（页对齐的起始位置）
        self.free_slots = torch.arange(0, num_pages * page_size, page_size)

    def allocate_paged(self, reqs):
        """为请求分配新页面"""
        needed_pages = sum(req.needed_pages for req in reqs)
        if needed_pages > len(self.free_slots):
            self.evict(...)  # 驱逐前缀缓存以腾出空间
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        # 写入 page_table
        self._write_page_table(allocated, reqs)
```

### 3.5 写入与读取 KV Cache

**写入**（在 attention forward 中）：

```python
# attention backend (fa.py / fi.py)
def forward(self, q, k, v, layer_id, batch):
    # 1. 将新计算的 K、V 写入缓存
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    #   batch.out_loc: 由 page_table 查表得到的物理位置

    # 2. 读取完整的 K、V 缓存用于 attention
    k_cache = self.kvcache.k_cache(layer_id)  # (total_tokens, heads, head_dim)
    v_cache = self.kvcache.v_cache(layer_id)

    # 3. 执行 attention（Flash Attention / FlashInfer）
    output = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
```

**store_kv 的 CUDA kernel**（高效写入）：

```python
# kernel/store.py → store.cu
# 每个 warp 处理一个 token 的写入
# indices[i] 指定第 i 个 token 写入缓存的哪个位置
def store_cache(k_cache, v_cache, indices, k, v):
    # k_cache[indices[i]] = k[i]  (向量化)
    # v_cache[indices[i]] = v[i]
```

---

## 4. 请求的生命周期

一个请求在 mini-sglang 中的完整流程：

```
1. 用户提交请求 "Hello, how are you?"

2. Tokenize → input_ids = [15339, 11, 1268, 527, 499, 30]

3. Prefill 阶段:
   a. CacheManager.match_req() → 检查前缀缓存（首次无匹配, cached_len=0）
   b. CacheManager.allocate_paged() → 分配页面
   c. 前向计算：6 个 token 一起算
      - 每层: Q,K,V = project(hidden)
      - store_kv(K, V, locations)  → 写入缓存
      - attention(Q, cached_K, cached_V) → 输出
   d. 采样得到第一个输出 token
   e. req.cached_len = 6, req.device_len = 7

4. Decode 阶段（循环）:
   a. 只输入 1 个新 token
   b. CacheManager.allocate_paged() → 分配 1 个新位置
   c. 前向计算：
      - 每层: q,k,v = project(hidden)  (1个token)
      - store_kv(k, v, new_location) → 追加到缓存
      - attention(q, all_cached_K, all_cached_V)
   d. 采样得到下一个 token
   e. 重复直到 EOS 或 max_tokens

5. 完成:
   a. CacheManager.cache_req() → 将前缀存入 PrefixCache
   b. 释放不再需要的页面
```

### 关键数据流

```python
# 请求对象跟踪的状态
class Req:
    cached_len: int      # 已在 KV cache 中的 token 数
    device_len: int      # 总 token 数（cached + 已生成）
    extend_len: int      # 本次需处理的 token 数 = device_len - cached_len
    remain_len: int      # 剩余可生成 token 数

# Prefill 后:
#   cached_len = input_len, extend_len = 0, device_len = input_len + 1

# 每步 Decode 后:
#   cached_len += 1, device_len += 1, remain_len -= 1
```

---

## 5. 显存预算计算

mini-sglang 根据可用显存动态计算能缓存多少 token：

```python
# engine.py
# 每页 KV cache 的显存占用
cache_per_page = (
    2                          # K + V
    * head_dim                 # 每个 head 的维度
    * num_kv_heads             # KV head 数量
    * page_size                # 每页 token 数
    * dtype.itemsize           # 每个元素的字节数 (bf16=2)
    * num_layers               # transformer 层数
)

# 以 Qwen3-0.6B 为例:
# head_dim=64, num_kv_heads=8, page_size=1, bf16, num_layers=28
# cache_per_page = 2 * 64 * 8 * 1 * 2 * 28 = 57,344 bytes ≈ 56 KB/token

# 可用显存
available = total_gpu_memory * 0.9 - model_memory
num_pages = available // cache_per_page
```

对于 Qwen3-0.6B（~1.2GB），在 24GB 显卡上：
- 可用 ≈ 24 × 0.9 - 1.2 ≈ 20.4 GB
- 每 token ≈ 56 KB
- 可缓存 ≈ 364K tokens

---

## 6. 从 aios 当前代码到 KV Cache 的改造路径

### 第一步：改造 forward 签名

当前 aios 的 attention 只接受完整序列。需要改为接受 `positions` 和 KV cache 参数：

```python
# 当前（无 KV cache）
def forward(self, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = split(qkv)
    q, k = apply_rotary(q, k, positions)  # positions 从序列长度推断
    attn_output = scaled_dot_product_attention(q, k, v)
    return self.o_proj(attn_output)

# 目标（有 KV cache）
def forward(self, hidden_states, positions, kv_cache, cache_locations):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = split(qkv)
    q, k = apply_rotary(q, k, positions)  # 显式传入 positions
    kv_cache.store(k, v, cache_locations)  # 写入缓存
    attn_output = attention_with_cache(q, kv_cache.k, kv_cache.v)
    return self.o_proj(attn_output)
```

### 第二步：改造生成循环

```python
# 当前
for _ in range(max_tokens):
    logits = model.forward(all_tokens)          # 重算所有
    next_token = sample(logits[:, -1, :])
    all_tokens = cat([all_tokens, next_token])

# 目标
# Prefill
logits = model.forward(input_ids, positions=range(len(input_ids)), kv_cache=cache)
next_token = sample(logits[:, -1, :])

# Decode loop
for step in range(max_tokens - 1):
    logits = model.forward(next_token, positions=[current_pos], kv_cache=cache)
    next_token = sample(logits[:, 0, :])
```

### 第三步：预分配 KV Cache

```python
# 简单版本：为每个请求预分配固定长度的缓存
class SimpleKVCache:
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, dtype, device):
        self.k_cache = torch.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_len = 0  # 当前已缓存的 token 数

    def store(self, layer_id, k, v):
        # k, v: (1, num_kv_heads, head_dim) — 单个 token
        self.k_cache[layer_id, self.seq_len] = k
        self.v_cache[layer_id, self.seq_len] = v

    def get(self, layer_id):
        return (
            self.k_cache[layer_id, :self.seq_len + 1],
            self.v_cache[layer_id, :self.seq_len + 1],
        )
```

### 演进路径

```
Lesson 3 (当前): 无 KV Cache，~5 tok/s
    │
    ▼
Lesson 4: 简单 KV Cache（预分配连续内存）
    │   - 改造 forward 签名
    │   - 实现 SimpleKVCache
    │   - 分离 prefill / decode
    │   - 预期 ~25 tok/s（~5x 提升）
    │
    ▼
Lesson 5: 分页 KV Cache（Paged Attention）
    │   - PageTable + CacheManager
    │   - 支持多请求并发
    │   - 动态分配/回收
    │
    ▼
Lesson 6: Continuous Batching
    │   - Prefill + Decode 交替调度
    │   - 请求级别的生命周期管理
    │
    ▼
Lesson 7: FlashAttention / Kernel 优化
        - 使用 perf.py 对比性能
        - 自定义 store kernel
```

---

## 7. 关键概念速查

| 概念 | 说明 |
|------|------|
| **Prefill** | 处理整个 prompt，填充 KV cache，compute-bound |
| **Decode** | 逐 token 生成，读取 KV cache，memory-bound |
| **Page Table** | 虚拟地址→物理地址映射，`page_table[req][pos] = cache_loc` |
| **Page Size** | 每页包含的 token 数，越大越高效但浪费越多（内部碎片） |
| **Prefix Cache** | 缓存相同前缀的 KV，避免重复 prefill |
| **Radix Tree** | 前缀缓存的数据结构，按 token 序列建立 trie |
| **store_kv** | 将新 K、V 写入缓存指定位置的操作（可用 CUDA kernel 加速） |
| **out_loc** | 由 page_table 查表得到的写入位置向量 |
| **TTFT** | Time To First Token，prefill 延迟 |
| **TPOT** | Time Per Output Token，decode 延迟 |

---

## 8. 用 Benchmark 验证优化效果

添加 KV Cache 前后，使用已实现的 benchmark 工具对比：

```bash
# 基线（Lesson 3，无 KV cache）
python benchmark/bench.py --model Qwen/Qwen3-0.6B --num-seqs 4 --max-input-len 128 --max-output-len 64
# 预期: ~5 tok/s

# 添加 KV cache 后
python benchmark/bench.py --model Qwen/Qwen3-0.6B --num-seqs 4 --max-input-len 128 --max-output-len 64
# 预期: ~25 tok/s

# 使用 perf.py 对比 kernel 性能
from aios.benchmark.perf import compare_memory_kernel_perf
compare_memory_kernel_perf(
    baseline=torch_store_fn,
    our_impl=custom_store_fn,
    memory_footprint=k.nbytes + v.nbytes,
    description="KV Store Kernel",
)
```
