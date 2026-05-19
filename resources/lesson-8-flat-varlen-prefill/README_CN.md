# Lesson 8：FlashInfer Flat Attention + Attention Backend

本课解决两个问题：

1. Lesson 7 的 prefill 很难把多个不同长度 prompt 合成一个 batch，只能退化成一次处理一个请求。
2. 模型层 `Qwen3Attention` 混入 KV cache、page table、FlashInfer metadata 等执行系统细节，后续接 CUDA graph、prefix cache、chunked prefill 会越来越乱。

本课的核心改动是：

- 用 **flat token** 表达 varlen prefill/decode。
- 接入 **FlashInfer paged KV attention**。
- 引入 **AttentionBackend**，把模型数学结构和推理执行细节拆开。
- 对齐 mini-sglang：scheduler 准备 batch metadata，attention backend 负责 KV 写入和 kernel dispatch。

---

## 1. 背景：当前还存在什么问题？

### 1.1 Prefill 的二维 batch 形状不适合 varlen prompt

传统 batch 会把 prompt padding 成二维矩阵：

```text
req0: [a0 a1 a2 a3]
req1: [b0 b1 PAD PAD]
req2: [c0 c1 c2 PAD]
```

这会浪费 prefill attention 计算，也不适合 FlashInfer 这类 varlen kernel。Lesson 8 改成 flat 表达：

```text
q = [req0_0 req0_1 req0_2 req0_3 req1_0 req1_1 req2_0 req2_1 req2_2]
```

请求边界用 cumulative length 表达：

```text
extend_lens = [4, 2, 3]
cu_seqlens_q = [0, 4, 6, 9]

req0 -> q[0:4]
req1 -> q[4:6]
req2 -> q[6:9]
```

### 1.2 模型层 attention 承担了太多系统职责

旧实现里 `Qwen3Attention.forward()` 同时负责：

```text
q/k/v projection
RoPE
KV cache 写入
page table 索引
FlashInfer metadata
prefill/decode kernel dispatch
```

mini-sglang 的边界更清晰：

```text
Qwen3Attention:
  只负责 q/k/v、QK norm、RoPE、o_proj

AttentionBackend:
  负责 KV cache 写入、metadata、FlashInfer wrapper plan/run
```

Lesson 8 把 AIOS 改成这个方向。

---

## 2. 原理：这节课做了什么？

### 2.1 Flat token + cumulative length

每个 batch 不再保存 `(B, seq_len)`，而是保存本轮需要计算的 flat token：

```text
batch.input_ids: (total_extend_tokens,)
batch.positions: (total_extend_tokens,)
batch.out_loc:   (total_extend_tokens,)
```

其中：

- `input_ids`：本轮要送进模型的 token。
- `positions`：对应 RoPE position。
- `out_loc`：当前 token 的 K/V 要写入 KV cache 的物理位置。

FlashInfer metadata 里最重要的是两组 cumulative length：

```text
cu_seqlens_q:
  query flat tensor 的请求边界

cu_seqlens_k:
  每个请求完整 KV 序列的边界
```

代码统一用一行计算：

```python
cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum_(dim=0)
```

decode 是 flat 表达的特例：每个请求本轮只扩展一个 token，所以 `seqlens_q = [1, 1, ...]`。

### 2.2 FlashInfer 两个 wrapper

本课使用 FlashInfer 的两个 paged KV attention wrapper：

```python
flashinfer.BatchPrefillWithPagedKVCacheWrapper
flashinfer.BatchDecodeWithPagedKVCacheWrapper
```

调用分两步：

```text
plan(metadata): 根据 batch 形状和 page table 准备 kernel
run(q, kv_cache): 每一层执行 attention
```

prefill wrapper 需要 `qo_indptr/paged_kv_indptr/paged_kv_indices`；decode wrapper 需要 `indptr/indices/seq_lens`。两者都依赖 page table 描述逻辑 token 到物理 KV slot 的映射。

### 2.3 Scheduler 和 AttentionBackend 的职责边界

本课的执行顺序：

```text
PrefillManager / DecodeManager
  -> 返回 Batch(reqs, phase)

Scheduler._prepare_batch
  -> 分配 KV slots
  -> 构造 input_ids / positions / out_loc
  -> attn_backend.prepare_metadata(batch)

Engine.run_batch
  -> with ctx.forward_batch(batch)
  -> model.forward()

Qwen3Attention
  -> q/k/v + RoPE
  -> attn_backend.forward(...)
```

也就是说，metadata 不再在模型层临时构造，而是在 scheduler 阶段准备好。

---

## 3. 具体实现：改了什么代码？

### 3.1 `python/aios/core.py`

新增 mini-sglang 风格的核心数据结构：

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    attn_metadata: BaseAttentionMetadata = field(init=False)
```

`Context.forward_batch(batch)` 用于在模型 forward 期间暴露当前 batch：

```python
with ctx.forward_batch(batch):
    logits = model.forward()
```

### 3.2 `python/aios/scheduler/common.py`

保留 mini-sglang 的 pending/running 分层：

```python
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
```

`PendingReq` 还没有 `table_idx/cached_len/device_len`；只有被 prefill admit 后才会变成真正的 `Req`。

### 3.3 `python/aios/scheduler/prefill.py`

`PrefillManager.schedule_next_batch()` 现在会选择多个 pending 请求组成 prefill batch：

```python
for pending in self.pending_list:
    if not self._can_admit(...):
        break
    selected.append(pending)
```

admit 时只做两件事：

```python
table_idx = self.table_manager.allocate()
token_pool[table_idx, :prompt_len] = pending.input_ids
```

KV page 不在这里分配，而是交给 `Scheduler._prepare_batch()`，这和 mini-sglang 的调用时机一致。

### 3.4 `python/aios/scheduler/decode.py`

DecodeManager 对齐 mini-sglang：

```python
running_reqs: Set[Req]

def schedule_next_batch(self) -> Batch | None:
    return Batch(reqs=list(self.running_reqs), phase="decode")
```

prefill 结束后，scheduler 通过 `filter_reqs()` 把仍可继续 decode 的请求加入 running set。

### 3.5 `python/aios/scheduler/scheduler.py`

`_prepare_batch()` 是调度侧最关键的函数：

```python
def _prepare_batch(self, batch: Batch) -> Batch:
    self.cache_manager.allocate_paged(batch.reqs, self.table_manager.page_table)
    batch.positions = _make_positions(batch, self.device)
    input_mapping = _make_input_tuple(batch, self.device)
    batch.input_ids = self.table_manager.token_pool[input_mapping].long()
    batch.out_loc = self.table_manager.page_table[input_mapping]
    self.attn_backend.prepare_metadata(batch)
    return batch
```

这里有三个映射：

```text
positions:
  [cached_len, device_len) 的 flat position

input_ids:
  token_pool[table_idx, position]

out_loc:
  page_table[table_idx, position]，用于写当前层 K/V
```

`out_loc` 只表示本轮新增 token 的 K/V 写入位置；attention 读取完整历史 KV 时使用的是 FlashInfer metadata 中的 `indices`。

### 3.6 `python/aios/attention/flashinfer.py`

`FlashInferAttentionBackend.prepare_metadata()` 构造本轮 batch 的 FlashInfer metadata：

```python
seqlens_q = [req.extend_len for req in reqs]
seqlens_k = [req.device_len for req in reqs]

cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(0)
cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(0)

indices = torch.cat([
    page_table[req.table_idx, :req.device_len]
    for req in reqs
])
```

`indices` 来自完整 page table，而不是 `out_loc`：

```text
out_loc:
  当前新 token 的写入位置

indices:
  当前请求完整上下文的 KV 读取位置
```

backend forward 负责写 K/V 并调用 FlashInfer：

```python
paged_kv_cache.store_kv(k, v, batch.out_loc.view(-1), layer_id)
attn_output = metadata.wrapper.run(q, kv_cache)
```

当前实现固定 `page_size=1`。后续如果支持 `page_size>1`，需要进一步区分 page index 和 page 内 offset。

### 3.7 `python/aios/models/qwen3.py`

模型层只保留 Qwen3 attention 数学结构：

```python
q = q_proj(hidden_states)
k = k_proj(hidden_states)
v = v_proj(hidden_states)
q, k = q_norm(q), k_norm(k)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
attn_output = attn_backend.forward(q, k, v, kv_cache, layer_id, batch)
```

模型不再构造 FlashInfer metadata，也不直接关心 page table。

### 3.8 `python/aios/layers/embedding.py`

prefill 阶段只对每个请求最后一个 hidden 做 vocab projection：

```python
if batch.is_prefill:
    indices = batch.attn_metadata.get_last_indices(batch.size)
    x = x[indices].contiguous()
```

因此 prefill 的 LMHead 输入从：

```text
(total_prompt_tokens, hidden)
```

变成：

```text
(batch_size, hidden)
```

### 3.9 `python/aios/llm/llm.py`

LLM 初始化全局 context：

```python
self.ctx = Context(page_size=1)
self.ctx.kv_cache = self.mha_kv_cache
self.ctx.attn_backend = self.model.attn_backend
set_global_ctx(self.ctx)
```

每次 `generate()` 创建当前请求使用的 page table，并交给 context：

```python
self.ctx.page_table = page_table
```

---

## 4. 验证结果

### 4.1 编译检查

```bash
python -m compileall -q python/aios
```

结果：通过。

### 4.2 真实 prompt continuous batching

命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-e2e \
PYTHONPATH=python \
python -m aios \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --prompt '2+3 等于多少？只回答数字。' \
  --prompt '用英文说 hello world。' \
  --max-tokens 4 \
  --temperature 0 \
  --max-running-reqs 2 \
  --device cuda
```

输出：

```text
=== prompt 0: '2+3 等于多少？只回答数字。' ===
2 + 3

=== prompt 1: '用英文说 hello world。' ===
Hello world!
```

### 4.3 Lesson 7 vs Lesson 8 吞吐对比

`run_lesson8.py` 在同一 workload 上跑两次：

- `LESSON7_COMPAT`：限制 `prefill_token_budget`，模拟 Lesson 7 中 prefill 基本一次只 admit 一个请求。
- `LESSON8_VARLEN`：不限制 prefill budget，启用 flat varlen prefill 合批。

| Workload | Lesson 7 compat | Lesson 8 varlen prefill | Speedup |
| --- | ---: | ---: | ---: |
| 16 seq, prompt 32..256, output 32 | 256.31 tok/s | 308.91 tok/s | 1.21x |
| 32 seq, prompt 128..512, output 8 | 217.79 tok/s | 408.53 tok/s | 1.88x |

Lesson 8 优化的是 prefill 合批，所以 prompt 越长、output 越短，端到端收益越明显；decode token 占比越高，speedup 会被 decode 阶段稀释。

### 4.4 当前仍未实现

- CUDA graph decode capture/replay。
- CUDA graph 所需的 batch padding。
- prefix cache/radix cache。
- chunked prefill。
- 多 attention backend 和 tensor parallel。
