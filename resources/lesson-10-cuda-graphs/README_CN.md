# Lesson 10：CUDA Graphs（Decode Replay）

## 1. 背景（Background / Why）

Lesson 9 已经完成 fused layers：QKV projection、gate/up projection、SwiGLU、RMSNorm 和 KV store 都被压缩成更少的 GPU 算子。单层 forward 已经更紧凑，但 decode 阶段还有一个新的瓶颈：

> 每生成 1 个 token，Python 都要重新调度整套 decode forward。

对 Qwen3-0.6B 这样的模型，一次 decode step 会经过 28 层。即使每层已经 fused，整个 step 仍然包含大量 kernel launch、FlashInfer plan/run、embedding、linear、norm、sampler 等调度动作。当 batch size 较小、每步只处理 1 个 token 时，CPU launch overhead 会占据很明显的比例。

CUDA Graphs 解决的不是数学计算量，而是**重复执行同一形状 decode step 时的 CPU 调度成本**。本课只捕获 decode，不捕获 prefill：

- prefill 是变长输入，shape 变化大，不适合作为本课第一版 graph。
- decode 每个请求每步只新增 1 个 token，shape 稳定，是 CUDA Graph 最自然的入口。
- Lesson 9 的 fused layer 已经让 decode 图更紧凑，适合在 Lesson 10 捕获。

## 2. 原理（Principle / What）

### 2.1 CUDA Graph 做了什么

普通 eager decode 每一步都是：

```text
Python
  -> launch embedding
  -> launch layer0 qkv/norm/attention/mlp
  -> launch layer1 ...
  -> ...
  -> launch lm_head
  -> sample
```

CUDA Graph capture 会先用固定 shape 跑一次，把这串 CUDA 操作记录成一张图。之后 replay 时，Python 不再逐个 launch kernel，而是一次性提交整张图：

```text
capture:
  static buffers + fixed batch shape -> CUDAGraph

replay:
  copy current input_ids/out_loc/positions -> static buffers
  update attention metadata
  graph.replay()
  read logits[:real_batch_size]
```

核心约束是：**graph replay 时设备指针和 tensor shape 必须稳定**。值可以变，地址不能随每步变化。

### 2.2 为什么要 padding decode batch

连续批处理中，每一步 decode 的真实 batch size 可能变化：

```text
step 0: 4 requests
step 1: 4 requests
step 2: 3 requests
step 3: 2 requests
```

CUDA Graph 不能用同一张图覆盖任意 batch size，所以我们捕获一组 bucket：

```text
graph_bs = [1, 2, 4]
```

真实 batch size 会 pad 到第一个足够大的 bucket：

```text
real bs=3 -> padded bs=4
real bs=2 -> padded bs=2
real bs=1 -> padded bs=1
```

padding 使用 dummy request。dummy request 的 page table 指向一个专门的 dummy KV page，不参与真实采样和结果回收。

### 2.3 Graph Capture Buffer

捕获时，模型不能直接使用每步新建的 `batch.input_ids`、`batch.out_loc`、`batch.positions`，因为这些 tensor 地址会变化。本课引入固定缓冲：

```text
GraphCaptureBuffer
  input_ids:  [max_graph_bs]
  out_loc:    [max_graph_bs]
  positions:  [max_graph_bs]
  logits:     [max_graph_bs, vocab_size]
```

capture 时，`batch.input_ids/out_loc/positions` 被替换成这些固定 buffer 的 slice。replay 前，只把当前 batch 的值 copy 进去：

```python
self.input_ids[:batch.padded_size].copy_(batch.input_ids)
self.out_loc[:batch.padded_size].copy_(batch.out_loc)
self.positions[:batch.padded_size].copy_(batch.positions)
```

这样 graph 看到的 tensor 地址不变，但内容随每步更新。

### 2.4 Attention metadata 也要 graph-aware

FlashInfer decode graph 使用 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`。它需要固定的 GPU buffers：

- `indptr_buffer`
- `indices_buffer`
- `last_page_len_buffer`

普通 decode 仍然使用 `BatchDecodeWithPagedKVCacheWrapper`。Graph replay 时，AIOS 会把普通 `prepare_metadata()` 生成的 metadata 重新绑定到 graph wrapper，然后调用 plan，把当前 page table / seq lens 更新到固定缓冲。

## 3. 具体实现（Implementation / How）

### 3.1 `python/aios/core.py`

`Batch` 新增 mini-sglang 对齐字段：

```python
padded_reqs: List[Req]

@property
def padded_size(self) -> int:
    return len(self.padded_reqs)
```

`batch.reqs` 仍表示真实请求；`batch.padded_reqs` 表示本次 forward 实际进入模型的请求列表，可能包含 dummy request。

同时新增 `clear_global_ctx()`，让 benchmark runner 可以在同一 Python 进程里先创建 eager LLM，再释放后创建 CUDA Graph LLM。

### 3.2 `python/aios/engine/graph.py`

本课新增 `GraphRunner`，职责与 mini-sglang 对齐：

```python
GraphRunner
  graph_bs_list
  buffer: GraphCaptureBuffer
  graph_map: dict[int, torch.cuda.CUDAGraph]

  pad_batch(batch)
  replay(batch)
  destroy_cuda_graphs()
```

捕获流程：

```python
for bs in sorted(graph_bs_list, reverse=True):
    batch = Batch(reqs=[dummy_req] * bs, phase="decode")
    batch.padded_reqs = batch.reqs
    attn_backend.prepare_for_capture(batch)
    buffer.set_batch(batch)
    with ctx.forward_batch(batch):
        buffer.logits[:bs].copy_(model.forward())   # warmup
        with torch.cuda.graph(graph, pool=pool):
            buffer.logits[:bs].copy_(model.forward())
```

`pool=graph.pool()` 会复用 CUDA graph memory pool，避免每个 batch size 捕获都重复占用过多显存。

replay 流程：

```python
buffer.copy_from(batch)
attn_backend.prepare_for_replay(batch)
graph_map[batch.padded_size].replay()
return buffer.logits[:batch.size]
```

注意返回只切真实 `batch.size`，dummy request 的 logits 不参与采样。

### 3.3 `python/aios/attention/base.py`

attention backend 接口新增三组方法：

```python
init_capture_graph(max_seq_len, bs_list)
prepare_for_capture(batch)
prepare_for_replay(batch)
```

`HybridAttentionBackend` 会把这些调用转发给 decode backend，因为本课只捕获 decode。

### 3.4 `python/aios/attention/flashinfer.py`

FlashInfer backend 新增 `FlashInferCaptureData`：

```python
seq_lens
cu_seqlens_k
cu_seqlens_q
page_table
indices = page_table.view(-1)
```

普通 `prepare_metadata()` 改为读取 `batch.padded_reqs`。这样 prefill 仍然是真实请求列表，decode graph 则可以包含 dummy request。

capture 时创建 graph wrapper：

```python
flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    use_tensor_cores=...,
    indptr_buffer=capture.cu_seqlens_k[:bs + 1],
    indices_buffer=capture.indices,
    last_page_len_buffer=capture.seq_lens[:bs],
)
```

replay 时把当前 batch 的 metadata 绑定到对应 bucket 的 graph wrapper：

```python
metadata.wrapper = self._graph_wrappers[batch.padded_size]
self._initialize_metadata_once(metadata)
```

这里保留 mini-sglang 的一个细节：FlashInfer plan 会使用 pinned host staging buffer，本课用 `torch.cuda.Event` 在连续 plan 之间同步，避免 host buffer 被下一次 plan 提前改写。

### 3.5 `python/aios/scheduler/scheduler.py`

`_prepare_batch()` 新增 padding 时机：

```python
if graph_runner is not None:
    graph_runner.pad_batch(batch)
else:
    batch.padded_reqs = batch.reqs
```

随后只给真实请求分配 KV page：

```python
cache_manager.allocate_paged(batch.reqs, page_table)
```

但构造模型输入时使用 `padded_reqs`：

```python
positions = _make_positions(batch)
input_mapping = _make_input_tuple(batch)
batch.input_ids = token_pool[input_mapping]
batch.out_loc = page_table[input_mapping]
```

这保证 dummy request 进入模型时也有合法的 `input_id/position/out_loc`。

### 3.6 `python/aios/llm/llm.py`

CUDA Graph 要求 page table 地址稳定，所以 `LLM.__init__()` 现在提前创建：

```python
page_table = torch.zeros((max_running_reqs + 1, max_seq_len), device="cuda")
dummy_table_idx = max_running_reqs
dummy_page = num_pages
page_table[dummy_table_idx].fill_(dummy_page)
```

KV cache 多分配 1 个 page 给 dummy request：

```python
MHAKVCache(num_pages=num_pages + 1)
CacheManager(num_pages)  # 真实请求只能分配真实页
```

启用方式：

```python
LLM(
    model_path,
    enable_cuda_graph=True,
    max_running_reqs=4,
    cuda_graph_bs=[1, 2, 4],
)
```

普通路径默认不启用 CUDA Graph，因此 Lesson 9 的 eager fused path 保持兼容。

### 3.7 `benchmark/bench.py`

主 benchmark 增加：

```bash
--cuda-graph
--cuda-graph-max-bs
```

用于直接比较同一 workload 下的 eager decode 与 graph replay。

### 3.8 `resources/lesson-10-cuda-graphs/run_lesson10.py`

runner 支持：

```bash
--suite e2e
--suite bench
--suite all
--cuda-visible-devices
```

`bench` 会分别创建 eager LLM 和 CUDA Graph LLM。模型加载和 graph capture 不计入生成耗时，capture time 单独报告。

## mini-sglang 对齐与差异

| 部分 | AIOS Lesson 10 | mini-sglang |
|---|---|---|
| `Batch.padded_reqs/padded_size` | 相同语义 | 相同 |
| `GraphRunner` 职责 | capture / pad / replay | 相同 |
| dummy request | 额外 table row + dummy KV page | 相同 |
| capture 范围 | decode only | decode only |
| FlashInfer graph wrapper | `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | 相同 |
| graph bucket | 用户指定或默认小 bucket | 可按显存扩到更大 bucket |
| CUDA stream | 使用当前 stream 的课程简化版 | 独立 engine stream |
| 分布式/TP | 未实现 | 已支持 |

本课的主要教学化简是：单 GPU、单 stream、只捕获 FlashInfer decode。它保留了 mini-sglang 的核心数据面和调用时机，但不提前引入 TP 和服务器循环。

## 4. 验证结果（Verify）

### 4.1 编译检查

```bash
PYTHONPATH=python python -m compileall -q \
  python/aios benchmark/bench.py resources/lesson-10-cuda-graphs/run_lesson10.py
```

结果：通过。

### 4.2 E2E CUDA Graph 生成

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-lesson10 \
PYTHONPATH=python \
python resources/lesson-10-cuda-graphs/run_lesson10.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --cuda-visible-devices 1 \
  --suite e2e \
  --max-running 4 \
  --cuda-graph-bs 1,2,4
```

实测输出：

```text
[E2E_CUDA_GRAPH] token_ids=[12555, 374, 279, 897]
```

### 4.3 Eager vs CUDA Graph benchmark

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-lesson10 \
PYTHONPATH=python \
python resources/lesson-10-cuda-graphs/run_lesson10.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --cuda-visible-devices 1 \
  --suite bench \
  --num-seqs 4 \
  --min-prompt-len 16 \
  --max-prompt-len 32 \
  --max-tokens 8 \
  --max-running 4 \
  --cuda-graph-bs 1,2,4
```

实测输出：

```text
Workload: num_seqs=4 prompt_len=16..32 max_tokens=8 max_running=4 graph_bs=[1, 2, 4]
[EAGER_DECODE] output_tokens=32 elapsed=0.11s tps=279.86
[CUDA_GRAPH] output_tokens=32 elapsed=0.04s tps=716.47
Summary: capture_time=1.24s speedup=2.56x eager_tps=279.86 graph_tps=716.47
```

这个 workload 很小，因此 capture time 不应该算进单次请求收益。生产系统通常在模型初始化阶段完成 capture，后续大量 decode step 复用 replay。

### 4.4 课程结论

Lesson 10 解决的是 decode 阶段的 CPU launch overhead：

```text
Lesson 9: 让单次 decode forward 更少、更融合
Lesson 10: 让重复 decode forward 用 graph replay 提交
```

完成本课后，AIOS 的 decode 路径具备了生产推理引擎常见的执行形态：scheduler 仍然动态管理请求，但进入 GPU 的 decode batch 会被 pad 到固定 bucket，并通过 CUDA Graph replay 执行。
