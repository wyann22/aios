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

### 2.1 CUDA Graph 优化的到底是什么

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

这里的瓶颈不一定在 GPU 计算本身。一次 Python 算子调用通常还要经过
Python dispatcher、PyTorch C++、CUDA runtime/driver，最后才把 kernel 提交到
GPU stream。如果 kernel 本身很短，host 可能来不及持续供给工作，GPU 时间线上就会
出现 launch gap。

CUDA Graph 把 kernel、memcpy、memset 等 GPU 操作表示为节点，把 stream 顺序或 event
表达为依赖边。它**不会融合 kernel，也不会减少模型 FLOPs**；它优化的是整组操作的
提交方式：先记录完整工作流，后续通过一次 `cudaGraphLaunch` 提交可执行图。
[NVIDIA 的入门示例](https://developer.nvidia.com/blog/cuda-graphs/)中，一个
2.9 μs 的短 kernel 在普通异步 launch 模式下平均耗时为 3.8 μs；使用 graph 后降为
3.4 μs。具体数值与硬件和 CUDA 版本有关，但它说明了为什么“kernel 很快”时 launch
开销反而更显眼。

![Eager 与 CUDA Graph 的提交时间线](./cuda_graph.png)

可以用一个简化模型表示两条路径：

```text
T_eager ≈ Σ T_kernel + N × T_launch + T_gap
T_graph ≈ Σ T_kernel + T_graph_launch + T_copy/plan
```

因此 CUDA Graph 最适合：同一拓扑重复次数多、单个 kernel 较短、CPU enqueue-bound
的工作负载。大 batch 或长 prefill 往往由 GPU 计算主导，graph 的相对收益会下降。

### 2.2 Definition、Instantiation 与 Execution

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
把 graph 工作流明确拆成三个阶段：

1. **Definition**：创建节点和依赖，得到 `cudaGraph_t`。可以逐个调用 Graph API，
   也可以用 stream capture 记录现有 CUDA 代码。本课采用后者。
2. **Instantiation**：校验 topology，并提前完成大量 launch 初始化，得到可重复执行的
   `cudaGraphExec_t`。这一步可能较慢，但只需要对每个 graph bucket 做一次。
3. **Execution**：把 executable graph launch 到一个 CUDA stream。同一个 graphExec
   可以反复执行，不需要再次实例化。

PyTorch 的 `torch.cuda.CUDAGraph` 和 `torch.cuda.graph(...)` 封装了底层对象：

```python
g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g, pool=shared_pool):
    static_logits.copy_(model.forward())  # capture

static_input.copy_(new_input)
g.replay()                               # repeated execution
```

stream 进入 capture mode 后，提交到该 stream 的工作会被记录为 graph 节点，而不是按
普通方式立即入队。capture 期间不能执行会破坏依赖记录的同步操作；CUDA 官方文档也
明确禁止在 active capture 上同步 stream/device 或使用某些同步 API。

### 2.3 为什么必须 warmup

第一次 forward 经常包含真正推理阶段不会重复出现的工作：

- FlashInfer/Triton JIT 编译与模块加载；
- cuBLAS、allocator、library handle 的 lazy initialization；
- workspace 和临时 tensor 的首次分配；
- CUDA Graph 第一次 launch 所需的 device upload。

如果这些一次性操作混入 capture，可能直接触发 capture error，也可能把不应该重复的
节点固化到 graph 中。[PyTorch 官方 CUDA Graph 说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
建议先在 side stream 上执行 warmup，再开始正式 capture。AIOS 的 `GraphRunner`
也在进入 `torch.cuda.graph(...)` 前先执行一次相同 shape 的 model forward。

### 2.4 静态地址：值可以变化，指针不能变化

graphExec 记录的不只是“调用哪个 kernel”，还包括 kernel 参数以及 tensor 指针。
PyTorch replay 会再次运行相同 kernels 和相同参数；对 pointer 参数而言，这意味着显存
地址必须与 capture 时一致。下面这种写法不能直接 replay：

```python
# Wrong: each step may produce tensors with different storage addresses.
batch.input_ids = new_input
g.replay()
```

正确做法是长期持有 capture 时使用的静态 tensor，只更新它们的内容：

```python
static_input.copy_(new_input)  # value changes, address does not
g.replay()
output = static_logits[:real_batch_size]
```

![CUDA Graph 生命周期与静态地址](./cuda_graph_lifecycle.png)

这正是 `GraphCaptureBuffer` 存在的原因。`input_ids`、`out_loc`、`positions` 和
`logits` 在 capture 后一直存活；replay 前 copy 新值，模型看到的 device pointer
始终不变。

### 2.5 Graph Memory Pool

capture 期间产生的 GPU allocation 也必须能在 replay 时回到兼容地址。PyTorch 为
CUDA Graph 管理私有 memory pool；不同 graph 如果确定不会并发 replay，可以共享
pool 以减少重复预留。AIOS 捕获最大 bucket 后保存 `graph.pool()`，后续 bucket 通过：

```python
with torch.cuda.graph(graph, pool=pool, stream=self.stream):
    ...
```

共享同一个 pool。这降低了 `[1, 2, 4, ...]` 多个 decode graph 的额外显存成本，但也
意味着这些 graph 不能在同一时刻并发使用这块共享内存。本课的单 engine stream
天然满足这一条件。

### 2.6 动态 batch 如何映射到静态 graph

graph replay 的 kernel topology 和 tensor shape 必须与 capture 兼容，但连续批处理的
真实 batch size 会变化：

```text
step 0: 4 requests
step 1: 4 requests
step 2: 3 requests
step 3: 2 requests
```

AIOS 不为每一个可能的 batch size 都捕获一张图，而是准备少量 bucket：

```text
graph_bs = [1, 2, 4]

real bs=3 -> padded bs=4
real bs=2 -> padded bs=2
real bs=1 -> padded bs=1
```

padding 使用 dummy request。dummy request 拥有独立 table row 和 dummy KV page，
所以它能安全经过 embedding、attention 和 MLP，但不会进入真实采样与资源回收。
这是一种典型的工程交换：用少量冗余 GPU 计算换取有限数量的静态 graph。

如果 shape、kernel 类型或控制流发生拓扑变化，需要选择另一个 bucket 或重新 capture。
CUDA 也支持在 topology 兼容时通过 `cudaGraphExecUpdate` 更新节点参数，但 AIOS 当前
通过静态 buffer 和多 bucket 解决动态输入，没有使用 Graph Update。

### 2.7 为什么只捕获 decode，不捕获 prefill

decode 每个请求每步只新增一个 token。在 bucket 固定后，模型输入 token 数、QKV shape、
attention kernel 配置和输出 shape 都稳定，而且一条请求会重复几十到上千个 decode
step，capture 成本容易摊薄。

prefill 则相反：prompt 长度和总 token 数高度动态，单步计算量也更大。强行 padding
prefill 会浪费大量算力，并需要更多 graph buckets；此时 GPU compute 往往已经能隐藏
大部分 host launch 开销。因此本课保持：

```text
prefill -> eager FlashInfer wrapper
decode  -> CUDA Graph + graph-aware FlashInfer wrapper
```

更精确地说，本课捕获的是从 embedding 到 LM head logits 的 `model.forward()`；
sampling 和 scheduler 的 CPU 状态更新仍在 graph 外执行。这样保持 graph topology
稳定，也不提前引入 Lesson 11 的 batch sampler。

### 2.8 Attention metadata 也必须 graph-aware

模型静态并不代表 attention metadata 静态。每个 decode step 的序列长度和 page table
内容仍在变化。FlashInfer 为此提供
[`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`](https://docs.flashinfer.ai/api/attention.html)，
构造时接收地址固定的：

- `indptr_buffer`
- `indices_buffer`
- `last_page_len_buffer`

普通 decode 使用 `BatchDecodeWithPagedKVCacheWrapper`；graph replay 时，AIOS 把本轮
`prepare_metadata()` 生成的长度和 page table 信息 plan 到 graph wrapper 的固定
buffers，然后 replay 外层 model graph。这里仍然遵守“值可变、地址不变”。

### 2.9 Capture 成本与盈亏平衡

CUDA Graph 不是免费优化。它增加了 warmup、capture、instantiation、静态 buffer 和
graph pool 显存。若 capture 成本为 `C`，每次 replay 相比 eager 节省 `ΔT`，至少需要：

```text
replay_count > C / ΔT
```

次 replay 才能摊平初始化成本。NVIDIA 的示例还指出第一次 graph launch 可能比后续
launch 更慢，因此 benchmark 应把 capture time 单独报告，并在 warmup 后测 steady
state。CUDA Graph 也不保证所有模型都加速：当工作负载 compute-bound、重复次数很少，
或者为了静态 shape 付出过多 padding 时，收益可能很小甚至为负。

本课 benchmark 因此同时报告：capture time、steady-state throughput、speedup，以及
eager/graph token ids 是否一致。

### 2.10 官方资料索引

- [NVIDIA CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)：节点、依赖、stream capture、instantiation、execution 与 graph update。
- [NVIDIA Technical Blog：Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)：launch overhead、capture/instantiate 示例与初始化成本。
- [PyTorch：Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)：静态 tensor、warmup、memory pool 和 replay 工作流。
- [FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)：普通 decode wrapper 与 CUDA Graph decode wrapper 的 buffer 接口。

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

`Req` 的设备状态更新也与 mini-sglang 对齐：Engine 在 forward 后调用
`complete_one()`，Scheduler 再把采样 token 追加到 CPU `input_ids`。这样
`cached_len/device_len/remain_len` 在 graph replay 和资源回收时具有一致语义。

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
        buffer.logits[:bs] = model.forward()        # warmup
        with torch.cuda.graph(graph, pool=pool):
            buffer.logits[:bs] = model.forward()
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

接口名称同步为 mini-sglang 的 `BaseAttnBackend`、`BaseAttnMetadata` 和
`HybridBackend`。`HybridBackend` 会把 graph 调用转发给 decode backend，因为本课只捕获 decode。

### 3.4 `python/aios/attention/fi.py`

FlashInfer 实现文件和类型名同步为 mini-sglang 的 `fi.py`、`FICaptureData`、
`FIMetadata`、`FlashInferBackend`，旧 `flashinfer.py` 已删除。capture data 包含：

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
metadata.wrapper = self.graph_wrappers[batch.padded_size]
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
cache_manager.allocate_paged(batch.reqs)
```

但构造模型输入时使用 `padded_reqs`：

```python
positions = _make_positions(batch)
input_mapping = _make_input_tuple(batch)
write_mapping = _make_write_tuple(batch)
batch.out_loc = page_table[input_mapping]
return ForwardInput(batch, input_mapping, write_mapping)
```

`ForwardInput` 的字段和写回时机与 mini-sglang 一致：真正执行前才从
`token_pool[input_mapping]` 读取输入；Engine forward 后通过 `write_mapping`
写入采样结果。dummy request 因此有合法的 `input_id/position/out_loc`，但不会被写回或采样。

### 3.6 `python/aios/engine/engine.py` 与 `python/aios/llm/llm.py`

资源所有权已按 mini-sglang 重构：`Engine` 统一拥有 model、KV cache、page
table、attention backend、CUDA stream 和 `GraphRunner`；`LLM` 只负责 tokenizer、
请求组织和同步生成循环。模型层不再保存 backend 或显式接收 KV cache，而是通过
全局 `Context` 访问当前 batch、backend 与 cache。

CUDA Graph 要求 page table 地址稳定，所以 `Engine.__init__()` 提前创建：

```python
page_table = torch.zeros((max_running_reqs + 1, max_seq_len), device="cuda")
dummy_table_idx = max_running_reqs
dummy_page = num_pages
page_table[dummy_table_idx].fill_(dummy_page)
```

KV cache 多分配 1 个 page 给 dummy request：

```python
MHAKVCache(num_pages=num_pages + 1)
CacheManager(num_pages, page_size, page_table)  # 真实请求只能分配真实页
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
| Engine 资源所有权 | model / KV / page table / backend / graph | 相同 |
| `ForwardInput` | input/write mapping 分离 | 相同 |
| attention 命名与文件 | `BaseAttnBackend` / `FIMetadata` / `fi.py` | 相同 |
| CUDA stream | 独立 engine stream；同步调度循环 | engine + scheduler 双 stream overlap |
| 分布式/TP | 未实现 | 已支持 |

本课保留的边界是：单 GPU、同步调度循环、只捕获 FlashInfer decode；不提前引入
prefix cache、chunked prefill、TP/MoE、进程通信和服务器循环。除此之外，核心
数据所有权、字段语义、page table 读写、padding、metadata prepare 以及 capture/replay
调用时机均按 mini-sglang 对齐。

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
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-refactor \
PYTHONPATH=python \
python resources/lesson-10-cuda-graphs/run_lesson10.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --cuda-visible-devices 1 \
  --suite e2e \
  --max-running 4 \
  --cuda-graph-bs 1,2,4 \
  --memory-ratio 0.2
```

实测输出：

```text
[E2E_CUDA_GRAPH] token_ids=[12555, 374, 279, 897]
```

### 4.3 Eager vs CUDA Graph benchmark

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
FLASHINFER_CACHE_DIR=/tmp/flashinfer-aios-refactor \
PYTHONPATH=python \
python resources/lesson-10-cuda-graphs/run_lesson10.py \
  --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
  --cuda-visible-devices 1 \
  --suite bench \
  --num-seqs 8 \
  --min-prompt-len 32 \
  --max-prompt-len 96 \
  --max-tokens 24 \
  --max-running 4 \
  --cuda-graph-bs 1,2,4 \
  --memory-ratio 0.2
```

实测输出：

```text
Workload: num_seqs=8 prompt_len=32..96 max_tokens=24 max_running=4 graph_bs=[1, 2, 4]
[EAGER_DECODE] output_tokens=192 elapsed=0.68s tps=282.30
[CUDA_GRAPH] output_tokens=192 elapsed=0.21s tps=928.52
[CORRECTNESS] eager and CUDA graph token ids match
Summary: capture_time=1.33s speedup=3.29x eager_tps=282.30 graph_tps=928.52
```

这个 workload 很小，因此 capture time 不应该算进单次请求收益。生产系统通常在模型初始化阶段完成 capture，后续大量 decode step 复用 replay。

### 4.4 课程结论

Lesson 10 解决的是 decode 阶段的 CPU launch overhead：

```text
Lesson 9: 让单次 decode forward 更少、更融合
Lesson 10: 让重复 decode forward 用 graph replay 提交
```

完成本课后，AIOS 的 decode 路径具备了生产推理引擎常见的执行形态：scheduler 仍然动态管理请求，但进入 GPU 的 decode batch 会被 pad 到固定 bucket，并通过 CUDA Graph replay 执行。
