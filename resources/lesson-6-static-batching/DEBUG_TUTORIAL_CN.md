# Lesson 6 调试教程：用 VS Code 单步追踪 Static Batching

> 本文配合 `.vscode/launch.json` 中的 **"Lesson 6: static-batch tutorial trace (3 prompts)"** 配置，带你用 VS Code 的 Python 调试器一步步走完一次完整的 static batching 执行。建议配合 [TECHNICAL_CN.md](./TECHNICAL_CN.md) 第 5 节一起看。

---

## 0. 准备

1. 确认模型路径：`/data4/home/yan.wang/huggingface/Qwen3-0.6B`。如不同请在 `launch.json` 中修改 `--model`。
2. VS Code 打开仓库根目录（会自动识别 `.vscode/launch.json`）。
3. 左侧 "Run and Debug" 面板选择 **Lesson 6: static-batch tutorial trace (3 prompts)**。
4. 打好下文的断点，按 F5 启动。

### 三个调试配置的定位

| 配置名 | 用途 |
|--------|------|
| `Lesson 6: static-batch tutorial trace (3 prompts)` | **主配置**。3 个不同长度的 prompt，触发 2 个 prefill group + 多请求 decode 循环 |
| `Lesson 6: static-batch single-group (2 prompts)` | 精简版：2 个 prompt 同组。适合聚焦 decode 循环，不关心 prefill 分组 |
| `Lesson 6: baseline dynamic-KV (3 prompts)` | 对照版：相同 prompt 走非批处理路径，贪心解码时输出应逐 byte 相同 |

> 配置里 `justMyCode: false` 是**故意的**——我们要能单步进入 `aios/` 自身的模块（PyTorch 内部仍然被 debugpy 默认跳过）。

---

## 1. 断点地图

本教程按执行顺序排布 8 个关键断点。所有文件路径相对于 `python/aios/`：

| # | 文件 | 行 | 作用 |
|---|------|----|------|
| ① | `scheduler/scheduler.py` | 64 | `init_requests` 循环入口——看 token_pool / page_table 初值被写入 |
| ② | `scheduler/scheduler.py` | 82 | `iter_prefill_batches` 入口——看按 prompt 长度分组 |
| ③ | `engine/engine.py` | 25 | `run_batch` 的 `model.forward`——入口/出口看 logits |
| ④ | `scheduler/scheduler.py` | 153 | `process_batch_output` 的 `complete_one` 之前——看状态推进 |
| ⑤ | `scheduler/scheduler.py` | 112 | `schedule_decode_batch` 入口——看构批逻辑 |
| ⑥ | `scheduler/scheduler.py` | 129 | decode 页分配行——看 `page_table[idxs, pos] = new_pages` |
| ⑦ | `models/qwen3.py` | 143 | `_batched_paged_attention` 中的 prefill/decode 分支 |
| ⑧ | `scheduler/scheduler.py` | 183 | `finalize_results` 释放页——看 `[:cached_len]` 的范围 |

> 行号对应 2026-04-18 版本代码。如有小幅偏移，按函数名/上下文定位。

---

## 2. 逐步追踪

### Step 1 — 请求初始化（断点 ①，`scheduler.py:64`）

**入口条件**：程序启动后第一次命中。此时 `self._states` 为空。

**在 "Variables" 面板观察**：

- `all_input_ids`：一个 list of tensor，len=3，对应 3 个 prompt 的 token。
- `params_list`：3 个 `SamplingParams`。

**单步 (F10) 走完循环**后展开：

- `self.table_manager.page_table`：此时 `page_table[0, :len0]`, `page_table[1, :len1]`, `page_table[2, :len2]` 三行前缀被填上了物理页号；后续列仍然是 0。
- `self.table_manager.token_pool`：三行前缀被填上了 prompt 的 token id。

**要理解的重点**（对照 TECHNICAL_CN.md §5.4 表格）：

```
prefill 前:  cached_len=0,  device_len=prompt_len
```

**prompt 对应的 `[0, prompt_len)` 页在这里一次性分配完成**。这就是 "alloc-at-schedule-time" 在 prefill 阶段的形式——由 `init_requests` 代劳，所以后面的 `iter_prefill_batches` 完全不需要再分配。

### Step 2 — Prefill 分组（断点 ②，`scheduler.py:82`）

**在 "Watch" 中加入**：`[len(s.req.input_ids) for s in self._states]`

**单步进入循环后观察**：

- `by_len`：`defaultdict(list, {2: [0, 2], 4: [1]})`（实际数字取决于 tokenizer）。相同长度的请求被归到一组。
- 每次 `yield` 的 `ScheduledBatch`：
  - `batch.input_ids.shape == (len(indices), prompt_len)`
  - `batch.positions.shape == (B, prompt_len)` 且值为 `[0, 1, ..., prompt_len-1]`
  - `batch.out_loc`：把各请求前 `prompt_len` 列的物理页号 stack 起来——这正是本次 prefill 要写入的 KV 物理页位置。

> **为什么 prefill 必须按长度分组**？因为 batched attention 要求 `(B, L)` 是个矩形 tensor。不同 L 无法简单 stack。这就是 static batching 的命名由来——批形状静态矩形。生产引擎用 chunked prefill 绕开这个限制。

### Step 3 — 首次 forward（断点 ③，`engine/engine.py:25`）

按 F11 **进入** `self.model.forward(...)`。调用栈会依次展开：

```
Engine.run_batch
 └─ Qwen3ForCausalLM.forward
     └─ Qwen3Model.forward      ← 这里会看到 position_ids = batch.positions
         └─ Qwen3DecoderLayer.forward × N
             └─ Qwen3Attention.forward
                 └─ _batched_paged_attention  ← Step 7 要看的地方
```

**退出**（Shift+F11）回到 `run_batch` 后观察：

- `logits.shape == (B, seq_len, vocab)`
- `last_logits.shape == (B, vocab)`
- `scheduled.samplers` 的长度等于 `B`——每个请求独立采样

### Step 4 — 处理 prefill 输出（断点 ④，`scheduler.py:153`）

进入 `process_batch_output`。在 `req.complete_one()` 调用**之前**暂停，查看：

- `req.cached_len == 0` （prefill 前）
- `req.device_len == prompt_len`

按 F10 走过 `complete_one()` 后：

- `req.cached_len == prompt_len`
- `req.device_len == prompt_len + 1` ← 预留了下一个 slot

再走一步，观察 `token_pool[table_idx, device_len - 1] = tok` 把采样出来的"prefill 首 token"写到 `token_pool[table_idx, prompt_len]`。

**关键对应关系**（记住这一条，后面 decode 就顺了）：

```
新采样的 token 写到 token_pool[device_len - 1]，
complete_one 之后 device_len - 1 == cached_len，
所以下一步 schedule_decode_batch 从 token_pool[cached_len] 读它作为新输入。
```

### Step 5 — 构建 decode 批（断点 ⑤，`scheduler.py:112`）

此时 prefill 全部结束，进入第一轮 decode。

**在 Debug Console 中输入**（用 Python 表达式）：

```python
[(s.req.table_idx, s.req.cached_len, s.req.device_len) for s in self._states]
```

你应该看到每个活跃请求 `device_len == cached_len + 1`。

按 F10 单步走到 `new_pages = self.cache_manager.allocate(B)` 之后：

- `new_pages.shape == (B,)`，每个元素是一个刚分配的物理页 id
- `table_idxs` / `positions_1d` 都是 `(B,)`，前者是每行的 slot，后者等于 `cached_len`

### Step 6 — 核心：二维 fancy indexing（断点 ⑥，`scheduler.py:129`）

**这是 static batching 里最抽象的一行**：

```python
self.table_manager.page_table[table_idxs, positions_1d] = new_pages
```

等价写法：

```python
for b in range(B):
    page_table[table_idxs[b], positions_1d[b]] = new_pages[b]
```

即"把 `B` 个新页写到 `B` 个 (行, 列) 位置"。所写的列 = 各自的 `cached_len`——正是这一步 extend 要覆盖的那个位置。

在 Debug Console 验证：

```python
page_table[table_idxs, positions_1d]       # 应该等于 new_pages
```

紧接着一行：

```python
input_ids = token_pool[table_idxs, positions_1d].long().unsqueeze(1)
```

这里读取的是**上一步 process_batch_output 写入的 token**（因为那次写入位置 `device_len - 1` 等于现在的 `cached_len`）。步进一行后检查 `input_ids.shape == (B, 1)` 且内容与上一轮 Step 4 写入的 token 一致——这是两个组件之间的"接力接口"。

### Step 7 — Attention 内部的变长 KV 对齐（断点 ⑦，`qwen3.py:143`）

F5 跑到这里。在 decode 批里：

- 展开 `batch.reqs`，查看每个 `req.cached_len`——通常**各不相同**（prompt 长度不同 + decode 步数不同）
- 单步走过 `else` 分支，观察：
  - `max_kv_len = max(r.cached_len + 1 for r in batch.reqs)`
  - `kv_len_i < max_kv_len` 时 `F.pad(...)` 被调用
  - `causal_mask` 被构造成 `(B, 1, 1, max_kv_len)`，仅前 `cached_len + 1` 位为 0，其余为 `-inf`

**在 Watch 中加入**：

```python
[r.cached_len + 1 for r in batch.reqs]
max_kv_len
```

核对 pad 的行数等于 `max_kv_len - (cached_len_i + 1)`。

### Step 8 — 资源回收（断点 ⑧，`scheduler.py:183`）

所有请求结束后进入 `finalize_results`。观察：

- `req.cached_len` vs `req.device_len`：**通常 `device_len == cached_len + 1`**——因为最后一次 `complete_one` 之后 `device_len` 多加了 1，但终止路径不再执行 `schedule_decode_batch`，**所以 `page_table[table_idx, device_len - 1]` 那个 slot 并没有被实际分配（仍是 0）**。
- 释放范围是 `[:cached_len]` 而**不是** `[:device_len]`——这正是对上述不对称的修正。

若想亲眼看这个"漏斗"：

```python
# 在 Debug Console 里，对任意一个 req：
page_table[req.table_idx, req.device_len - 1]   # 往往是 0
page_table[req.table_idx, req.cached_len - 1]   # 最后一个真实分配的页
```

---

## 3. 常见问题

**Q1：断点命中顺序和我预期不一样？**
prefill 阶段会**一次处理一组**（group），`iter_prefill_batches` 是生成器。3 prompt 场景下断点 ② 会被命中两次（2 个 group）。

**Q2：为什么我的 `new_pages` 是整数而不是 tensor？**
`CacheManager.allocate(B)` 返回 `(B,)` 的 int32 tensor；`allocate(1)` 返回 `(1,)`。两者都能被 fancy index 接受。

**Q3：单步 F11 进不到 aios 代码？**
确认配置里 `"justMyCode": false`。如果仍然跳过，检查 VS Code 是否识别 `python/aios/` 为 workspace 内容——`PYTHONPATH` 环境变量已在 launch.json 中设为 `${workspaceFolder}/python`。

**Q4：我想对比 static batch 和非批处理的输出是否一致？**
切换到 `Lesson 6: baseline dynamic-KV (3 prompts)` 配置运行一次（不调试也行），把 stdout 拷出来；再跑 static-batch 配置，比较两份输出。温度为 0 且无其他随机性的情况下，两者应该 byte-for-byte 相同。

---

## 4. 延伸阅读

- [TECHNICAL_CN.md §5](./TECHNICAL_CN.md)：Scheduler 每个函数的完整实现讲解
- [TECHNICAL_CN.md §7.3](./TECHNICAL_CN.md#73-_batched_paged_attention批量-paged-attention核心)：`_batched_paged_attention` 的 mask / pad 细节
- mini-sglang 参考实现：`.claude/skills/course/references/mini-sglang/python/minisgl/scheduler/scheduler.py`
