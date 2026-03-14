# Lesson 3：从单文件到推理框架 — 将 run.py 重构为 aios/ 包

## 概述

在 Lesson 2 中，我们用 **251 行**的单文件 `run_qwen3.py` 实现了一个完整的 Qwen3 推理引擎。它能跑，但存在三个核心问题：

1. **依赖 `nn.Module`** — PyTorch 的 `nn.Module` 为训练设计（梯度追踪、参数注册、hook 机制），推理用不到这些，反而增加了开销
2. **依赖 `transformers`** — 用 `AutoConfig` 读配置、用 `AutoModelForCausalLM` 中转权重，多加载了一份完整模型到内存
3. **单文件不可扩展** — 所有代码堆在一个文件里，无法独立替换某一层的实现（比如后续换 FlashAttention）

本课的目标是：**保持推理逻辑完全不变**，仅改造代码架构，将 `run_qwen3` 重构为一个模块化的 `aios` 推理框架（参考mini-sglang）。

## 改造前后对比

```
改造前：run.py (251行，单文件)                改造后：python/aios/ (20个文件，模块化包)
─────────────────────────────                ─────────────────────────────────
nn.Module 基类                               → BaseOP 自定义基类（无梯度开销）
nn.Parameter 包装权重                         → torch.empty() 原始张量
nn.Linear                                   → Linear(BaseOP) + F.linear
nn.Embedding                                → Embedding(BaseOP) + F.embedding
nn.ModuleList                                → OPList（自定义列表容器）
register_buffer (RoPE)                       → StateLessOP + 私有属性 _cache
AutoConfig.from_pretrained()                 → ModelConfig.from_json()（直读 JSON）
AutoModelForCausalLM → state_dict 中转       → safetensors 直接加载
load_state_dict(strict=False)                → BaseOP.load_state_dict() 严格消费
generate() 独立函数                           → LLM 类封装
```

## 核心设计思想：BaseOP 替代 nn.Module

这是本课最关键的改变。来看为什么推理不需要 `nn.Module`：

### nn.Module 做了什么？

```python
class RMSNorm(nn.Module):           # 继承 nn.Module
    def __init__(self, dim, eps):
        super().__init__()           # 初始化 nn.Module 内部机制
        self.weight = nn.Parameter(  # 包装成 Parameter（自动追踪梯度）
            torch.ones(dim)          # 分配真实内存
        )
        self.eps = eps
```

`nn.Module` 在幕后做了大量工作：
- 维护 `_parameters` 字典追踪所有 `nn.Parameter`
- 维护 `_modules` 字典追踪所有子模块
- 维护 `_buffers` 字典追踪所有 buffer
- 支持 `register_forward_hook`、`register_backward_hook`
- 支持 `train()`/`eval()` 模式切换
- `nn.Parameter` 自动设置 `requires_grad=True`

**推理时这些全部不需要。** 我们只需要两件事：
1. 存储权重（`torch.Tensor`）
2. 执行前向计算（`forward()`）

### BaseOP 的极简设计

```python
class BaseOP:
    def forward(self, *args, **kwargs): ...   # 唯一的抽象接口

    def state_dict(self, ...):                # 递归收集所有 Tensor 属性
        for name, param in self.__dict__.items():
            if name.startswith("_"):          # 跳过私有属性（非权重）
                continue
            if isinstance(param, torch.Tensor):
                result[key] = param           # 直接收集
            elif isinstance(param, BaseOP):
                param.state_dict(...)         # 递归进入子模块

    def load_state_dict(self, state_dict):    # 递归加载，pop 消费每个 key
        ...                                   # 最终校验 state_dict 为空
```

**核心规则：**
- 公有属性（不以 `_` 开头）如果是 `torch.Tensor` → 权重，参与 `state_dict`
- 公有属性如果是 `BaseOP` → 子模块，递归处理
- 私有属性（以 `_` 开头） → 跳过（用于存放 scale、cache 等非权重数据）

---

## 改造步骤

### 总览：目录结构（对标 mini-sglang）

```
python/aios/                          ← 对标 mini-sglang 的 python/minisgl/
├── __init__.py                       导出 LLM, Sampler, SamplingParams
├── __main__.py                       CLI 入口 (python -m aios)
├── core.py                           SamplingParams（对标 minisgl/core.py）
├── engine/                           采样引擎
│   ├── __init__.py                   导出 Sampler
│   └── sample.py                     Sampler 类（对标 minisgl/engine/sample.py）
├── llm/                              用户接口
│   ├── __init__.py                   导出 LLM
│   └── llm.py                        LLM 入口类（对标 minisgl/llm/llm.py）
├── models/                           模型实现
│   ├── __init__.py                   create_model() 工厂函数
│   ├── base.py                       BaseLLMModel(ABC, BaseOP)
│   ├── config.py                     ModelConfig dataclass + from_json()
│   ├── qwen3.py                      完整 Qwen3 模型（5 个类）
│   └── weight.py                     safetensors 直接加载
└── layers/                           基础算子层
    ├── __init__.py                   导出所有层
    ├── base.py                       BaseOP / StateLessOP / OPList
    ├── linear.py                     Linear(BaseOP)
    ├── norm.py                       RMSNorm(BaseOP)
    ├── rotary.py                     RotaryEmbedding(StateLessOP)
    ├── attention.py                  rotate_half / apply_rotary_pos_emb / repeat_kv
    ├── activation.py                 silu_and_mul
    └── embedding.py                  Embedding + LMHead
```

### 文件依赖关系

```
┌──────────────────────────────────────────────────────────────────┐
│                      aios/__init__.py                            │
│                 导出 LLM, Sampler, SamplingParams                 │
│                              │                                   │
│              ┌───────────────┼───────────────┐                   │
│              │               │               │                   │
│        llm/llm.py      engine/sample.py   core.py               │
│              │               │          (SamplingParams)         │
│    ┌─────────┼─────────┐     │                                   │
│    │         │         │     │                                   │
│  models/   models/  models/  │                                   │
│  __init__  weight   config   │                                   │
│    │                         │                                   │
│  models/qwen3.py ← models/base.py                               │
│    │                                                             │
│  layers/__init__.py                                              │
│   ├── base.py (BaseOP, StateLessOP, OPList)                      │
│   ├── linear.py       (Linear)                                   │
│   ├── norm.py         (RMSNorm)                                  │
│   ├── rotary.py       (RotaryEmbedding)                          │
│   ├── attention.py    (apply_rotary_pos_emb, repeat_kv)          │
│   ├── activation.py   (silu_and_mul)                             │
│   └── embedding.py    (Embedding, LMHead)                        │
└──────────────────────────────────────────────────────────────────┘
```

---

### 第 1 步：创建 BaseOP 基类体系 — `layers/base.py`

**关键理解 — `state_dict` 的 key 是怎么拼出来的：**

```
Qwen3ForCausalLM                     prefix = ""
  ├── model (Qwen3Model)             prefix = "model"
  │     ├── embed_tokens (Embedding) prefix = "model.embed_tokens"
  │     │     └── weight             key = "model.embed_tokens.weight"  ✓ 对应 safetensors
  │     ├── layers (OPList)          prefix = "model.layers"
  │     │     ├── [0] (DecoderLayer) prefix = "model.layers.0"
  │     │     │     ├── self_attn    prefix = "model.layers.0.self_attn"
  │     │     │     │     ├── q_proj prefix = "model.layers.0.self_attn.q_proj"
  │     │     │     │     │   └── weight  key = "model.layers.0.self_attn.q_proj.weight"  ✓
```

这些 key **恰好**与 safetensors 文件中的 key 一致，所以可以直接加载！

**对应 run.py 中的**：所有 `nn.Module` 继承

**做什么**：创建三个基类，替代 `nn.Module` 和 `nn.ModuleList`

```python
# layers/base.py

def _concat_prefix(prefix: str, name: str) -> str:
    """拼接 prefix 和 name，如 _concat_prefix("model.layers", "0") → "model.layers.0" """
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """替代 nn.Module 的推理基类"""

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def state_dict(self, *, prefix="", result=None):
        result = result if result is not None else {}
        for name, param in self.__dict__.items():
            if name.startswith("_"):      # 规则：_ 开头的属性不是权重
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)
        return result

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))  # pop 消费
                assert param.shape == item.shape    # 校验形状
                setattr(self, name, item)            # 替换为真实数据
            elif isinstance(param, BaseOP):
                param.load_state_dict(state_dict, prefix=_concat_prefix(prefix, name),
                                      _internal=True)
        # 只有最外层调用才校验是否有多余 key
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """无参数的算子（如 RotaryEmbedding），state_dict 永远为空"""
    def load_state_dict(self, state_dict, *, prefix="", _internal=False): ...
    def state_dict(self, *, prefix="", result=None):
        return result if result is not None else {}


class OPList(BaseOP, Generic[T]):
    """替代 nn.ModuleList，用整数索引做 prefix"""
    def __init__(self, ops: List[T]):
        self.op_list = ops

    def state_dict(self, *, prefix="", result=None):
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)),
                               _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys: {list(state_dict.keys())}")
```

---

### 第 2 步：替换 nn.Linear — `layers/linear.py`

**对应 run.py 中的**：所有 `nn.Linear(..., bias=False)`

**改造对比**：

```python
# ❌ run.py（nn.Module 版本）
self.q_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
# nn.Linear 内部做了什么：
#   self.weight = nn.Parameter(torch.empty(out, in))  ← 追踪梯度
#   self.weight.requires_grad = True                   ← 推理不需要

# ✅ aios 版本（BaseOP）
self.q_proj = Linear(config.hidden_size, num_heads * head_dim)
# Linear 内部：
#   self.weight = torch.empty(out, in)   ← 原始张量，无梯度开销
```

```python
# layers/linear.py
class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)  # 裸张量
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

> **为什么用 `torch.empty` 而不是 `torch.zeros`？**
>
> 因为这些权重会在 `load_state_dict` 时被完全替换。用 `empty` 不做初始化，速度更快。
> 更关键的是，在 meta device 上创建时（第 9 步），`torch.empty` 不分配任何内存。

---

### 第 3 步：替换 RMSNorm — `layers/norm.py`

**对应 run.py 中的**：`class RMSNorm(nn.Module)`

**改造对比**：

```python
# ❌ run.py
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()                    # nn.Module 初始化
        self.weight = nn.Parameter(torch.ones(dim))  # Parameter 包装
        self.eps = eps

# ✅ aios 版本
class RMSNorm(BaseOP):
    def __init__(self, size, eps=1e-6):
        self.eps = eps                        # 普通 float，以 "eps" 命名不以 _ 开头...
        self.weight = torch.empty(size)       # 裸张量
```

> **注意 `eps` 的处理**：`eps` 是 float，不是 `torch.Tensor`，也不是 `BaseOP`，所以 `state_dict` 遍历时会自动跳过它。`BaseOP` 只关心两种类型：`torch.Tensor`（权重）和 `BaseOP`（子模块）。

forward 逻辑完全不变：
```python
def forward(self, x):
    x_float = x.float()
    rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
    return (x_float * rms).to(x.dtype) * self.weight
```

---

### 第 4 步：替换 RotaryEmbedding — `layers/rotary.py`

**对应 run.py 中的**：`class RotaryEmbedding(nn.Module)` + `register_buffer`

这是改动最大的一个层，因为 run.py 用了 `register_buffer` 存放 `inv_freq`：

```python
# ❌ run.py
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, ...) / head_dim))
        self.register_buffer('inv_freq', inv_freq)  # 每次 forward 现算 cos/sin

    def forward(self, position_ids):
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)  # 现算
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
```

**改造思路**：
1. 继承 `StateLessOP`（RoPE 没有可学习参数）
2. 预计算所有位置的 cos/sin，存为 `_cos_cache` / `_sin_cache`（`_` 前缀 → 跳过 state_dict）
3. 在 meta device 上下文中，强制用 CPU 计算（meta device 不能做数学运算）

```python
# ✅ aios 版本
class RotaryEmbedding(StateLessOP):
    def __init__(self, head_dim, max_position_embeddings, base=1000000.0):
        super().__init__()
        # 即使外层有 with torch.device("meta")，这里强制用 CPU 计算
        with torch.device("cpu"):
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, ...) / head_dim))
            t = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cache = emb.cos()  # (max_pos, head_dim) — 预计算！
            self._sin_cache = emb.sin()

    def forward(self, position_ids):
        # 直接查表，不再现算
        cos = self._cos_cache[position_ids].unsqueeze(1)  # (B, 1, S, head_dim)
        sin = self._sin_cache[position_ids].unsqueeze(1)
        return cos, sin

    def set_device(self, device):
        self._cos_cache = self._cos_cache.to(device)
        self._sin_cache = self._sin_cache.to(device)
```

**关键变化**：
- run.py：每次 forward 都现算 cos/sin → 预计算所有位置，存为 cache
- `register_buffer` → `_` 前缀的普通属性
- 新增 `set_device` 方法，在加载完权重后将 cache 移到 GPU

---

### 第 5 步：提取辅助函数 — `layers/attention.py`

**对应 run.py 中的**：`apply_rotary_pos_emb()`、`rotate_half()`、`repeat_kv()`

这些纯函数直接从 run.py 搬出来，**几乎无改动**：

```python
# layers/attention.py
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)
```

唯一变化：`rotate_half` 从 `apply_rotary_pos_emb` 内部的嵌套函数变成了模块级函数。

---

### 第 6 步：替换 Embedding 和 LMHead — `layers/embedding.py`

**对应 run.py 中的**：`nn.Embedding` 和 `nn.Linear`（用作 lm_head）

```python
# ❌ run.py
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# ✅ aios 版本
self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
self.lm_head = LMHead(config.vocab_size, config.hidden_size,
                       tie_word_embeddings=config.tie_word_embeddings,
                       tied_embedding=self.embed_tokens if config.tie_word_embeddings else None)
```

`LMHead` 需要处理 **tie_word_embeddings**（权重共享）的情况。像 Qwen3-0.6B 这样的小模型，`lm_head.weight` 和 `embed_tokens.weight` 共享同一份数据：

```python
class LMHead(BaseOP):
    def __init__(self, num_embeddings, embedding_dim,
                 tie_word_embeddings=False, tied_embedding=None):
        self._tie_word_embeddings = tie_word_embeddings  # _ 开头 → 不参与 state_dict
        self._tied_embedding = tied_embedding            # _ 开头 → 不参与 state_dict
        if not tie_word_embeddings:
            self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, x):
        w = self._tied_embedding.weight if self._tie_word_embeddings else self.weight
        return F.linear(x, w)

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        if self._tie_word_embeddings:
            # safetensors 中有 lm_head.weight，但 tie 时不需要加载，直接 pop 掉
            key = _concat_prefix(prefix, "weight")
            if key in state_dict:
                state_dict.pop(key)
        else:
            super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
```

> **为什么 `_tie_word_embeddings` 要以 `_` 开头？**
>
> 因为它是 bool 类型，如果不以 `_` 开头，`BaseOP.state_dict` 遍历 `__dict__` 时虽然会跳过（不是 Tensor 也不是 BaseOP），`load_state_dict` 也会跳过。但以 `_` 开头是一个更明确的约定：**_这个属性不是模型权重的一部分_**。

---

### 第 7 步：导出所有层 — `layers/__init__.py`

```python
from .base import BaseOP, StateLessOP, OPList, _concat_prefix
from .linear import Linear
from .norm import RMSNorm
from .rotary import RotaryEmbedding
from .attention import apply_rotary_pos_emb, repeat_kv, rotate_half
from .embedding import Embedding, LMHead
```

---

### 第 8 步：定义模型基类 — `models/base.py`

```python
from abc import ABC, abstractmethod
from aios.layers import BaseOP

class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
```

与 run.py 的 `Qwen3ForCausalLM(nn.Module)` 对比，这里用 `BaseLLMModel(ABC, BaseOP)` 作为所有 LLM 模型的抽象基类。后续支持其他模型（如 LLaMA）时，只需新增一个继承 `BaseLLMModel` 的类。

---

### 第 9 步：消除 transformers.AutoConfig — `models/config.py`

**对应 run.py 中的**：`config = AutoConfig.from_pretrained(args.model)`

run.py 用 `AutoConfig` 读取模型配置，这依赖 `transformers` 库。实际上配置就是模型目录下的一个 `config.json` 文件：

```json
{
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "vocab_size": 151936,
  "tie_word_embeddings": true,
  ...
}
```

直接读 JSON 即可：

```python
@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int         # config.json 中叫 num_attention_heads
    num_kv_heads: int         # config.json 中叫 num_key_value_heads
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, model_path: str) -> "ModelConfig":
        with open(os.path.join(model_path, "config.json")) as f:
            data = json.load(f)
        return cls(
            num_layers=data["num_hidden_layers"],
            num_qo_heads=data["num_attention_heads"],
            num_kv_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            ...
        )
```

> **`frozen=True`** 表示配置创建后不可修改，避免运行时意外改动。

---

### 第 10 步：迁移模型实现 — `models/qwen3.py`

**对应 run.py 中的**：`Qwen3Attention`、`Qwen3MLP`、`Qwen3DecoderLayer`、`Qwen3Model`、`Qwen3ForCausalLM`

这是代码量最大的一步（~120 行），但逻辑与 run.py **完全一致**，只是基类和调用方式变了。

**逐个类的改造对比**：

#### Qwen3Attention

```python
# ❌ run.py
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()                                        # nn.Module
        self.scale = self.head_dim ** -0.5                        # 公有属性
        self.q_proj = nn.Linear(hidden, heads * dim, bias=False)  # nn.Linear

    def forward(self, hidden_states, position_embeddings, mask):
        q = self.q_proj(hidden_states)    # nn.Module.__call__ 触发 forward

# ✅ aios 版本
class Qwen3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx):
        self._scale = config.head_dim ** -0.5                     # _ 前缀 → 非权重
        self._layer_idx = layer_idx                               # _ 前缀
        self.q_proj = Linear(hidden, heads * dim)                 # 自定义 Linear

    def forward(self, hidden_states, position_embeddings, mask):
        q = self.q_proj.forward(hidden_states)   # 显式调用 .forward()
```

> **为什么用 `.forward()` 而不是 `()`？**
>
> `nn.Module` 的 `__call__` 会做额外处理（hook、autograd context），`BaseOP` 没有实现 `__call__`，所以直接调用 `.forward()` 更清晰。

#### Qwen3Model

```python
# ❌ run.py
self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(N)])
self.rotary_emb = RotaryEmbedding(head_dim, rope_theta)  # 公有 → 参与 state_dict?

for layer in self.layers:       # nn.ModuleList 直接迭代
    hidden = layer(hidden, ...)  # __call__

# ✅ aios 版本
self.layers = OPList([Qwen3DecoderLayer(config, i) for i in range(N)])
self._rotary_emb = RotaryEmbedding(...)   # _ 前缀 → 跳过 state_dict（无可学习参数）

for layer in self.layers.op_list:           # OPList.op_list 迭代
    hidden = layer.forward(hidden, ...)     # 显式 .forward()
```

> **为什么 `_rotary_emb` 要以 `_` 开头？**
>
> `RotaryEmbedding` 继承 `StateLessOP`，它没有可学习参数。如果命名为 `rotary_emb`（公有），`BaseOP.state_dict` 会尝试递归进入它，虽然 `StateLessOP.state_dict` 会返回空字典不会出错，但 `load_state_dict` 在迭代 `__dict__` 时也会进入它。用 `_` 前缀直接跳过，更干净。

---

### 第 11 步：safetensors 直接加载 — `models/weight.py`

**对应 run.py 中的**：`load_weights_from_hf()` 函数

这是消除 `transformers` 依赖（除 tokenizer 外）的关键步骤。

```python
# ❌ run.py — 需要加载两份模型！
def load_weights_from_hf(model, model_path, device, dtype):
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, ...)  # 加载完整 HF 模型（占一份内存）
    hf_state_dict = hf_model.state_dict()                              # 提取 state_dict
    model.load_state_dict(hf_state_dict, strict=False)                 # 复制到我们的模型
    del hf_model     # 才能释放 HF 模型的内存
    model.to(device)  # 再把我们的模型移到 GPU

# ✅ aios 版本 — 直读 safetensors，只需一份内存
def load_weights(model, model_path, device, dtype):
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    state_dict = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)
    state_dict = {k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)  # BaseOP 会 pop 每个 key，最后校验为空
```

**内存对比**（以 Qwen3-0.6B, bfloat16 为例）：

| 方式 | 峰值内存 | 说明 |
|------|---------|------|
| run.py | ~2.4 GB | HF 模型 1.2G + 我们的模型 1.2G 同时在内存 |
| aios | ~1.2 GB | safetensors → 直接加载到我们的模型 |

---

### 第 12 步：模型工厂函数 — `models/__init__.py`

```python
def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM(config)
    raise ValueError(f"Unsupported model: {model_path}")
```

这样做的好处：后续支持新模型时只需：
1. 新增 `models/llama.py`
2. 在 `create_model()` 加一个 `elif "llama" in model_name`

---

### 第 13 步：采样参数与采样器 — `core.py` + `engine/sample.py`

**对应 run.py 中的**：`generate()` 函数的 `temperature`、`top_k` 等散落参数

`core.py` 对标 mini-sglang 的 `core.py`，`engine/sample.py` 对标 `engine/sample.py`：

```python
# core.py — 采样参数（对标 minisgl/core.py）
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0

# engine/sample.py — 采样器（对标 minisgl/engine/sample.py）
@dataclass
class Sampler:
    sampling_params: SamplingParams

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sampling_params.is_greedy:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / self.sampling_params.temperature
        # top_k filtering + softmax + multinomial sampling
        ...
```

---

### 第 14 步：构建 LLM 入口类 — `llm/llm.py`

**对应 run.py 中的**：`main()` 函数 + `generate()` 函数

这是整个架构的入口。将 run.py 的 `main()` 中的初始化逻辑和 `generate()` 函数合并为一个 `LLM` 类：

```python
class LLM:
    def __init__(self, model_path, dtype=torch.bfloat16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # 0. 支持 HuggingFace Hub ID（如 "Qwen/Qwen3-0.6B"）
        model_path = _resolve_model_path(model_path)

        # 1. 读取配置 (替代 AutoConfig)
        config = ModelConfig.from_json(model_path)

        # 2. 在 meta device 上创建模型骨架（零内存开销！）
        with torch.device("meta"):
            self.model = create_model(model_path, config)

        # 3. 从 safetensors 直接加载权重
        load_weights(self.model, model_path, self.device, self.dtype)

        # 4. 将 RoPE cache 移到 GPU
        self.model.model._rotary_emb.set_device(self.device)

        # 5. Tokenizer（仍用 transformers，这是唯一的外部依赖）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @torch.no_grad()
    def generate(self, prompts, sampling_params=None):
        # 自回归生成逻辑与 run.py 的 generate() 完全一致
        ...
```

> **什么是 meta device？**
>
> `torch.device("meta")` 是 PyTorch 提供的虚拟设备。在 meta device 上创建张量只记录形状和类型，**不分配任何内存**。
>
> ```python
> with torch.device("meta"):
>     t = torch.empty(1000, 1000)  # 0 字节内存！只记录 shape=(1000,1000)
> ```
>
> 这让我们可以先构建完整的模型结构（确定每一层的形状），然后再通过 `load_state_dict` 用真实数据替换每个张量。

---

### 第 15 步：包入口 — `__init__.py`

```python
from aios.llm import LLM
from aios.engine import Sampler
from aios.core import SamplingParams
```

最终用户只需：

```python
from aios import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0, max_tokens=32))
print(outputs[0]["text"])
```

---

## safetensors Key 与 BaseOP state_dict 对应关系

这是整个架构能工作的根本原因——**命名完全对齐**：

```
safetensors 文件中的 key                        BaseOP 模型中的路径
──────────────────────────────────────────      ──────────────────────────────
model.embed_tokens.weight                      model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight         model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight         model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight         model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight         model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight         model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight         model.layers.0.self_attn.k_norm.weight
model.layers.0.input_layernorm.weight          model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight            model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight              model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight            model.layers.0.mlp.down_proj.weight
...（重复 28 层）...
model.norm.weight                              model.norm.weight
lm_head.weight                                 lm_head.weight (tie 时被 pop 掉)
```

**Qwen3-0.6B 共 311 个 key**，`BaseOP.load_state_dict` 用 `pop()` 逐个消费，最终断言 `state_dict` 为空。如果有任何一个 key 对不上，会立即报错——**这比 `nn.Module` 的 `strict=False` 更安全**。

---

## 完整文件清单

按创建顺序（从底层到顶层），所有文件位于 `python/aios/` 下：

| # | 文件 | 行数 | 说明 |
|---|------|------|------|
| 1 | `layers/base.py` | ~111 | BaseOP / StateLessOP / OPList 基类体系 |
| 2 | `layers/linear.py` | ~16 | Linear(BaseOP) — 替代 nn.Linear |
| 3 | `layers/norm.py` | ~17 | RMSNorm(BaseOP) — 纯 PyTorch |
| 4 | `layers/rotary.py` | ~30 | RotaryEmbedding(StateLessOP) — 预计算 cos/sin |
| 5 | `layers/attention.py` | ~24 | rotate_half / apply_rotary_pos_emb / repeat_kv |
| 6 | `layers/activation.py` | ~10 | silu_and_mul（fused gate*up） |
| 7 | `layers/embedding.py` | ~60 | Embedding + LMHead（含 tie_word_embeddings） |
| 8 | `layers/__init__.py` | ~14 | 导出所有层 |
| 9 | `models/base.py` | ~15 | BaseLLMModel(ABC, BaseOP) |
| 10 | `models/config.py` | ~63 | ModelConfig dataclass + from_json() / from_hf() |
| 11 | `models/qwen3.py` | ~153 | 完整 Qwen3 模型（5 个类） |
| 12 | `models/weight.py` | ~30 | safetensors 直接加载 |
| 13 | `models/__init__.py` | ~16 | create_model() 工厂函数 |
| 14 | `core.py` | ~16 | SamplingParams dataclass（对标 minisgl/core.py） |
| 15 | `engine/__init__.py` | ~3 | 导出 Sampler |
| 16 | `engine/sample.py` | ~32 | Sampler 采样器（对标 minisgl/engine/sample.py） |
| 17 | `llm/__init__.py` | ~3 | 导出 LLM |
| 18 | `llm/llm.py` | ~78 | LLM 入口类（对标 minisgl/llm/llm.py） |
| 19 | `__init__.py` | ~5 | 包入口 |
| 20 | `__main__.py` | ~48 | CLI 入口（python -m aios） |

**总计 ~700 行**（vs run.py 的 251 行），增加的是模块化结构和注释，推理逻辑零改动。

---

## 安装与验证

```bash
# 安装（pyproject.toml 已配置 package-dir 指向 python/）
pip install -e .

# 1. 导入测试
python -c "from aios import LLM, Sampler, SamplingParams; print('Import OK')"

# 2. 端到端生成 — 支持本地路径和 HuggingFace Hub ID
python -m aios --model /path/to/Qwen3-0.6B --prompt "Who are you?" --temperature 0 --max-tokens 32
python -m aios --model Qwen/Qwen3-0.6B --prompt "Who are you?" --temperature 0 --max-tokens 32

# 3. 验证要点：
#   ✓ 311 个 safetensors key 全部被正确消费
#   ✓ tie_word_embeddings 时 lm_head.weight 被 pop 掉
#   ✓ meta device 创建 → safetensors 加载 → GPU 推理 正常工作
#   ✓ 输出与 run.py 一致（相同 prompt + greedy → 相同输出）
#   ✓ Chat template 自动应用，模型以问答形式回复
```

---

## 踩坑记录

### 1. meta device 上的 dtype 不匹配

**问题**：`torch.empty()` 在 meta device 上默认创建 `float32` 张量，但 safetensors 权重是 `bfloat16`。`BaseOP.load_state_dict` 如果同时校验 shape 和 dtype，会报错。

**解决**：只校验 shape，不校验 dtype。meta tensor 的 dtype 无意义（没有真实数据），真实 dtype 由加载的权重决定。

### 2. RotaryEmbedding 在 meta device 上无法计算

**问题**：`with torch.device("meta")` 上下文中，`torch.arange()` 和 `torch.outer()` 创建的是 meta tensor，不能做数学运算。

**解决**：在 `RotaryEmbedding.__init__` 中用 `with torch.device("cpu")` 强制在 CPU 上计算 cos/sin cache，然后通过 `set_device()` 移到 GPU。

### 3. tie_word_embeddings 处理

**问题**：Qwen3-0.6B 的 `lm_head.weight` 和 `embed_tokens.weight` 共享同一份数据。safetensors 中同时存在两个 key，但我们只需要加载一份。

**解决**：`LMHead.load_state_dict()` 在 tie 模式下直接 `pop` 掉 `lm_head.weight`，forward 时使用 `embed_tokens.weight`。

---

## 下一步：Lesson 4 — KV Cache

本课完成后，每次生成一个 token 都要重新计算整个序列的 attention（`O(n²)` 复杂度）。Lesson 4 将引入 KV Cache，将解码阶段的复杂度降到 `O(n)`。
