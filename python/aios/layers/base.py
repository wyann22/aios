from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]
### BaseOP design
### class Qwen3Attention(BaseOP):
###   methods:
###     def __init__(self, ...) 初始化
###     def forward(self, ...) 推理
###     def state_dict(self, ...) 获取权重
###     def load_state_dict(self, ...) 加载权重
###   variables:
###     weights: 权重成员变量，必须是 torch.Tensor 类型，且不以 _ 开头命名, 
###     _others: 其他成员变量，必须以 _ 开头命名，且不包含在 state_dict 中
def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
        result = {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                result.update(param.state_dict(prefix=_concat_prefix(prefix, name)))

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape, (
                    f"Shape mismatch for {_concat_prefix(prefix, name)}: "
                    f"expected {param.shape}, got {item.shape}"
                )
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
        return {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
        result = {}
        for i, op in enumerate(self.op_list):
            result.update(op.state_dict(prefix=_concat_prefix(prefix, str(i))))
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
