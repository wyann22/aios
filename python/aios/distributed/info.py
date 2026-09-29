from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int
    local_rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.size:
            raise ValueError(f"rank must be in [0, {self.size}), got {self.rank}")

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO = DistributedInfo(rank=0, size=1, local_rank=0)


def set_tp_info(rank: int, size: int, local_rank: int) -> None:
    global _TP_INFO
    _TP_INFO = DistributedInfo(rank=rank, size=size, local_rank=local_rank)


def get_tp_info() -> DistributedInfo:
    return _TP_INFO


def reset_tp_info() -> None:
    set_tp_info(rank=0, size=1, local_rank=0)
