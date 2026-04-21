from __future__ import annotations

import torch


class TableManager:
    """Manages page_table + token_pool + request slot allocation (mini-sglang style)."""

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table  # (max_running_reqs, max_seq_len), int32, GPU
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
