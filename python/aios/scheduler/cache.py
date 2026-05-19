from __future__ import annotations

import torch
from aios.kvcache import create_naive_cache_manager

from ..core import Req


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_naive_cache_manager(device=device)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate_paged(self, reqs: list[Req], page_table: torch.Tensor) -> None:
        for req in reqs:
            assert req.table_idx is not None
            extend_len = req.extend_len
            if extend_len == 0:
                continue
            indices = self.allocate(extend_len)
            page_table[
                req.table_idx,
                req.cached_len : req.device_len,
            ] = indices

    # Temporarily disabled: these methods support prefix cache handle management
    # and integrity checks, but they are unused by the current lesson-6 paged
    # KV path. Keep the implementation below commented for future re-enable.
    #
    # def lock(self, handle: BaseCacheHandle) -> None:
    #     self.manager.lock_handle(handle, unlock=False)
    #
    # def unlock(self, handle: BaseCacheHandle) -> None:
    #     self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated
        raise NotImplementedError("CacheManager eviction is not implemented.")
        # # NOTE: len(evicted) + free_len >= needed_len
        # evicted = self.manager.evict(needed_len - free_len)
        # merged = torch.cat([self._free_slots, evicted])
        # assert len(merged) >= needed_len, "Eviction did not free enough space."

        # allocated = merged[:needed_len]
        # self._free_slots = merged[needed_len:]
        # return allocated

    # def free_and_cache_finished_req(
    #     self,
    #     old_handle: BaseCacheHandle,
    #     input_ids: torch.Tensor,
    #     indices: torch.Tensor,
    # ) -> None:
    #     in_cache_len = self.manager.insert_prefix(input_ids, indices)
    #     self._free(indices[old_handle.cached_len : in_cache_len])
    #     self.unlock(old_handle)
    #
    # def check_integrity(self) -> None:
    #     self.manager.check_integrity()
    #     if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
    #         raise RuntimeError(
    #             "CacheManager integrity check failed:"
    #             f" free_slots({len(self._free_slots)}) +"
    #             f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
    #         )
