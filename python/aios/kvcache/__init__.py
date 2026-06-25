from __future__ import annotations

import torch
from .base import BaseKVCache, BaseCacheHandle, BaseCacheManager, SizeInfo
from .mha_pool import MHAKVCache
from .naive_manager import NaiveCacheManager

def create_naive_cache_manager(device: torch.device) -> BaseCacheManager:
    return NaiveCacheManager(device=device)

__all__ = [
    "BaseKVCache",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "MHAKVCache",
    "NaiveCacheManager",
    "create_naive_cache_manager",
]
