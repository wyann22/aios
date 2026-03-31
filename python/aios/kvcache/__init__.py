from __future__ import annotations

import torch
from .base import BaseKVCache, KVCacheLayout, BaseCacheHandle, BaseCacheManager, SizeInfo
from .dynamic import DynamicKVCache
from .mha_pool import MHAKVCache
from .naive_manager import NaiveCacheManager

def create_naive_cache_manager(device: torch.device) -> BaseCacheManager:
    return NaiveCacheManager(device=device)

__all__ = [
    "BaseKVCache",
    "KVCacheLayout",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "DynamicKVCache",
    "MHAKVCache",
    "NaiveCacheManager",
    "create_naive_cache_manager",
]
