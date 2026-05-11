import math
import os
import time

import numpy as np
import torch
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .kvcache_config import KVCacheConfig
from .kvcache_utils import KVCacheOffloadMode, KVLookupResult, KVIndexMeta
from .kvcache_metadata import KVCacheMetadata


class SecondaryTaskStatus(Enum):
    UNINITIALIZED = "uninitialized"
    SKIPPED = "skipped"
    LAUNCHED = "launched"
    READY = "ready"
    EVENT_READY = "event_ready"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SecondaryErrorCode(str, Enum):
    SDK_IMPORT_FAILED = "sdk_import_failed"
    SDK_INIT_FAILED = "sdk_init_failed"
    LOOKUP_FAILED = "lookup_failed"
    LOOKUP_MISSING_TOKENS = "lookup_missing_tokens"
    ONBOARD_TASK_NOT_FOUND = "onboard_task_not_found"
    ONBOARD_WAIT_FAILED = "onboard_wait_failed"
    ONBOARD_TIMEOUT = "onboard_timeout"
    OFFLOAD_TASK_NOT_FOUND = "offload_task_not_found"
    OFFLOAD_WAIT_FAILED = "offload_wait_failed"
    OFFLOAD_TIMEOUT = "offload_timeout"
    CANCEL_FAILED = "cancel_failed"

@dataclass
class SecondaryTaskHandle:
    backend: str
    user_ids: Optional[torch.Tensor] = None  # TODO(junyiq): check if needed
    handle: Optional[Any] = None
    status: SecondaryTaskStatus = SecondaryTaskStatus.UNINITIALIZED
    metadata: Optional[Dict[str, Any]] = None
    time_launched: Optional[float] = None

@dataclass
class SecondaryWaitResult:
    status: SecondaryTaskStatus
    ready: bool
    message: str = ""
    failed_mask: Optional[torch.Tensor] = None
    failed_user_ids: Optional[List[int]] = None

class SecondaryKVCacheManagerBase(ABC):
    @abstractmethod
    def register_gpu_cache_table(self, cache_table: torch.Tensor) -> None:
        ...
    
    @abstractmethod
    def build_index_meta(self, user_ids: torch.Tensor, sequence_lengths: torch.Tensor) -> KVIndexMeta:
        ...
    
    @abstractmethod
    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        ...

    @abstractmethod
    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> SecondaryTaskHandle:
        ...

    @abstractmethod
    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...

    @abstractmethod
    def offload_launch_kvcache(
        self,
        offload_user_ids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_page_indices_list: List[torch.Tensor], 
    ) -> Optional[SecondaryTaskHandle]:
        ...

    @abstractmethod
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...
    
    @abstractmethod
    def finish_task(self, task_handle: SecondaryTaskHandle) -> bool:
        ...

    @abstractmethod
    def cancel_task(self, task_handle: SecondaryTaskHandle) -> bool:
        ...
    
    @abstractmethod
    def evict(self, user_ids: torch.Tensor):
        ...

    @abstractmethod
    def evict_all(self, user_ids: torch.Tensor):
        ...



# class NopSecondaryKVCacheManager(SecondaryKVCacheManagerBase):
#     def lookup_kvcache(self, index_meta: KVIndexMeta):
#         return {"backend": "nop", "hit_mask": None}

#     def onboard_launch_kvcache(self, index_meta, restore_slot_mapping):
#         return SecondaryTaskHandle(
#             backend="nop",
#             handle=None,
#             status=SecondaryTaskStatus.SKIPPED,
#             metadata={"reason": "nop backend"},
#         )

#     def onboard_wait_kvcache(self, task_handle):
#         return SecondaryWaitResult(
#             status=SecondaryTaskStatus.READY,
#             ready=True,
#             message="nop onboard wait ready",
#         )

#     def offload_launch_kvcache(
#         self,
#         index_meta,
#         append_slot_mapping,
#         append_slot_indptr=None,
#     ):
#         return SecondaryTaskHandle(
#             backend="nop",
#             handle=None,
#             status=SecondaryTaskStatus.SKIPPED,
#             metadata={"reason": "nop backend"},
#         )

#     def offload_wait_kvcache(self, task_handle):
#         return SecondaryWaitResult(
#             status=SecondaryTaskStatus.READY,
#             ready=True,
#             message="nop offload wait ready",
#         )

#     def cancel_task(self, task_handle):
#         return None
