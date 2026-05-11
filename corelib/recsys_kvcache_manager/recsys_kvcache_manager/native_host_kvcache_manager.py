import math
import os
import time

from kvcache_cpp import (
    HostKVStorageImpl,
    KVOnloadHandle,
    KVOffloadHandle,
)

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .hierarchical_kvcache_manager import (
    SecondaryKVCacheManagerBase,
    SecondaryErrorCode,
    SecondaryTaskStatus,
    SecondaryTaskHandle,
    SecondaryWaitResult,
)

from .kvcache_config import KVCacheConfig
from .kvcache_utils import KVCacheOffloadMode, KVLookupResult, KVIndexMeta
from .kvcache_metadata import KVCacheMetadata

class NativeHostKVCacheManager(SecondaryKVCacheManagerBase):
    def __init__(self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        bytes_capacity_per_layer: int,
        max_batch_size: int,
        max_sequence_length: int,
        onload_timeout_ms: float,
        offload_timeout_ms: float,
        dtype: torch.dtype,
        device_idx: int,
    ):
        self.backend_name = "native"

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.bytes_capacity_per_layer = bytes_capacity_per_layer
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device_idx = device_idx

        self.kvcache_mananger_impl = HostKVStorageImpl(
            self.num_layers,
            self.num_heads, 
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.bytes_capacity_per_layer,
            self.max_batch_size,
            self.max_sequence_length,
            self.device_idx
        )

        self._onload_timeout_ms = onload_timeout_ms
        self._offload_timeout_ms = offload_timeout_ms
    
    def register_gpu_cache_table(self, cache_table: torch.Tensor):
        self.kvcache_mananger_impl.register_gpu_cache_table(
            [ cache_table[idx] for idx in range(self.num_layers) ]
        )
    
    def build_index_meta(self, user_ids: torch.Tensor, sequence_lengths: torch.Tensor) -> KVIndexMeta:
        index_meta = KVIndexMeta(
            user_ids=user_ids,
            seq_lengths=sequence_lengths,
        )
        return index_meta

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        cached_lengths = self.kvcache_mananger_impl.lookup(index_meta.user_ids)
        cached_start_indices = torch.zeros_like(cached_lengths)
        return KVLookupResult(
            user_ids=index_meta.user_ids,
            host_cached_start_indices=cached_start_indices,
            host_cached_lengths=cached_lengths,
        )

    def onboard_launch_kvcache(self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata
    ) -> SecondaryTaskHandle:
        g_end_idxs = lookup_result.gpu_cached_start_indices + lookup_result.gpu_cached_lengths
        h_longer = g_end_idxs < lookup_result.host_cached_lengths

        onload_start_indices = torch.where(
            torch.logical_and(lookup_result.gpu_cached_start_indices == 0, h_longer),
            g_end_idxs,
            0,
        )
        onload_lengths = torch.where(
            h_longer,
            lookup_result.host_cached_lengths,
            lookup_result.gpu_cached_start_indices,
        )

        onload_paged_ids_list = [
            kvcache_metadata.kv_indices.narrow(
                0,                                                          # dim
                kvcache_metadata.kv_indptr[seq_idx] 
                + onload_start_indices[seq_idx].item() // self.page_size,   # start: gpu, this is allowed to be Tensor. TODO(junyiq): check if there is d2h
                onload_lengths[seq_idx].item() // self.page_size,           # length
            ) for seq_idx in range(index_meta.user_ids.size(0))      
        ]
        if torch.sum(onload_lengths).item() == 0:
            # No data to onboard, skip the task.
            return SecondaryTaskHandle(
                backend="native",
                handle=None,
                status=SecondaryTaskStatus.SKIPPED,
            )

        native_handle = KVOnloadHandle(self.num_layers)
        self.kvcache_mananger_impl.onload_kvcache(
            index_meta.user_ids,
            onload_paged_ids_list,
            native_handle)
        return SecondaryTaskHandle(
            backend="native",
            handle=native_handle,
            status=SecondaryTaskStatus.LAUNCHED,
            metadata={
                "onboard_start_indices": onload_start_indices,
                "onboard_lengths": onload_lengths,
            },
        )

    def onboard_wait_kvcache(self, task_handle):
        # Use layerwise data transfer and sync for native backend.
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.SKIPPED,
            ready=False,
        )
    
    def onboard_wait_kvcache_by_layer(self, task_handle, layer_idx: int):  # wait for a single layer
        task_handle.handle.wait_host(layer_idx)
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.EVENT_READY,  # the following get on the default stream will be synced
            ready=True,
        )

    def offload_launch_kvcache(
        self,
        offload_user_ids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_page_indices_list: List[torch.Tensor], 
        index_meta: Optional[KVIndexMeta] = None,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        native_handle = KVOffloadHandle(self.num_layers)
        ret = self.kvcache_mananger_impl.offload_kvcache(
            offload_user_ids, 
            offload_start_indices, 
            offload_page_indices_list, 
            native_handle)
        if not ret:
            return None

        return SecondaryTaskHandle(
            backend="native",
            handle=native_handle,
            status=SecondaryTaskStatus.LAUNCHED,
            time_launched=time.perf_counter_ns(),
        )

    def offload_wait_kvcache(self, task_handle):
        is_ready = task_handle.handle.try_wait_host(-1)
        elapsed_time = (time.perf_counter_ns() - task_handle.time_launched) / 1000_000.
        # print(f"[DEBUG] Offload elapsed time: {elapsed_time} ms")
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY if is_ready else 
                   SecondaryTaskStatus.TIMEOUT if elapsed_time > self._offload_timeout_ms else 
                   SecondaryTaskStatus.LAUNCHED,
            ready=is_ready,
        )
    
    def finish_task(self, task_handle):
        if isinstance(task_handle.handle, KVOnloadHandle):
            raise NotImplementedError("Finish onload by layer is supported, but not the whole task at once, since the native implementation uses layerwise synchronization.")
        elif isinstance(task_handle.handle, KVOffloadHandle):
            return self.kvcache_mananger_impl.finish_offload(task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")
        return False

    def cancel_task(self, task_handle):
        if isinstance(task_handle.handle, KVOnloadHandle):
            raise NotImplementedError("Cancel onload is not supported in the current native implementation.")
        elif isinstance(task_handle.handle, KVOffloadHandle):
            return self.kvcache_mananger_impl.cancel_offload(task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")
        return False
    
    def evict(self, user_ids: torch.Tensor):
        for uid in user_ids.tolist():
            self.kvcache_mananger_impl.evict(uid)
    
    def evict_all(self,):
        self.kvcache_mananger_impl.evict_all()
    
    @staticmethod
    def get_offload_handle_metadata(task_handle):
        return (
            task_handle.handle.get_user_ids(),
            task_handle.handle.get_start_indices(),
            task_handle.handle.get_lengths(),
        )