import math
import os
import time

from kvcache_cpp import GPUKVCacheManagerImpl
import paged_kvcache_ops

import numpy as np
import torch
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .kvcache_config import KVCacheConfig
from .kvcache_utils import KVCacheOffloadMode, KVLookupResult, KVIndexMeta
from .kvcache_metadata import KVCacheMetadata, get_kvcache_metadata_buffer

class GPUKVCacheManager:
    def __init__(self, 
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        num_primary_cache_pages: int,
        num_buffer_pages: int,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device_idx: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device_idx = device_idx

        self.gpu_kvcache_table: torch.Tensor = torch.empty(
            [
                self.num_layers,
                self.num_primary_cache_pages,
                2,  # k/v
                self.page_size,
                self.num_heads,
                self.head_dim,
            ],
            dtype=self.dtype,
            device=self.device_idx,
        )
        self.kvcache_manager_impl = GPUKVCacheManagerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.num_primary_cache_pages,
            self.num_buffer_pages,
            self.max_batch_size,
            self.max_sequence_length,
            self.device_idx,
        )
        # Auxiliary metadata
        self.num_sms = torch.cuda.get_device_properties(self.device_idx).multi_processor_count
    
    def lookup(self, uids: torch.Tensor) -> KVLookupResult:
        cached_start_indices, cached_lengths = self.kvcache_manager_impl.lookup(uids)
        return KVLookupResult(
            user_ids=uids,
            gpu_cached_start_indices=cached_start_indices,
            gpu_cached_lengths=cached_lengths,
        )

    def allocate(self, 
        uids: torch.Tensor, 
        seq_hist_lengths: torch.Tensor,  # total history lengths in the sequence of current batch
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None
    ) -> KVCacheMetadata:
        new_hist_lengths = seq_hist_lengths - lookup_results.cached_lengths

        if output_kvcache_metadata is None:
            batch_size = uids.size(0)
            num_new_tokens = torch.sum(new_hist_lengths, dtype=torch.int32).item()
            num_total_pages = torch.sum(torch.ceil(seq_hist_lengths / self.page_size), dtype=torch.int32).item()
            output_kvcache_metadata = get_kvcache_metadata_buffer(
                batch_size,
                num_new_tokens,
                num_total_pages,
            )
        output_kvcache_metadata.max_seqlen = max(seq_hist_lengths).item()
        self.kvcache_manager_impl.allocate(
            uids, 
            seq_hist_lengths,
            lookup_results.host_cached_lengths,
            output_kvcache_metadata.page_ids_gpu_buffer,
            output_kvcache_metadata.metadata_gpu_buffer)
        return output_kvcache_metadata
    
    def evict(self, uids: torch.Tensor) -> None:
        for uid in uids.tolist():
            self.kvcache_manager_impl.evict(uid)
    
    def evict_all(self) -> None:
        self.kvcache_manager_impl.evict_all()
    
    # def invalid(self, uids) -> None:
    #     for uid in uids.tolist():
    #         self.kvcache_manager_impl.invalid(uid)
    
    def put(self, k, v, layer_idx, kvcache_metadata: KVCacheMetadata, append_offsets: Optional[torch.Tensor] = None) -> None:
        assert k.shape == v.shape, f"key and value shape mismatch: {k.shape} vs {v.shape}"
        if k.size(0) == self.num_layers:
            raise NotImplementedError("Only support layer-wise in this implementation.")
        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_table[layer_idx].unbind(dim=1)
        assert k.shape[-2:] == paged_k_cache.shape[-2:], f"input k/v shape {k.shape} mismatch with cache shape {paged_k_cache.shape}"
        batch_size = kvcache_metadata.kv_indptr.size(0) - 1
        paged_kvcache_ops.append_kvcache(
            k, v, 
            kvcache_metadata.batch_indices,
            kvcache_metadata.position,
            append_offsets if append_offsets is not None else torch.zeros((batch_size,), dtype=torch.int32, device=self.device_idx),
            kvcache_metadata.new_history_nnz_cuda,
            kvcache_metadata.new_history_nnz,
            paged_k_cache,
            paged_v_cache,
            kvcache_metadata.kv_indices,
            kvcache_metadata.kv_indptr,
            kvcache_metadata.kv_last_page_len,
            0,  # NHD layout
            self.num_sms)
    
    def get(self, page_ids, last_page_lens, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: debug interface (for symmetry). Not used in inference pipeline.
        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_table[layer_idx].unbind(dim=1)

        k = paged_k_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        v = paged_v_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        k = k[ : k.size(0) - (self.page_size - int(last_page_lens)), ... ]
        v = v[ : v.size(0) - (self.page_size - int(last_page_lens)), ... ]
        return k, v

    def revoke_onboard_pages(self, user_ids, onboard_page_starts, num_onboard_pages):
        self.kvcache_manager_impl.revoke_onboard_pages(user_ids, onboard_page_starts, num_onboard_pages)
    
    # [[ offload critirea ]]
    def check_for_offload(self, uids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.kvcache_manager_impl.check_for_offload(
            uids if uids is not None else torch.tensor([], dtype=torch.int64))
    
    def acquire_offload_pages(self, uids: torch.Tensor, offloaded_lengths: torch.Tensor, always_offload: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        return self.kvcache_manager_impl.acquire_offload_pages(uids, offloaded_lengths, always_offload)
    
    def release_offload_pages(self, 
        uids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_lengths: torch.Tensor,
        offloaded: bool,
    ) -> None:
        return self.kvcache_manager_impl.release_offload_pages(
            uids, offload_start_indices, offload_lengths, offloaded)