# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU-only HSTU cache with native page metadata and VMM-backed layer tensors.

The native HSTU append/attention kernels receive ordinary contiguous per-layer
Tensors. Their base addresses stay fixed on growth/shrink. The root is not a
compact [layers, pages, ...] Tensor: each layer has its own reserved VA span.
No FlexKV/host registration, CUDA graphs, IPC, or external raw-pointer consumers
may use this backend. Build the accompanying kvcache_cpp metadata hooks first.
"""

import math
import threading
import time
import traceback
import weakref
from contextlib import contextmanager

from .pool import VmmKvPool


class _CudaArrayOwner:
    """PyTorch retains this owner until all aliases of its Storage disappear.

    CUDA-array-interface import uses from_blob with a PyObject retaining deleter:
    https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_numpy.cpp
    The pool outlives every tensor, and explicit close/shrink checks these owners.
    An int16 view permits lossless bf16 reinterpretation without NumPy bf16.
    """

    def __init__(self, pool, pointer, shape, typestr):
        self.pool = pool
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": typestr,
            "data": (pointer, False),
            "strides": None,
            # Pool growth synchronizes before publishing any pointer.
            "stream": None,
        }


def _torch_layer(torch, owner, dtype, device):
    tensor = torch.as_tensor(owner, device=f"cuda:{device}").view(dtype)
    if tensor.data_ptr() != owner.__cuda_array_interface__["data"][0]:
        raise RuntimeError("CUDA array import copied the VMM layer")
    if not tensor.is_contiguous():
        raise RuntimeError("HSTU VMM layer must be contiguous")
    return tensor


class HstuVmmBackend:
    """Native page bookkeeping, VMM allocation, and serialized batch lifetime.

    ``batch()`` must surround the complete model forward, including append.
    Resize/close acquire the same lock and reject reentry from an active batch.
    All escaped Tensor/Storage aliases must be released before resizing: a stale
    larger tensor could otherwise address unmapped pages even with a stable base.
    Reusable graph metadata is deliberately unsupported. Growth changes capacity
    only; new pages become cached history only through subsequent real forwards.
    """

    backend_name = "gpu_only_vmm"
    host_kvstorage_manager = None

    def __init__(
        self,
        config,
        *,
        max_pages,
        chunk_bytes=None,
        driver=None,
        _torch=None,
        _impl_factory=None,
        _table_factory=None,
    ):
        if config.host_kvstorage_backend != "nop":
            raise ValueError("VMM requires host_kvstorage_backend='nop'")
        if config.host_capacity_per_layer != 0 or config.num_buffer_pages != 0:
            raise ValueError("VMM requires zero host capacity and zero buffer pages")
        for name in (
            "num_layers",
            "num_heads",
            "head_dim",
            "page_size",
            "offload_chunksize",
            "num_primary_cache_pages",
            "max_batch_size",
            "max_seq_len",
        ):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if config.offload_chunksize % config.page_size:
            raise ValueError("offload_chunksize must be a multiple of page_size")
        if (
            type(max_pages) is not int
            or not config.num_primary_cache_pages <= max_pages < 2**31
        ):
            raise ValueError("max_pages must cover initial pages and fit native int32")
        if _torch is None:
            import torch as _torch
        if _impl_factory is None:
            from kvcache_cpp import GPUKVCacheManagerImpl

            _impl_factory = GPUKVCacheManagerImpl
        if not all(
            hasattr(_impl_factory, name)
            for name in (
                "_vmm_snapshot",
                "_resize_vmm_metadata",
                "_enable_vmm_events",
                "_drain_vmm_capacity_evictions",
            )
        ):
            raise RuntimeError(
                "Rebuild recsys_kvcache_manager: VMM metadata hooks are missing"
            )
        self._torch = _torch
        if config.dtype not in (_torch.float16, _torch.bfloat16, _torch.float32):
            raise ValueError("HSTU VMM supports float16, bfloat16 and float32")
        self._element_size = 4 if config.dtype == _torch.float32 else 2
        self._typestr = "<f4" if self._element_size == 4 else "<i2"
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.page_size = config.page_size
        self.chunk_size = config.offload_chunksize
        self.max_batch_size = config.max_batch_size
        self.max_sequence_length = config.max_seq_len
        self.num_buffer_pages = 0
        self.dtype = config.dtype
        self.device_idx = config.device
        self._lock = threading.RLock()
        self._active_batch = False
        self._batch_thread = None
        self._poisoned = False
        self._closed = False
        self._owners = weakref.WeakSet()
        self._tables = []
        self._table_factory = _table_factory or _torch_layer
        self._evictions = []
        self._pool = VmmKvPool(
            self.num_layers,
            2 * self.page_size * self.num_heads * self.head_dim * self._element_size,
            max_pages,
            device=self.device_idx,
            driver=driver,
            chunk_bytes=chunk_bytes,
        )
        self.minimum_pages = self._pool._round_pages(
            self.max_batch_size * math.ceil(self.max_sequence_length / self.page_size)
        )
        try:
            if self._pool.max_mapped_pages >= 2**31:
                raise ValueError(
                    "Rounded VMM reservation exceeds native int32 page IDs"
                )
            if (
                self._pool._round_pages(config.num_primary_cache_pages)
                < self.minimum_pages
            ):
                raise ValueError(
                    "Initial VMM capacity cannot hold a maximum batch history"
                )
            self._pool.resize(config.num_primary_cache_pages, at_batch_barrier=True)
            self.num_primary_cache_pages = self._pool.stats()["mapped_pages"]
            if self.num_primary_cache_pages >= 2**31:
                raise ValueError("Rounded VMM page capacity exceeds int32")
            self.impl_ = _impl_factory(
                self.num_layers,
                self.num_heads,
                self.head_dim,
                self.page_size,
                self.chunk_size,
                self.num_primary_cache_pages,
                0,
                self.max_batch_size,
                self.max_sequence_length,
                self.device_idx,
            )
            self.impl_._enable_vmm_events()
            self._make_tables()
        except BaseException as error:
            self._poisoned = True
            self._tables = []
            # Failed import/wrapping frames can retain temporary layer tensors
            # and owners solely through the traceback. Retire those completed
            # frames, but never unmap storage still owned by an escaped alias.
            traceback.clear_frames(error.__traceback__)
            try:
                if self._owners:
                    raise RuntimeError("Constructor left escaped VMM Tensor aliases")
                self._pool.close(at_batch_barrier=True)
                self._closed = True
            except BaseException as cleanup_error:
                # Construction returned no backend. Preserve an explicit owner
                # for safe retry after aliases are retired/driver errors clear.
                error.vmm_cleanup_owner = self
                error.vmm_cleanup_error = cleanup_error
            raise

    @classmethod
    def from_config(cls, config, *, max_pages, **kwargs):
        return cls(config, max_pages=max_pages, **kwargs)

    @property
    def gpu_kvcache_mgr(self):
        return self

    @property
    def gpu_kvcache_tables(self):
        self._check_usable()
        return self._tables

    def get_cache_tables(self, layer_idx=-1):
        return (
            self.gpu_kvcache_tables
            if layer_idx == -1
            else self.gpu_kvcache_tables[layer_idx]
        )

    def _check_usable(self):
        if self._closed or self._poisoned:
            raise RuntimeError("HSTU VMM backend is closed/poisoned")
        self._pool._require_open()

    def _require_batch(self):
        self._check_usable()
        if not self._active_batch or self._batch_thread != threading.get_ident():
            raise RuntimeError("HSTU VMM cache access must run inside backend.batch()")

    @contextmanager
    def batch(self):
        with self._lock:
            self._check_usable()
            if self._active_batch:
                raise RuntimeError("Nested HSTU VMM batches are unsupported")
            self._active_batch = True
            self._batch_thread = threading.get_ident()
            try:
                yield
            except BaseException:
                # Metadata can already include history that the failed forward
                # never appended. Do not reuse that history after any failure.
                self._poisoned = True
                raise
            finally:
                self._active_batch = False
                self._batch_thread = None

    @contextmanager
    def resize_guard(self):
        """Hold the same lock as forward across a coupled capacity transaction."""
        with self._lock:
            self._check_usable()
            if self._active_batch:
                raise RuntimeError("Cannot change capacity during an active batch")
            yield

    def _make_tables(self):
        shape = (
            self.num_primary_cache_pages,
            2,
            self.page_size,
            self.num_heads,
            self.head_dim,
        )
        tables = []
        for pointer in self._pool.layer_ptrs:
            owner = _CudaArrayOwner(self._pool, pointer, shape, self._typestr)
            self._owners.add(owner)
            tables.append(
                self._table_factory(self._torch, owner, self.dtype, self.device_idx)
            )
        self._tables = tables

    def _retire_tables(self):
        self._tables = []
        if self._owners:
            # An escaped storage is alive. Keep the mapping, restore our current
            # views, and refuse mutation. Aliases need no unsafe refcount guesses.
            if self._pool.stats()["state"] == "open" and not self._poisoned:
                self._make_tables()
            raise RuntimeError(
                "Release outstanding KV metadata/Tensor/Storage aliases before resize/close"
            )

    def _barrier(self, at_batch_barrier, cuda_graphs_in_use):
        self._pool._barrier(at_batch_barrier, cuda_graphs_in_use)
        if self._active_batch:
            raise RuntimeError("Cannot resize/close during a batch")
        self._pool._driver.synchronize()

    @property
    def capacity_payload_bytes(self):
        return self._pool.stats()["mapped_physical_bytes"]

    @property
    def resident_payload_bytes(self):
        return (
            sum(len(row[3]) for row in self.impl_._vmm_snapshot())
            * self._pool.page_bytes
            * self.num_layers
        )

    def plan_pages(self, pages):
        """Pure capacity plan; rounded mapped pages count toward the budget."""
        with self._lock:
            self._check_usable()
            if type(pages) is not int or not 0 < pages <= self._pool.max_pages:
                raise ValueError("Requested pages outside reserved positive range")
            mapped = self._pool._round_pages(pages)
            if mapped < self.minimum_pages:
                raise ValueError("VMM target cannot hold a maximum batch history")
            return {
                "requested_pages": pages,
                "mapped_pages": mapped,
                "capacity_payload_bytes": mapped
                * self._pool.page_bytes
                * self.num_layers,
                "minimum_pages": self.minimum_pages,
                "max_requested_pages": self._pool.max_pages,
                "chunk_pages": self._pool.chunk_pages,
                "page_payload_bytes": self._pool.page_bytes * self.num_layers,
            }

    def plan_resize_bytes(self, payload_bytes, *, round_up=False):
        """Return a chunk-aligned physical plan without changing cache state."""
        if type(payload_bytes) is not int or payload_bytes <= 0:
            raise ValueError("KV payload_bytes must be positive")
        quantum = self._pool.chunk_bytes * self.num_layers
        chunks = (
            (payload_bytes + quantum - 1) // quantum
            if round_up
            else payload_bytes // quantum
        )
        mapped = chunks * self._pool.chunk_pages
        if mapped <= 0 or mapped > self._pool.max_mapped_pages:
            raise ValueError("KV budget outside reserved positive VMM range")
        requested = min(mapped, self._pool.max_pages)
        plan = self.plan_pages(requested)
        return dict(
            plan,
            target_pages=requested,
            payload_bytes=plan["capacity_payload_bytes"],
            allocated_physical_bytes=plan["capacity_payload_bytes"],
            extra_peak_bytes=0,
        )

    def stats(self):
        with self._lock:
            result = self._pool.stats()
            result.update(
                backend=self.backend_name,
                poisoned=self._poisoned,
                closed=self._closed,
                capacity_payload_bytes=result["mapped_physical_bytes"],
                payload_bytes=result["mapped_physical_bytes"],
            )
            if not self._closed and not self._poisoned:
                result["resident_payload_bytes"] = self.resident_payload_bytes
            if self._poisoned:
                result["state"] = "poisoned"
            if self._closed:
                result["state"] = "closed"
            return result

    def resize_pages(self, pages, *, at_batch_barrier, cuda_graphs_in_use=False):
        with self._lock:
            self._check_usable()
            plan = self.plan_pages(pages)
            self._barrier(at_batch_barrier, cuda_graphs_in_use)
            before = self.num_primary_cache_pages
            started = time.perf_counter()
            if plan["mapped_pages"] == before:
                self._pool.resize(pages, at_batch_barrier=True)
                return dict(
                    self.stats(),
                    action_ms=0.0,
                    evicted_users=[],
                    preserves_survivors=True,
                )
            snapshot = self.impl_._vmm_snapshot()
            self._retire_tables()
            # Once metadata or mappings mutate, an unexpected error poisons the
            # complete backend; restoring a quota alone cannot recover lost KV.
            self._poisoned = True
            if plan["mapped_pages"] < before:
                evicted = self.impl_._resize_vmm_metadata(
                    plan["mapped_pages"], before, snapshot
                )
                live = [page for row in self.impl_._vmm_snapshot() for page in row[3]]
                self._pool.resize(pages, at_batch_barrier=True, live_page_ids=live)
            else:
                self._pool.resize(pages, at_batch_barrier=True)
                evicted = self.impl_._resize_vmm_metadata(
                    plan["mapped_pages"], before, snapshot
                )
            self.num_primary_cache_pages = plan["mapped_pages"]
            self._make_tables()
            self._evictions.extend(
                (uid, length, "resize_tail") for uid, length in evicted
            )
            self._poisoned = False
            return dict(
                self.stats(),
                action_ms=1000 * (time.perf_counter() - started),
                evicted_users=evicted,
                preserves_survivors=True,
            )

    def close(self, *, at_batch_barrier, cuda_graphs_in_use=False):
        with self._lock:
            if self._closed:
                return self.stats()
            self._barrier(at_batch_barrier, cuda_graphs_in_use)
            self._retire_tables()
            self._poisoned = True
            self._pool.close(at_batch_barrier=True)
            self._closed = True
            return self.stats()

    def _index(self, user_ids, sequence_lengths):
        torch = self._torch
        for tensor, name, dtypes in (
            (user_ids, "user_ids", (torch.int64,)),
            (sequence_lengths, "sequence_lengths", (torch.int32, torch.int64)),
        ):
            if (
                tensor.device.type != "cpu"
                or tensor.dtype not in dtypes
                or tensor.dim() != 1
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be a contiguous CPU integer vector")
        ids, lengths = user_ids.tolist(), sequence_lengths.tolist()
        if (
            not 0 < len(ids) <= self.max_batch_size
            or len(lengths) != len(ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Invalid batch size or duplicate cache UID")
        if any(n < 0 or n > self.max_sequence_length for n in lengths):
            raise ValueError("History length outside configured bounds")
        if (
            sum(math.ceil(n / self.page_size) for n in lengths)
            > self.num_primary_cache_pages
        ):
            raise ValueError(
                "Complete batch histories exceed current KV capacity; split the batch"
            )
        return ids, lengths

    def lookup_kvcache(self, user_ids, sequence_lengths):
        self._require_batch()
        from recsys_kvcache_manager.kvcache_utils import KVIndexMeta, KVLookupResult

        _, lengths = self._index(user_ids, sequence_lengths)
        starts, cached = self.impl_.lookup(user_ids)
        if any(starts.tolist()) or any(
            k < 0 or k > n for k, n in zip(cached.tolist(), lengths)
        ):
            raise ValueError(
                "History changed/truncated: evict UID explicitly before reuse"
            )
        zero = self._torch.zeros_like(cached)
        result = KVLookupResult(
            user_ids=user_ids,
            cached_start_indices=starts,
            cached_lengths=cached,
            gpu_cached_start_indices=starts,
            gpu_cached_lengths=cached,
            host_cached_start_indices=zero,
            host_cached_lengths=zero,
        )
        return KVIndexMeta(user_ids, sequence_lengths), result

    def allocate_kvcache(
        self, index_meta, lookup_results, output_kvcache_metadata=None
    ):
        self._require_batch()
        if output_kvcache_metadata is not None:
            raise ValueError(
                "VMM uses fresh batch metadata; graph/reusable metadata is unsupported"
            )
        from recsys_kvcache_manager.kvcache_metadata import get_kvcache_metadata_buffer

        ids, lengths = self._index(index_meta.user_ids, index_meta.seq_lengths)
        _, current = self.impl_.lookup(index_meta.user_ids)
        if not self._torch.equal(current, lookup_results.cached_lengths):
            raise ValueError("Stale KV lookup; lookup again before allocation")
        if lookup_results.user_ids.tolist() != ids:
            raise ValueError("Foreign KV lookup")
        for field in (
            "cached_start_indices",
            "gpu_cached_start_indices",
            "host_cached_start_indices",
            "host_cached_lengths",
        ):
            if any(getattr(lookup_results, field).tolist()):
                raise ValueError("Suffix or host-backed KV lookup")
        if not self._torch.equal(current, lookup_results.gpu_cached_lengths):
            raise ValueError("Invalid GPU cached lengths")
        cached = current.tolist()
        if any(k < 0 or k > n for k, n in zip(cached, lengths)):
            raise ValueError("Cached prefix exceeds requested history")
        new_tokens = sum(n - k for n, k in zip(lengths, cached))
        pages = sum(math.ceil(n / self.page_size) for n in lengths)
        metadata = get_kvcache_metadata_buffer(
            len(ids), new_tokens, pages, device=self.device_idx
        )
        metadata.kv_cache_table = list(self._tables)
        metadata.max_seqlen = max(lengths)
        self.impl_.allocate(
            index_meta.user_ids,
            index_meta.seq_lengths.to(dtype=self._torch.int64).contiguous(),
            lookup_results.host_cached_lengths,
            metadata.page_ids_gpu_buffer,
            metadata.metadata_gpu_buffer,
        )
        self._evictions.extend(
            (uid, length, "capacity")
            for uid, length in self.impl_._drain_vmm_capacity_evictions()
        )
        return metadata

    def drain_evictions(self):
        with self._lock:
            events, self._evictions = self._evictions, []
            return events

    def evict(self, user_ids, for_gpu=False, for_host=False):
        with self._lock:
            self._check_usable()
            if self._active_batch:
                raise RuntimeError("Cannot invalidate KV during a batch")
            if for_host:
                raise ValueError("VMM has no host cache")
            if for_gpu:
                self._pool._driver.synchronize()
                for uid in user_ids.tolist():
                    self.impl_.evict(uid)

    def evict_all(self, for_gpu=False, for_host=False):
        with self._lock:
            self._check_usable()
            if self._active_batch:
                raise RuntimeError("Cannot invalidate KV during a batch")
            if for_host:
                raise ValueError("VMM has no host cache")
            if for_gpu:
                self._pool._driver.synchronize()
                self.impl_.evict_all()

    def onboard_launch(self, index_meta, lookup_result, kvcache_metadata):
        self._require_batch()
        return None

    def onboard_wait(self, kv_index_meta, task_handle):
        self._require_batch()
        if task_handle is not None:
            raise ValueError("VMM cannot consume a host onload task")
        return None

    onboard_try_wait = onboard_wait

    def onboard_wait_layer(self, task_handle, layer_idx):
        return self.onboard_wait(None, task_handle)

    def offload_launch(self, index_meta, kvcache_metadata=None):
        self._require_batch()
        return None

    def offload_try_wait(self):
        with self._lock:
            self._check_usable()

    offload_reap_completed = offload_try_wait
    offload_flush = offload_try_wait

    def prefetch_kvcache(self, index_meta):
        self._require_batch()
        return None
