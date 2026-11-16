# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Experimental eight-way linear-hash embedding cache with VMM value storage.

This is a separate per-row LRU prototype, not the NVEmb LFU backend. The CUDA
library owns fixed maximum-size metadata; VmmKvPool owns the external values.
All native calls complete their GPU work before returning. Values returned by
lookup are copies, so callers never retain aliases into the resizable arena.
"""

import ctypes
import threading
import time

from .pool import VmmKvPool, _count


class _Native:
    """The synchronous C ABI in native/vmm_embedding.cu."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        pointer, u64 = ctypes.c_void_p, ctypes.c_uint64
        signatures = {
            "create": [pointer, u64, u64, u64, ctypes.c_int,
                       ctypes.POINTER(pointer), ctypes.POINTER(u64)],
            "lookup": [pointer, pointer, u64, pointer, pointer],
            "insert": [pointer, pointer, pointer, u64],
            "resize": [pointer, u64, u64, ctypes.POINTER(u64)],
            "stats": [pointer, ctypes.POINTER(u64), ctypes.POINTER(u64),
                      ctypes.POINTER(u64)],
            "destroy": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, "vmm_embedding_" + name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.lib.vmm_embedding_last_error.argtypes = []
        self.lib.vmm_embedding_last_error.restype = ctypes.c_char_p

    def _call(self, name, *args):
        if getattr(self.lib, "vmm_embedding_" + name)(*args):
            error = self.lib.vmm_embedding_last_error()
            message = error.decode("utf-8", errors="replace") if error else "unknown error"
            raise RuntimeError(f"Native embedding {name} failed: {message}")

    def create(self, pointer, row_bytes, max_sets, initial_sets, device):
        handle, metadata = ctypes.c_void_p(), ctypes.c_uint64()
        self._call("create", pointer, row_bytes, max_sets, initial_sets, device,
                   ctypes.byref(handle), ctypes.byref(metadata))
        if not handle.value:
            raise RuntimeError("Native embedding create returned a null handle")
        return handle.value, metadata.value

    def lookup(self, handle, keys, count, values, hits):
        self._call("lookup", handle, keys, count, values, hits)

    def insert(self, handle, keys, values, count):
        self._call("insert", handle, keys, values, count)

    def resize(self, handle, old_sets, new_sets):
        evicted = ctypes.c_uint64()
        self._call("resize", handle, old_sets, new_sets, ctypes.byref(evicted))
        return evicted.value

    def stats(self, handle):
        active, occupied, clock = (ctypes.c_uint64() for _ in range(3))
        self._call("stats", handle, ctypes.byref(active), ctypes.byref(occupied),
                   ctypes.byref(clock))
        return active.value, occupied.value, clock.value

    def destroy(self, handle):
        self._call("destroy", handle)


class VmmEmbeddingCache:
    """Opaque byte rows, nonnegative int64 keys, and explicit serialized resize.

    Payload limits and resize targets must be exact multiples of ``chunk_bytes``.
    Metadata is allocated for the maximum capacity once and reported separately.
    Grow maps and zeroes new payload before splitting buckets. Shrink merges
    buckets, retaining the eight most recently used rows of each pair, before
    unmapping its tail. The native library reports the resulting eviction count.

    CUDA graphs, external raw-pointer users, and concurrent operations outside
    this wrapper are unsupported. Call close explicitly; there is no destructor
    that might free storage while a consumer is running. Private constructor
    injections support CPU coordination tests without importing torch or CUDA.
    """

    backend_name = "vmm_embedding_linear_hash_lru_prototype"

    def __init__(
        self, row_bytes, max_payload_bytes, initial_payload_bytes, chunk_bytes,
        library_path, device=0, *, _torch=None, _driver=None, _native=None,
    ):
        for value, name in ((row_bytes, "row_bytes"),
                            (max_payload_bytes, "max_payload_bytes"),
                            (initial_payload_bytes, "initial_payload_bytes"),
                            (chunk_bytes, "chunk_bytes")):
            _count(value, name, 1)
        _count(device, "device")
        if row_bytes % 4 or chunk_bytes % (8 * row_bytes):
            raise ValueError("Rows must be a multiple of four bytes; chunks must hold whole eight-row sets")
        if max_payload_bytes % chunk_bytes or initial_payload_bytes % chunk_bytes:
            raise ValueError("Payload capacities must be exact multiples of chunk_bytes")
        if initial_payload_bytes > max_payload_bytes:
            raise ValueError("Initial payload exceeds maximum payload")
        max_sets = max_payload_bytes // (8 * row_bytes)
        if max_sets >= 2**31:
            raise ValueError("Maximum set count must fit signed int32")
        self._native = _native if _native is not None else _Native(library_path)
        if _torch is None:
            import torch as _torch
        self._torch = _torch
        self.row_bytes, self.device = row_bytes, device
        self.max_payload_bytes, self.chunk_bytes = max_payload_bytes, chunk_bytes
        self._lock = threading.RLock()
        self._handle = None
        self._metadata_bytes = 0
        self._quarantined_metadata_bytes = 0
        self._active_sets = 0
        self._closed = False
        self._poisoned = False
        self._generation = 0
        self._pool = None
        with self._torch.cuda.device(self.device):
            # Establish the runtime's primary context before creating CudaDriver.
            context_probe = self._torch.empty(1, device=f"cuda:{device}")
            self._pool = VmmKvPool(
                1, 8 * row_bytes, max_sets, device=device, driver=_driver,
                chunk_bytes=chunk_bytes,
            )
            del context_probe
            try:
                self._active_sets = initial_payload_bytes // (8 * row_bytes)
                self._pool.resize(self._active_sets, at_batch_barrier=True)
                self._handle, self._metadata_bytes = self._native.create(
                    self._pool.layer_ptrs[0], row_bytes, max_sets,
                    self._active_sets, device,
                )
            except BaseException as error:
                self._poisoned = True
                try:
                    self.close()
                except BaseException as cleanup_error:
                    error.vmm_cleanup_owner = self
                    error.vmm_cleanup_error = cleanup_error
                raise

    def _check_usable(self):
        if self._closed or self._poisoned:
            raise RuntimeError("Embedding VMM cache is closed or poisoned; only cleanup is allowed")

    def _keys(self, keys):
        if (keys.dtype != self._torch.int64 or keys.dim() != 1
                or not keys.is_contiguous() or keys.device.type != "cuda"
                or keys.device.index != self.device):
            raise ValueError("keys must be a contiguous CUDA int64 vector on the cache device")

    def _values(self, values, count):
        if (values.dtype != self._torch.uint8 or values.dim() != 2
                or tuple(values.shape) != (count, self.row_bytes)
                or not values.is_contiguous() or values.device.type != "cuda"
                or values.device.index != self.device):
            raise ValueError("values must be contiguous CUDA uint8 [count, row_bytes] on the cache device")

    def lookup(self, keys):
        """Return (uint8 [N, row_bytes] values, bool [N] hits); misses are zero."""
        with self._lock, self._torch.cuda.device(self.device):
            self._check_usable()
            self._keys(keys)
            # Validate the owning context and drain producers on other streams
            # before the native ABI uses its default stream.
            self._pool._driver.synchronize()
            if keys.numel() and keys.min().item() < 0:
                raise ValueError("Embedding keys must be nonnegative")
            values = self._torch.empty(
                (keys.numel(), self.row_bytes), dtype=self._torch.uint8,
                device=f"cuda:{self.device}",
            )
            hits = self._torch.empty(
                keys.numel(), dtype=self._torch.bool, device=f"cuda:{self.device}",
            )
            try:
                self._native.lookup(self._handle, keys.data_ptr(), keys.numel(),
                                    values.data_ptr(), hits.data_ptr())
            except BaseException:
                # Lookup updates per-row recency; partial mutation is unusable.
                self._poisoned = True
                raise
            return values, hits

    def insert(self, keys, values):
        """Insert/update byte rows. The final occurrence wins for duplicate keys."""
        with self._lock, self._torch.cuda.device(self.device):
            self._check_usable()
            self._keys(keys)
            self._values(values, keys.numel())
            self._pool._driver.synchronize()
            if keys.numel() and keys.min().item() < 0:
                raise ValueError("Embedding keys must be nonnegative")
            try:
                self._native.insert(self._handle, keys.data_ptr(),
                                    values.data_ptr(), keys.numel())
            except BaseException:
                self._poisoned = True
                raise

    def stats(self, *, include_occupancy=False):
        """Accounting without a GPU scan; opt in to native occupancy diagnostics."""
        with self._lock:
            pool = self._pool.stats()
            metadata = (self._metadata_bytes if self._handle is not None
                        else self._quarantined_metadata_bytes)
            active = None if self._poisoned and not self._closed else self._active_sets
            result = dict(
                pool, backend=self.backend_name, policy="per_row_lru",
                state="closed" if self._closed else "poisoned" if self._poisoned else "open",
                poisoned=self._poisoned, closed=self._closed,
                row_bytes=self.row_bytes, max_payload_bytes=self.max_payload_bytes,
                active_sets=active, capacity_rows=None if active is None else 8 * active,
                payload_bytes=pool["mapped_physical_bytes"],
                fixed_metadata_bytes=metadata,
                metadata_accounting="requested_cuda_malloc_bytes",
                metadata_cleanup_uncertain=bool(self._quarantined_metadata_bytes),
                device_pool_bytes=pool["allocated_physical_bytes"] + metadata,
                generation=self._generation,
            )
            if include_occupancy:
                self._check_usable()
                with self._torch.cuda.device(self.device):
                    self._pool._driver.synchronize()
                    active, occupied, clock = self._native.stats(self._handle)
                if active != self._active_sets:
                    self._poisoned = True
                    raise RuntimeError("Native embedding capacity disagrees with the VMM wrapper")
                result.update(occupied_rows=occupied, resident_rows=occupied, clock=clock)
            return result

    def resize(self, target_payload_bytes, *, at_batch_barrier=True,
               cuda_graphs_in_use=False):
        """Return completed pool/migration/total timings; no occupancy scan runs."""
        with self._lock, self._torch.cuda.device(self.device):
            self._check_usable()
            self._pool._barrier(at_batch_barrier, cuda_graphs_in_use)
            _count(target_payload_bytes, "target_payload_bytes", 1)
            if target_payload_bytes > self.max_payload_bytes or target_payload_bytes % self.chunk_bytes:
                raise ValueError("Target payload must be chunk-aligned and within the reserved maximum")
            old_sets = self._active_sets
            new_sets = target_payload_bytes // (8 * self.row_bytes)
            started = time.perf_counter()
            self._pool._driver.synchronize()
            barrier_ms = 1000 * (time.perf_counter() - started)
            pool_ms = migration_ms = 0.0
            evicted = 0
            if new_sets != old_sets:
                self._poisoned = True
                native_started = False
                try:
                    if new_sets > old_sets:
                        phase = time.perf_counter()
                        self._pool.resize(new_sets, at_batch_barrier=True)
                        pool_ms = 1000 * (time.perf_counter() - phase)
                    phase = time.perf_counter()
                    native_started = True
                    evicted = self._native.resize(self._handle, old_sets, new_sets)
                    migration_ms = 1000 * (time.perf_counter() - phase)
                    if new_sets < old_sets:
                        phase = time.perf_counter()
                        # Native merge has emptied the tail. All remaining sets
                        # form a conservative complete superset of live pages.
                        self._pool.resize(new_sets, at_batch_barrier=True,
                                          live_page_ids=range(new_sets))
                        pool_ms = 1000 * (time.perf_counter() - phase)
                except BaseException:
                    pool = self._pool.stats()
                    if (not native_started and pool["state"] == "open"
                            and pool["mapped_pages"] == old_sets):
                        self._poisoned = False  # Pool growth fully rolled back.
                    raise
                self._active_sets = new_sets
                self._generation += 1
                self._poisoned = False
            result = self.stats()
            result.update(pool_ms=pool_ms, migration_ms=migration_ms,
                          barrier_ms=barrier_ms, evicted_rows=evicted,
                          preserves_survivor_values=True,
                          mapped_delta_bytes=(new_sets - old_sets) * 8 * self.row_bytes)
            result["total_ms"] = 1000 * (time.perf_counter() - started)
            result["action_ms"] = result["total_ms"]
            return result

    def close(self, *, at_batch_barrier=True, cuda_graphs_in_use=False):
        """Drain work, release native metadata, then close the value pool."""
        with self._lock, self._torch.cuda.device(self.device):
            self._pool._barrier(at_batch_barrier, cuda_graphs_in_use)
            if self._closed:
                return self.stats()
            self._pool._driver.synchronize()
            self._poisoned = True
            if self._handle is not None:
                handle = self._handle
                # Native destroy consumes the handle even on error. Failed
                # cudaFree resources may be quarantined until context teardown.
                self._quarantined_metadata_bytes = self._metadata_bytes
                self._handle = None
                self._native.destroy(handle)
                self._quarantined_metadata_bytes = 0
            self._pool.close(at_batch_barrier=True)
            self._active_sets = 0
            self._closed = True
            return self.stats()
