# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A bounded, single-device VMM arena with injectable CUDA driver operations.

CUDA requires complete map ranges to be unmapped, so each releasable chunk is
created and mapped independently. See the official Driver API:
https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html

This module manages storage only. It does not own native HSTU page metadata,
evict users, move live pages, wrap tensors, resize NVEmb, or execute a ratio
policy. No CUDA package is imported until a real driver is requested.
"""

from dataclasses import dataclass
from math import lcm


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class _Chunk:
    address: int
    size: int
    handle: object
    layer: int
    first_page: int
    mapped: bool = False


class VmmKvPool:
    """Reserve fixed layer addresses and commit physical memory in whole chunks.

    ``page_bytes`` is one layer's [2, tokens, heads, channels] KV page size.
    A single VA reservation contains ``num_layers`` fixed-size layer spans.
    Each span is rounded to lcm(page_bytes, driver allocation granularity).
    ``max_pages`` limits requested pages; actual mapped capacity can be larger
    because of rounding, and is always reported separately. All mapped pages,
    including rounding slack, are usable capacity; a same-chunk resize releases
    no physical memory. New chunks are zeroed before publication.

    ``chunk_bytes`` optionally selects a larger physical allocation unit. It
    must be a multiple of both the CUDA minimum and ``page_bytes``. Larger units
    reduce mapping calls but coarsen capacity rounding; ``granularity_bytes``
    continues to report the actual CUDA minimum, not the selected chunk size.

    Every resize/close requires an exclusive completed-batch barrier, no graphs,
    and no concurrently launched consumers. Synchronization drains queued work;
    it does not acquire the caller's lock. A shrink requires a complete supplied
    set of live page IDs; this arena cannot independently verify native metadata.
    Pointers remain stable until close, but old larger views become unsafe after
    shrinking. The caller must retire them and renew batch metadata. There is no
    Tensor ownership bridge: callers must close explicitly after all consumers
    finish. No destructor silently unmaps possibly live addresses.

    Growth failures roll back newly allocated chunks. Failed rollback or a
    partially completed shrink poisons the pool; only close may then run. Close
    tracks unreleased resources and can be retried after a driver error.
    """

    def __init__(
        self, num_layers, page_bytes, max_pages, *, device=0, driver=None,
        chunk_bytes=None,
    ):
        _count(num_layers, "num_layers", 1)
        _count(page_bytes, "page_bytes", 1)
        _count(max_pages, "max_pages", 1)
        _count(device, "device")
        if driver is None:
            from .cuda_driver import CudaDriver

            driver = CudaDriver(device)
        self._driver = driver
        self.num_layers = num_layers
        self.page_bytes = page_bytes
        self.max_pages = max_pages
        self.granularity_bytes = driver.granularity()
        _count(self.granularity_bytes, "driver granularity", 1)
        minimum_chunk = lcm(page_bytes, self.granularity_bytes)
        if chunk_bytes is not None:
            _count(chunk_bytes, "chunk_bytes", 1)
            if chunk_bytes % minimum_chunk:
                raise ValueError(
                    "chunk_bytes must be a multiple of both page size and CUDA granularity"
                )
        self.chunk_bytes = minimum_chunk if chunk_bytes is None else chunk_bytes
        self.chunk_pages = self.chunk_bytes // page_bytes
        self.max_mapped_pages = self._round_pages(max_pages)
        self.layer_stride_bytes = self.max_mapped_pages * page_bytes
        self._reservation_bytes = num_layers * self.layer_stride_bytes
        if self._reservation_bytes > (1 << 63) - 1:
            raise ValueError("VA reservation exceeds signed 64-bit size")
        self._base = driver.reserve(self._reservation_bytes, self.granularity_bytes)
        self._chunks = {}
        self._mapped_pages = 0
        self._requested_pages = 0
        self._state = "open"

    def _round_pages(self, pages):
        return ((pages + self.chunk_pages - 1) // self.chunk_pages) * self.chunk_pages

    @property
    def layer_ptrs(self):
        """Stable per-layer base pointers, valid only while the pool is open."""
        self._require_open()
        return tuple(
            self._base + i * self.layer_stride_bytes for i in range(self.num_layers)
        )

    def stats(self):
        """Return actual mapping/accounting; poisoned pools have no valid capacity.

        Driver allocations are outside PyTorch's caching allocator accounting.
        ``mapped_physical_bytes`` counts current mappings, while
        ``allocated_physical_bytes`` also includes handles awaiting cleanup.
        Neither number is the reserved virtual address size.
        """
        return {
            "state": self._state,
            "requested_pages": self._requested_pages,
            "mapped_pages": self._mapped_pages if self._state != "poisoned" else None,
            "mapped_physical_bytes": sum(
                c.size for c in self._chunks.values() if c.mapped
            ),
            "allocated_physical_bytes": sum(c.size for c in self._chunks.values()),
            "reserved_va_bytes": self._reservation_bytes if self._base else 0,
            "base_ptr": self._base,
            "layer_stride_bytes": self.layer_stride_bytes,
            "chunk_pages": self.chunk_pages,
            "chunk_bytes": self.chunk_bytes,
            "granularity_bytes": self.granularity_bytes,
            "max_requested_pages": self.max_pages,
            "max_mapped_pages": self.max_mapped_pages,
        }

    def _require_open(self):
        if self._state != "open":
            raise RuntimeError(f"VMM pool is {self._state}; no cache access is allowed")

    @staticmethod
    def _barrier(at_batch_barrier, cuda_graphs_in_use):
        if at_batch_barrier is not True or cuda_graphs_in_use is not False:
            raise ValueError(
                "An exclusive completed-batch barrier without CUDA graphs is required"
            )

    def _drop_chunk(self, chunk):
        if chunk.mapped:
            self._driver.unmap(chunk.address, chunk.size)
            chunk.mapped = False
        self._driver.release(chunk.handle)
        del self._chunks[chunk.address]

    def _rollback_growth(self, added):
        failures = []
        for chunk in reversed(added):
            try:
                self._drop_chunk(chunk)
            except BaseException as error:
                failures.append(error)
        if failures:
            self._state = "poisoned"
            raise RuntimeError(
                "Growth rollback failed; pool is poisoned; retry close"
            ) from failures[0]

    def resize(
        self,
        requested_pages,
        *,
        at_batch_barrier,
        live_page_ids=None,
        cuda_graphs_in_use=False,
    ):
        """Commit a rounded capacity without eviction or live-page relocation.

        ``live_page_ids`` must be complete and current at the caller's barrier.
        Page IDs are common to all layers. Any ID in a physically removed tail
        rejects the operation before a driver call. No proof is needed when
        rounding leaves the actual mapped capacity unchanged.
        """
        self._require_open()
        self._barrier(at_batch_barrier, cuda_graphs_in_use)
        _count(requested_pages, "requested_pages")
        if requested_pages > self.max_pages:
            raise ValueError("Requested pages exceed the reserved maximum")
        target = self._round_pages(requested_pages)
        if isinstance(live_page_ids, range):
            # A native cache may prove that every survivor is in a contiguous
            # prefix. Validate that conservative range without constructing a
            # Python set with millions of IDs on the resize path.
            if live_page_ids:
                first, last = live_page_ids[0], live_page_ids[-1]
                if min(first, last) < 0 or max(first, last) >= self._mapped_pages:
                    raise ValueError("Live page ID is outside current mapped capacity")
                live = {max(first, last)}
            else:
                live = set()
        elif live_page_ids is not None:
            live = set()
            for page in live_page_ids:
                _count(page, "live page ID")
                if page >= self._mapped_pages:
                    raise ValueError("Live page ID is outside current mapped capacity")
                live.add(page)
        else:
            live = None
        if target < self._mapped_pages:
            if live is None:
                raise ValueError(
                    "Shrink requires an explicit complete live_page_ids set"
                )
            if any(page >= target for page in live):
                raise ValueError(
                    "Shrink tail contains live KV pages; no eviction or copying is performed"
                )
        if target == self._mapped_pages:
            self._requested_pages = requested_pages
            return self.stats()
        self._driver.synchronize()
        layer_ptrs = self.layer_ptrs
        # Fail closed even if an interrupt arrives between mutation and commit.
        self._state = "poisoned"
        if target > self._mapped_pages:
            added = []
            try:
                for layer, ptr in enumerate(layer_ptrs):
                    for page in range(self._mapped_pages, target, self.chunk_pages):
                        handle = self._driver.create(self.chunk_bytes)
                        chunk = _Chunk(
                            ptr + page * self.page_bytes,
                            self.chunk_bytes,
                            handle,
                            layer,
                            page,
                        )
                        self._chunks[chunk.address] = chunk
                        added.append(chunk)
                        self._driver.map(chunk.address, chunk.size, chunk.handle)
                        chunk.mapped = True
                        self._driver.set_access(chunk.address, chunk.size)
                        self._driver.zero(chunk.address, chunk.size)
                self._driver.synchronize()
            except BaseException:
                # zero may have queued work even if a later operation failed.
                try:
                    self._driver.synchronize()
                except BaseException:
                    self._state = "poisoned"
                    raise RuntimeError(
                        "Could not drain failed growth; pool is poisoned; retry close"
                    )
                self._rollback_growth(added)
                self._state = "open"
                raise
        else:
            tail = [c for c in self._chunks.values() if c.first_page >= target]
            try:
                for chunk in reversed(tail):
                    self._drop_chunk(chunk)
            except BaseException:
                # Released backing cannot be restored without losing contents.
                self._state = "poisoned"
                raise
        self._mapped_pages = target
        self._requested_pages = requested_pages
        self._state = "open"
        return self.stats()

    def close(self, *, at_batch_barrier, cuda_graphs_in_use=False):
        """Drain users, unmap complete chunks, release handles, then free VA.

        Idempotent after success; retryable on failure. The caller must retire
        every raw pointer/tensor view before closing, even when a pool is poisoned.
        """
        self._barrier(at_batch_barrier, cuda_graphs_in_use)
        if self._state == "closed":
            return self.stats()
        self._driver.synchronize()
        self._state = "poisoned"
        failures = []
        for chunk in reversed(list(self._chunks.values())):
            try:
                self._drop_chunk(chunk)
            except BaseException as error:
                failures.append(error)
        if failures:
            self._state = "poisoned"
            raise failures[0]
        try:
            self._driver.free_address(self._base, self._reservation_bytes)
        except BaseException:
            self._state = "poisoned"
            raise
        self._base = 0
        self._mapped_pages = 0
        self._requested_pages = 0
        self._state = "closed"
        return self.stats()
