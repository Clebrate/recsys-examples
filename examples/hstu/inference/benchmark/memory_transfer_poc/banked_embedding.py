# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PoC embedding module using real NVE GPU banks and the existing NVE CPU PS.

Construct after checkpoint loading, then replace one existing collection entry:
    collection.embeddings[table_name] = BankedNVEEmbedding(ps, width, bank_spans)

A span is (CUDA pointer, bytes, owner). For two transferable page ranges, call
kv_tail_spans once per range; each layer produces a bank. Prepend a permanent
bank span. Retire those KV pages BEFORE constructing the NVE banks. After
draining requests and FlexKV, retire a group here, then restore its KV pages.
For the reverse direction, retire empty KV pages first, then restore the group.

The primary policy hashes each key with the SplitMix64 finalizer, then takes
its unsigned high 32 bits modulo the original bank count. This avoids coupling
bank routing to NVE's key % num_sets mapping. Raw lookup keys are unchanged.
By default retired partitions
route to the first live bank, whose misses read the real NVE PS and populate
that bank. Active primary mappings stay unchanged. Restoring a partition can
leave duplicate cached entries in the fallback bank; no caches are cleared.
Fixed-ratio comparisons must use this same module, routing, and insert policy.
The Python routing and real NVE CPU misses add overhead; this is a mechanism
POC, not a performance-equivalent replacement for the normal NVEmbedding.
"""

from contextlib import nullcontext
from typing import Sequence

import torch
from torch.utils.dlpack import to_dlpack

import memory_transfer_poc


_HIT_COUNT_SOURCE = (
    "gpu_hits/ps_hits are llround(NVE float table hit rate * query count) per bank batch; "
    "queries include duplicate keys, not raw counters or unique MGET miss counts"
)


def primary_bank_ids(keys: torch.Tensor, bank_count: int) -> torch.Tensor:
    """Stable partitions using the high bits of the SplitMix64 finalizer.

    Torch int64 multiplication wraps modulo 2**64. Signed constants encode
    0xbf58476d1ce4e5b9 and 0x94d049bb133111eb; masks turn arithmetic shifts
    into logical shifts. Hash only the route, never the key passed to NVE/PS.
    """
    mixed = (keys ^ ((keys >> 30) & ((1 << 34) - 1))) * -4658895280553007687
    mixed = (mixed ^ ((mixed >> 27) & ((1 << 37) - 1))) * -7723592293110705685
    mixed = mixed ^ ((mixed >> 31) & ((1 << 33) - 1))
    return ((mixed >> 32) & 0xffffffff).remainder(bank_count)


def tensor_span(tensor: torch.Tensor):
    """Borrow exactly a contiguous tensor view while retaining its storage owner."""
    if not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() == 0:
        raise ValueError("A bank span must be a nonempty contiguous CUDA tensor")
    return (tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor)


def kv_tail_spans(kv_tensor: torch.Tensor, first_page: int, num_pages: int):
    """Return one contiguous NVE bank span per layer in a BF16 KV page range."""
    if kv_tensor.dtype != torch.bfloat16 or kv_tensor.ndim != 6 or kv_tensor.shape[2] != 2:
        raise ValueError("Expected BF16 KV tensor [layer, page, 2, token, head, dim]")
    if not kv_tensor.is_cuda or not kv_tensor.is_contiguous():
        raise ValueError("KV tensor must be contiguous and on CUDA")
    if first_page < 0 or num_pages <= 0 or first_page + num_pages > kv_tensor.shape[1]:
        raise ValueError("Invalid KV page range")
    return [tensor_span(layer.narrow(0, first_page, num_pages)) for layer in kv_tensor.unbind(0)]


class BankedNVEEmbedding(torch.nn.Module):
    def __init__(self, parameter_server, row_elements: int, bank_spans: Sequence,
                 device=None, max_batch_keys: int = 4096, retired_policy: str = "first_live"):
        super().__init__()
        if row_elements <= 0 or max_batch_keys <= 0 or not bank_spans:
            raise ValueError("Need a positive width/batch limit and at least one bank")
        self.embedding_size = row_elements
        self.data_type = torch.float32
        self.max_batch_keys = max_batch_keys
        self.bank_count = len(bank_spans)  # Never changes when banks retire.
        if retired_policy not in {"first_live", "ps_only"}:
            raise ValueError("retired_policy must be first_live or ps_only")
        self.retired_policy = retired_policy
        self._live_banks = [True] * self.bank_count
        self._route_ids = list(range(self.bank_count))
        self._collect_lookup_stats = False
        self.profile = False
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Banked NVE requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        ps_table = getattr(parameter_server, "parameter_server", parameter_server)
        self._backend = memory_transfer_poc.BankedNVE(
            ps_table, row_elements, list(bank_spans), self.device.index, max_batch_keys
        )
        self.register_buffer("_bank_routes", torch.arange(self.bank_count, dtype=torch.int64,
                                                         device=self.device), persistent=False)
        self.last_lookup_stats = {}

    @torch.no_grad()
    def forward(self, keys: torch.Tensor, *, populate: bool = True) -> torch.Tensor:
        if keys.dtype != torch.int64 or keys.ndim != 1 or keys.device != self.device:
            raise ValueError("Expected a CUDA int64 key vector on the bank device")
        result = torch.empty((keys.numel(), self.embedding_size), dtype=torch.float32, device=self.device)
        totals = {"queries": 0, "gpu_hits": 0, "ps_hits": 0}
        bank_results = []
        profile_range = torch.profiler.record_function if self.profile else nullcontext
        # Primary partitions never change. Only retired partitions can borrow
        # the first live bank; this is an explicit policy, not retained hits.
        with profile_range("recsys.embedding.banked.route_and_gather"):
            primary_routes = primary_bank_ids(keys, self.bank_count)
            routes = self._bank_routes[primary_routes]
        stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        for bank_id in range(self.bank_count):
            with profile_range("recsys.embedding.banked.route_and_gather"):
                positions = torch.nonzero(routes == bank_id, as_tuple=False).flatten()
            for start in range(0, positions.numel(), self.max_batch_keys):
                with profile_range("recsys.embedding.banked.route_and_gather"):
                    selected = positions[start:start + self.max_batch_keys]
                    bank_keys = keys.index_select(0, selected).contiguous()
                    bank_output = torch.empty((selected.numel(), self.embedding_size),
                                              dtype=torch.float32, device=self.device)
                with profile_range("recsys.embedding.banked.native_lookup_and_populate"):
                    stats = self._backend.lookup_bank(
                        bank_id, to_dlpack(bank_keys), to_dlpack(bank_output), stream_ptr, populate
                    )
                with profile_range("recsys.embedding.banked.scatter"):
                    result.index_copy_(0, selected, bank_output)
                bank_results.append(stats)
                for field in totals:
                    totals[field] += stats[field]
        totals["gpu_hit_rate"] = totals["gpu_hits"] / totals["queries"] if totals["queries"] else 0.0
        totals["banks"] = bank_results
        totals["hit_count_source"] = _HIT_COUNT_SOURCE
        self.last_lookup_stats = totals
        return result

    def retire(self, bank_ids):
        """Drain these banks, release their spans, and redirect retired partitions."""
        bank_ids = list(bank_ids)
        if self.retired_policy == "first_live" and not any(
            live and bank not in bank_ids for bank, live in enumerate(self._live_banks)
        ):
            raise ValueError("first_live policy requires at least one permanent live bank")
        self._backend.retire(bank_ids)
        for bank in bank_ids:
            self._live_banks[bank] = False
        self._refresh_routes()

    def restore(self, bank_ids):
        """Recreate only these banks after their KV page ranges are retired."""
        bank_ids = list(bank_ids)
        self._backend.restore(bank_ids)
        for bank in bank_ids:
            self._live_banks[bank] = True
        self._refresh_routes()

    def _refresh_routes(self):
        first_live = next((bank for bank, live in enumerate(self._live_banks) if live), None)
        self._route_ids = [
            bank if live or self.retired_policy == "ps_only" else first_live
            for bank, live in enumerate(self._live_banks)
        ]
        self._bank_routes.copy_(torch.tensor(self._route_ids, dtype=torch.int64, device=self.device))

    @property
    def emb_layer(self):
        # Compatibility with collection statistics without registering a module
        # as its own child (which would create a state_dict/traversal cycle).
        return self

    def set_collect_lookup_stats(self, enabled: bool):
        self._collect_lookup_stats = bool(enabled)

    def get_last_lookup_stats(self):
        if not self._collect_lookup_stats or not self.last_lookup_stats:
            return (0, [])
        stats = self.last_lookup_stats
        count = stats["queries"]
        return (count, [stats["gpu_hits"] / count if count else 0.0,
                        stats["ps_hits"] / count if count else 0.0])

    def wait(self):
        self._backend.wait()

    def stats(self):
        result = self._backend.stats()
        result["hit_count_source"] = _HIT_COUNT_SOURCE
        result["primary_routing_policy"] = "splitmix64_finalizer(key).unsigned_high32 % original_bank_count"
        result["retired_policy"] = self.retired_policy
        result["physical_bank_for_primary_partition"] = self._route_ids[:]
        result["restore_policy"] = "original primary mapping resumes; fallback-bank duplicates are not cleared"
        return result

    def close(self):
        self._backend.close()
