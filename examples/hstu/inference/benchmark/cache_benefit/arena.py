# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-allocation GPU cache, with actual hit/miss execution."""
import time

from costs import Costs, wait_ms
from metadata import KVLRU, EmbeddingLRU


class Parameters:
    def __init__(self, dim, mode="default"):
        if mode not in ("default", "pattern"):
            raise ValueError("Parameter mode must be default or pattern")
        self.dim, self.mode = dim, mode

    def fetch(self, ids):
        import torch

        # The reference sweep does not load a checkpoint; its empty CPU PS
        # supplies the default zero value. Pattern mode detects data corruption.
        values = torch.zeros((len(ids), self.dim), dtype=torch.float32)
        if self.mode == "pattern" and ids:
            keys = torch.tensor(ids, dtype=torch.int64)
            values[:, 0] = (keys % 97).float() / 97
            values[:, 1] = (keys % 31).float() / 31
            values[:, 2] = keys.float() / 2**25
        return values


class Arena:
    def __init__(self, geometry, fraction, parameters, costs=Costs()):
        import torch

        self.geometry, self.costs, self.parameters = geometry, costs, parameters
        self.buffer = torch.empty(
            geometry.actual_bytes, dtype=torch.uint8, device="cuda"
        )
        self.layers = list(self.buffer.split(geometry.layer_bytes))
        self.pages, self.slots, self.boundary = geometry.partition(fraction)
        self.kv = KVLRU(self.pages, geometry.page_tokens)
        self.emb = [
            EmbeddingLRU(
                geometry.layer_bytes // geometry.row_bytes,
                self.boundary // geometry.row_bytes,
            )
            for _ in self.layers
        ]
        physical_pages = geometry.layer_bytes // geometry.page_bytes
        self.kv_tables = [
            layer.view(torch.bfloat16).view(
                physical_pages,
                2,
                geometry.page_tokens,
                geometry.heads,
                geometry.head_dim,
            )
            for layer in self.layers
        ]
        self.embedding_tables = [
            layer.view(torch.float32).view(-1, geometry.embedding_dim)
            for layer in self.layers
        ]
        self.current_ids = None
        self.waits_enabled = False
        self.backend_calls = self.cpu_fetch_ids = 0
        self.resize_records = []
        self.begin()

    @property
    def actual_fraction(self):
        return self.boundary / self.geometry.layer_bytes

    def begin(self):
        self.last_embedding = {}
        self.kv_wait_elapsed_ms = 0.0

    def embedding_lookup(self, gpu_keys):
        import torch

        ids, g = self.current_ids, self.geometry
        if ids is None or len(ids) != gpu_keys.numel():
            raise RuntimeError("Missing or inconsistent immutable request context")
        plans, all_missing = [], []
        for shard, cache in enumerate(self.emb):
            positions = [i for i, key in enumerate(ids) if key % g.layers == shard]
            keys = [ids[i] for i in positions]
            slots, missing, destinations = cache.touch(keys)
            plans.append((positions, slots, missing, destinations))
            all_missing.extend(missing)
        # Exactly one logical backend call for the complete missing key set.
        # A 7.5ms MGET is 7.5ms total, not 7.5ms multiplied by key/shard count.
        target = (
            self.costs.embedding_ms(len(all_missing)) if self.waits_enabled else 0.0
        )
        executed = wait_ms(target)
        fetch_ms, fetched = 0.0, None
        if all_missing:
            started = time.perf_counter()
            fetched = self.parameters.fetch(all_missing).to(gpu_keys.device)
            fetch_ms = (time.perf_counter() - started) * 1000
            self.backend_calls += 1
            self.cpu_fetch_ids += len(all_missing)
        result = torch.empty(
            (len(ids), g.embedding_dim), dtype=torch.float32, device=gpu_keys.device
        )
        offset = 0
        for table, (positions, slots, missing, destinations) in zip(
            self.embedding_tables, plans
        ):
            if missing:
                table.index_copy_(
                    0,
                    torch.tensor(destinations, device=gpu_keys.device),
                    fetched[offset : offset + len(missing)],
                )
                offset += len(missing)
            if positions:
                selected = table.index_select(
                    0, torch.tensor(slots, device=gpu_keys.device)
                )
                result.index_copy_(
                    0, torch.tensor(positions, device=gpu_keys.device), selected
                )
        self.last_embedding = {
            "embedding_lookup_ids": len(ids),
            "embedding_miss_ids": len(all_missing),
            "embedding_backend_calls": int(bool(all_missing)),
            "embedding_wait_target_ms": target,
            "embedding_wait_executed_ms": executed,
            "cpu_fetch_and_h2d_ms": fetch_ms,
        }
        return result

    def drop_embedding(self, ids):
        return sum(
            cache.drop([key for key in ids if key % self.geometry.layers == shard])
            for shard, cache in enumerate(self.emb)
        )

    def resize(self, fraction):
        import torch

        pages, slots, boundary = self.geometry.partition(fraction)
        if pages == self.pages:
            return 0.0
        started = time.perf_counter()
        torch.cuda.synchronize()
        old = self.actual_fraction
        kv_evicted = self.kv.resize(pages)
        emb_evicted = sum(
            cache.resize(boundary // self.geometry.row_bytes) for cache in self.emb
        )
        self.pages, self.slots, self.boundary = pages, slots, boundary
        elapsed = (time.perf_counter() - started) * 1000
        self.resize_records.append(
            {
                "old_fraction": old,
                "actual_fraction": self.actual_fraction,
                "elapsed_ms": elapsed,
                "kv_users_evicted": kv_evicted,
                "embedding_ids_evicted": emb_evicted,
                "bytes_migrated": 0,
            }
        )
        return elapsed

    def occupancy(self):
        g = self.geometry
        return {
            "kv_capacity_bytes": self.boundary * g.layers,
            "embedding_capacity_bytes": (g.layer_bytes - self.boundary) * g.layers,
            "kv_used_pages": self.pages - len(self.kv.free),
            "kv_cached_users": len(self.kv.entries),
            "embedding_cached_ids": sum(len(cache.keys) for cache in self.emb),
        }


class KVAdapter:
    """Single-request paged-KV interface consumed by the unchanged HSTU kernels."""

    def __init__(self, arena):
        self.arena = arena
        self.backend = self
        self.gpu_kvcache_mgr = self
        self.host_kvstorage_manager = None
        self.ongoing_offload_tasks = []
        self.last_lookup = None

    @property
    def gpu_kvcache_tables(self):
        return self.arena.kv_tables

    def lookup_kvcache(self, user_ids, sequence_lengths):
        import torch
        from recsys_kvcache_manager.kvcache_utils import KVIndexMeta, KVLookupResult

        if user_ids.numel() != 1:
            raise ValueError("This first executor implements one request at a time")
        uid = int(user_ids.reshape(-1)[0])
        entry = self.arena.kv.lookup(uid)
        length = entry["tokens"] if entry else 0
        lengths = torch.tensor([length], dtype=torch.int64)
        zero = torch.zeros(1, dtype=torch.int64)
        result = KVLookupResult(
            user_ids=user_ids,
            cached_start_indices=zero,
            cached_lengths=lengths,
            gpu_cached_start_indices=zero,
            gpu_cached_lengths=lengths,
            host_cached_start_indices=zero,
            host_cached_lengths=zero,
        )
        self.last_lookup = {
            "user_id": uid,
            "cached_tokens": length,
            "required_tokens": int(sequence_lengths.reshape(-1)[0]),
        }
        self.arena.kv_wait_elapsed_ms = 0.0
        if self.arena.waits_enabled and length < self.last_lookup["required_tokens"]:
            self.arena.kv_wait_elapsed_ms = wait_ms(self.arena.costs.kv_miss_ms)
        return KVIndexMeta(user_ids=user_ids, seq_lengths=sequence_lengths), result

    def allocate_kvcache(
        self, index_meta, lookup_results, output_kvcache_metadata=None
    ):
        import torch
        from recsys_kvcache_manager.kvcache_metadata import get_kvcache_metadata_buffer

        uid = int(index_meta.user_ids.reshape(-1)[0])
        history = int(index_meta.seq_lengths.reshape(-1)[0])
        cached = int(lookup_results.cached_lengths.reshape(-1)[0])
        entry = self.arena.kv.allocate(uid, history)
        pages = entry["pages"]
        new = history - cached
        if new < 0:
            raise RuntimeError("Cached history is longer than requested history")
        last = (
            history % self.arena.geometry.page_tokens or self.arena.geometry.page_tokens
        )
        packed = [0, len(pages), last, history, 0, history, new, 0, new]
        packed += [0] * new + list(range(cached, history))
        metadata = get_kvcache_metadata_buffer(
            1,
            new,
            len(pages),
            page_ids_gpu_buffer=torch.tensor(pages, dtype=torch.int32, device="cuda"),
            metadata_gpu_buffer=torch.tensor(packed, dtype=torch.int32, device="cuda"),
        )
        metadata.kv_cache_table = self.arena.kv_tables
        metadata.max_seqlen = history
        return metadata

    def onboard_launch(self, index_meta, lookup_result, metadata):
        metadata.kv_onload_handle = None
        return None

    def offload_try_wait(self):
        pass

    def offload_launch(self, *args):
        pass

    def evict_all(self, **kwargs):
        self.arena.kv = KVLRU(self.arena.pages, self.arena.geometry.page_tokens)


def make_embedding_module(arena):
    import torch

    class ArenaEmbedding(torch.nn.Module):
        def forward(self, keys):
            return arena.embedding_lookup(keys)

    return ArenaEmbedding()
