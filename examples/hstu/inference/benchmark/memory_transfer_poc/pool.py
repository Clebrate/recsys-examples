# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Three-ratio POC: exchange two KV page ranges with fixed-size NVEMB banks."""

import gc
import time

import torch
from banked_embedding import BankedNVEEmbedding, kv_tail_spans, tensor_span


class CachePool:
    ratios = (0.25, 0.5, 0.75)

    def __init__(self, model, initial_ratio):
        self.kv = model.dense_module.kvcache
        self.device_cache = self.kv.gpu_kvcache_mgr
        if not self.kv.host_kvstorage_manager._no_host_capacity:
            raise ValueError("Shared backing POC requires FlexKV CPU capacity 0")
        if model.cache_budget_report["kv_cache_ratio"] != 0.75:
            raise ValueError("Construct the model at ratio .75 before installing CachePool")
        if self.device_cache.page_stats()["used_pages"]:
            raise ValueError("Install CachePool before warming either cache")
        if initial_ratio not in self.ratios:
            raise ValueError(f"POC ratios are {self.ratios}")
        self.tensor = self.device_cache.gpu_kvcache_tensor
        # Paged attention loads the last page's V padding before multiplying
        # by zero attention weights. Uninitialized NaNs would still propagate.
        with torch.inference_mode():
            self.tensor.zero_()
        torch.cuda.synchronize(self.tensor.device)
        layers, pages = self.tensor.shape[:2]
        if pages < 3:
            raise ValueError("Need at least three KV pages")
        self.bounds = (pages // 3, 2 * pages // 3, pages)
        self.page_bytes = self.tensor.numel() * self.tensor.element_size() // pages
        collection = model.sparse_module._dynamic_embedding_collection
        base_bytes = collection.gpu_cache_bytes_by_table["video_id"]
        # Release the original video cache. The PS belongs to collection and stays.
        del collection.embeddings["video_id"]
        gc.collect()
        base = torch.empty(base_bytes, dtype=torch.uint8, device=self.tensor.device)
        # Each layer owns a contiguous KV slice; no KV stride/pointer changes.
        edges = [(i * base_bytes // layers) // 16 * 16 for i in range(layers)] + [base_bytes]
        spans = [tensor_span(base[edges[i]:edges[i + 1]]) for i in range(layers)]
        self.groups = []
        for first, last in zip(self.bounds[:-1], self.bounds[1:]):
            self.device_cache.retire_empty_pages(first, last - first)
            bank_ids = list(range(len(spans), len(spans) + layers))
            spans.extend(kv_tail_spans(self.tensor, first, last - first))
            self.groups.append((first, last - first, bank_ids))
        config = next(c for c in collection._embedding_configs if c.name == "video_id")
        self.embedding = BankedNVEEmbedding(
            collection._owned_parameter_servers["video_id"], config.embedding_dim,
            spans, device=self.tensor.device, max_batch_keys=16384,
        )
        collection.embeddings["video_id"] = self.embedding
        # ModuleDict order feeds feature_names by position. Restore config order.
        collection.embeddings = torch.nn.ModuleDict({
            c.name: collection.embeddings[c.name] for c in collection._embedding_configs
        })
        self.level = 0
        self.base_embedding_bytes = base_bytes
        self.user_embedding_bytes = sum(v for k, v in collection.gpu_cache_bytes_by_table.items() if k != "video_id")
        self.events = []
        self.set_ratio(initial_ratio, request_id=-1)
        self.events.clear()  # Initialization is setup time, not an online action.

    def capacities(self):
        kv_bytes = self.bounds[self.level] * self.page_bytes
        # The same physical tensor bytes are owned by exactly one side at a time.
        video_bytes = self.base_embedding_bytes + (self.tensor.shape[1] - self.bounds[self.level]) * self.page_bytes
        return {
            "ratio": self.ratios[self.level], "kv_payload_bytes": kv_bytes,
            "video_embedding_backing_bytes": video_bytes,
            "user_embedding_bytes": self.user_embedding_bytes,
            "shared_backing_bytes": self.tensor.numel() * self.tensor.element_size() + self.base_embedding_bytes + self.user_embedding_bytes,
            "kv_pages": self.device_cache.page_stats(),
        }

    def set_ratio(self, ratio, request_id):
        if ratio not in self.ratios:
            raise ValueError(f"POC ratios are {self.ratios}")
        target = self.ratios.index(ratio)
        if target == self.level:
            return None
        started = time.perf_counter()
        before = self.capacities()
        self.kv.offload_flush()
        torch.cuda.synchronize(self.tensor.device)
        drained = time.perf_counter()
        snapshot = self.device_cache.page_snapshot()["user_pages"]
        evicted = set()
        transferred = 0
        initialized = 0
        evict_ms = bank_ms = page_ms = 0.0
        initialize_ms = 0.0
        while self.level != target:
            growing = target > self.level
            group = self.level if growing else self.level - 1
            first, count, banks = self.groups[group]
            if growing:
                tick = time.perf_counter()
                self.embedding.retire(banks)
                bank_ms += (time.perf_counter() - tick) * 1000
                tick = time.perf_counter()
                # Finite FP32 embedding bytes can encode BF16 NaNs. Clear only
                # the returned range, including padding, before KV can reuse it.
                with torch.inference_mode():
                    self.tensor[:, first:first + count].zero_()
                torch.cuda.synchronize(self.tensor.device)
                initialize_ms += (time.perf_counter() - tick) * 1000
                initialized += count * self.page_bytes
                tick = time.perf_counter()
                self.device_cache.restore_pages(first, count)
                page_ms += (time.perf_counter() - tick) * 1000
                self.level += 1
            else:
                tick = time.perf_counter()
                users = self.device_cache.page_snapshot()["user_pages"]
                victims = [uid for uid, pages in users.items()
                           if any(first <= page < first + count for page in pages)]
                self.device_cache.evict(torch.tensor(victims, dtype=torch.int64))
                evicted.update(victims)
                evict_ms += (time.perf_counter() - tick) * 1000
                tick = time.perf_counter()
                self.device_cache.retire_empty_pages(first, count)
                page_ms += (time.perf_counter() - tick) * 1000
                tick = time.perf_counter()
                self.embedding.restore(banks)
                bank_ms += (time.perf_counter() - tick) * 1000
                self.level -= 1
            transferred += count * self.page_bytes
        after = self.device_cache.page_snapshot()["user_pages"]
        preserved = {uid: pages for uid, pages in snapshot.items() if uid not in evicted}
        if any(after.get(uid) != pages for uid, pages in preserved.items()):
            raise RuntimeError("Unrelated KV page mappings changed during transfer")
        record = {
            "request_id": request_id, "from_ratio": before["ratio"], "to_ratio": ratio,
            "transition_wall_ms": (time.perf_counter() - started) * 1000,
            "drain_ms": (drained - started) * 1000, "evict_ms": evict_ms,
            "bank_rebuild_ms": bank_ms, "page_ownership_ms": page_ms,
            "kv_initialize_ms": initialize_ms, "initialized_payload_bytes": initialized,
            "transferred_backing_bytes": transferred, "copied_payload_bytes": 0,
            "evicted_users": len(evicted),
            "evicted_kv_bytes": sum(len(snapshot[uid]) for uid in evicted) * self.page_bytes,
            "preserved_users": len(preserved), "unchanged_page_mappings": True,
            "before": before, "after": self.capacities(),
        }
        self.events.append(record)
        return record

    def close(self):
        self.kv.offload_flush()
        torch.cuda.synchronize(self.tensor.device)
        self.embedding.close()
