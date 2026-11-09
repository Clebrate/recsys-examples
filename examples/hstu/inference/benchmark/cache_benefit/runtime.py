# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse the reference request builder and HSTU forward; time the full request."""
import hashlib
import os
import time
from pathlib import Path

from arena import Arena, KVAdapter, Parameters, make_embedding_module
from metadata import Geometry


class Runtime:
    def __init__(
        self,
        pool,
        budget_bytes,
        fraction,
        costs,
        gin_config,
        seed=1,
        kv_history_items=128,
        parameter_mode="default",
    ):
        os.environ["HSTU_INFERENCE_ONLY"] = "1"
        os.environ["CACHE_RATIO_EMB_WAIT_US"] = "0"
        os.environ["CACHE_RATIO_KV_WAIT_MS"] = "0"
        os.environ["CACHE_RATIO_KV_CACHE_HISTORY"] = str(kv_history_items)
        import cache_ratio_sweep as reference
        import gin
        import torch

        self.reference, self.torch, self.pool = reference, torch, pool
        self.kv_history_items = kv_history_items
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        gin.parse_config_file(str(gin_config))
        network = reference.NetworkArgs()
        geometry = Geometry(
            budget_bytes,
            network.num_layers,
            network.num_attention_heads,
            network.kv_channels,
            reference._page_size(),
            network.hidden_size,
        )
        configs = reference._configs(
            pool.metadata["video_vocab"],
            pool.metadata["action_vocab"],
            network.hidden_size,
        )
        total_sequence = 2 * (
            pool.metadata["history_cap"] + pool.metadata["candidates"]
        )
        self.model, _ = reference.build_model(
            configs,
            8 * 2**20,
            0,
            64 * 2**20,
            0,
            0,
            total_sequence,
            use_flexkv=False,
        )
        self.arena = Arena(
            geometry, fraction, Parameters(network.hidden_size, parameter_mode), costs
        )
        self.model.dense_module.kvcache = KVAdapter(self.arena)
        dynamic = reference._emb_wait_collection(self.model)
        # Both the clean repository collection and the earlier wait-sidecar
        # variant work. All extra waits belong to this benchmark's miss path.
        dynamic._emb_wait_us = 0
        dynamic._emb_wait_sidecars = {}
        dynamic._emb_wait_pending = []
        for name in dynamic.embeddings:
            dynamic.embeddings[name] = make_embedding_module(self.arena)
        static = self.model.sparse_module._static_embedding_collection
        for config in static._embedding_configs:
            layer = torch.nn.Embedding(
                config.num_embeddings,
                config.embedding_dim,
                dtype=torch.float32,
                device="cuda",
            )
            torch.nn.init.normal_(layer.weight, 0.0, 0.02)
            static.embeddings[config.name] = layer
        self.model.eval()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        self.provenance = {
            "backend": "experimental_fixed_page_arena",
            "model_state_sha256": digest.hexdigest(),
            "model_seed": seed,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "parameter_mode": parameter_mode,
            "checkpoint_loaded": False,
            "action_initializer": "seeded normal(std=0.02)",
            "gin_sha256": hashlib.sha256(Path(gin_config).read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(
                Path(reference.__file__).read_bytes()
            ).hexdigest(),
            "kv_history_items": kv_history_items,
            "cache_payload_bytes": geometry.actual_bytes,
            "cpu_kv_cache": False,
            "ssd": False,
            "resize_policy": "normal LRU; physical victims during ownership changes",
            "metric": "CPU request materialization/H2D through synchronized prediction; executed miss waits and due resize included; CSV loading/logging, network and service queues excluded",
        }

    def forward(self, index, resize=None):
        torch, reference, arena = self.torch, self.reference, self.arena
        started = time.perf_counter()
        resize_ms = arena.resize(resize) if resize is not None else 0.0
        arena.begin()
        item = self.pool.records[index]
        ids = item.videos.tolist()
        batch = reference._one_revisit_batch(
            item.videos.astype("int64"),
            item.actions.astype("int64"),
            item.history,
            self.pool.metadata["candidates"],
        )
        arena.current_ids = ids
        batch = batch.to(device=torch.cuda.current_device())
        history_tokens = 2 * min(item.history, self.kv_history_items)
        with torch.inference_mode():
            logits = reference._forward_with_kv_window(
                self.model,
                batch,
                torch.tensor([item.uid], dtype=torch.int64),
                torch.tensor([history_tokens], dtype=torch.int64),
                self.kv_history_items,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        lookup = self.model.dense_module.kvcache.last_lookup
        row = {
            "pool_index": index,
            "e2e_ms": elapsed,
            "kv_full_hit": int(lookup["cached_tokens"] >= history_tokens),
            "kv_cached_tokens": lookup["cached_tokens"],
            "history_tokens": history_tokens,
            **arena.last_embedding,
            "kv_wait_executed_ms": arena.kv_wait_elapsed_ms,
            "actual_kv_fraction": arena.actual_fraction,
            "resize_ms": resize_ms,
            **arena.occupancy(),
        }
        return logits, row

    def prepare_conditions(self, index, condition):
        """Unmeasured fixture setup; each timed counterfactual starts warm."""
        was_enabled = self.arena.waits_enabled
        self.arena.waits_enabled = False
        self.forward(index)
        self.arena.waits_enabled = was_enabled
        record = self.pool.records[index]
        if condition[0] == "M":
            self.arena.kv.drop(record.uid)
        if condition[1] == "M":
            self.arena.drop_embedding(record.videos.tolist())

    def all_hit_capacity(self):
        import numpy as np

        g = self.arena.geometry
        pages = sum(
            (2 * min(r.history, self.kv_history_items) + g.page_tokens - 1)
            // g.page_tokens
            for r in self.pool.records
        )
        shards = np.bincount(self.pool.unique_keys() % g.layers, minlength=g.layers)
        if pages > self.arena.pages or int(shards.max()) > self.arena.slots:
            raise ValueError("GPU allocation cannot hold the complete working set")
        return {
            "required_kv_pages": pages,
            "available_kv_pages": self.arena.pages,
            "required_embedding_rows_per_shard": shards.tolist(),
            "available_embedding_rows_per_shard": self.arena.slots,
            "kv_working_set_bytes": pages * g.page_bytes * g.layers,
            "embedding_working_set_bytes": int(shards.sum()) * g.row_bytes,
        }
