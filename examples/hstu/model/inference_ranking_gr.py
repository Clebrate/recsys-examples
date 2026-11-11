# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from contextlib import nullcontext
from typing import Dict

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceHSTUConfig, RankingConfig
from modules.inference_dense_module import InferenceDenseModule
from modules.inference_embedding import InferenceEmbedding
from recsys_kvcache_manager.kvcache_config import KVCacheConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

try:
    import hstu_cuda_ops  # noqa: F401 - registers torch.ops.hstu_cuda_ops.*
except ImportError:
    hstu_cuda_ops = None

try:
    _STRIP_CACHED_TOKENS_OP = torch.ops.hstu_cuda_ops.strip_cached_tokens
    _HAS_STRIP_CACHED_TOKENS_OP = hstu_cuda_ops is not None
except (AttributeError, RuntimeError):
    _STRIP_CACHED_TOKENS_OP = None
    _HAS_STRIP_CACHED_TOKENS_OP = False
    raise ImportError(
        "hstu_cuda_ops.strip_cached_tokens is not available. "
        "Please ensure that the hstu_cuda_ops extension is built and installed correctly."
    )

if _HAS_STRIP_CACHED_TOKENS_OP:
    import commons.ops.cuda_ops.fake_hstu_cuda_ops  # noqa: F401


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        sparse_module (InferenceHSTUConfig): The HSTU configuration.
        dense_module (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        sparse_module: torch.nn.Module,
        dense_module: torch.nn.Module,
    ):
        super().__init__()
        self.sparse_module = sparse_module
        self.dense_module = dense_module
        self.collect_cache_stats = False
        self.profile_cache = False
        self.last_cache_stats = {}

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self.dense_module.half()
        return self

    def get_num_class(self):
        return self.dense_module.get_num_class()

    def get_num_tasks(self):
        return self.dense_module.get_num_tasks()

    def get_metric_types(self):
        return self.dense_module.get_metric_types()

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]

        self.sparse_module.load_checkpoint(checkpoint_dir, model_state_dict)
        self.dense_module.load_state_dict(model_state_dict, strict=False)

    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)
        feature_order = list(range(num_context + 2))
        lengths = batch.features.lengths()
        cached_i64 = origin_num_cached.to(device=lengths.device, dtype=torch.int64)
        new_values, new_lengths = _STRIP_CACHED_TOKENS_OP(
            batch.features.values(),
            lengths,
            batch.features.offsets(),
            cached_i64,
            feature_order,
        )

        new_features = KeyedJaggedTensor(
            values=new_values,
            lengths=new_lengths,
            keys=batch.features.keys(),
        )
        batch.features = new_features

        torch.cuda.nvtx.range_pop()
        return batch

    def forward_with_kvcache(
        self,
        batch: HSTUBatch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        profile_range = (
            torch.profiler.record_function if self.profile_cache else nullcontext
        )
        self.last_cache_stats = {}
        self.dense_module.collect_cache_stats = self.collect_cache_stats
        self.dense_module.profile_cache = self.profile_cache
        dynamic_collection = getattr(
            self.sparse_module, "_dynamic_embedding_collection", None
        )
        if hasattr(dynamic_collection, "collect_cache_stats"):
            dynamic_collection.collect_cache_stats = self.collect_cache_stats
        with torch.inference_mode():
            # lookup and allocate for kv cache
            with profile_range("recsys.kvcache.lookup"):
                index_meta, lookup_res = self.dense_module.kvcache.lookup_kvcache(
                    user_ids,
                    total_history_lengths,
                )
            with profile_range("recsys.kvcache.allocate"):
                kvcache_metadata = self.dense_module.kvcache.allocate_kvcache(
                    index_meta, lookup_res
                )

            # Async onboarding overlaps token stripping and embedding lookup.
            with profile_range("recsys.kvcache.onboard_launch"):
                self.dense_module.kvcache.onboard_launch(
                    index_meta, lookup_res, kvcache_metadata
                )

            old_cached_lengths = lookup_res.cached_lengths
            with profile_range("recsys.kvcache.strip"):
                striped_batch = self.strip_cached_tokens(
                    batch,
                    old_cached_lengths,
                )

            torch.cuda.nvtx.range_push("HSTU embedding")
            try:
                with profile_range("recsys.embedding.lookup"):
                    embeddings = self.sparse_module(striped_batch.features)
            finally:
                torch.cuda.nvtx.range_pop()

            kvcache_info = (index_meta, lookup_res, kvcache_metadata)
            logits = self.dense_module.forward_with_kvcache(
                striped_batch,
                embeddings,
                user_ids,
                total_history_lengths,
                kvcache_info,
            )

            if self.collect_cache_stats:
                # Lookup metadata is already on CPU. Never copy GPU counters or
                # synchronize a stream just to collect these optional statistics.
                history = index_meta.seq_lengths
                cached = lookup_res.cached_lengths
                gpu_cached = lookup_res.gpu_cached_lengths
                host_cached = lookup_res.host_cached_lengths
                gpu_start = lookup_res.gpu_cached_start_indices
                host_start = lookup_res.host_cached_start_indices
                if all(
                    tensor.device.type == "cpu"
                    for tensor in (
                        history,
                        cached,
                        gpu_cached,
                        host_cached,
                        gpu_start,
                        host_start,
                    )
                ):
                    reused = torch.minimum(cached.clamp_min(0), history)
                    gpu_reused = (
                        torch.minimum(gpu_start + gpu_cached, history)
                        - gpu_start.clamp_min(0)
                    ).clamp_min(0)
                    host_reused = (
                        torch.minimum(host_start + host_cached, history)
                        - host_start.clamp_min(0)
                    ).clamp_min(0)
                    history_tokens = int(history.sum())
                    cached_tokens = int(reused.sum())
                    self.last_cache_stats.update(
                        history_tokens=history_tokens,
                        kv_cached_tokens=cached_tokens,
                        kv_gpu_cached_tokens=int(gpu_reused.sum()),
                        kv_host_cached_tokens=int(host_reused.sum()),
                        kv_hit_rate=(
                            cached_tokens / history_tokens if history_tokens else None
                        ),
                        kv_full_hit_requests=int(
                            ((reused == history) & (history > 0)).sum()
                        ),
                        kv_partial_hit_requests=int(
                            ((reused > 0) & (reused < history)).sum()
                        ),
                        kv_zero_hit_requests=int(((reused == 0) & (history > 0)).sum()),
                        kv_empty_history_requests=int((history == 0).sum()),
                        kv_cached_lengths=cached.tolist(),
                        gpu_cached_lengths=gpu_cached.tolist(),
                        host_cached_lengths=host_cached.tolist(),
                    )
                else:
                    self.last_cache_stats["kv_stats_unavailable_reason"] = (
                        "Lookup metadata is not on CPU; collection would synchronize."
                    )
                # GPU and host cached tokens overlap; only the merged count is
                # the number of history tokens reused by this forward.
                self.last_cache_stats.update(
                    batch_size=batch.batch_size,
                    embedding_keys=striped_batch.features.values().numel(),
                    embedding_hit_rate=None,
                    embedding_hit_rate_unavailable_reason=(
                        "The embedding backend exposes no Python hit counter."
                    ),
                    **self.dense_module.last_cache_stats,
                )
                self.last_cache_stats.update(
                    getattr(dynamic_collection, "last_cache_stats", {})
                )

        return logits

    def forward_nokvcache(
        self,
        batch: HSTUBatch,
    ):
        profile_range = (
            torch.profiler.record_function if self.profile_cache else nullcontext
        )
        self.last_cache_stats = {}
        self.dense_module.collect_cache_stats = self.collect_cache_stats
        self.dense_module.profile_cache = self.profile_cache
        dynamic_collection = getattr(
            self.sparse_module, "_dynamic_embedding_collection", None
        )
        if hasattr(dynamic_collection, "collect_cache_stats"):
            dynamic_collection.collect_cache_stats = self.collect_cache_stats
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            try:
                with profile_range("recsys.embedding.lookup"):
                    embeddings = self.sparse_module(batch.features)
            finally:
                torch.cuda.nvtx.range_pop()
            logits = self.dense_module.forward_nokvcache(batch, embeddings)
            if self.collect_cache_stats:
                self.last_cache_stats.update(
                    batch_size=batch.batch_size,
                    embedding_keys=batch.features.values().numel(),
                    embedding_hit_rate=None,
                    embedding_hit_rate_unavailable_reason=(
                        "The embedding backend exposes no Python hit counter."
                    ),
                    **self.dense_module.last_cache_stats,
                )
                self.last_cache_stats.update(
                    getattr(dynamic_collection, "last_cache_stats", {})
                )

        return logits

    def forward(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module(batch, embeddings)
        return logits


def get_inference_ranking_gr(
    hstu_config: InferenceHSTUConfig,
    kvcache_config: KVCacheConfig,
    task_config: RankingConfig,
    use_cudagraph=False,
    cudagraph_configs=None,
    sparse_shareables=None,
    embedding_backend=None,
    gpu_cache_bytes=None,
):
    for ebc_config in task_config.embedding_configs:
        assert (
            ebc_config.dim == hstu_config.hidden_size
        ), "hstu layer hidden size should equal to embedding dim"

    inference_sparse = InferenceEmbedding(
        task_config.embedding_configs,
        embedding_backend=(
            hstu_config.embedding_backend
            if embedding_backend is None
            else embedding_backend
        ),
        sparse_shareables=sparse_shareables,
        gpu_cache_bytes=gpu_cache_bytes,
    )
    inference_dense = InferenceDenseModule(
        hstu_config,
        kvcache_config,
        task_config,
        use_cudagraph,
        cudagraph_configs,
    )

    return InferenceRankingGR(inference_sparse, inference_dense)


def apply_inference(
    training_model: torch.nn.Module,
    dynamic_table_configs: Dict[str, int],
    trained_emb_table_sizes: Dict[str, int],
    checkpoint_dir: str,
):
    from dynamicemb.exportable_tables import apply_inference_embedding_collection
    from modules.exportable_embedding import apply_inference_sparse
    from modules.inference_dense_module import apply_inference_hstu_dense

    # Step.1 - [General] Convert ModuleDict[Embedding] to InferenceEmbeddingCollection
    model = apply_inference_embedding_collection(
        training_model,
        dynamic_table_configs,
        trained_emb_table_sizes,
    )

    # Step.2 - [Recsys Example Structure Specific] Apply model specific training to inference conversion
    #          Convert ShardingEmbedding into ExportableEmbedding
    #          Convert dense module into InferenceDenseModule
    #          Convert RankingGR into InferenceRankingGR
    sparse_module = apply_inference_sparse(model._embedding_collection)
    dense_module = apply_inference_hstu_dense(
        hstu_config=model._hstu_config,
        kvcache_config=None,  # kvcache_config is not needed for export
        task_config=model._task_config,
        use_exportable=True,
        hstu_block=model._hstu_block,
        mlp=model._mlp,
    )

    inference_model = InferenceRankingGR(
        sparse_module,
        dense_module,
    )
    if model._hstu_config.bf16:
        inference_model.bfloat16()
    elif model._hstu_config.fp16:
        inference_model.half()

    inference_model.load_checkpoint(checkpoint_dir)
    inference_model = inference_model.eval()

    return inference_model
