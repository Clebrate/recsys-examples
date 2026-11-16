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
import math
import os
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import get_embedding_names_by_table
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    import pynve.torch.nve_ps as nve_ps
    from modules.nve_compat import (
        gpu_only_constructor_kwargs,
        hierarchical_constructor_kwargs,
    )
    from pynve.torch.nve_layers import NVEmbedding

    def get_nve_local_ps(vocab_size, embedding_dim, torch_dtype):
        return nve_ps.NVEParameterServer(vocab_size, embedding_dim, torch_dtype)

    class InferenceNVEEmbeddingCollection(torch.nn.Module):
        def __init__(
            self,
            configs: List[EmbeddingConfig],
            device: Optional[torch.device] = None,
            use_gpu_only: bool = False,
            gpu_cache_ratio: float = 0.01,
            is_weighted: bool = False,
            sparse_shareables=None,
            gpu_cache_bytes: Optional[int] = None,
        ):
            super().__init__()
            self._is_weighted = is_weighted
            self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
            self._embedding_configs = configs
            if device is None:
                self._device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self._device = torch.device("cuda", device)
            else:
                self._device = torch.device(device)
            self._lengths_per_embedding: List[int] = []
            self._owned_parameter_servers = {}

            if not math.isfinite(gpu_cache_ratio) or not 0 < gpu_cache_ratio <= 1:
                raise ValueError("gpu_cache_ratio must be finite and in (0, 1]")
            if gpu_cache_bytes is not None:
                if type(gpu_cache_bytes) is not int or gpu_cache_bytes <= 0:
                    raise ValueError("gpu_cache_bytes must be a positive integer")
                if use_gpu_only:
                    raise ValueError(
                        "Cache byte budgets apply only to hierarchical tables"
                    )
            if len({config.name for config in configs}) != len(configs):
                raise ValueError("Duplicate embedding table names")

            row_bytes = [
                config.embedding_dim
                * torch.empty((), dtype=config.data_type).element_size()
                for config in configs
            ]
            table_bytes = [
                config.num_embeddings * row for config, row in zip(configs, row_bytes)
            ]
            if gpu_cache_bytes is not None:
                if not configs or any(n <= 0 for n in table_bytes + row_bytes):
                    raise ValueError(
                        "Cache budget requires nonempty, positive-size tables"
                    )
                remaining = gpu_cache_bytes - sum(row_bytes)
                if remaining < 0:
                    raise ValueError(
                        "GPU cache must hold at least one row per dynamic table"
                    )
                # Reserve one row per table, then divide the rest by table size.
                weight_sum = sum(table_bytes)
                gpu_sizes = [
                    row + (remaining * weight // (weight_sum * row)) * row
                    for weight, row in zip(table_bytes, row_bytes)
                ]
            else:
                gpu_sizes = [
                    int(config.num_embeddings * gpu_cache_ratio) * row
                    for config, row in zip(configs, row_bytes)
                ]
            if not use_gpu_only and any(size <= 0 for size in gpu_sizes):
                raise ValueError(
                    "GPU cache must hold at least one row per dynamic table"
                )
            self.gpu_cache_bytes_by_table = (
                dict(zip((config.name for config in configs), gpu_sizes))
                if not use_gpu_only
                else {}
            )

            # The opt-in native VMM storage reserves separately from its active
            # value payload. Initial native cache_size includes the original
            # per-set tag/counter/age layout; physical VMM metadata is reported
            # by the binding and is not silently charged as usable values.
            vmm_max_payload = int(os.environ.get("NVE_NVEMB_VMM_MAX_PAYLOAD_BYTES", "0"))
            native_gpu_sizes = list(gpu_sizes)
            if vmm_max_payload and not use_gpu_only:
                if gpu_cache_bytes is None:
                    raise ValueError("NVEmb VMM requires an explicit value payload budget")
                chunk = int(os.environ.get("NVE_NVEMB_VMM_CHUNK_BYTES", str(2 * 1024**2)))
                if chunk <= 0 or vmm_max_payload % chunk or any(chunk % (8 * row) for row in row_bytes):
                    raise ValueError("NVEmb VMM chunks must align to native sets and the reservation")
                available_chunks = gpu_cache_bytes // chunk
                if available_chunks < len(configs):
                    raise ValueError("NVEmb VMM budget must hold at least one chunk per table")
                remaining_chunks = available_chunks - len(configs)
                weight_sum = sum(table_bytes)
                table_chunks = [1 + remaining_chunks * weight // weight_sum for weight in table_bytes]
                remainder_order = sorted(
                    range(len(configs)),
                    key=lambda i: (-(remaining_chunks * table_bytes[i] % weight_sum), i),
                )
                for i in remainder_order[:available_chunks - sum(table_chunks)]:
                    table_chunks[i] += 1
                gpu_sizes = [count * chunk for count in table_chunks]
                if any(size > vmm_max_payload for size in gpu_sizes):
                    raise ValueError("NVEmb VMM initial payload exceeds its reservation")
                native_gpu_sizes = [
                    (payload // (8 * row)) * (8 * row + 104)
                    for payload, row in zip(gpu_sizes, row_bytes)
                ]
                self.gpu_cache_bytes_by_table = dict(zip((c.name for c in configs), gpu_sizes))
            self.gpu_cache_configured_bytes_by_table = (
                dict(zip((c.name for c in configs), native_gpu_sizes)) if not use_gpu_only else {}
            )

            for table_idx, embedding_config in enumerate(configs):
                if not use_gpu_only:
                    gpu_cache_size = native_gpu_sizes[table_idx]
                    storage = (
                        sparse_shareables[embedding_config.name]
                        if sparse_shareables
                        else get_nve_local_ps(
                            0, embedding_config.embedding_dim, torch.float32
                        )
                    )
                    if not sparse_shareables:
                        self._owned_parameter_servers[embedding_config.name] = storage
                    self.embeddings[embedding_config.name] = NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        gpu_cache_size=gpu_cache_size,
                        host_cache_size=0,
                        optimize_for_training=False,
                        device=self._device,
                        **hierarchical_constructor_kwargs(storage),
                    )
                else:
                    self.embeddings[embedding_config.name] = NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        optimize_for_training=False,
                        device=self._device,
                        **gpu_only_constructor_kwargs(),
                    )

                if not embedding_config.feature_names:
                    embedding_config.feature_names = [embedding_config.name]
                self._lengths_per_embedding.extend(
                    len(embedding_config.feature_names)
                    * [embedding_config.embedding_dim]
                )

            self._embedding_names: List[str] = [
                embedding
                for embeddings in get_embedding_names_by_table(configs)
                for embedding in embeddings
            ]
            self._feature_names: List[List[str]] = [
                table.feature_names for table in configs
            ]

        def set_feature_splits(self, features_split_size, features_split_indices):
            pass

        def load_checkpoint(self, checkpoint_dir, model_state_dict=None):
            # Shared stores are loaded by their owner (e.g. the Triton launcher).
            # Standalone inference owns its stores and must load actual values.
            self.load_checkpoint_into_ps(self._owned_parameter_servers, checkpoint_dir)

        @classmethod
        def load_checkpoint_into_ps(
            self, ps_dict, checkpoint_dir=None, rank=0, world_size=1
        ):
            if checkpoint_dir is None:
                return

            for table_name in ps_dict:
                ps_dict[table_name].load_from_file(
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_keys.rank_{rank}.world_size_{world_size}.dyn",
                    ),
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_values.rank_{rank}.world_size_{world_size}.dyn",
                    ),
                )

        def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
            """
            Run the EmbeddingCollection forward pass. This method takes in a `KeyedJaggedTensor`
            and returns a `dict` of `JaggedTensor`, which is the result embeddings for each feature.

            Args:
                features (KeyedJaggedTensor): Input features
            Returns:
                Dict[str, JaggedTensor]
            """
            result_embeddings: Dict[str, torch.Tensor] = dict()
            feature_dict = features.to_dict()
            for i, embedding in enumerate(self.embeddings.values()):
                for feature_name in self._feature_names[i]:
                    f = feature_dict[feature_name]
                    keys = f.values()
                    if keys.numel() == 0:
                        config = self._embedding_configs[i]
                        res = torch.empty(
                            (0, config.embedding_dim),
                            device=self._device,
                            dtype=config.data_type,
                        )
                    else:
                        res = embedding(keys.contiguous())
                    result_embeddings[feature_name] = JaggedTensor(
                        values=res, lengths=f.lengths()
                    )

            return result_embeddings

        def embedding_configs(
            self,
        ):
            return self._embedding_configs

        def is_weighted(self) -> bool:
            return self._is_weighted

except ImportError:
    print("NV-Embeddings is not installed. NVEMB backend is not supported.")
    nve_layers = None
    InferenceNVEEmbeddingCollection = None  # type: ignore
