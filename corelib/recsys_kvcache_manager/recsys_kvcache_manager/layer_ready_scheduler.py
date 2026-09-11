# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unified Recsys onboard wait API.

P0 still uses one FlexKV ``client.launch``. GPU-ready is:

1. ``gpu_ready_events[i]`` → ``compute_stream.wait_event`` (target FlexKV API)
2. else backend ``handle.wait_layer`` (eventfd or Native ``cudaStreamWaitEvent``)

SSD span futures and Recsys-driven ``ssd_read`` / ``h2d_layer`` land in P1.
"""

from typing import Any, Optional, Protocol, runtime_checkable

import torch

from .host_kvstorage_manager import HostKVTaskHandle
from .layer_ready_plan import Span


@runtime_checkable
class CpuSpanFuture(Protocol):
    def done(self) -> bool:
        ...

    def wait(self) -> None:
        ...


@runtime_checkable
class FlexKVTransport(Protocol):
    """Narrow FlexKV primitives. Not wired in P0 (still one launch + eventfd)."""

    def ssd_read(self, task: Any, span: Span) -> CpuSpanFuture:
        ...

    def h2d_layer(
        self, task: Any, layer: int, stream: torch.cuda.Stream
    ) -> torch.cuda.Event:
        ...

    def gds_read(
        self, task: Any, span: Span, stream: torch.cuda.Stream
    ) -> torch.cuda.Event:
        ...

    def cpu_layer_ready(self, task: Any, layer: int) -> bool:
        ...


def wait_layer_gpu_ready(
    task_handle: Optional[HostKVTaskHandle],
    layer_idx: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Block the compute stream until original layer ``layer_idx`` is on GPU."""
    if task_handle is None:
        return
    task_handle.stream_wait_layer(layer_idx, stream=stream)


def wait_naive_gpu_ready(
    task_handle: Optional[HostKVTaskHandle],
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Wait the last GPU-ready event. Used when ``plan.ready == naive``.

    P0 has no per-layer CUDA events on FlexKV naive GET, so this is a no-op
    here; ``inference_dense_module`` still calls ``onboard_wait`` for the
    whole task. Once FlexKV returns ``gpu_ready_events``, this waits the
    last event instead of a CPU ``client.wait``.
    """
    if task_handle is None or task_handle.gpu_ready_events is None:
        return
    last_idx = len(task_handle.gpu_ready_events) - 1
    task_handle.stream_wait_layer(last_idx, stream=stream)
