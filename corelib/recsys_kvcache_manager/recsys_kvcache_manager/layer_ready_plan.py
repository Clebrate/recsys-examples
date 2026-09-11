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

"""Recsys-owned onboard plan. Transport details stay in FlexKV."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Transport = Literal["uring_ssd", "gds"]
ReadyMode = Literal["naive", "layerwise"]
PrefetchMode = Literal["off", "whole", "layer"]


@dataclass(frozen=True)
class Span:
    """Half-open original-layer range [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    def covers(self, layer: int) -> bool:
        return self.start <= layer < self.end

    @property
    def num_layers(self) -> int:
        return self.end - self.start


def even_spans(num_layers: int, layer_batch: int) -> Tuple[Span, ...]:
    """Split [0, num_layers) into equal chunks of ``layer_batch`` (last may be shorter)."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if layer_batch <= 0:
        raise ValueError(f"layer_batch must be positive, got {layer_batch}")
    batch = min(layer_batch, num_layers)
    return tuple(
        Span(start, min(start + batch, num_layers))
        for start in range(0, num_layers, batch)
    )


@dataclass(frozen=True)
class LayerReadyPlan:
    transport: Transport
    ready: ReadyMode
    spans: Tuple[Span, ...]
    prefetch: PrefetchMode = "off"
    prefetch_depth: int = 1

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("LayerReadyPlan.spans must be non-empty")
        if self.prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")
        if self.transport == "gds" and self.prefetch != "off":
            raise ValueError("GDS prefetch is out of scope; keep prefetch='off'")
        expected = 0
        for span in self.spans:
            if span.start != expected:
                raise ValueError(
                    f"spans must cover [0, L) contiguously, got gap at {expected}: {self.spans}"
                )
            expected = span.end

    @property
    def num_layers(self) -> int:
        return self.spans[-1].end

    def span_covering(self, layer: int) -> Optional[Span]:
        for span in self.spans:
            if span.covers(layer):
                return span
        return None

    def is_whole_cache_wait(self) -> bool:
        return self.ready == "naive"


def default_onboard_plan(
    *,
    num_layers: int,
    layerwise: bool,
    transport: Transport = "uring_ssd",
    layer_batch: Optional[int] = None,
    prefetch: PrefetchMode = "off",
) -> LayerReadyPlan:
    """Plan that matches PR 428 launch: one FlexKV GET.

    ``layerwise=False`` → naive whole-cache wait.
    ``layerwise=True`` and ``layer_batch is None`` → one SSD/GDS span [0, L),
    per-layer GPU-ready wait (today: eventfd; later: cudaEvent).
    ``layer_batch`` set → even-N spans for a future Recsys-driven ssd_read loop.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    ready: ReadyMode = "layerwise" if layerwise else "naive"
    if layer_batch is None:
        spans = (Span(0, num_layers),)
    else:
        spans = even_spans(num_layers, layer_batch)
    return LayerReadyPlan(
        transport=transport,
        ready=ready,
        spans=spans,
        prefetch=prefetch,
    )
