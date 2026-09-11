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

import unittest

from recsys_kvcache_manager.layer_ready_plan import (
    LayerReadyPlan,
    Span,
    default_onboard_plan,
    even_spans,
)


class LayerReadyPlanTest(unittest.TestCase):
    def test_even_spans_n1_n2_n4(self) -> None:
        self.assertEqual(even_spans(8, 1), tuple(Span(i, i + 1) for i in range(8)))
        self.assertEqual(
            even_spans(8, 2), (Span(0, 2), Span(2, 4), Span(4, 6), Span(6, 8))
        )
        self.assertEqual(even_spans(8, 4), (Span(0, 4), Span(4, 8)))
        self.assertEqual(even_spans(8, 8), (Span(0, 8),))

    def test_default_onboard_plan_matches_pr428(self) -> None:
        naive = default_onboard_plan(num_layers=8, layerwise=False)
        self.assertEqual(naive.ready, "naive")
        self.assertEqual(naive.transport, "uring_ssd")
        self.assertEqual(naive.spans, (Span(0, 8),))
        self.assertTrue(naive.is_whole_cache_wait())

        layerwise = default_onboard_plan(num_layers=8, layerwise=True)
        self.assertEqual(layerwise.ready, "layerwise")
        self.assertEqual(layerwise.spans, (Span(0, 8),))
        self.assertFalse(layerwise.is_whole_cache_wait())

        even2 = default_onboard_plan(num_layers=8, layerwise=True, layer_batch=2)
        self.assertEqual(even2.spans, even_spans(8, 2))
        self.assertEqual(even2.span_covering(3), Span(2, 4))

    def test_gds_prefetch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "GDS prefetch"):
            LayerReadyPlan(
                transport="gds",
                ready="layerwise",
                spans=(Span(0, 8),),
                prefetch="whole",
            )


try:
    import torch
    from recsys_kvcache_manager.host_kvstorage_manager import (
        HostKVTaskHandle,
        HostKVTaskStatus,
    )
    from recsys_kvcache_manager.layer_ready_scheduler import wait_layer_gpu_ready

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _FakeLayerHandle:
    def __init__(self) -> None:
        self.called = []

    def wait_layer(self, layer_idx: int) -> None:
        self.called.append(layer_idx)


@unittest.skipUnless(_HAS_TORCH, "torch is required for handle wait tests")
class LayerReadyHandleTest(unittest.TestCase):
    def test_needs_whole_onboard_wait(self) -> None:
        skipped = HostKVTaskHandle(
            backend="flexkv",
            handle=None,
            status=HostKVTaskStatus.SKIPPED,
        )
        self.assertFalse(skipped.needs_whole_onboard_wait())

        naive = HostKVTaskHandle(
            backend="flexkv",
            user_ids=torch.tensor([1]),
            handle=_FakeLayerHandle(),
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=False,
            plan=default_onboard_plan(num_layers=8, layerwise=False),
        )
        self.assertTrue(naive.needs_whole_onboard_wait())

        layerwise = HostKVTaskHandle(
            backend="flexkv",
            user_ids=torch.tensor([1]),
            handle=_FakeLayerHandle(),
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=True,
            plan=default_onboard_plan(num_layers=8, layerwise=True),
        )
        self.assertFalse(layerwise.needs_whole_onboard_wait())

        pr428_fallback = HostKVTaskHandle(
            backend="flexkv",
            user_ids=torch.tensor([1]),
            handle=_FakeLayerHandle(),
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=False,
        )
        self.assertTrue(pr428_fallback.needs_whole_onboard_wait())

    def test_stream_wait_layer_eventfd_fallback(self) -> None:
        backend = _FakeLayerHandle()
        handle = HostKVTaskHandle(
            backend="flexkv",
            user_ids=torch.tensor([1]),
            handle=backend,
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=True,
            plan=default_onboard_plan(num_layers=8, layerwise=True),
        )
        wait_layer_gpu_ready(handle, 3)
        self.assertEqual(backend.called, [3])

    def test_stream_wait_layer_prefers_cuda_event(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        backend = _FakeLayerHandle()
        event = torch.cuda.Event()
        event.record()
        handle = HostKVTaskHandle(
            backend="flexkv",
            user_ids=torch.tensor([1]),
            handle=backend,
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=True,
            gpu_ready_events=[event],
        )
        wait_layer_gpu_ready(handle, 0)
        self.assertEqual(backend.called, [])


if __name__ == "__main__":
    unittest.main()
