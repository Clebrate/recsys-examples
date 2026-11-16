# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from types import SimpleNamespace

from examples.hstu.inference.cache_ratio.memory_manager.nvemb import (
    NvEmbMemoryManager,
    NvEmbResizeError,
)


class Binding:
    def __init__(self, name, calls, payload=1024):
        self.name, self.calls, self.payload = name, calls, payload
        self.generation = 0
        self.fail = False
        self.rejection = None
        self.resident_value = b"native-cache-value"
        self.ps = object()
        self.handler = object()

    def cache_resize_capabilities(self):
        return json.dumps(
            dict(
                version=1,
                preserves_values=True,
                preserves_counter_bits=True,
                preserves_pending_inserts=True,
            )
        )

    def cache_resize_state(self):
        return json.dumps(
            dict(
                state="ready",
                generation=self.generation,
                payload_bytes=self.payload,
                device_pool_bytes=self.payload + self.payload // 8,
                configured_bytes=self.payload + self.payload // 8,
                ways=8,
                row_bytes=16,
            )
        )

    def plan_cache_resize(self, payload, host_limit):
        if self.rejection:
            return json.dumps(dict(status="rejected", reason=self.rejection))
        return json.dumps(
            dict(
                status="planned",
                before=json.loads(self.cache_resize_state()),
                requested_payload_bytes=payload,
                payload_bytes=payload,
                device_pool_bytes=payload + payload // 8,
                generation=self.generation,
                extra_peak_bytes=0,
            )
        )

    def apply_cache_resize(self, payload, generation, pool_limit, host_limit):
        self.calls.append((self.name, payload, pool_limit))
        if self.fail:
            raise RuntimeError("native allocation failed after releasing old pool")
        if generation != self.generation:
            return json.dumps(dict(status="rejected", reason="stale_generation"))
        if payload + payload // 8 > pool_limit:
            return json.dumps(dict(status="rejected", reason="device_pool_budget"))
        self.payload = payload
        self.generation += 1
        return json.dumps(
            dict(
                status="applied",
                after=json.loads(self.cache_resize_state()),
                survivor_rows=3,
                evicted_rows=0,
            )
        )


class Collection:
    def __init__(self, payloads=(1024, 1024)):
        self.calls = []
        self.embeddings = {
            name: SimpleNamespace(
                emb_layer=Binding(name, self.calls, payload), gpu_cache_size=payload
            )
            for name, payload in zip(("user", "video"), payloads)
        }
        self.gpu_cache_bytes_by_table = dict(zip(self.embeddings, payloads))

    def embedding_configs(self):
        return [
            SimpleNamespace(name=n, num_embeddings=100, embedding_dim=4)
            for n in self.embeddings
        ]


class NvEmbTests(unittest.TestCase):
    def setUp(self):
        self.collection = Collection()
        self.manager = NvEmbMemoryManager(self.collection)

    def test_requires_real_native_capability_before_any_plan(self):
        self.collection.embeddings["user"].emb_layer = object()
        with self.assertRaisesRegex(RuntimeError, "native preserving resize"):
            NvEmbMemoryManager(self.collection)
        self.assertEqual(self.collection.calls, [])

    def test_pool_metadata_is_distinct_from_value_payload(self):
        state = self.manager.stats()
        self.assertEqual(state["payload_bytes"], 2048)
        self.assertEqual(state["device_pool_bytes"], 2304)

    def test_plan_does_not_mutate_native_objects_or_capacity(self):
        before = self.manager.stats()
        plan = self.manager.plan_resize(3073)
        self.assertEqual(self.manager.stats(), before)
        self.assertEqual(self.collection.calls, [])
        self.assertLessEqual(plan["payload_bytes"], 3073)
        self.assertGreater(plan["device_pool_bytes"], plan["payload_bytes"])

    def test_small_budget_rejects_before_native_calls(self):
        with self.assertRaisesRegex(ValueError, "one native set"):
            self.manager.plan_resize(255)
        self.assertEqual(self.collection.calls, [])

    def test_native_rehash_rejection_does_not_mutate_any_table(self):
        self.collection.embeddings["video"].emb_layer.rejection = "mixed_age_merge"
        before = self.manager.stats()
        with self.assertRaisesRegex(ValueError, "mixed_age_merge"):
            self.manager.plan_resize(1024)
        self.assertEqual(before, self.manager.stats())
        self.assertEqual(self.collection.calls, [])

    def test_barrier_and_graph_guards(self):
        plan = self.manager.plan_resize(3072)
        for kwargs in (
            {"at_batch_barrier": False},
            {"at_batch_barrier": True, "cuda_graphs_in_use": True},
        ):
            with self.assertRaises(ValueError):
                self.manager.resize(plan, device_pool_limit_bytes=10000, **kwargs)
        self.assertEqual(self.collection.calls, [])

    def test_stale_later_table_prevents_earlier_mutation(self):
        plan = self.manager.plan_resize(3072)
        self.collection.embeddings["video"].emb_layer.generation += 1
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.manager.resize(
                plan, at_batch_barrier=True, device_pool_limit_bytes=10000
            )
        self.assertEqual(self.collection.calls, [])

    def test_physical_pool_envelope_rejects_before_first_mutation(self):
        plan = self.manager.plan_resize(3072)
        with self.assertRaisesRegex(ValueError, "physical pool envelope"):
            self.manager.resize(
                plan, at_batch_barrier=True, device_pool_limit_bytes=3072
            )
        self.assertEqual(self.collection.calls, [])

    def test_shrink_first_can_rebalance_at_constant_physical_limit(self):
        self.collection = Collection((512, 1536))
        self.manager = NvEmbMemoryManager(self.collection)
        plan = self.manager.plan_resize(2048)
        original = {
            name: (
                emb.emb_layer,
                emb.emb_layer.ps,
                emb.emb_layer.handler,
                emb.emb_layer.resident_value,
            )
            for name, emb in self.collection.embeddings.items()
        }
        receipt = self.manager.resize(
            plan, at_batch_barrier=True, device_pool_limit_bytes=2304
        )
        self.assertEqual([c[0] for c in self.collection.calls], ["video", "user"])
        self.assertEqual(receipt["after"]["device_pool_bytes"], 2304)
        for name, emb in self.collection.embeddings.items():
            self.assertEqual(
                original[name],
                (
                    emb.emb_layer,
                    emb.emb_layer.ps,
                    emb.emb_layer.handler,
                    emb.emb_layer.resident_value,
                ),
            )
            self.assertEqual(self.collection.gpu_cache_bytes_by_table[name], 1152)

    def test_partial_failure_is_not_a_successful_rollback(self):
        plan = self.manager.plan_resize(3072)
        self.collection.embeddings["video"].emb_layer.fail = True
        with self.assertRaises(NvEmbResizeError) as raised:
            self.manager.resize(
                plan, at_batch_barrier=True, device_pool_limit_bytes=10000
            )
        self.assertEqual(set(raised.exception.receipt["completed_tables"]), {"user"})
        self.assertEqual(raised.exception.receipt["failed_table"], "video")
        self.assertEqual(self.manager.stats()["state"], "poisoned")
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            self.manager.plan_resize(2048)

    def test_interrupted_apply_poisoned_and_receipt_retained(self):
        plan = self.manager.plan_resize(3072)

        def interrupted(*args):
            raise KeyboardInterrupt()

        self.collection.embeddings["user"].emb_layer.apply_cache_resize = interrupted
        with self.assertRaises(NvEmbResizeError):
            self.manager.resize(
                plan, at_batch_barrier=True, device_pool_limit_bytes=10000
            )
        self.assertEqual(self.manager.last_receipt["status"], "failed")
        self.assertEqual(self.manager.stats()["state"], "poisoned")


if __name__ == "__main__":
    unittest.main()
