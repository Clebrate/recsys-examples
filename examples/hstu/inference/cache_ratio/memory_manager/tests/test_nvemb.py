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


class PreparedBinding(Binding):
    """Native protocol double: preparation freezes access and tokens are one-use."""

    def __init__(self, name, calls, events, payload=1024):
        super().__init__(name, calls, payload)
        self.events = events
        self.pending = None
        self.serial = 0
        self.reject_commit = False
        self.fail_cancel = False
        self.legacy_apply_calls = 0
        self.host_limits = []
        self.prepared_host_bytes = 64
        self.host_peak = 96
        self.commit_host_peak = 96

    def cache_resize_capabilities(self):
        caps = json.loads(super().cache_resize_capabilities())
        caps.update(prepared_resize=True, prepared_resize_version=1)
        return json.dumps(caps)

    def cache_resize_state(self):
        state = json.loads(super().cache_resize_state())
        state["prepared"] = self.pending is not None
        if self.pending is not None:
            state["prepared_token"] = self.pending["token"]
        return json.dumps(state)

    def prepare_cache_resize(self, payload, host_limit):
        self.events.append(("prepare", self.name))
        self.host_limits.append(("prepare", host_limit))
        if host_limit < self.host_peak:
            return dict(status="rejected", reason="host_staging_budget")
        if self.pending is not None:
            return dict(status="rejected", reason="already_prepared")
        plan = json.loads(super().plan_cache_resize(payload, host_limit))
        if plan["status"] == "planned":
            self.serial += 1
            plan.update(status="prepared", token=str(self.serial),
                        prepared_host_bytes=self.prepared_host_bytes,
                        host_staging_peak_bound_bytes=self.host_peak,
                        commit_host_staging_peak_bound_bytes=self.commit_host_peak)
            self.pending = plan
        return plan

    def commit_cache_resize(self, token, pool_limit, host_limit):
        self.events.append(("commit", self.name))
        self.host_limits.append(("commit", host_limit))
        if self.pending is None or self.pending["token"] != token:
            return dict(status="rejected", reason="invalid_token", token_consumed=False)
        plan, self.pending = self.pending, None
        if self.reject_commit or host_limit < self.commit_host_peak:
            return dict(status="rejected", reason="host_staging_budget", token_consumed=True)
        receipt = json.loads(Binding.apply_cache_resize(self, plan["requested_payload_bytes"],
                                                       plan["generation"], pool_limit, host_limit))
        receipt["token_consumed"] = True
        return receipt

    def cancel_cache_resize(self, token):
        self.events.append(("cancel", self.name))
        if self.fail_cancel:
            raise RuntimeError("native cancellation failed")
        if self.pending is None or self.pending["token"] != token:
            return dict(status="rejected", reason="invalid_token")
        self.pending = None
        return dict(status="cancelled")

    def apply_cache_resize(self, *args):
        self.legacy_apply_calls += 1
        raise AssertionError("Prepared binding must not replan during apply")

    def forward(self):
        if self.pending is not None:
            raise RuntimeError("prepared plan blocks forward")


class PreparedCollection(Collection):
    def __init__(self):
        super().__init__()
        self.events = []
        for name, embedding in self.embeddings.items():
            embedding.emb_layer = PreparedBinding(name, self.calls, self.events)


class PreparedNvEmbTests(unittest.TestCase):
    def setUp(self):
        self.collection = PreparedCollection()
        self.manager = NvEmbMemoryManager(self.collection)

    def bindings(self):
        return [embedding.emb_layer for embedding in self.collection.embeddings.values()]

    def test_prepares_all_then_commits_once_without_legacy_reexport(self):
        self.assertTrue(self.manager.supports_prepared_resize)
        plan = self.manager.plan_resize(3072)
        self.assertEqual(plan["resize_mode"], "prepared")
        self.assertEqual(self.collection.events, [("prepare", "user"), ("prepare", "video")])
        for binding in self.bindings():
            with self.assertRaisesRegex(RuntimeError, "blocks forward"):
                binding.forward()
        receipt = self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertEqual(self.collection.events[-2:], [("commit", "user"), ("commit", "video")])
        self.assertEqual(receipt["resize_mode"], "prepared")
        self.assertEqual(receipt["prepare_ms"], plan["prepare_ms"])
        self.assertGreaterEqual(receipt["apply_ms"], 0)
        for binding in self.bindings():
            self.assertEqual(binding.legacy_apply_calls, 0)
            binding.forward()
        calls = list(self.collection.events)
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertEqual(calls, self.collection.events)

    def test_cancel_restores_ledger_and_consumes_plan(self):
        before = self.manager.stats()
        plan = self.manager.plan_resize(3072)
        with self.assertRaisesRegex(RuntimeError, "outstanding"):
            self.manager.plan_resize(2048)
        self.assertEqual(self.manager.cancel_resize(plan)["status"], "cancelled")
        self.assertEqual(self.manager.stats(), before)
        self.assertEqual(self.manager.cancel_resize(plan)["status"], "unchanged")
        with self.assertRaisesRegex(ValueError, "cancelled"):
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        for binding in self.bindings():
            binding.forward()
            token = plan["tables"][binding.name]["token"]
            self.assertEqual(binding.commit_cache_resize(token, 4000, 1024)["status"], "rejected")

    def test_second_table_prepare_rejection_cancels_first_before_returning(self):
        self.bindings()[1].rejection = "mixed_age_merge"
        before = self.manager.stats()
        with self.assertRaisesRegex(ValueError, "mixed_age_merge"):
            self.manager.plan_resize(3072)
        self.assertEqual(self.manager.stats(), before)
        self.assertEqual(self.collection.events, [("prepare", "user"), ("prepare", "video"),
                                                   ("cancel", "user")])
        self.assertEqual(self.collection.calls, [])

    def test_pool_preflight_rejection_releases_all_preparations(self):
        before = self.manager.stats()
        plan = self.manager.plan_resize(3072)
        with self.assertRaisesRegex(ValueError, "physical pool envelope"):
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=3072)
        self.assertEqual(self.manager.stats(), before)
        self.assertEqual(self.collection.calls, [])
        self.assertTrue(all(binding.pending is None for binding in self.bindings()))

    def test_explicit_first_commit_rejection_stays_usable_and_cancels_remaining(self):
        before = self.manager.stats()
        plan = self.manager.plan_resize(3072)
        self.bindings()[0].reject_commit = True
        with self.assertRaises(NvEmbResizeError) as raised:
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertTrue(raised.exception.receipt["unchanged"])
        self.assertEqual(before, self.manager.stats())
        self.assertEqual(self.collection.events[-2:], [("commit", "user"), ("cancel", "video")])

    def test_later_rejection_after_first_commit_poisoned(self):
        plan = self.manager.plan_resize(3072)
        self.bindings()[1].reject_commit = True
        with self.assertRaises(NvEmbResizeError) as raised:
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertFalse(raised.exception.receipt["unchanged"])
        self.assertEqual(set(raised.exception.receipt["completed_tables"]), {"user"})
        self.assertEqual(self.manager.stats()["state"], "poisoned")
        self.assertTrue(all(binding.pending is None for binding in self.bindings()))

    def test_native_throw_still_cancels_other_table_and_poisoned(self):
        plan = self.manager.plan_resize(3072)
        self.bindings()[0].fail = True
        with self.assertRaises(NvEmbResizeError):
            self.manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertIsNone(self.bindings()[1].pending)
        self.assertEqual(self.manager.stats()["state"], "poisoned")

    def test_cancel_failure_attempts_other_table_and_poisoned(self):
        plan = self.manager.plan_resize(3072)
        self.bindings()[0].fail_cancel = True
        with self.assertRaises(NvEmbResizeError):
            self.manager.cancel_resize(plan)
        self.assertIsNone(self.bindings()[1].pending)
        self.assertEqual(self.manager.stats()["state"], "poisoned")

    def test_global_host_limit_counts_all_retained_snapshots(self):
        manager = NvEmbMemoryManager(self.collection, host_staging_limit_bytes=160)
        plan = manager.plan_resize(3072)
        self.assertEqual(plan["prepared_host_bytes"], 128)
        self.assertEqual(plan["host_staging_peak_bound_bytes"], 160)
        manager.resize(plan, at_batch_barrier=True, device_pool_limit_bytes=4000)
        self.assertEqual(self.bindings()[0].host_limits, [("prepare", 160), ("commit", 96)])
        self.assertEqual(self.bindings()[1].host_limits, [("prepare", 96), ("commit", 160)])

    def test_prepare_aggregate_host_overflow_cancels_previous_snapshot(self):
        manager = NvEmbMemoryManager(self.collection, host_staging_limit_bytes=159)
        before = manager.stats()
        with self.assertRaisesRegex(ValueError, "host_staging_budget"):
            manager.plan_resize(3072)
        self.assertEqual(manager.stats(), before)
        self.assertIsNone(self.bindings()[0].pending)

    def test_commit_host_bound_checked_before_returning_prepared_plan(self):
        manager = NvEmbMemoryManager(self.collection, host_staging_limit_bytes=160)
        self.bindings()[0].commit_host_peak = 128
        before = manager.stats()
        with self.assertRaisesRegex(ValueError, "commit exceeds the global host"):
            manager.plan_resize(3072)
        self.assertEqual(manager.stats(), before)
        self.assertEqual(self.collection.calls, [])
        self.assertTrue(all(binding.pending is None for binding in self.bindings()))

    def test_malformed_prepared_accounting_cancels_all_tokens(self):
        binding = self.bindings()[1]
        original = binding.prepare_cache_resize
        def malformed(*args):
            plan = original(*args)
            del plan["payload_bytes"]
            return plan
        binding.prepare_cache_resize = malformed
        before = self.manager.stats()
        with self.assertRaisesRegex(RuntimeError, "Invalid native resize accounting"):
            self.manager.plan_resize(3072)
        self.assertEqual(self.manager.stats(), before)
        self.assertTrue(all(binding.pending is None for binding in self.bindings()))

    def test_legacy_and_mixed_capabilities_are_not_reported_as_optimized(self):
        legacy = NvEmbMemoryManager(Collection())
        self.assertFalse(legacy.supports_prepared_resize)
        self.assertEqual(legacy.plan_resize(2048)["resize_mode"], "legacy_replanned")
        self.collection.embeddings["video"].emb_layer = Binding("video", self.collection.calls)
        mixed = NvEmbMemoryManager(self.collection)
        self.assertFalse(mixed.supports_prepared_resize)
        plan = mixed.plan_resize(2048)
        self.assertEqual(plan["resize_mode"], "mixed")
        mixed.cancel_resize(plan)



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

    def enable_vmm(self, row_bytes=(16, 16)):
        for binding, width in zip(self.manager._bindings.values(), row_bytes):
            original = binding.cache_resize_state

            def state(original=original, width=width):
                result = json.loads(original())
                result.update(vmm=True, chunk_bytes=256, reserved_value_bytes=4096,
                              row_bytes=width)
                return json.dumps(result)

            binding.cache_resize_state = state

    def test_vmm_whole_chunks_largest_remainder_and_tie_order(self):
        self.enable_vmm()
        plan = self.manager.plan_resize(9 * 256 + 17)
        self.assertEqual(plan["tables"]["user"]["payload_bytes"], 5 * 256)
        self.assertEqual(plan["tables"]["video"]["payload_bytes"], 4 * 256)
        self.assertEqual(plan["payload_bytes"], 9 * 256)
        self.assertEqual(plan["requested_payload_bytes"] - plan["payload_bytes"], 17)

    def test_vmm_weights_are_vocabulary_bytes_including_dtype_width(self):
        self.enable_vmm((16, 32))
        plan = self.manager.plan_resize(9 * 256)
        self.assertEqual(plan["tables"]["user"]["payload_bytes"], 3 * 256)
        self.assertEqual(plan["tables"]["video"]["payload_bytes"], 6 * 256)

    def test_vmm_bounds_and_native_observation_missing_remain_explicit(self):
        self.enable_vmm()
        self.assertEqual(self.manager.payload_bounds(), dict(minimum_payload_bytes=512,
                                                             maximum_payload_bytes=8192))
        for result in self.manager.policy_observation()["tables"].values():
            self.assertFalse(result["complete"])
            self.assertEqual(result["reason"], "missing_native_observation")

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
