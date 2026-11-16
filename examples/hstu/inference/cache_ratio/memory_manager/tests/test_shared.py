# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import unittest
from contextlib import contextmanager

from examples.hstu.inference.cache_ratio.memory_manager.shared import SharedCacheMemory


class Kv:
    def __init__(self, events):
        self.size = 500
        self.events = events
        self.fail = False
        self.closed = False

    def stats(self):
        return dict(
            payload_bytes=self.size,
            allocated_physical_bytes=self.size,
            resident_payload_bytes=self.size,
            state="open",
        )

    def plan_resize_bytes(self, size, round_up=False):
        target = ((size + 9) // 10 if round_up else size // 10) * 10
        if target < 100:
            raise ValueError("Maximum batch cannot fit")
        return dict(
            payload_bytes=target,
            allocated_physical_bytes=target,
            target_pages=target,
            extra_peak_bytes=0,
        )

    def resize_pages(self, pages, at_batch_barrier):
        self.events.append("kv")
        if self.fail:
            raise RuntimeError("Failed native growth")
        self.size = pages
        return dict(self.stats(), preserves_survivors=True)

    @contextmanager
    def batch(self):
        yield

    @contextmanager
    def resize_guard(self):
        yield

    def close(self, at_batch_barrier):
        self.closed = True


class Emb:
    def __init__(self, events):
        self.size = 496
        self.events = events
        self.fail_plan = False
        self.fail = False
        self.peak = 0

    def stats(self):
        return dict(
            payload_bytes=self.size, device_pool_bytes=self.size + 16, state="ready"
        )

    def plan_resize(self, size):
        if self.fail_plan:
            raise ValueError("Native rehash cannot retain survivors")
        target = size // 8 * 8
        return dict(
            payload_bytes=target,
            device_pool_bytes=target + 16,
            extra_peak_bytes=self.peak,
        )

    def resize(self, plan, at_batch_barrier, device_pool_limit_bytes):
        self.events.append("embedding")
        if self.fail:
            raise RuntimeError("Failed native migration")
        assert plan["device_pool_bytes"] <= device_pool_limit_bytes
        self.size = plan["payload_bytes"]
        return dict(self.stats(), preserves_survivors=True)


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.kv, self.emb = Kv(self.events), Emb(self.events)
        self.memory = SharedCacheMemory(self.kv, self.emb, 1000, 1100)

    def test_grow_kv_releases_embedding_first_and_reports_slack(self):
        result = self.memory.resize_ratio(0.61, at_batch_barrier=True)
        self.assertEqual(self.events, ["embedding", "kv"])
        self.assertEqual(result["after"]["kv"]["payload_bytes"], 610)
        self.assertEqual(result["after"]["embedding"]["payload_bytes"], 384)
        self.assertEqual(result["after"]["unused_payload_bytes"], 6)
        self.assertEqual(result["after"]["device_pool_bytes"], 1010)

    def test_grow_embedding_releases_kv_first(self):
        result = self.memory.resize_ratio(0.401, at_batch_barrier=True)
        self.assertEqual(self.events, ["kv", "embedding"])
        self.assertEqual(self.kv.size, 410)  # Ceil toward current, no overshoot.
        self.assertEqual(self.emb.size, 584)
        self.assertEqual(result["status"], "applied")

    def test_unsupported_nvemb_fails_before_kv_mutation(self):
        self.emb.fail_plan = True
        with self.assertRaises(ValueError):
            self.memory.resize_ratio(0.4, at_batch_barrier=True)
        self.assertEqual(self.events, [])
        with self.memory.batch():
            pass

    def test_failure_after_donor_mutation_poisoned(self):
        self.kv.fail = True
        with self.assertRaises(RuntimeError):
            self.memory.resize_ratio(0.6, at_batch_barrier=True)
        self.assertEqual(self.events, ["embedding", "kv"])
        self.assertEqual(self.emb.size, 400)
        self.assertEqual(self.memory.stats()["state"], "poisoned")
        with self.assertRaises(RuntimeError):
            with self.memory.batch():
                pass
        self.memory.close()
        self.assertTrue(self.kv.closed)

    def test_over_peak_preflight_has_no_mutations(self):
        self.emb.peak = 1000
        with self.assertRaises(ValueError):
            self.memory.resize_ratio(0.6, at_batch_barrier=True)
        self.assertEqual(self.events, [])

    def test_resize_during_forward_rejected_before_native_calls(self):
        with self.memory.batch():
            with self.assertRaises(RuntimeError):
                self.memory.resize_ratio(0.6, at_batch_barrier=True)
        self.assertEqual(self.events, [])

    def test_forward_exception_poisoned(self):
        with self.assertRaises(RuntimeError):
            with self.memory.batch():
                raise RuntimeError("Append did not finish")
        self.assertEqual(self.memory.stats()["state"], "poisoned")

    def test_noop_and_small_quantized_move_do_not_count_action(self):
        result = self.memory.resize_ratio(0.5, at_batch_barrier=True)
        self.assertEqual(result["status"], "rounded_noop")
        self.assertEqual(self.memory.actions, 0)
        self.assertEqual(self.events, [])

    def test_donor_only_shrink_rejected(self):
        self.emb.size = 500
        self.emb.plan_resize = lambda budget: dict(
            payload_bytes=500, device_pool_bytes=516, extra_peak_bytes=0
        )
        with self.assertRaisesRegex(ValueError, "only shrink"):
            self.memory.resize_ratio(0.48, at_batch_barrier=True)
        self.assertEqual(self.events, [])

    def test_native_misreport_is_fail_closed(self):
        original = self.kv.resize_pages

        def incorrect(pages, **kwargs):
            return original(pages + 10, **kwargs)

        self.kv.resize_pages = incorrect
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.memory.resize_ratio(0.6, at_batch_barrier=True)
        self.assertEqual(self.memory.stats()["state"], "poisoned")

    def test_explicit_barrier_and_supported_bounds(self):
        for ratio in (0, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                self.memory.resize_ratio(ratio, at_batch_barrier=True)
        with self.assertRaises(ValueError):
            self.memory.resize_ratio(0.6, at_batch_barrier=False)
        self.assertEqual(self.events, [])

    def test_native_minimum_batch_capacity_checked_before_mutation(self):
        with self.assertRaises(ValueError):
            self.memory.resize_ratio(0.02, at_batch_barrier=True)
        self.assertEqual(self.events, [])

    def test_failed_native_release_preserves_unknown_accounting(self):
        self.emb.stats = lambda: dict(
            state="poisoned", payload_bytes=None, device_pool_bytes=None
        )
        report = self.memory.stats()
        self.assertEqual(report["state"], "poisoned")
        self.assertIsNone(report["unused_payload_bytes"])
        self.assertIsNone(report["device_pool_bytes"])
        with self.assertRaises(RuntimeError):
            self.memory.resize_ratio(0.6, at_batch_barrier=True)


if __name__ == "__main__":
    unittest.main()
