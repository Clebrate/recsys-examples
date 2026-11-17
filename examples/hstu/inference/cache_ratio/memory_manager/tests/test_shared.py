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


class WholeChunkKv(Kv):
    minimum_pages = 1024
    quantum = 3 * 64 * 1024**2

    def __init__(self, events):
        super().__init__(events)
        self.size = 42 * self.quantum

    def stats(self):
        return dict(super().stats(), backend="gpu_only_vmm")

    def plan_pages(self, pages):
        return dict(chunk_pages=1024, page_payload_bytes=3 * 64 * 1024)

    def plan_resize_bytes(self, size, round_up=False):
        target = ((size + self.quantum - 1) // self.quantum if round_up
                  else size // self.quantum) * self.quantum
        if target < self.quantum or target > 16 * 1024**3:
            raise ValueError("KV budget outside reserved positive VMM range")
        return dict(payload_bytes=target, allocated_physical_bytes=target,
                    target_pages=target, extra_peak_bytes=0)


class WholeChunkEmb(Emb):
    chunk = 64 * 1024**2

    def __init__(self, events):
        super().__init__(events)
        self.size = 16 * 1024**3 - 42 * WholeChunkKv.quantum
        self.pending = None

    def stats(self):
        return dict(super().stats(), tables={name: dict(vmm=True, chunk_bytes=self.chunk)
                                            for name in ("user", "video")})

    def plan_resize(self, size):
        if self.pending is not None:
            raise RuntimeError("Outstanding preparation")
        if size < 2 * self.chunk:
            raise ValueError("Embedding payload must hold one VMM chunk per table")
        self.events.append("prepare_embedding")
        target = size // self.chunk * self.chunk
        self.pending = dict(payload_bytes=target, device_pool_bytes=target + 16,
                            extra_peak_bytes=self.peak, resize_mode="prepared")
        return self.pending

    def resize(self, plan, **kwargs):
        if plan is not self.pending:
            raise RuntimeError("Invalid prepared plan")
        result = super().resize(plan, **kwargs)
        self.pending = None
        return result

    def cancel_resize(self, plan):
        if plan is self.pending:
            self.events.append("cancel_embedding")
            self.pending = None


class ByteTransferTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.kv, self.emb = WholeChunkKv(self.events), WholeChunkEmb(self.events)
        self.budget = 16 * 1024**3
        self.memory = SharedCacheMemory(self.kv, self.emb, self.budget, self.budget + 1024)

    def test_sub_quantum_request_is_explicit_noop_before_preparation(self):
        self.assertEqual(self.memory.transfer_quantum_bytes(), 192 * 1024**2)
        before = self.memory.stats()
        for delta in (128 * 1024**2, -128 * 1024**2):
            result = self.memory.resize_by_bytes(delta, at_batch_barrier=True)
            self.assertEqual(result["status"], "rounded_noop")
            self.assertEqual(result["reason"], "below_transfer_quantum")
            self.assertEqual(result["requested_delta_bytes"], delta)
            self.assertEqual(result["actual_delta_bytes"], 0)
            self.assertEqual(result["unapplied_delta_bytes"], delta)
        self.assertEqual(self.events, [])
        self.assertEqual(self.memory.stats(), before)

    def test_nonmultiple_request_rounds_toward_current_without_hidden_accumulation(self):
        for delta, actual in ((256, 192), (-256, -192)):
            result = self.memory.resize_by_bytes(delta * 1024**2, at_batch_barrier=True)
            self.assertEqual(result["actual_delta_bytes"], actual * 1024**2)
            self.assertEqual(result["unapplied_delta_bytes"], (delta - actual) * 1024**2)
            self.assertEqual(result["after"]["unused_payload_bytes"], 0)
        before = self.memory.stats()
        self.events.clear()
        for _ in range(2):
            self.memory.resize_by_bytes(128 * 1024**2, at_batch_barrier=True)
        self.assertEqual(self.memory.stats(), before)
        self.assertEqual(self.events, [])

    def test_bidirectional_exact_transfer_and_inclusive_timings(self):
        before = self.memory.stats()
        quantum = self.memory.transfer_quantum_bytes()
        grow = self.memory.resize_by_bytes(quantum, at_batch_barrier=True)
        self.assertEqual(self.events, ["prepare_embedding", "embedding", "kv"])
        self.assertEqual(grow["after"]["kv"]["payload_bytes"], before["kv"]["payload_bytes"] + quantum)
        self.assertEqual(grow["after"]["embedding"]["payload_bytes"], before["embedding"]["payload_bytes"] - quantum)
        self.assertEqual(grow["after"]["unused_payload_bytes"], 0)
        self.assertEqual(grow["kv_delta_bytes"], quantum)
        self.assertEqual(grow["embedding_resize_mode"], "prepared")
        self.assertAlmostEqual(grow["prepare_ms"] + grow["apply_ms"], grow["action_ms"])
        self.events.clear()
        shrink = self.memory.resize_by_bytes(-quantum, at_batch_barrier=True)
        self.assertEqual(self.events, ["prepare_embedding", "kv", "embedding"])
        self.assertEqual(shrink["after"]["kv"]["payload_bytes"], before["kv"]["payload_bytes"])
        self.assertEqual(shrink["after"]["embedding"]["payload_bytes"], before["embedding"]["payload_bytes"])
        self.assertEqual(shrink["after"]["unused_payload_bytes"], 0)

    def test_budget_invalid_types_and_barrier_rejected(self):
        for delta in (True, 128.0, self.budget * 3, -self.budget * 3):
            with self.assertRaises(ValueError):
                self.memory.resize_by_bytes(delta, at_batch_barrier=True)
        with self.assertRaises(ValueError):
            self.memory.resize_by_bytes(0, at_batch_barrier=False)
        self.assertEqual(self.events, [])

    def test_equal_byte_transfers_preserve_existing_budget_slack(self):
        self.emb.size -= self.emb.chunk
        before = self.memory.stats()
        self.assertEqual(before["unused_payload_bytes"], self.emb.chunk)
        for delta in (self.kv.quantum, -self.kv.quantum):
            receipt = self.memory.resize_by_bytes(delta, at_batch_barrier=True)
            self.assertEqual(receipt["actual_delta_bytes"], delta)
            self.assertEqual(receipt["after"]["unused_payload_bytes"], self.emb.chunk)
        self.assertEqual(self.memory.stats()["kv"], before["kv"])
        self.assertEqual(self.memory.stats()["embedding"], before["embedding"])

    def test_embedding_donor_exhaustion_rejected_before_prepare_despite_slack(self):
        self.emb.size = self.kv.quantum
        before = self.memory.stats()
        self.assertGreater(before["unused_payload_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "embedding donor"):
            self.memory.resize_by_bytes(self.kv.quantum, at_batch_barrier=True)
        self.assertEqual(self.events, [])
        self.assertEqual(self.memory.stats(), before)

    def test_shared_peak_failure_cancels_before_any_donor_mutation(self):
        before = self.memory.stats()
        self.emb.peak = self.budget
        with self.assertRaisesRegex(ValueError, "peak envelope"):
            self.memory.resize_by_bytes(-self.kv.quantum, at_batch_barrier=True)
        self.assertEqual(self.events, ["prepare_embedding", "cancel_embedding"])
        self.assertIsNone(self.emb.pending)
        self.assertEqual(self.memory.stats(), before)

    def test_ratio_noop_releases_preparation_before_next_batch(self):
        result = self.memory.resize_ratio(self.kv.size / self.budget, at_batch_barrier=True)
        self.assertEqual(result["status"], "rounded_noop")
        self.assertEqual(self.events, ["prepare_embedding", "cancel_embedding"])
        self.assertEqual(self.memory.actions, 0)
        self.assertIsNone(self.emb.pending)
        with self.memory.batch():
            pass

    def test_kv_donor_failure_cancels_unconsumed_embedding_plan_and_poisoned(self):
        self.kv.fail = True
        with self.assertRaises(RuntimeError):
            self.memory.resize_by_bytes(-self.kv.quantum, at_batch_barrier=True)
        self.assertEqual(self.events, ["prepare_embedding", "kv", "cancel_embedding"])
        self.assertIsNone(self.emb.pending)
        self.assertEqual(self.memory.stats()["state"], "poisoned")


class SharedPreparedIntegrationTests(unittest.TestCase):
    def setUp(self):
        from examples.hstu.inference.cache_ratio.memory_manager.tests.test_nvemb import PreparedCollection
        from examples.hstu.inference.cache_ratio.memory_manager.nvemb import NvEmbMemoryManager

        self.collection = PreparedCollection()
        self.kv = Kv(self.collection.events)
        self.kv.size = 1024
        self.emb = NvEmbMemoryManager(self.collection)
        self.memory = SharedCacheMemory(self.kv, self.emb, 3072, 4000)

    def test_every_native_table_prepares_before_donor_mutation(self):
        self.memory.resize_ratio(0.5, at_batch_barrier=True)
        self.assertEqual(self.collection.events, [("prepare", "user"), ("prepare", "video"),
                                                 ("commit", "user"), ("commit", "video"), "kv"])

    def test_native_preflight_rejection_keeps_shared_runtime_usable(self):
        before = self.memory.stats()
        self.collection.embeddings["user"].emb_layer.reject_commit = True
        with self.assertRaises(RuntimeError):
            self.memory.resize_ratio(0.5, at_batch_barrier=True)
        self.assertEqual(before, self.memory.stats())
        self.assertEqual(self.memory.last_action["status"], "rejected")
        with self.memory.batch():
            pass

    def test_embedding_preflight_rejection_after_kv_donor_is_partial_failure(self):
        self.collection.embeddings["user"].emb_layer.reject_commit = True
        with self.assertRaises(RuntimeError):
            self.memory.resize_ratio(0.2, at_batch_barrier=True)
        self.assertEqual(self.collection.events[:3], [("prepare", "user"), ("prepare", "video"), "kv"])
        self.assertEqual(self.memory.stats()["state"], "poisoned")
        self.assertTrue(all(e.emb_layer.pending is None for e in self.collection.embeddings.values()))



if __name__ == "__main__":
    unittest.main()
