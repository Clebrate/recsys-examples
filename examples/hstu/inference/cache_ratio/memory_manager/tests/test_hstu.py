# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU lifetime/coordination tests, not native HSTU or CUDA correctness claims."""

import threading
import unittest
from types import SimpleNamespace

from examples.hstu.inference.cache_ratio.memory_manager.hstu import HstuVmmBackend
from examples.hstu.inference.cache_ratio.memory_manager.tests.test_pool import (
    FakeDriver,
)


class FakeImpl:
    def __init__(self, *args):
        self.pages = args[5]
        self.rows = {}
        self.fail_commit = False

    def _vmm_snapshot(self):
        return sorted(
            (uid, 0, length, list(pages)) for uid, (length, pages) in self.rows.items()
        )

    def _resize_vmm_metadata(self, target, expected_pages, expected):
        if self.fail_commit:
            raise RuntimeError("Injected metadata failure")
        if self.pages != expected_pages or self._vmm_snapshot() != expected:
            raise RuntimeError("Stale snapshot")
        removed = [
            (uid, length)
            for uid, (length, pages) in self.rows.items()
            if any(p >= target for p in pages)
        ]
        for uid, _ in removed:
            del self.rows[uid]
        self.pages = target
        return removed

    def _enable_vmm_events(self):
        pass

    def _drain_vmm_capacity_evictions(self):
        return []

    def evict(self, uid):
        self.rows.pop(uid, None)

    def evict_all(self):
        self.rows.clear()


class FakeLayer:
    def __init__(self, torch, owner, dtype, device):
        self.owner = owner
        self.pointer = owner.__cuda_array_interface__["data"][0]
        self.shape = owner.__cuda_array_interface__["shape"]
        self.dtype = dtype


TORCH = SimpleNamespace(
    float16="fp16", bfloat16="bf16", float32="fp32", int64="i64", int32="i32"
)


class Vector:
    device = SimpleNamespace(type="cpu")

    def __init__(self, values, dtype="i64"):
        self.values, self.dtype = values, dtype

    def dim(self):
        return 1

    def is_contiguous(self):
        return True

    def tolist(self):
        return self.values


def config(**changes):
    values = dict(
        host_kvstorage_backend="nop",
        host_capacity_per_layer=0,
        num_buffer_pages=0,
        num_layers=2,
        num_heads=1,
        head_dim=1,
        page_size=2,
        offload_chunksize=2,
        num_primary_cache_pages=2,
        max_batch_size=1,
        max_seq_len=4,
        dtype="bf16",
        device=0,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class HstuTests(unittest.TestCase):
    def make(self, **changes):
        d = FakeDriver(granularity=16)
        backend = HstuVmmBackend(
            config(**changes),
            max_pages=8,
            driver=d,
            _torch=TORCH,
            _impl_factory=FakeImpl,
            _table_factory=FakeLayer,
        )
        self.addCleanup(lambda: backend.close(at_batch_barrier=True))
        return backend, d

    def test_growth_retains_native_prefix_ids_values_and_layer_addresses(self):
        b, d = self.make()
        ptrs = b._pool.layer_ptrs
        b.impl_.rows = {7: (4, [0, 1])}
        for ptr in ptrs:
            d.bytes_at(ptr)[:] = b"x" * 16
        receipt = b.resize_pages(4, at_batch_barrier=True)
        self.assertEqual(b.impl_.rows, {7: (4, [0, 1])})
        self.assertEqual(b._pool.layer_ptrs, ptrs)
        self.assertEqual(receipt["payload_bytes"], 64)
        self.assertEqual(receipt["resident_payload_bytes"], 32)
        self.assertTrue(receipt["preserves_survivors"])
        for ptr in ptrs:
            self.assertEqual(d.bytes_at(ptr), b"x" * 16)
            self.assertEqual(d.bytes_at(ptr + 16), bytes(16))

    def test_shrink_evicts_tail_users_only_and_releases_physical_backing(self):
        b, d = self.make(num_primary_cache_pages=6)
        b.impl_.rows = {1: (4, [0, 1]), 2: (2, [4]), 3: (2, [3])}
        ptrs = b._pool.layer_ptrs
        for ptr in ptrs:
            d.bytes_at(ptr)[:] = b"s" * 16
        receipt = b.resize_pages(3, at_batch_barrier=True)
        self.assertEqual(receipt["mapped_pages"], 4)
        self.assertEqual(receipt["evicted_users"], [(2, 2)])
        self.assertEqual(b.impl_.rows, {1: (4, [0, 1]), 3: (2, [3])})
        self.assertEqual(receipt["allocated_physical_bytes"], 64)
        self.assertEqual(b.drain_evictions(), [(2, 2, "resize_tail")])
        for ptr in ptrs:
            self.assertEqual(d.bytes_at(ptr), b"s" * 16)
        self.assertEqual(len(d.handles), 4)

    def test_escaped_tensor_storage_owner_blocks_shrink_and_close(self):
        b, d = self.make(num_primary_cache_pages=4)
        alias = b.gpu_kvcache_tables[0]
        before = dict(d.mappings)
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            b.resize_pages(2, at_batch_barrier=True)
        self.assertEqual(d.mappings, before)
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            b.close(at_batch_barrier=True)
        self.assertEqual(d.mappings, before)
        del alias
        b.resize_pages(2, at_batch_barrier=True)

    def test_tensor_owner_keeps_pool_alive(self):
        b, _ = self.make()
        self.assertIs(b.gpu_kvcache_tables[0].owner.pool, b._pool)
        self.assertEqual(b.gpu_kvcache_tables[0].dtype, "bf16")

    def test_resize_during_batch_and_graph_use_rejected(self):
        b, _ = self.make()
        with b.batch():
            b._require_batch()
            with self.assertRaisesRegex(RuntimeError, "during a batch"):
                b.resize_pages(4, at_batch_barrier=True)
        with self.assertRaises(ValueError):
            b.resize_pages(4, at_batch_barrier=True, cuda_graphs_in_use=True)
        with self.assertRaisesRegex(RuntimeError, "inside backend.batch"):
            b._require_batch()

    def test_forward_failure_poison_prevents_validity_of_unwritten_history(self):
        b, _ = self.make()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with b.batch():
                raise RuntimeError("forward failed")
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            b.resize_pages(4, at_batch_barrier=True)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            with b.batch():
                pass

    def test_native_metadata_commit_failure_poison_and_cleanup(self):
        b, d = self.make()
        b.impl_.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "metadata failure"):
            b.resize_pages(4, at_batch_barrier=True)
        self.assertTrue(b.stats()["poisoned"])
        b.close(at_batch_barrier=True)
        self.assertEqual(d.mappings, {})
        self.assertEqual(d.handles, {})

    def test_driver_shrink_failure_poison_and_cleanup(self):
        b, d = self.make(num_primary_cache_pages=4)
        b.impl_.rows = {2: (2, [3])}
        d.fail("unmap")
        with self.assertRaisesRegex(RuntimeError, "unmap failure"):
            b.resize_pages(2, at_batch_barrier=True)
        self.assertTrue(b.stats()["poisoned"])
        b.close(at_batch_barrier=True)
        self.assertEqual(d.mappings, {})

    def test_byte_plan_rounding_includes_all_layer_physical_granularity(self):
        b, _ = self.make()
        self.assertEqual(b.plan_resize_bytes(63)["payload_bytes"], 32)
        self.assertEqual(b.plan_resize_bytes(63, round_up=True)["payload_bytes"], 64)
        self.assertEqual(b.plan_resize_bytes(64)["target_pages"], 4)
        with self.assertRaises(ValueError):
            b.plan_resize_bytes(1)
        with self.assertRaises(ValueError):
            b.plan_resize_bytes(129, round_up=True)

    def test_same_chunk_resize_keeps_views_and_storage(self):
        b, d = self.make()
        alias = b.gpu_kvcache_tables[0]
        previous = list(d.mappings)
        receipt = b.resize_pages(1, at_batch_barrier=True)
        self.assertIs(alias, b.gpu_kvcache_tables[0])
        self.assertEqual(receipt["mapped_pages"], 2)
        self.assertEqual(receipt["action_ms"], 0)
        self.assertEqual(list(d.mappings), previous)

    def test_configuration_rejects_host_storage_before_driver_allocation(self):
        for changes in (
            {"host_kvstorage_backend": "flexkv"},
            {"num_buffer_pages": 1},
            {"host_capacity_per_layer": 1},
        ):
            d = FakeDriver()
            with self.assertRaises(ValueError):
                HstuVmmBackend(
                    config(**changes),
                    max_pages=8,
                    driver=d,
                    _torch=TORCH,
                    _impl_factory=FakeImpl,
                    _table_factory=FakeLayer,
                )
            self.assertEqual(d.events, [])

    def test_missing_native_extension_hooks_fail_before_driver_allocation(self):
        d = FakeDriver()
        with self.assertRaisesRegex(RuntimeError, "Rebuild"):
            HstuVmmBackend(
                config(), max_pages=8, driver=d, _torch=TORCH, _impl_factory=object
            )
        self.assertEqual(d.events, [])

    def test_full_history_capacity_and_uid_batch_constraints(self):
        b, _ = self.make(max_batch_size=2, max_seq_len=2)
        self.assertEqual(b._index(Vector([1, 2]), Vector([2, 2])), ([1, 2], [2, 2]))
        for ids, lengths in (([1, 1], [2, 2]), ([1, 2], [2, 3]), ([1], [-1])):
            with self.assertRaises(ValueError):
                b._index(Vector(ids), Vector(lengths))

    def test_minimum_capacity_guards_next_maximum_batch(self):
        b, _ = self.make(num_primary_cache_pages=6, max_seq_len=8)
        self.assertEqual(b.minimum_pages, 4)
        with self.assertRaisesRegex(ValueError, "maximum batch"):
            b.resize_pages(2, at_batch_barrier=True)
        with self.assertRaisesRegex(ValueError, "maximum batch"):
            b.plan_resize_bytes(32)
        b.resize_pages(4, at_batch_barrier=True)

    def test_init_capacity_failure_releases_reserved_address(self):
        d = FakeDriver(granularity=16)
        with self.assertRaisesRegex(ValueError, "maximum batch"):
            HstuVmmBackend(
                config(max_seq_len=8),
                max_pages=8,
                driver=d,
                _torch=TORCH,
                _impl_factory=FakeImpl,
                _table_factory=FakeLayer,
            )
        self.assertEqual(d.reservations, {})
        self.assertEqual(d.handles, {})

    def test_explicit_invalidation_is_forbidden_inside_forward(self):
        b, _ = self.make()
        with b.batch():
            with self.assertRaisesRegex(RuntimeError, "during a batch"):
                b.evict(Vector([1]), for_gpu=True)
            with self.assertRaisesRegex(RuntimeError, "during a batch"):
                b.evict_all(for_gpu=True)

    def test_constructor_tensor_bridge_failure_retires_traceback_owners(self):
        driver = FakeDriver(granularity=16)
        calls = []

        def failing_bridge(torch, owner, dtype, device):
            layer = FakeLayer(torch, owner, dtype, device)
            calls.append(owner.__cuda_array_interface__["data"][0])
            if len(calls) == 2:
                raise RuntimeError("Tensor bridge failed")
            return layer

        with self.assertRaisesRegex(RuntimeError, "Tensor bridge failed") as caught:
            HstuVmmBackend(
                config(),
                max_pages=8,
                driver=driver,
                _torch=TORCH,
                _impl_factory=FakeImpl,
                _table_factory=failing_bridge,
            )
        self.assertFalse(hasattr(caught.exception, "vmm_cleanup_owner"))
        self.assertEqual(driver.mappings, {})
        self.assertEqual(driver.handles, {})
        self.assertEqual(driver.reservations, {})

    def test_constructor_escaped_alias_never_unmapped_and_returns_cleanup_owner(self):
        driver, escaped = FakeDriver(granularity=16), []

        def escaping_bridge(torch, owner, dtype, device):
            escaped.append(FakeLayer(torch, owner, dtype, device))
            raise RuntimeError("Escaped bridge alias")

        with self.assertRaisesRegex(RuntimeError, "Escaped bridge alias") as caught:
            HstuVmmBackend(
                config(),
                max_pages=8,
                driver=driver,
                _torch=TORCH,
                _impl_factory=FakeImpl,
                _table_factory=escaping_bridge,
            )
        self.assertTrue(driver.mappings)
        cleanup = caught.exception.vmm_cleanup_owner
        self.assertIn("escaped", str(caught.exception.vmm_cleanup_error))
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            cleanup.close(at_batch_barrier=True)
        escaped.clear()
        cleanup.close(at_batch_barrier=True)
        self.assertEqual(driver.reservations, {})

    def test_constructor_cleanup_driver_failure_retains_retry_owner(self):
        driver = FakeDriver(granularity=16)

        def failing_bridge(torch, owner, dtype, device):
            driver.fail("unmap")
            raise RuntimeError("Bridge failed before cleanup")

        with self.assertRaisesRegex(RuntimeError, "Bridge failed") as caught:
            HstuVmmBackend(
                config(),
                max_pages=8,
                driver=driver,
                _torch=TORCH,
                _impl_factory=FakeImpl,
                _table_factory=failing_bridge,
            )
        cleanup = caught.exception.vmm_cleanup_owner
        self.assertIn("unmap", str(caught.exception.vmm_cleanup_error))
        cleanup.close(at_batch_barrier=True)
        self.assertEqual(driver.mappings, {})
        self.assertEqual(driver.handles, {})
        self.assertEqual(driver.reservations, {})

    def test_batch_lock_serializes_resize(self):
        b, _ = self.make()
        started, done = threading.Event(), threading.Event()
        errors = []

        def resize():
            started.set()
            try:
                b.resize_pages(4, at_batch_barrier=True)
            except BaseException as error:
                errors.append(error)
            done.set()

        with b.batch():
            thread = threading.Thread(target=resize)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(done.wait(0.02))
        thread.join(1)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
