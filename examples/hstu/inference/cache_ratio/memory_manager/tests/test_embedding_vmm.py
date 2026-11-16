# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU coordination/fault tests; these do not establish native CUDA correctness."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from examples.hstu.inference.cache_ratio.memory_manager.embedding_vmm import VmmEmbeddingCache
from examples.hstu.inference.cache_ratio.memory_manager.tests.test_pool import FakeDriver


class Tensor:
    def __init__(self, shape, dtype="uint8", device="cuda:0", data=None):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype
        kind, _, index = device.partition(":")
        self.device = SimpleNamespace(type=kind, index=int(index or 0))
        self.data = data if data is not None else []

    def dim(self):
        return len(self.shape)

    def numel(self):
        product = 1
        for n in self.shape:
            product *= n
        return product

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return id(self)

    def min(self):
        return SimpleNamespace(item=lambda: min(self.data))


TORCH = SimpleNamespace(
    cuda=SimpleNamespace(device=lambda device: nullcontext()),
    empty=lambda shape, dtype="uint8", device="cuda:0": Tensor(shape, dtype, device),
    uint8="uint8", int64="int64", bool="bool",
)


class Native:
    def __init__(self, driver):
        self.driver = driver
        self.events = driver.events
        self.fail = None
        self.handle = object()
        self.active = 0
        self.occupied = 3

    def _call(self, name, *arguments):
        self.events.append((name, *arguments))
        if self.fail == name:
            self.fail = None
            raise RuntimeError("Injected native " + name)

    def create(self, pointer, row_bytes, max_sets, initial_sets, device):
        self._call("native_create", pointer, max_sets, initial_sets)
        self.active = initial_sets
        return self.handle, 1024

    def resize(self, handle, old_sets, new_sets):
        self._call("native_resize", old_sets, new_sets)
        if old_sets != self.active:
            raise RuntimeError("Unexpected native capacity")
        # Both source and destination must remain mapped throughout migration.
        expected = max(old_sets, new_sets) * 32
        assert sum(size for size, _ in self.driver.mappings.values()) == expected
        self.active = new_sets
        return 2 if new_sets < old_sets else 0

    def lookup(self, handle, keys, count, values, hits):
        self._call("native_lookup", count)

    def insert(self, handle, keys, values, count):
        self._call("native_insert", count)

    def stats(self, handle):
        self._call("native_stats")
        return self.active, self.occupied, 19

    def destroy(self, handle):
        self._call("native_destroy")


class EmbeddingVmmTests(unittest.TestCase):
    def make(self, initial=128):
        driver = FakeDriver(granularity=32)
        native = Native(driver)
        cache = VmmEmbeddingCache(
            row_bytes=4, max_payload_bytes=256, initial_payload_bytes=initial,
            chunk_bytes=64, library_path="unused", _torch=TORCH,
            _driver=driver, _native=native,
        )
        self.addCleanup(cache.close)
        driver.events.clear()
        return cache, driver, native

    def test_grow_maps_before_migration_and_shrink_migrates_before_unmap(self):
        cache, driver, _ = self.make()
        pointer = cache.stats()["base_ptr"]
        grow = cache.resize(256)
        names = [event[0] for event in driver.events]
        self.assertLess(names.index("zero"), names.index("native_resize"))
        self.assertEqual(grow["mapped_delta_bytes"], 128)
        self.assertEqual(grow["evicted_rows"], 0)
        self.assertEqual(grow["device_pool_bytes"], 256 + 1024)
        driver.events.clear()
        shrink = cache.resize(64)
        names = [event[0] for event in driver.events]
        self.assertLess(names.index("native_resize"), names.index("unmap"))
        self.assertEqual(shrink["base_ptr"], pointer)
        self.assertEqual(shrink["capacity_rows"], 16)
        self.assertEqual(shrink["evicted_rows"], 2)
        self.assertEqual(shrink["mapped_delta_bytes"], -192)
        for name in ("pool_ms", "migration_ms", "barrier_ms", "total_ms"):
            self.assertGreaterEqual(shrink[name], 0)
        self.assertGreaterEqual(shrink["total_ms"], shrink["pool_ms"] + shrink["migration_ms"])

    def test_shrink_migration_failure_never_unmaps_and_poison_blocks_access(self):
        cache, driver, native = self.make()
        native.fail = "native_resize"
        with self.assertRaisesRegex(RuntimeError, "native_resize"):
            cache.resize(64)
        self.assertFalse(any(event[0] == "unmap" for event in driver.events))
        self.assertEqual(cache.stats()["payload_bytes"], 128)
        self.assertEqual(cache.stats()["state"], "poisoned")
        self.assertIsNone(cache.stats()["active_sets"])
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            cache.resize(256)

    def test_failed_pool_growth_rolls_back_without_native_mutation(self):
        cache, driver, _ = self.make()
        driver.fail("access")
        with self.assertRaisesRegex(RuntimeError, "access"):
            cache.resize(256)
        self.assertFalse(any(event[0] == "native_resize" for event in driver.events))
        self.assertEqual(cache.stats()["state"], "open")
        self.assertEqual(cache.stats()["payload_bytes"], 128)
        cache.resize(256)

    def test_failed_native_growth_keeps_backing_and_poisons(self):
        cache, _, native = self.make()
        native.fail = "native_resize"
        with self.assertRaises(RuntimeError):
            cache.resize(256)
        self.assertEqual(cache.stats()["payload_bytes"], 256)
        self.assertTrue(cache.stats()["poisoned"])

    def test_failed_unmap_after_native_shrink_poisons_and_close_retries(self):
        cache, driver, native = self.make()
        driver.fail("unmap")
        with self.assertRaisesRegex(RuntimeError, "unmap"):
            cache.resize(64)
        self.assertEqual(native.active, 2)
        self.assertTrue(cache.stats()["poisoned"])
        cache.close()
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)

    def test_noop_and_invalid_resize_do_not_modify_resources(self):
        cache, driver, _ = self.make()
        for arguments in ({"target_payload_bytes": 65},
                          {"target_payload_bytes": 0},
                          {"target_payload_bytes": 512},
                          {"target_payload_bytes": 64, "at_batch_barrier": False},
                          {"target_payload_bytes": 64, "cuda_graphs_in_use": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                cache.resize(**arguments)
        self.assertEqual(driver.events, [])
        receipt = cache.resize(128)
        self.assertEqual(driver.events, [("sync",)])
        self.assertEqual(receipt["migration_ms"], 0)
        self.assertEqual(receipt["pool_ms"], 0)
        self.assertEqual(receipt["generation"], 0)

    def test_occupancy_is_explicit_and_not_in_resize_receipt(self):
        cache, driver, _ = self.make()
        cache.stats()
        cache.resize(256)
        self.assertFalse(any(event[0] == "native_stats" for event in driver.events))
        result = cache.stats(include_occupancy=True)
        self.assertEqual(result["resident_rows"], 3)
        self.assertEqual(result["clock"], 19)

    def test_input_validation_and_context_barrier_precede_native_access(self):
        cache, driver, _ = self.make()
        keys = Tensor((2,), "int64", data=[1, 2])
        values = Tensor((2, 4))
        driver.fail("sync")
        with self.assertRaisesRegex(RuntimeError, "sync"):
            cache.insert(keys, values)
        self.assertFalse(any(event[0] == "native_insert" for event in driver.events))
        with self.assertRaises(ValueError):
            cache.lookup(Tensor((1,), "int64", data=[-1]))
        with self.assertRaises(ValueError):
            cache.insert(keys, Tensor((2, 3)))
        cache.insert(keys, values)
        output, hits = cache.lookup(keys)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(hits.dtype, "bool")

    def test_destroy_failure_consumes_handle_and_reports_uncertain_metadata(self):
        cache, driver, native = self.make()
        native.fail = "native_destroy"
        with self.assertRaisesRegex(RuntimeError, "native_destroy"):
            cache.close()
        self.assertTrue(cache.stats()["metadata_cleanup_uncertain"])
        self.assertEqual(cache.stats()["fixed_metadata_bytes"], 1024)
        cache.close()
        self.assertEqual(sum(e[0] == "native_destroy" for e in driver.events), 1)
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)

    def test_constructor_native_failure_releases_pool(self):
        driver = FakeDriver(granularity=32)
        native = Native(driver)
        native.fail = "native_create"
        with self.assertRaisesRegex(RuntimeError, "native_create"):
            VmmEmbeddingCache(4, 256, 128, 64, "unused", _torch=TORCH,
                              _driver=driver, _native=native)
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)


if __name__ == "__main__":
    unittest.main()
