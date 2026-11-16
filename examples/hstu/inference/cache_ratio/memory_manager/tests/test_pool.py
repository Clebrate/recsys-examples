# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only driver simulation; does not establish CUDA or HSTU correctness."""

import unittest
from unittest.mock import patch

from examples.hstu.inference.cache_ratio.memory_manager import VmmKvPool
from examples.hstu.inference.cache_ratio.memory_manager.pool import _Chunk


class FakeDriver:
    """Enforce full-map unmapping and handle lifetime, with fault injection."""

    def __init__(self, granularity=8):
        self.unit = granularity
        self.events = []
        self.handles = {}
        self.mappings = {}
        self.reservations = {}
        self.serial = 0
        self.failures = {}

    def fail(self, operation, after=0, times=1, exception=RuntimeError):
        self.failures[operation] = ([after] + [0] * (times - 1), exception)

    def _event(self, operation, *args):
        self.events.append((operation, *args))
        if operation in self.failures:
            countdowns, exception = self.failures[operation]
            if countdowns[0] == 0:
                countdowns.pop(0)
                if not countdowns:
                    del self.failures[operation]
                raise exception(f"Injected {operation} failure")
            countdowns[0] -= 1

    def granularity(self):
        return self.unit

    def reserve(self, size, alignment):
        self._event("reserve", size, alignment)
        assert size % self.unit == 0
        ptr = self.unit * 1000
        self.reservations[ptr] = size
        return ptr

    def create(self, size):
        self._event("create", size)
        assert size > 0 and size % self.unit == 0
        self.serial += 1
        self.handles[self.serial] = bytearray([171]) * size
        return self.serial

    def map(self, address, size, handle):
        self._event("map", address, size, handle)
        assert address % self.unit == size % self.unit == 0
        assert size == len(self.handles[handle])
        assert any(
            base <= address and address + size <= base + length
            for base, length in self.reservations.items()
        )
        assert not any(
            address < ptr + length and ptr < address + size
            for ptr, (length, _) in self.mappings.items()
        )
        self.mappings[address] = (size, handle)

    def set_access(self, address, size):
        self._event("access", address, size)
        assert self.mappings[address][0] == size

    def zero(self, address, size):
        self._event("zero", address, size)
        length, handle = self.mappings[address]
        assert length == size
        self.handles[handle][:] = bytes(size)

    def synchronize(self):
        self._event("sync")

    def unmap(self, address, size):
        self._event("unmap", address, size)
        assert self.mappings[address][0] == size, "Partial unmap is forbidden"
        del self.mappings[address]

    def release(self, handle):
        self._event("release", handle)
        assert not any(h == handle for _, h in self.mappings.values())
        del self.handles[handle]

    def free_address(self, address, size):
        self._event("free", address, size)
        assert self.reservations[address] == size
        assert not self.mappings and not self.handles
        del self.reservations[address]

    def bytes_at(self, address):
        return self.handles[self.mappings[address][1]]


class PoolTests(unittest.TestCase):
    def test_contiguous_survivor_proof_checks_both_range_endpoints(self):
        pool, driver = self.make_pool()
        self.resize(pool, 8)
        for live in (range(-1, 4), range(8, 0, -1), range(0, 8)):
            before = driver.events[:]
            with self.assertRaises(ValueError):
                self.resize(pool, 4, live_page_ids=live)
            self.assertEqual(driver.events, before)
        self.resize(pool, 4, live_page_ids=range(3, -1, -1))
        self.assertEqual(pool.stats()["mapped_pages"], 4)

    def test_larger_chunk_preserves_cuda_minimum_and_rounding(self):
        driver = FakeDriver(granularity=8)
        pool = VmmKvPool(2, 6, 24, driver=driver, chunk_bytes=48)
        self.addCleanup(lambda: pool.close(at_batch_barrier=True))
        result = pool.resize(9, at_batch_barrier=True)
        self.assertEqual(result["granularity_bytes"], 8)
        self.assertEqual(result["chunk_bytes"], 48)
        self.assertEqual(result["mapped_pages"], 16)
        self.assertEqual(result["mapped_physical_bytes"], 192)
        before = len(driver.events)
        pool.resize(10, at_batch_barrier=True)
        self.assertEqual(len(driver.events), before)
        pool.resize(8, at_batch_barrier=True, live_page_ids=[0, 7])
        self.assertEqual(pool.stats()["mapped_physical_bytes"], 96)

    def test_invalid_chunk_rejected_before_reservation(self):
        for chunk in (0, -1, 16, 25, 24.0, True):
            driver = FakeDriver(granularity=8)
            with self.assertRaises(ValueError):
                VmmKvPool(1, 6, 24, driver=driver, chunk_bytes=chunk)
            self.assertFalse(driver.reservations)

    def make_pool(self, **kwargs):
        driver = FakeDriver()
        pool = VmmKvPool(
            kwargs.get("num_layers", 2),
            kwargs.get("page_bytes", 6),
            kwargs.get("max_pages", 9),
            driver=driver,
        )
        self.addCleanup(lambda: pool.close(at_batch_barrier=True))
        return pool, driver

    def resize(self, pool, pages, **kwargs):
        return pool.resize(pages, at_batch_barrier=True, **kwargs)

    def test_fixed_layer_stride_and_stable_addresses_across_growth(self):
        pool, driver = self.make_pool()
        ptrs = pool.layer_ptrs
        self.assertEqual(ptrs[1] - ptrs[0], 72)
        first = self.resize(pool, 1)
        self.assertEqual((first["requested_pages"], first["mapped_pages"]), (1, 4))
        self.assertEqual(first["mapped_physical_bytes"], 48)
        self.assertEqual(first["reserved_va_bytes"], 144)
        for ptr in ptrs:
            driver.bytes_at(ptr)[:] = bytes([91]) * 24
        second = self.resize(pool, 5)
        self.assertEqual(second["mapped_pages"], 8)
        self.assertEqual(pool.layer_ptrs, ptrs)
        for ptr in ptrs:
            self.assertEqual(driver.bytes_at(ptr), bytes([91]) * 24)
            self.assertEqual(driver.bytes_at(ptr + 24), bytes(24))

    def test_lcm_quantization_and_same_chunk_resize(self):
        pool, driver = self.make_pool()
        self.assertEqual((pool.chunk_bytes, pool.chunk_pages), (24, 4))
        self.resize(pool, 3)
        before = driver.events[:]
        result = self.resize(pool, 1)
        self.assertEqual(driver.events, before)
        self.assertEqual((result["requested_pages"], result["mapped_pages"]), (1, 4))
        result = self.resize(pool, 9)
        self.assertEqual((result["mapped_pages"], result["max_mapped_pages"]), (12, 12))

    def test_shrink_releases_only_complete_chunks_and_retains_low_pages(self):
        pool, driver = self.make_pool()
        self.resize(pool, 9)
        ptrs = pool.layer_ptrs
        for ptr in ptrs:
            driver.bytes_at(ptr)[0] = 35
        driver.events.clear()
        result = self.resize(pool, 3, live_page_ids={0, 3})
        self.assertEqual(result["mapped_pages"], 4)
        self.assertEqual(result["mapped_physical_bytes"], 48)
        for event in driver.events:
            if event[0] == "unmap":
                self.assertEqual(event[2], 24)
        self.assertEqual(sum(e[0] == "unmap" for e in driver.events), 4)
        for ptr in ptrs:
            self.assertEqual(driver.bytes_at(ptr)[0], 35)

    def test_shrink_requires_live_page_proof_and_rejects_occupied_tail(self):
        pool, driver = self.make_pool()
        self.resize(pool, 8)
        before = driver.events[:]
        with self.assertRaises(ValueError):
            self.resize(pool, 4)
        with self.assertRaises(ValueError):
            self.resize(pool, 4, live_page_ids=[0, 7])
        self.assertEqual(driver.events, before)
        self.assertEqual(pool.stats()["mapped_pages"], 8)

    def test_resize_cycles_and_close_do_not_search_chunks_by_equality(self):
        pool, driver = self.make_pool(num_layers=3)
        self.resize(pool, 9)
        ptrs = pool.layer_ptrs
        for ptr in ptrs:
            driver.bytes_at(ptr)[:] = bytes([57]) * 24
        # Removing blocks must not trigger comparisons against other blocks.
        with patch.object(_Chunk, "__eq__", side_effect=AssertionError("chunk scan")):
            for pages in (4, 8, 4, 12):
                result = self.resize(pool, min(pages, 9), live_page_ids=[0])
                self.assertEqual(result["mapped_pages"], pages)
                self.assertEqual(result["allocated_physical_bytes"], pages * 6 * 3)
                self.assertEqual(pool.layer_ptrs, ptrs)
                for ptr in ptrs:
                    self.assertEqual(driver.bytes_at(ptr), bytes([57]) * 24)
            pool.close(at_batch_barrier=True)
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)

    def test_shrink_to_zero_keeps_reservation_and_regrowth_zeroes(self):
        pool, driver = self.make_pool()
        self.resize(pool, 4)
        ptrs = pool.layer_ptrs
        for ptr in ptrs:
            driver.bytes_at(ptr)[:] = bytes([99]) * 24
        result = self.resize(pool, 0, live_page_ids=[])
        self.assertEqual(result["mapped_physical_bytes"], 0)
        self.assertEqual(result["reserved_va_bytes"], 144)
        self.resize(pool, 4)
        self.assertEqual(pool.layer_ptrs, ptrs)
        for ptr in ptrs:
            self.assertEqual(driver.bytes_at(ptr), bytes(24))

    def test_growth_failures_roll_back_without_changing_old_contents(self):
        for operation in ("create", "map", "access", "zero", "sync"):
            with self.subTest(operation=operation):
                pool, driver = self.make_pool()
                self.resize(pool, 4)
                ptrs = pool.layer_ptrs
                for ptr in ptrs:
                    driver.bytes_at(ptr)[0] = 73
                old_handles = set(driver.handles)
                # Fail after some new resources exist; final sync is second sync.
                driver.fail(operation, after=1)
                with self.assertRaises(RuntimeError):
                    self.resize(pool, 8)
                self.assertEqual(pool.stats()["state"], "open")
                self.assertEqual(pool.stats()["mapped_pages"], 4)
                self.assertEqual(set(driver.handles), old_handles)
                for ptr in ptrs:
                    self.assertEqual(driver.bytes_at(ptr)[0], 73)

    def test_failed_growth_cleanup_poison_retains_resources_for_close(self):
        pool, driver = self.make_pool()
        self.resize(pool, 4)
        driver.fail("access")
        driver.fail("unmap")
        with self.assertRaises(RuntimeError):
            self.resize(pool, 8)
        self.assertEqual(pool.stats()["state"], "poisoned")
        self.assertIsNone(pool.stats()["mapped_pages"])
        with self.assertRaises(RuntimeError):
            self.resize(pool, 4)
        result = pool.close(at_batch_barrier=True)
        self.assertEqual(result["allocated_physical_bytes"], 0)

    def test_failed_growth_drain_keeps_resources_without_unmapping(self):
        pool, driver = self.make_pool(num_layers=3)
        self.resize(pool, 4)
        driver.events.clear()
        driver.fail("sync", after=1, times=2)
        with self.assertRaises(RuntimeError):
            self.resize(pool, 8)
        self.assertEqual(pool.stats()["state"], "poisoned")
        self.assertEqual(pool.stats()["mapped_physical_bytes"], 144)
        self.assertFalse(any(event[0] == "unmap" for event in driver.events))
        pool.close(at_batch_barrier=True)

    def test_growth_interrupt_rolls_back_and_reraises_interrupt(self):
        for interrupt in (KeyboardInterrupt, SystemExit):
            with self.subTest(interrupt=interrupt):
                pool, driver = self.make_pool()
                self.resize(pool, 4)
                driver.fail("zero", exception=interrupt)
                with self.assertRaises(interrupt):
                    self.resize(pool, 8)
                self.assertEqual(pool.stats()["state"], "open")
                self.assertEqual(pool.stats()["mapped_pages"], 4)
                self.assertEqual(pool.stats()["allocated_physical_bytes"], 48)

    def test_shrink_and_close_interrupts_preserve_cleanup_accounting(self):
        for interrupt in (KeyboardInterrupt, SystemExit):
            for operation in ("shrink", "close"):
                with self.subTest(operation=operation, interrupt=interrupt):
                    pool, driver = self.make_pool()
                    self.resize(pool, 8)
                    driver.fail("release", exception=interrupt)
                    with self.assertRaises(interrupt):
                        if operation == "shrink":
                            self.resize(pool, 4, live_page_ids=[])
                        else:
                            pool.close(at_batch_barrier=True)
                    self.assertEqual(pool.stats()["state"], "poisoned")
                    self.assertTrue(driver.handles)
                    pool.close(at_batch_barrier=True)
                    self.assertFalse(
                        driver.handles or driver.mappings or driver.reservations
                    )

    def test_failed_shrink_poison_and_close_retry(self):
        pool, driver = self.make_pool()
        self.resize(pool, 8)
        driver.fail("release")
        with self.assertRaises(RuntimeError):
            self.resize(pool, 4, live_page_ids=[])
        result = pool.stats()
        self.assertEqual(result["state"], "poisoned")
        self.assertGreater(
            result["allocated_physical_bytes"], result["mapped_physical_bytes"]
        )
        pool.close(at_batch_barrier=True)
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)

    def test_partial_shrink_release_failure_keeps_only_unreleased_handles(self):
        pool, driver = self.make_pool(num_layers=3)
        self.resize(pool, 9)
        before = set(driver.handles)
        driver.events.clear()
        driver.fail("release", after=2)
        with self.assertRaises(RuntimeError):
            self.resize(pool, 4, live_page_ids=[0])
        released = before - set(driver.handles)
        self.assertEqual(len(released), 2)
        state = pool.stats()
        self.assertEqual(state["state"], "poisoned")
        self.assertEqual(state["allocated_physical_bytes"], 7 * 24)
        self.assertEqual(state["mapped_physical_bytes"], 6 * 24)
        driver.events.clear()
        pool.close(at_batch_barrier=True)
        self.assertFalse(
            any(e[0] == "release" and e[1] in released for e in driver.events)
        )
        self.assertFalse(driver.handles or driver.mappings or driver.reservations)

    def test_close_release_order_and_idempotence(self):
        pool, driver = self.make_pool()
        self.resize(pool, 5)
        driver.events.clear()
        result = pool.close(at_batch_barrier=True)
        self.assertEqual(driver.events[0], ("sync",))
        self.assertEqual(driver.events[-1][0], "free")
        operations = [event[0] for event in driver.events[1:-1]]
        self.assertEqual(operations, ["unmap", "release"] * 4)
        self.assertEqual(result["state"], "closed")
        self.assertEqual(result["reserved_va_bytes"], 0)
        before = driver.events[:]
        pool.close(at_batch_barrier=True)
        self.assertEqual(driver.events, before)
        with self.assertRaises(RuntimeError):
            _ = pool.layer_ptrs

    def test_close_failure_does_not_free_va_before_remaining_handles(self):
        pool, driver = self.make_pool()
        self.resize(pool, 4)
        driver.fail("release")
        with self.assertRaises(RuntimeError):
            pool.close(at_batch_barrier=True)
        self.assertTrue(driver.reservations)
        self.assertEqual(pool.stats()["state"], "poisoned")
        pool.close(at_batch_barrier=True)
        self.assertFalse(driver.reservations or driver.handles)

    def test_address_free_failure_is_retryable_without_double_freeing_handles(self):
        pool, driver = self.make_pool()
        self.resize(pool, 4)
        driver.fail("free")
        with self.assertRaises(RuntimeError):
            pool.close(at_batch_barrier=True)
        self.assertEqual(pool.stats()["state"], "poisoned")
        self.assertFalse(driver.handles or driver.mappings)
        self.assertTrue(driver.reservations)
        before = sum(event[0] == "release" for event in driver.events)
        pool.close(at_batch_barrier=True)
        self.assertEqual(sum(event[0] == "release" for event in driver.events), before)

    def test_invalid_geometry_counts_pages_and_barriers_reject_before_mutation(self):
        for arguments in ((0, 6, 9), (2, 0, 9), (2, 6, 0), (True, 6, 9)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                VmmKvPool(*arguments, driver=FakeDriver())
        pool, driver = self.make_pool()
        self.resize(pool, 4)
        before = driver.events[:]
        for pages in (-1, True, 1.0, 10):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                self.resize(pool, pages)
        for live in ([-1], [True], [1.5], [4]):
            with self.subTest(live=live), self.assertRaises(ValueError):
                self.resize(pool, 0, live_page_ids=live)
        with self.assertRaises(ValueError):
            pool.resize(8, at_batch_barrier=False)
        with self.assertRaises(ValueError):
            pool.resize(8, at_batch_barrier=True, cuda_graphs_in_use=True)
        with self.assertRaises(ValueError):
            pool.close(at_batch_barrier=False)
        self.assertEqual(driver.events, before)


if __name__ == "__main__":
    unittest.main()
