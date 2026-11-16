# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of eviction provenance, prefix overlap, expiry and bounded state."""

import unittest

from examples.hstu.inference.cache_ratio.ratio_policy.signals import GhostHistory


class GhostHistoryTests(unittest.TestCase):
    def test_cold_miss_has_no_reuse_signal(self):
        history = GhostHistory()
        self.assertEqual(history.reused_units(123, 0, 100, 0), 0)
        self.assertEqual(history.reused_units(123, 0, 100, 1), 0)

    def test_prefix_overlap_excludes_already_cached_tokens(self):
        history = GhostHistory()
        history.record_eviction(123, 8, 0)
        # min(missing=12, evicted=8) would incorrectly return eight here.
        self.assertEqual(history.reused_units(123, 8, 20, 1), 0)
        self.assertEqual(history.reused_units(123, 6, 20, 2), 2)
        self.assertEqual(history.reused_units(123, 2, 5, 3), 3)
        self.assertEqual(history.reused_units(123, 0, 20, 4), 8)
        self.assertEqual(history.reused_units(123, 9, 5, 5), 0)
        self.assertEqual(history.reused_units(123, 0, 0, 6), 0)

    def test_embedding_namespaces_and_repeated_misses(self):
        history = GhostHistory()
        key = ("video_id", 17)
        history.record_eviction(key, 1, 2)
        self.assertEqual(history.reused_units(("user_id", 17), 0, 1, 3), 0)
        self.assertEqual(history.reused_units(key, 0, 1, 3), 1)
        self.assertEqual(history.reused_units(key, 0, 1, 3), 1)
        self.assertEqual(history.reused_units(key, 1, 1, 4), 0)
        self.assertEqual(history.reused_units(key, 0, 1, 5), 1)

    def test_ttl_boundary_and_lookups_do_not_extend_eviction_time(self):
        history = GhostHistory(ttl_s=10)
        history.record_eviction(1, 12, 5)
        self.assertEqual(history.reused_units(1, 0, 12, 14.999), 12)
        self.assertEqual(history.reused_units(1, 0, 12, 15), 0)
        self.assertEqual(history.reused_units(1, 0, 12, 16), 0)

    def test_capacity_evicts_oldest_record_not_least_recently_queried(self):
        history = GhostHistory(max_entries=2)
        history.record_eviction(1, 1, 0)
        history.record_eviction(2, 2, 1)
        self.assertEqual(history.reused_units(1, 0, 10, 2), 1)
        history.record_eviction(3, 3, 3)
        self.assertEqual(history.reused_units(1, 0, 10, 3), 0)
        self.assertEqual(history.reused_units(2, 0, 10, 3), 2)
        self.assertEqual(history.reused_units(3, 0, 10, 3), 3)

    def test_re_eviction_replaces_prefix_time_and_capacity_order(self):
        history = GhostHistory(max_entries=2, ttl_s=10)
        history.record_eviction(1, 20, 0)
        history.record_eviction(2, 8, 1)
        history.record_eviction(1, 3, 8)
        history.record_eviction(3, 6, 9)
        self.assertEqual(history.reused_units(2, 0, 100, 10), 0)
        self.assertEqual(history.reused_units(1, 0, 100, 10), 3)
        self.assertEqual(history.reused_units(1, 0, 100, 18), 0)
        self.assertEqual(history.reused_units(3, 0, 100, 18), 6)

    def test_invalidation_removes_evidence_without_creating_new_ghost(self):
        history = GhostHistory()
        old_version, new_version = ("video_id", 1, "v1"), ("video_id", 1, "v2")
        history.record_eviction(old_version, 1, 0)
        history.invalidate(old_version)
        history.invalidate(old_version)
        self.assertEqual(history.reused_units(old_version, 0, 1, 1), 0)
        self.assertEqual(history.reused_units(new_version, 0, 1, 1), 0)
        history.record_eviction(new_version, 1, 2)
        self.assertEqual(history.reused_units(new_version, 0, 1, 2), 1)

    def test_event_clock_is_shared_between_recording_and_queries(self):
        history = GhostHistory()
        history.record_eviction(1, 8, 5)
        self.assertEqual(history.reused_units(1, 0, 8, 6), 8)
        with self.assertRaises(ValueError):
            history.record_eviction(2, 1, 5)
        with self.assertRaises(ValueError):
            history.reused_units(1, 0, 8, 5)
        history.record_eviction(2, 1, 6)
        self.assertEqual(history.reused_units(2, 0, 1, 6), 1)
        # A cold lookup also observes time and must prevent backdated evictions.
        self.assertEqual(history.reused_units(99, 0, 1, 7), 0)
        with self.assertRaises(ValueError):
            history.record_eviction(3, 1, 6)

    def test_large_integer_clock_comparisons_do_not_round_backwards_time(self):
        history = GhostHistory()
        history.record_eviction(1, 1, 2**53 + 1)
        with self.assertRaises(ValueError):
            history.reused_units(1, 0, 1, 2**53)

    def test_invalid_constructor_options(self):
        for count in (0, -1, True, False, 1.0, "2", None):
            with self.subTest(max_entries=count), self.assertRaises(ValueError):
                GhostHistory(max_entries=count)
        for ttl in (0, -1, True, "60", None, float("nan"), float("inf")):
            with self.subTest(ttl_s=ttl), self.assertRaises(ValueError):
                GhostHistory(ttl_s=ttl)

    def test_invalid_keys_rejected_by_every_operation(self):
        history = GhostHistory()
        invalid = (
            None,
            True,
            False,
            -1,
            1.0,
            float("nan"),
            "",
            (),
            [],
            {},
            ("video", -1),
            ("video", []),
        )
        for key in invalid:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    history.record_eviction(key, 1, 0)
                with self.assertRaises(ValueError):
                    history.reused_units(key, 0, 1, 0)
                with self.assertRaises(ValueError):
                    history.invalidate(key)

    def test_invalid_prefix_units_do_not_change_records_or_clock(self):
        history = GhostHistory()
        history.record_eviction(1, 5, 0)
        for units in (0, -1, True, 1.5, "1", None):
            with self.subTest(evicted=units), self.assertRaises(ValueError):
                history.record_eviction(1, units, 100)
        for units in (-1, True, 1.5, "1", None):
            with self.subTest(units=units):
                with self.assertRaises(ValueError):
                    history.reused_units(1, units, 5, 100)
                with self.assertRaises(ValueError):
                    history.reused_units(1, 0, units, 100)
        self.assertEqual(history.reused_units(1, 0, 5, 1), 5)

    def test_invalid_event_times_do_not_change_records_or_clock(self):
        history = GhostHistory()
        history.record_eviction(1, 5, 0)
        for now in (
            -1,
            True,
            None,
            "1",
            float("nan"),
            float("inf"),
            -float("inf"),
            10**1000,
        ):
            with self.subTest(now=now):
                with self.assertRaises(ValueError):
                    history.record_eviction(1, 2, now)
                with self.assertRaises(ValueError):
                    history.reused_units(1, 0, 5, now)
        self.assertEqual(history.reused_units(1, 0, 5, 1), 5)


if __name__ == "__main__":
    unittest.main()
