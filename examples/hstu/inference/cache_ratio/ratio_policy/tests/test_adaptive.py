# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow tests; synthetic signals establish no runtime speedup."""

import copy
from dataclasses import replace
import math
import unittest

from examples.hstu.inference.cache_ratio.ratio_policy.adaptive import (
    AdaptiveRatioPolicy,
    CacheStats,
    Config,
    ResizeReceipt,
    Window,
)


def window(
    start=0.0,
    end=2.0,
    *,
    kv_bytes=500000,
    total=1000000,
    direction=1,
    kv_resident=None,
    embedding_resident=None,
    queue_start=0,
    queue_end=0,
    tokens=100,
    arrivals=None,
    completed=None,
    resize_cost=0.01
):
    embedding_bytes = total - kv_bytes
    arrivals = round(50 * (end - start)) if arrivals is None else arrivals
    completed = arrivals if completed is None else completed
    kv_work, embedding_work = (10000, 10) if direction > 0 else (10, 10000)
    if direction == 0:
        kv_work, embedding_work = kv_bytes / 100, embedding_bytes / 100
    return Window(
        start,
        end,
        completed,
        arrivals,
        tokens,
        queue_start,
        queue_end,
        CacheStats(
            kv_bytes, kv_bytes if kv_resident is None else kv_resident, 100, 80, kv_work
        ),
        CacheStats(
            embedding_bytes,
            embedding_bytes if embedding_resident is None else embedding_resident,
            100,
            80,
            embedding_work,
        ),
        resize_cost,
    )


def receipt(policy, *, completed_s=None, **changes):
    decision, observed = policy.pending
    total = observed.kv.capacity_bytes + observed.embedding.capacity_bytes
    raw = decision.target_ratio * total
    kv_bytes = (
        math.floor(raw) if decision.target_ratio > policy.ratio else math.ceil(raw)
    )
    value = ResizeReceipt(
        kv_bytes,
        total - kv_bytes,
        observed.end_s + 0.01 if completed_s is None else completed_s,
        1.0,
        True,
        total,
        total + total // 2,
    )
    return replace(value, **changes)


def back_off(policy):
    start = 0.0
    for _ in range(12):
        end = max(start + policy.config.min_interval_s, policy.next_check_s)
        result = policy.observe(window(start, end, direction=0))
        assert result.target_ratio is None
        start = end
        if policy.interval_s == policy.config.max_interval_s:
            return start
    raise AssertionError("Stable balanced observations did not reach maximum interval")


class AdaptivePolicyTests(unittest.TestCase):
    def test_fixed_budget_keeps_rounding_slack_out_of_cache_capacity(self):
        policy = AdaptiveRatioPolicy()
        observed = replace(
            window(total=900000, kv_bytes=500000), payload_budget_bytes=1000000
        )
        result = policy.observe(observed)
        self.assertGreater(result.target_ratio, 0.5)
        self.assertEqual(policy.ratio, 0.5)
        self.assertEqual(policy.total_bytes, 1000000)
        self.assertEqual(policy.unused_payload_bytes, 100000)
        self.assertEqual(policy.pending[1].embedding.capacity_bytes, 400000)
        self.assertEqual(policy.pending[1].unused_payload_bytes, 100000)

    def test_payload_ceiling_and_slack_survive_window_accumulation(self):
        policy = AdaptiveRatioPolicy()
        for start in (0, 1):
            result = policy.observe(
                replace(
                    window(start, start + 1, total=900000), payload_budget_bytes=1000000
                )
            )
        self.assertIsNotNone(result.target_ratio)
        self.assertEqual(policy.pending[1].payload_budget_bytes, 1000000)
        self.assertEqual(policy.pending[1].unused_payload_bytes, 100000)

    def test_receipt_accepts_real_slack_and_physical_peak_below_unused_ceiling(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(replace(window(total=900000), payload_budget_bytes=1000000))
        # 4 KiB growth and a separately rounded 8 KiB donor release.
        applied = ResizeReceipt(504096, 391808, 2.01, 1, True, 910000, 1200000, 104096)
        policy.applied(applied)
        self.assertEqual(policy.ratio, 0.504096)
        self.assertEqual(policy.unused_payload_bytes, 104096)
        self.assertEqual(policy.applied_actions, 1)
        self.assertIsNotNone(
            policy.observe(
                replace(
                    window(2, 4.01, total=895904, kv_bytes=504096),
                    payload_budget_bytes=1000000,
                )
            ).target_ratio
        )

    def test_missing_or_invented_slack_in_receipt_is_rejected_without_mutation(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(replace(window(total=900000), payload_budget_bytes=1000000))
        valid = ResizeReceipt(504096, 391808, 2.01, 1, True, 910000, 1200000, 104096)
        before = copy.deepcopy(vars(policy))
        for unused in (0, 1, True, -1, 104095):
            with self.subTest(unused=unused), self.assertRaises(ValueError):
                policy.applied(replace(valid, unused_payload_bytes=unused))
            self.assertEqual(vars(policy), before)

    def test_embedding_change_without_receipt_is_detected_with_fixed_ceiling(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(replace(window(total=900000), payload_budget_bytes=1000000))
        before = copy.deepcopy(vars(policy))
        with self.assertRaisesRegex(ValueError, "Capacity changed"):
            policy.observe(
                replace(window(2, 4, total=900001), payload_budget_bytes=1000000)
            )
        self.assertEqual(vars(policy), before)

    def test_invalid_budget_or_actual_payload_overflow_rejected_before_mutation(self):
        for budget in (True, 0, -1, 899999, 1000000.0):
            policy = AdaptiveRatioPolicy()
            before = copy.deepcopy(vars(policy))
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                policy.observe(
                    replace(window(total=900000), payload_budget_bytes=budget)
                )
            self.assertEqual(vars(policy), before)

    def test_partial_donor_only_shrink_is_not_a_successful_resize(self):
        for direction, kv, embedding in ((1, 500000, 390000), (-1, 490000, 400000)):
            policy = AdaptiveRatioPolicy()
            policy.observe(
                replace(
                    window(total=900000, direction=direction),
                    payload_budget_bytes=1000000,
                )
            )
            before = copy.deepcopy(vars(policy))
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                policy.applied(
                    ResizeReceipt(
                        kv,
                        embedding,
                        2.01,
                        1,
                        True,
                        900000,
                        1200000,
                        1000000 - kv - embedding,
                    )
                )
            self.assertEqual(vars(policy), before)

    def test_unused_budget_can_fund_growth_without_fake_donor_capacity(self):
        for direction in (1, -1):
            policy = AdaptiveRatioPolicy()
            policy.observe(
                replace(
                    window(total=900000, direction=direction),
                    payload_budget_bytes=1000000,
                )
            )
            kv, embedding = (504000, 400000) if direction > 0 else (500000, 404000)
            policy.applied(
                ResizeReceipt(kv, embedding, 2.01, 1, True, 910000, 1200000, 96000)
            )
            self.assertEqual(policy.applied_actions, 1)
            self.assertEqual(policy._fill[0], direction)
            self.assertEqual(
                policy._fill[1], (500000 if direction > 0 else 400000) + 3200
            )

    def test_cold_misses_do_not_allocate_more_capacity(self):
        observed = window()
        observed = replace(
            observed,
            kv=replace(observed.kv, evicted_reuse_units=0, miss_work_ms=10**9),
            embedding=replace(observed.embedding, evicted_reuse_units=0),
        )
        policy = AdaptiveRatioPolicy()
        result = policy.observe(observed)
        self.assertIsNone(result.target_ratio)
        self.assertEqual(result.reason, "insufficient_evicted_reuse")
        self.assertEqual(policy.applied_actions, 0)

    def test_direction_uses_eviction_reuse_cost_not_raw_miss_count(self):
        observed = window()
        # Many expensive but cold KV misses versus fewer reusable embedding misses.
        observed = replace(
            observed,
            kv=replace(
                observed.kv,
                miss_units=1000000,
                evicted_reuse_units=0,
                miss_work_ms=1000000,
            ),
            embedding=replace(
                observed.embedding,
                miss_units=100,
                evicted_reuse_units=80,
                miss_work_ms=1000,
            ),
        )
        result = AdaptiveRatioPolicy().observe(observed)
        self.assertLess(result.target_ratio, 0.5)
        self.assertLess(result.pressure_balance, 0)

    def test_missing_observations_on_either_cache_are_not_zero_pressure(self):
        for cache in ("kv", "embedding"):
            for field in ("evicted_reuse_units", "miss_work_ms"):
                with self.subTest(cache=cache, field=field):
                    observed = window()
                    observed = replace(
                        observed,
                        **{cache: replace(getattr(observed, cache), **{field: None})}
                    )
                    result = AdaptiveRatioPolicy().observe(observed)
                    self.assertIsNone(result.target_ratio)
                    self.assertEqual(result.reason, "missing_cache_observation")
        observed = window()
        result = AdaptiveRatioPolicy().observe(
            replace(
                observed,
                kv=replace(observed.kv, evicted_reuse_units=None),
                embedding=replace(observed.embedding, miss_work_ms=None),
            )
        )
        self.assertEqual(result.reason, "missing_cache_observation")

    def test_pressure_generates_continuous_ratios_with_supported_bounds(self):
        config = Config(min_ratio=0.2, max_ratio=0.8)
        proposals = []
        for work in (100, 1000, 10000):
            observed = window(kv_bytes=373000)
            result = AdaptiveRatioPolicy(config).observe(
                replace(observed, kv=replace(observed.kv, miss_work_ms=work))
            )
            self.assertTrue(0.373 < result.target_ratio <= 0.8)
            self.assertNotIn(result.target_ratio, (0.25, 0.5, 0.75))
            proposals.append(result.target_ratio)
        self.assertEqual(len(set(proposals)), 3)
        upper = AdaptiveRatioPolicy(config).observe(window(kv_bytes=799000))
        self.assertEqual(upper.target_ratio, 0.8)
        bound = AdaptiveRatioPolicy(config).observe(window(kv_bytes=800000))
        self.assertIsNone(bound.target_ratio)
        self.assertEqual(bound.reason, "capacity_bound")
        lower = AdaptiveRatioPolicy(config).observe(
            window(kv_bytes=201000, direction=-1)
        )
        self.assertEqual(lower.target_ratio, 0.2)

    def test_unknown_or_dominating_resize_cost_holds(self):
        for cost, reason in [
            (None, "missing_resize_cost"),
            (10**9, "resize_cost_dominates"),
        ]:
            with self.subTest(cost=cost):
                result = AdaptiveRatioPolicy().observe(window(resize_cost=cost))
                self.assertIsNone(result.target_ratio)
                self.assertEqual(result.reason, reason)
        self.assertIsNotNone(
            AdaptiveRatioPolicy().observe(window(resize_cost=0.01)).target_ratio
        )

    def test_unused_receiver_capacity_prevents_another_grant(self):
        result = AdaptiveRatioPolicy().observe(window(kv_resident=100000))
        self.assertIsNone(result.target_ratio)
        self.assertEqual(result.reason, "receiver_has_free_capacity")

    def test_queue_growth_shortens_backoff_but_respects_minimum_interval(self):
        policy = AdaptiveRatioPolicy()
        end = back_off(policy)
        old_due = policy.next_check_s
        result = policy.observe(window(end, end + 1, queue_end=10))
        self.assertEqual(result.reason, "collecting_window")
        self.assertEqual(result.next_check_s, end + policy.config.min_interval_s)
        self.assertLess(result.next_check_s, old_due)
        result = policy.observe(window(end + 1, end + 2, queue_start=10, queue_end=20))
        self.assertIsNotNone(result.target_ratio)

    def test_traffic_drift_shortens_backoff_without_queue_growth(self):
        policy = AdaptiveRatioPolicy()
        end = back_off(policy)
        old_due = policy.next_check_s
        result = policy.observe(window(end, end + 2, tokens=1000))
        self.assertIsNotNone(result.target_ratio)
        self.assertLess(result.next_check_s, old_due)
        self.assertEqual(policy.interval_s, policy.config.min_interval_s)

    def test_initial_short_windows_do_not_starve_evaluation_forever(self):
        policy = AdaptiveRatioPolicy()
        first = policy.observe(window(0, 1))
        self.assertEqual(first.reason, "collecting_window")
        second = policy.observe(window(1, 2))
        self.assertIsNotNone(second.target_ratio)

    def test_short_windows_accumulate_counts_work_and_offered_token_mean(self):
        policy = AdaptiveRatioPolicy()
        first = window(
            0,
            1,
            kv_resident=100000,
            arrivals=13,
            completed=12,
            tokens=10,
            queue_start=2,
            queue_end=3,
            resize_cost=None,
        )
        second = window(
            1,
            2,
            arrivals=87,
            completed=88,
            tokens=110,
            queue_start=3,
            queue_end=9,
            resize_cost=0.01,
        )
        second = replace(
            second,
            kv=replace(
                second.kv, miss_units=900, evicted_reuse_units=0, miss_work_ms=0
            ),
            embedding=replace(second.embedding, evicted_reuse_units=0, miss_work_ms=0),
        )
        self.assertEqual(policy.observe(first).reason, "collecting_window")
        result = policy.observe(second)
        self.assertGreater(result.target_ratio, 0.5)
        accumulated = policy.pending[1]
        self.assertEqual((accumulated.start_s, accumulated.end_s), (0, 2))
        self.assertEqual((accumulated.completed, accumulated.arrivals), (100, 100))
        self.assertEqual(accumulated.mean_request_tokens, 97)
        self.assertEqual((accumulated.queue_at_start, accumulated.queue_at_end), (2, 9))
        self.assertEqual(accumulated.kv, CacheStats(500000, 500000, 1000, 80, 10000))
        self.assertEqual(accumulated.embedding.miss_units, 200)
        self.assertEqual(accumulated.resize_cost_ms, 0.01)

    def test_unknown_parts_propagate_until_accumulated_window_is_consumed(self):
        for cache in ("kv", "embedding"):
            for field in ("evicted_reuse_units", "miss_work_ms"):
                for unknown_index in (0, 1):
                    with self.subTest(
                        cache=cache, field=field, unknown_index=unknown_index
                    ):
                        policy = AdaptiveRatioPolicy()
                        parts = [window(0, 1), window(1, 2)]
                        unknown = parts[unknown_index]
                        parts[unknown_index] = replace(
                            unknown,
                            **{cache: replace(getattr(unknown, cache), **{field: None})}
                        )
                        self.assertEqual(
                            policy.observe(parts[0]).reason, "collecting_window"
                        )
                        result = policy.observe(parts[1])
                        self.assertIsNone(result.target_ratio)
                        self.assertEqual(result.reason, "missing_cache_observation")
                        self.assertIsNotNone(policy.observe(window(2, 4)).target_ratio)

    def test_latest_occupancy_and_resize_estimate_are_not_summed(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window(0, 1, resize_cost=10**9))
        result = policy.observe(window(1, 2, kv_resident=100000, resize_cost=0.01))
        self.assertEqual(result.reason, "receiver_has_free_capacity")
        policy = AdaptiveRatioPolicy()
        policy.observe(window(0, 1, resize_cost=10**9))
        self.assertIsNotNone(
            policy.observe(window(1, 2, resize_cost=0.01)).target_ratio
        )
        policy = AdaptiveRatioPolicy()
        policy.observe(window(0, 1, resize_cost=0.01))
        self.assertEqual(
            policy.observe(window(1, 2, resize_cost=None)).reason, "missing_resize_cost"
        )

    def test_arrival_gap_starts_fresh_observation_instead_of_inventing_zero_traffic(
        self,
    ):
        policy = AdaptiveRatioPolicy()
        first = window(0, 1)
        first = replace(first, embedding=replace(first.embedding, miss_work_ms=None))
        policy.observe(first)
        self.assertIsNotNone(policy.observe(window(10, 12)).target_ratio)
        observed = policy.pending[1]
        self.assertEqual((observed.start_s, observed.end_s), (10, 12))
        self.assertEqual(observed.kv.miss_units, 100)
        self.assertEqual(observed.arrivals, 100)

    def test_full_fill_threshold_uses_integer_byte_growth(self):
        policy = AdaptiveRatioPolicy(Config(fill_fraction=1))
        policy.observe(window())
        applied = receipt(policy, kv_bytes=500005, embedding_bytes=499995)
        policy.applied(applied)
        result = policy.observe(
            window(2, applied.completed_s + 2, kv_bytes=500005, kv_resident=500005)
        )
        self.assertIsNotNone(result.target_ratio)
        self.assertNotEqual(result.reason, "waiting_for_fill")

    def test_long_observed_action_increases_next_observation_interval(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        applied = receipt(policy, completed_s=12, action_ms=10000)
        policy.applied(applied)
        self.assertEqual(policy.interval_s, 20)
        self.assertEqual(policy.next_check_s, 32)

    def test_proposal_does_not_change_ratio_before_receipt(self):
        policy = AdaptiveRatioPolicy()
        result = policy.observe(window())
        self.assertGreater(result.target_ratio, 0.5)
        self.assertEqual(policy.ratio, 0.5)
        self.assertEqual(policy.applied_actions, 0)
        again = policy.observe(window(2, 3))
        self.assertEqual(again.reason, "awaiting_resize_receipt")
        self.assertEqual(policy.ratio, 0.5)
        self.assertEqual(policy.proposals, 1)

    def test_unacknowledged_capacity_changes_are_rejected_exactly(self):
        for total, change in [(1000000, 1000), (10**12, 1)]:
            with self.subTest(total=total, change=change):
                policy = AdaptiveRatioPolicy()
                policy.observe(window(total=total, kv_bytes=total // 2))
                before = copy.deepcopy(vars(policy))
                with self.assertRaisesRegex(ValueError, "Capacity changed"):
                    policy.observe(
                        window(2, 4, total=total, kv_bytes=total // 2 + change)
                    )
                self.assertEqual(vars(policy), before)

    def test_reverse_pressure_reduces_step_and_can_reverse_before_fill(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        applied = receipt(policy)
        policy.applied(applied)
        step = policy.step
        result = policy.observe(
            window(
                2,
                applied.completed_s + 2,
                kv_bytes=applied.kv_bytes,
                direction=-1,
                kv_resident=500000,
            )
        )
        self.assertLess(result.target_ratio, applied.kv_bytes / 1000000)
        self.assertLess(policy.step, step)

    def test_unfilled_growth_blocks_same_direction_and_timeout_is_not_warm(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        applied = receipt(policy)
        policy.applied(applied)
        end = applied.completed_s + 2
        result = policy.observe(
            window(2, end, kv_bytes=applied.kv_bytes, kv_resident=500000)
        )
        self.assertEqual(result.reason, "waiting_for_fill")
        self.assertIsNone(result.target_ratio)
        step = policy.step
        deadline = applied.completed_s + policy.config.fill_timeout_s
        result = policy.observe(
            window(end, deadline + 1, kv_bytes=applied.kv_bytes, kv_resident=500000)
        )
        self.assertEqual(result.reason, "growth_not_used")
        self.assertIsNone(result.target_ratio)
        self.assertLess(policy.step, step)
        reduced_step = policy.step
        result = policy.observe(
            window(
                deadline + 1,
                deadline + 10,
                kv_bytes=applied.kv_bytes,
                kv_resident=500000,
            )
        )
        self.assertEqual(result.reason, "growth_not_used")
        self.assertEqual(policy.step, reduced_step)

    def test_actual_fill_observation_allows_same_direction_again(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        applied = receipt(policy)
        policy.applied(applied)
        resident = math.ceil(
            500000 + policy.config.fill_fraction * (applied.kv_bytes - 500000)
        )
        result = policy.observe(
            window(
                2,
                applied.completed_s + 2,
                kv_bytes=applied.kv_bytes,
                kv_resident=resident,
            )
        )
        self.assertGreater(result.target_ratio, policy.ratio)

    def test_quantization_noop_is_not_an_action_and_enlarges_next_probe(self):
        policy = AdaptiveRatioPolicy()
        first = policy.observe(window())
        step = policy.step
        applied = receipt(policy, kv_bytes=500000, embedding_bytes=500000)
        policy.applied(applied)
        self.assertEqual(policy.ratio, 0.5)
        self.assertEqual(policy.applied_actions, 0)
        self.assertGreater(policy.step, step)
        second = policy.observe(window(2, applied.completed_s + 2))
        self.assertGreater(second.target_ratio, first.target_ratio)

    def test_many_completed_moves_have_no_lifetime_trial_cap(self):
        policy = AdaptiveRatioPolicy(Config(ewma_alpha=1))
        start, end, kv_bytes = 0.0, 2.0, 500000
        for index in range(160):
            result = policy.observe(
                window(
                    start, end, kv_bytes=kv_bytes, direction=1 if index % 2 == 0 else -1
                )
            )
            self.assertIsNotNone(result.target_ratio, (index, result))
            applied = receipt(policy)
            policy.applied(applied)
            start, end, kv_bytes = end, applied.completed_s + 2, applied.kv_bytes
        self.assertEqual(policy.applied_actions, 160)
        self.assertEqual(policy.proposals, 160)

    def test_invalid_receipts_leave_every_policy_field_unchanged(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        valid = receipt(policy)
        invalid = [
            replace(valid, kv_bytes=True),
            replace(valid, kv_bytes=-1),
            replace(valid, kv_bytes=valid.kv_bytes + 1),
            replace(valid, preserves_survivors=False),
            replace(valid, preserves_survivors=1),
            replace(valid, peak_cache_bytes=999999),
            replace(valid, peak_cache_bytes=valid.peak_limit_bytes + 1),
            replace(valid, peak_limit_bytes=-1),
            replace(valid, completed_s=1),
            replace(valid, completed_s=math.nan),
            replace(valid, action_ms=100),
            replace(valid, action_ms=math.inf),
            replace(valid, action_ms=-1),
            replace(valid, kv_bytes=400000, embedding_bytes=600000),
            replace(valid, kv_bytes=900000, embedding_bytes=100000),
        ]
        before = copy.deepcopy(vars(policy))
        for value in invalid:
            with self.subTest(receipt=value):
                with self.assertRaises(ValueError):
                    policy.applied(value)
                self.assertEqual(vars(policy), before)
        policy.applied(valid)
        self.assertEqual(policy.applied_actions, 1)

    def test_receipts_record_allocator_rounded_capacity(self):
        policy = AdaptiveRatioPolicy()
        result = policy.observe(window())
        actual_kv = 501000
        self.assertLess(actual_kv / 1000000, result.target_ratio)
        policy.applied(
            receipt(policy, kv_bytes=actual_kv, embedding_bytes=1000000 - actual_kv)
        )
        self.assertEqual(policy.ratio, actual_kv / 1000000)

    def test_ack_does_not_exclude_arrivals_during_resize(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        applied = receipt(policy, completed_s=3, action_ms=1000)
        policy.applied(applied)
        # This window includes the entire resize pause on the arrival clock.
        result = policy.observe(
            window(2, 5, kv_bytes=applied.kv_bytes, direction=-1, queue_end=50)
        )
        self.assertIsNotNone(result.target_ratio)

    def test_preexecution_rejection_keeps_capacity_and_continuous_arrival_window(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window())
        before = copy.deepcopy(vars(policy))
        for value in (1, -1, math.nan, True):
            with self.subTest(time=value):
                with self.assertRaises(ValueError):
                    policy.rejected(value)
                self.assertEqual(vars(policy), before)
        policy.rejected(3)
        self.assertEqual(policy.ratio, 0.5)
        self.assertEqual(policy.applied_actions, 0)
        self.assertIsNotNone(policy.observe(window(2, 5)).target_ratio)

    def test_overlapping_or_invalid_windows_do_not_mutate_state(self):
        policy = AdaptiveRatioPolicy()
        policy.observe(window(direction=0))
        before = copy.deepcopy(vars(policy))
        for invalid in [
            window(1, 3),
            replace(window(2, 4), end_s=1),
            replace(window(2, 4), arrivals=True),
            replace(window(2, 4), kv=CacheStats(500000, 500001, 1, 1, 1)),
            replace(window(2, 4), kv=CacheStats(500000, 500000, 1, 2, 1)),
        ]:
            with self.subTest(window=invalid):
                with self.assertRaises(ValueError):
                    policy.observe(invalid)
                self.assertEqual(vars(policy), before)

    def test_receipt_and_rejection_without_pending_proposal_fail(self):
        policy = AdaptiveRatioPolicy()
        with self.assertRaises(ValueError):
            policy.applied(ResizeReceipt(500000, 500000, 3, 1, True, 1000000, 1000000))
        with self.assertRaises(ValueError):
            policy.rejected(3)


if __name__ == "__main__":
    unittest.main()
