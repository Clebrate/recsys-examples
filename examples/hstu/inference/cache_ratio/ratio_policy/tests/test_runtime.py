import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from examples.hstu.inference.cache_ratio.memory_manager.shared import SharedCacheMemory
from examples.hstu.inference.cache_ratio.ratio_policy.adaptive import Config
from examples.hstu.inference.cache_ratio.ratio_policy.runtime import PolicyRuntime


class Vector(list):
    def tolist(self):
        return list(self)


class Event:
    syncs = 0

    def __init__(self, enable_timing):
        assert enable_timing

    def record(self):
        pass

    def synchronize(self):
        Event.syncs += 1

    def elapsed_time(self, other):
        return 10.0


class Kv:
    minimum_pages = 100

    def __init__(self):
        self.size, self.events = 500, []

    def stats(self):
        return dict(payload_bytes=self.size, allocated_physical_bytes=self.size,
                    resident_payload_bytes=self.size, state="open")

    def plan_pages(self, n):
        return dict(capacity_payload_bytes=n, max_requested_pages=900)

    def plan_resize_bytes(self, n, round_up=False):
        n = ((n + 9) // 10 if round_up else n // 10) * 10
        return dict(payload_bytes=n, allocated_physical_bytes=n,
                    target_pages=n, extra_peak_bytes=0)

    def resize_pages(self, n, at_batch_barrier):
        self.size = n
        return {"preserves_survivors": True}

    @contextmanager
    def resize_guard(self):
        yield

    def lookup_kvcache(self, ids, lengths):
        return (SimpleNamespace(user_ids=ids, seq_lengths=lengths),
                SimpleNamespace(cached_lengths=Vector([0] * len(ids))))

    def drain_evictions(self):
        result, self.events = self.events, []
        return result


class Emb:
    def __init__(self):
        self.size, self.generation, self.serial = 496, 0, 0
        self.next = {}
        self.decline = self.fail_after_mutation = False

    def stats(self):
        return dict(payload_bytes=self.size, device_pool_bytes=self.size + 16,
                    state="ready", tables={"video": dict(payload_bytes=self.size,
                    generation=self.generation)})

    def payload_bounds(self):
        return dict(minimum_payload_bytes=8, maximum_payload_bytes=1000)

    def policy_observation(self, reset=True):
        self.serial += 1
        result = dict(enabled=True, complete=True, overflow=False, observation_serial=self.serial,
                      resident_rows=self.size, capacity_rows=self.size, row_bytes=1,
                      ps_find_enabled=True, ps_find_complete=True, ps_find_overflow=False,
                      ps_find_keys=[], ps_find_count=0, ps_find_work_ms=0.0,
                      admission_eviction_keys=[], capacity_eviction_keys=[])
        result.update(self.next)
        self.next = {}
        return {"tables": {"video": result}}

    def plan_resize(self, n):
        if self.decline:
            raise ValueError("unsupported geometry")
        n = n // 8 * 8
        return dict(payload_bytes=n, device_pool_bytes=n + 16, extra_peak_bytes=0)

    def resize(self, plan, **kwargs):
        self.size = plan["payload_bytes"]
        self.generation += 1
        if self.fail_after_mutation:
            raise RuntimeError("native failed after mutation")
        return {"preserves_survivors": True}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.kv, self.emb = Kv(), Emb()
        self.memory = SharedCacheMemory(self.kv, self.emb, 1000, 1100,
                                       clock=lambda: self.now)
        self.dense = SimpleNamespace(forward_with_kvcache=lambda: "logits")
        self.model = SimpleNamespace(cache_memory=self.memory, dense_module=self.dense)
        self.runtimes = []

    def tearDown(self):
        for runtime in self.runtimes:
            runtime.close()

    def make(self, mode="observe", cost=0.01, config=None):
        runtime = PolicyRuntime(self.model, mode=mode, resize_cost_ms=cost,
                                config=config,
                                clock=lambda: self.now,
                                _torch=SimpleNamespace(cuda=SimpleNamespace(Event=Event)))
        runtime.start(lambda: self.now)
        self.runtimes.append(runtime)
        return runtime

    def batch(self, runtime, now, completed, *, final=False):
        runtime.before_batch()
        self.kv.lookup_kvcache(Vector([7]), Vector([100]))
        self.dense.forward_with_kvcache()
        self.now = now
        return runtime.after_batch(completed_count=completed,
                                   arrival_snapshot=lambda t: (completed, completed * 100),
                                   final=final)

    def seed_reuse(self, runtime):
        self.kv.events.append((7, 100, "capacity"))
        first = self.batch(runtime, 1, 1)
        self.assertEqual(first["window"]["kv"]["evicted_reuse_units"], 0)
        return first

    def test_real_eviction_only_credits_subsequent_miss(self):
        runtime = self.make()
        self.seed_reuse(runtime)
        second = self.batch(runtime, 3, 2)
        self.assertEqual(second["window"]["kv"]["evicted_reuse_units"], 100)
        self.assertEqual(second["rejection"]["reason"], "observation_only")
        self.assertIsNone(runtime.policy.pending)
        self.assertEqual(self.memory.actions, 0)

    def test_adaptive_records_actual_rounded_allocation_and_same_clock(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["action"]["status"], "applied")
        self.assertEqual(runtime.policy.applied_actions, 1)
        self.assertEqual(runtime.policy.ratio, self.kv.size / 1000)
        self.assertLess(runtime.policy.next_check_s, 10)
        self.assertEqual(self.memory.stats()["state"], "open")

    def test_safe_native_rejection_continues_with_unchanged_ledger(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        self.emb.decline = True
        before = self.memory.stats()
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["rejection"]["reason"], "native_preflight_declined")
        self.assertEqual(self.memory.stats(), before)
        self.assertIsNone(runtime.policy.pending)

    def test_partial_native_failure_is_not_a_rejection(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        self.emb.fail_after_mutation = True
        with self.assertRaisesRegex(RuntimeError, "after mutation"):
            self.batch(runtime, 3, 2)
        self.assertEqual(self.memory.stats()["state"], "poisoned")

    def test_mutating_preflight_cannot_be_reported_as_safe_rejection(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)

        def broken_plan(n):
            self.emb.generation += 1
            raise ValueError("broken native preflight")

        self.emb.plan_resize = broken_plan
        with self.assertRaisesRegex(ValueError, "broken native"):
            self.batch(runtime, 3, 2)
        self.assertEqual(self.memory.stats()["state"], "poisoned")

    def test_resize_victims_are_available_to_immediate_next_batch(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        original = self.emb.resize

        def resize(plan, **kwargs):
            result = original(plan, **kwargs)
            self.emb.next = dict(capacity_eviction_keys=[123])
            return result

        self.emb.resize = resize
        self.batch(runtime, 3, 2)
        self.emb.next = dict(ps_find_keys=[123], ps_find_count=1, ps_find_work_ms=2.0)
        result = self.batch(runtime, 6, 3)
        self.assertEqual(result["window"]["embedding"]["evicted_reuse_units"], 1)

    def test_missing_or_overflowed_native_observation_forces_hold(self):
        runtime = self.make()
        self.seed_reuse(runtime)
        self.emb.next = dict(overflow=True, complete=False)
        result = self.batch(runtime, 3, 2)
        self.assertIsNone(result["window"]["embedding"]["evicted_reuse_units"])
        self.assertIsNone(result["window"]["embedding"]["miss_work_ms"])
        self.assertEqual(result["decision"]["reason"], "missing_cache_observation")
        later = self.batch(runtime, 6, 3)
        self.assertFalse(later["batch_observation"]["embedding_complete"])

    def test_embedding_ghost_uses_actual_ps_miss_occurrences(self):
        runtime = self.make()
        self.emb.next = dict(admission_eviction_keys=[99])
        self.batch(runtime, 1, 1)
        self.emb.next = dict(ps_find_keys=[99, 99, 100], ps_find_count=3, ps_find_work_ms=4.5)
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["window"]["embedding"]["miss_units"], 3)
        self.assertEqual(result["window"]["embedding"]["evicted_reuse_units"], 2)
        self.assertEqual(result["window"]["embedding"]["miss_work_ms"], 4.5)

    def test_no_resize_cost_does_not_invent_zero(self):
        runtime = self.make(cost=None)
        self.seed_reuse(runtime)
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["decision"]["reason"], "missing_resize_cost")

    def test_diagnostic_applies_without_calibration_and_preserves_none(self):
        runtime = self.make("adaptive", cost=None,
                            config=Config(resize_cost_mode="diagnostic"))
        self.seed_reuse(runtime)
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["action"]["status"], "applied")
        self.assertIsNone(result["window"]["resize_cost_ms"])
        self.assertIsNone(runtime.stats()["resize_cost_ms"])
        self.assertEqual(result["decision"]["resize_cost_mode"], "diagnostic")
        self.assertEqual(runtime.stats()["resize_cost_mode"], "diagnostic")
        self.assertAlmostEqual(runtime.stats()["completion_rate_rps"], 2 / 3)
        self.assertTrue(runtime.stats()["last_window_throughput"]["input_limited"])

    def test_diagnostic_missing_observation_still_holds(self):
        runtime = self.make("adaptive", cost=None,
                            config=Config(resize_cost_mode="diagnostic"))
        self.seed_reuse(runtime)
        self.emb.next = dict(complete=False)
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["decision"]["reason"], "missing_cache_observation")
        self.assertIsNone(result["action"])
        self.assertEqual(self.memory.actions, 0)

    def test_resize_pause_includes_post_observation_and_preserves_call_interval(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        original_observe = self.memory.observe
        original_native = self.emb.policy_observation

        def timed_observe(*args, **kwargs):
            result = original_observe(*args, **kwargs)
            self.now = 3.25
            result["resize_interval"] = {"start_s": 3.0, "end_s": 3.25}
            return result

        def delayed_post_observation(**kwargs):
            result = original_native(**kwargs)
            if self.memory.actions:
                self.now += 0.5
            return result

        self.memory.observe = timed_observe
        self.emb.policy_observation = delayed_post_observation
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["resize_interval"], {"start_s": 3.0, "end_s": 3.25})
        self.assertEqual(result["resize_pause"]["start_s"], 3.0)
        self.assertEqual(result["resize_pause"]["end_s"], 3.75)

    def test_real_shared_interval_covers_prepare_and_both_cache_mutations(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        plan, emb_resize, kv_resize = self.emb.plan_resize, self.emb.resize, self.kv.resize_pages
        phases = []

        def timed_plan(n):
            phases.append("prepare")
            self.now += 0.125
            return plan(n)

        def timed_embedding(plan, **kwargs):
            phases.append("embedding")
            self.now += 0.25
            return emb_resize(plan, **kwargs)

        def timed_kv(n, **kwargs):
            phases.append("kv")
            self.now += 0.125
            return kv_resize(n, **kwargs)

        self.emb.plan_resize = timed_plan
        self.emb.resize = timed_embedding
        self.kv.resize_pages = timed_kv
        # Keep SharedCacheMemory.observe and resize_ratio unmodified: their
        # actual interval and duration accounting are what this test checks.
        result = self.batch(runtime, 3, 2)
        self.assertEqual(phases, ["prepare", "embedding", "kv"])
        self.assertEqual(result["action"]["status"], "applied")
        self.assertEqual(result["resize_interval"]["start_s"], 3)
        self.assertEqual(result["resize_interval"]["end_s"], 3.5)
        self.assertEqual(result["action"]["prepare_ms"], 125)
        self.assertEqual(result["action"]["apply_ms"], 375)
        self.assertEqual(result["action"]["action_ms"], 500)
        self.assertEqual(result["resize_pause"]["end_s"], 3.5)

    def test_real_shared_rounded_noop_records_planning_interval(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        before = self.memory.stats()
        original = self.emb.plan_resize

        def coarse_kv(n, round_up=False):
            n = ((n + 99) // 100 if round_up else n // 100) * 100
            return dict(payload_bytes=n, allocated_physical_bytes=n,
                        target_pages=n, extra_peak_bytes=0)

        def timed_plan(n):
            self.now += 0.25
            return original(n)

        self.kv.plan_resize_bytes = coarse_kv
        self.emb.plan_resize = timed_plan
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["action"]["status"], "rounded_noop")
        self.assertEqual(result["action"]["completed"], [])
        self.assertEqual(self.memory.stats(), before)
        self.assertEqual(result["resize_interval"]["start_s"], 3)
        self.assertEqual(result["resize_interval"]["end_s"], 3.25)
        self.assertEqual(result["action"]["action_ms"], 250)
        self.assertEqual(runtime.policy.applied_actions, 0)

    def test_real_shared_rejected_preflight_records_elapsed_time_without_mutation(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        before = self.memory.stats()

        def declined(n):
            self.now += 0.5
            raise ValueError("native preflight declined before mutation")

        self.emb.plan_resize = declined
        result = self.batch(runtime, 3, 2)
        self.assertIsNone(result["action"])
        self.assertTrue(result["rejection"]["unchanged"])
        self.assertEqual(self.memory.stats(), before)
        self.assertEqual(result["resize_interval"]["start_s"], 3)
        self.assertEqual(result["resize_interval"]["end_s"], 3.5)
        self.assertEqual(result["resize_pause"]["end_s"], 3.5)

    def test_real_shared_hold_does_not_plan_or_emit_interval(self):
        runtime = self.make("adaptive", cost=None)
        self.seed_reuse(runtime)

        def unexpected_plan(n):
            raise AssertionError("HOLD must not call resize preflight")

        self.emb.plan_resize = unexpected_plan
        result = self.batch(runtime, 3, 2)
        self.assertEqual(result["decision"]["reason"], "missing_resize_cost")
        self.assertIsNone(result["action"])
        self.assertNotIn("resize_interval", result)
        self.assertNotIn("resize_pause", result)

    def test_rejected_preflight_interval_is_retained_without_action(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        self.emb.decline = True
        original = self.memory.observe

        def timed_rejection(*args, **kwargs):
            result = original(*args, **kwargs)
            self.now = 3.5
            result["resize_interval"] = {"start_s": 3.0, "end_s": 3.5}
            return result

        self.memory.observe = timed_rejection
        result = self.batch(runtime, 3, 2)
        self.assertIsNone(result["action"])
        self.assertEqual(result["resize_pause"]["start_s"], 3.0)
        self.assertEqual(result["resize_pause"]["end_s"], 3.5)

    def test_legacy_executor_does_not_invent_resize_pause(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        original = self.memory.observe

        def legacy(*args, **kwargs):
            result = original(*args, **kwargs)
            result.pop("resize_interval", None)
            return result

        self.memory.observe = legacy
        result = self.batch(runtime, 3, 2)
        self.assertIsNotNone(result["action"])
        self.assertNotIn("resize_pause", result)

    def test_final_batch_never_executes_resize(self):
        runtime = self.make("adaptive")
        self.seed_reuse(runtime)
        result = self.batch(runtime, 3, 2, final=True)
        self.assertEqual(result["rejection"]["reason"], "end_of_stream")
        self.assertEqual(self.memory.actions, 0)

    def test_arrivals_during_pause_are_in_next_window(self):
        runtime = self.make()
        self.batch(runtime, 1, 1)
        self.now = 2  # Unobserved work never resets the arrival clock.
        runtime.before_batch()
        self.kv.lookup_kvcache(Vector([7]), Vector([100]))
        self.dense.forward_with_kvcache()
        self.now = 3
        result = runtime.after_batch(completed_count=2,
                                    arrival_snapshot=lambda t: (5, 500))
        self.assertEqual(result["window"]["start_s"], 1)
        self.assertEqual(result["window"]["arrivals"], 4)
        self.assertEqual(result["window"]["queue_at_end"], 3)
        signals = runtime.stats()["last_window_throughput"]
        self.assertEqual(signals["arrival_rate_rps"], 2)
        self.assertEqual(signals["completion_rate_rps"], 0.5)
        self.assertEqual(signals["completion_deficit_rps"], 1.5)
        self.assertFalse(signals["input_limited"])

    def test_close_restores_instance_slots_and_rejects_foreign_hooks(self):
        original = self.dense.forward_with_kvcache
        runtime = self.make()
        runtime.close()
        self.assertIs(self.dense.forward_with_kvcache, original)
        self.assertNotIn("lookup_kvcache", self.kv.__dict__)


if __name__ == "__main__":
    unittest.main()
