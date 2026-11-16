import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from examples.hstu.inference.cache_ratio.memory_manager.shared import SharedCacheMemory
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

    def make(self, mode="observe", cost=0.01):
        runtime = PolicyRuntime(self.model, mode=mode, resize_cost_ms=cost,
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

    def test_close_restores_instance_slots_and_rejects_foreign_hooks(self):
        original = self.dense.forward_with_kvcache
        runtime = self.make()
        runtime.close()
        self.assertIs(self.dense.forward_with_kvcache, original)
        self.assertNotIn("lookup_kvcache", self.kv.__dict__)


if __name__ == "__main__":
    unittest.main()
