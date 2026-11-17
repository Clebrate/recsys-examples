# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measured replay adapter for the existing policy; no synthetic cost or hits.

KV work is the CUDA-stream span of dense forward in a batch with missing KV
tokens, including candidate work. PS work is the native CPU find-call span.
These disjoint stage proxies are approximate attribution, not marginal savings
or a decomposition of request latency. Observation, synchronization, planning,
resize and refill remain on the replay's original arrival clock.
"""

from dataclasses import asdict, replace
from functools import wraps
from math import isfinite
from time import perf_counter

from .adaptive import AdaptiveRatioPolicy, CacheStats, Config, Window
from .signals import GhostHistory


def _count(value):
    return type(value) is int and value >= 0


def _time(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


class PolicyRuntime:
    """Install instance-only hooks, then start/before_batch/after_batch/close.

    ``arrival_snapshot(t)`` must return cumulative (arrivals, offered_tokens)
    using only requests whose original timestamp is <= t. ``completed_count``
    is cumulative. Resize never resets this origin. Only previously drained
    eviction events can credit a miss; within-batch eviction/reuse is omitted
    conservatively because native insertion clocks are not wall timestamps.
    """

    def __init__(self, model, *, mode="observe", resize_cost_ms=None,
                 config=None, clock=perf_counter, ghost_max_entries=100000,
                 ghost_ttl_s=60, _torch=None):
        if mode not in ("observe", "adaptive"):
            raise ValueError("Runtime mode must be observe or adaptive; fixed uses no runtime")
        if resize_cost_ms is not None and not _time(resize_cost_ms):
            raise ValueError("Resize calibration must be finite and nonnegative or None")
        self.model, self.memory = model, model.cache_memory
        self.mode, self.resize_cost_ms, self._wall = mode, resize_cost_ms, clock
        self.bounds = self.memory.policy_ratio_bounds()
        settings = replace(config or Config(), min_ratio=self.bounds["min_ratio"],
                           max_ratio=self.bounds["max_ratio"])
        self.policy = AdaptiveRatioPolicy(settings)
        self._kv_ghost = GhostHistory(ghost_max_entries, ghost_ttl_s)
        self._emb_ghost = GhostHistory(ghost_max_entries, ghost_ttl_s)
        self._ghost_ttl = ghost_ttl_s
        self._emb_incomplete_until = 0.0
        self._hooks, self._serials = [], {}
        self._active = self._closed = False
        self._replay_clock = None
        self._last_end = 0.0
        self._arrivals = self._tokens = self._completed = self._queue = 0
        self._last_throughput = None
        self.batches = self.safe_rejections = self.observation_gaps = 0
        self._kv_rows, self._dense_calls = [], 0
        if _torch is None:
            import torch as _torch
        self._torch = _torch
        self._start_event = _torch.cuda.Event(enable_timing=True)
        self._end_event = _torch.cuda.Event(enable_timing=True)
        try:
            self._wrap(self.memory.kv, "lookup_kvcache", self._lookup)
            self._wrap(model.dense_module, "forward_with_kvcache", self._dense)
            # Startup probes precede the measured replay. Start with empty
            # ghost history, not fabricated eviction timestamps for those probes.
            self.memory.embedding.policy_observation(reset=True)
            self.memory.kv.drain_evictions()
        except BaseException:
            self.close()
            raise

    def _wrap(self, owner, name, callback):
        original = getattr(owner, name)
        had_slot, old_slot = name in owner.__dict__, owner.__dict__.get(name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            return callback(original, *args, **kwargs)

        setattr(owner, name, wrapped)
        self._hooks.append((owner, name, had_slot, old_slot, wrapped))

    def _lookup(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if self._active:
            index, lookup = result
            ids = index.user_ids.tolist()
            requested = index.seq_lengths.tolist()
            cached = lookup.cached_lengths.tolist()
            if not (len(ids) == len(requested) == len(cached)):
                raise RuntimeError("KV observation vectors have different lengths")
            for uid, want, have in zip(ids, requested, cached):
                if not all(_count(x) for x in (uid, want, have)) or have > want:
                    raise RuntimeError("Invalid actual KV prefix observation")
                self._kv_rows.append((uid, have, want))
        return result

    def _dense(self, original, *args, **kwargs):
        if not self._active:
            return original(*args, **kwargs)
        self._dense_calls += 1
        if self._dense_calls != 1:
            raise RuntimeError("Runtime expects one dense forward per completed batch")
        self._start_event.record()
        try:
            return original(*args, **kwargs)
        finally:
            self._end_event.record()

    def start(self, replay_clock):
        if self._closed or self._replay_clock is not None or not callable(replay_clock):
            raise RuntimeError("Runtime start requires a fresh runtime and arrival clock")
        self._replay_clock = replay_clock

    def before_batch(self):
        if self._closed or self._replay_clock is None or self._active:
            raise RuntimeError("Invalid runtime batch boundary")
        self._kv_rows, self._dense_calls = [], 0
        self._active = True

    def _native(self, observation, actual, now):
        tables = observation.get("tables", {})
        expected = actual["embedding"].get("tables", {})
        complete = bool(tables) and set(tables) == set(expected)
        resident, misses, work, keys = 0, 0, 0.0, []
        reasons = []
        for name, native in tables.items():
            state = expected.get(name, {})
            valid = native.get("enabled") is True and native.get("complete") is True
            valid &= native.get("overflow") is False
            valid &= native.get("ps_find_enabled") is True and native.get("ps_find_complete") is True
            valid &= native.get("ps_find_overflow") is False
            occupied, capacity, row = (native.get(k) for k in ("resident_rows", "capacity_rows", "row_bytes"))
            geometry = all(_count(x) for x in (occupied, capacity, row)) and row > 0 and occupied <= capacity
            geometry = geometry and capacity * row == state.get("payload_bytes")
            valid &= geometry
            if geometry:
                resident += occupied * row
            serial = native.get("observation_serial")
            if not _count(serial) or serial <= self._serials.get(name, -1):
                valid = False
            if _count(serial):
                self._serials[name] = serial
            count, duration, found = (native.get(k) for k in ("ps_find_count", "ps_find_work_ms", "ps_find_keys"))
            found_valid = isinstance(found, list) and all(_count(key) for key in found)
            valid &= _count(count) and _time(duration) and found_valid
            if _count(count):
                misses += count
            if _time(duration):
                work += duration
            if found_valid:
                keys.extend((name, key) for key in found)
                valid &= count == len(found)
            if not valid:
                reasons.append(name + ":incomplete_native_observation")
            if native.get("overflow") or native.get("ps_find_overflow"):
                self._emb_incomplete_until = max(self._emb_incomplete_until, now + self._ghost_ttl)
            complete &= valid
        if now < self._emb_incomplete_until:
            complete = False
            reasons.append("bounded_event_overflow_history_gap")
        return resident, misses, work, keys, complete, reasons

    def _record_evictions(self, observation, now):
        kv_events = self.memory.kv.drain_evictions()
        for uid, length, reason in kv_events:
            if reason not in ("capacity", "resize_tail"):
                self._kv_ghost.invalidate(uid)
            elif length > 0:
                self._kv_ghost.record_eviction(uid, length, now)
        count = 0
        for name, native in observation.get("tables", {}).items():
            for kind in ("admission", "capacity"):
                keys = native.get(kind + "_eviction_keys", [])
                if not isinstance(keys, list) or any(not _count(key) for key in keys):
                    raise RuntimeError("Invalid native capacity eviction keys")
                for key in keys:
                    self._emb_ghost.record_eviction((name, key), 1, now)
                count += len(keys)
        return {"kv_events": len(kv_events), "embedding_events": count}

    def after_batch(self, *, completed_count, arrival_snapshot, final=False):
        if self._closed or not self._active:
            raise RuntimeError("No active completed batch to observe")
        started = self._wall()
        self._active = False
        dense_ms = None
        if self._dense_calls == 1:
            self._end_event.synchronize()
            dense_ms = float(self._start_event.elapsed_time(self._end_event))
            if not _time(dense_ms):
                raise RuntimeError("Invalid dense CUDA event duration")
        observation = self.memory.embedding.policy_observation(reset=True)
        actual = self.memory.stats()
        if actual["state"] != "open":
            raise RuntimeError("Native cache is poisoned; stop replay")
        now = self._replay_clock()
        arrivals, tokens = arrival_snapshot(now)
        if (not all(_count(x) for x in (arrivals, tokens, completed_count))
                or arrivals < self._arrivals or tokens < self._tokens
                or not self._completed < completed_count <= arrivals
                or not _time(now) or now <= self._last_end):
            raise ValueError("Cumulative arrival/completion clock moved backwards")
        resident, misses, ps_ms, miss_keys, complete, reasons = self._native(observation, actual, now)
        kv_misses = sum(want - have for _, have, want in self._kv_rows)
        kv_reuse = sum(self._kv_ghost.reused_units(uid, have, want, now)
                       for uid, have, want in self._kv_rows if have < want)
        emb_reuse = sum(self._emb_ghost.reused_units(key, 0, 1, now) for key in miss_keys) if complete else None
        kv_complete = bool(self._kv_rows) and dense_ms is not None
        if not complete or not kv_complete:
            self.observation_gaps += 1
        events = self._record_evictions(observation, now)
        offered = arrivals - self._arrivals
        window = Window(self._last_end, now, completed_count - self._completed,
                        offered, (tokens - self._tokens) / offered if offered else 0.0,
                        self._queue, arrivals - completed_count,
                        CacheStats(actual["kv"]["payload_bytes"], actual["kv"]["resident_payload_bytes"],
                                   kv_misses, kv_reuse if kv_complete else None,
                                   (dense_ms if kv_misses else 0.0) if kv_complete else None),
                        CacheStats(actual["embedding"]["payload_bytes"], resident, misses,
                                   emb_reuse, ps_ms if complete else None),
                        self.resize_cost_ms, self.memory.payload_budget_bytes)
        self._last_throughput = window.throughput()
        self._last_end, self._arrivals, self._tokens = now, arrivals, tokens
        self._completed, self._queue = completed_count, arrivals - completed_count
        if self.mode == "adaptive" and not final:
            result = self.memory.observe(self.policy, window, replay_clock=self._replay_clock)
        else:
            decision = self.policy.observe(window)
            result = {"decision": asdict(decision), "action": None}
            if decision.target_ratio is not None:
                reason = "end_of_stream" if final else "observation_only"
                self.policy.rejected(self._replay_clock(), reason)
                result["rejection"] = {"reason": reason, "unchanged": True}
        if result.get("rejection", {}).get("reason") == "native_preflight_declined":
            self.safe_rejections += 1
        if result.get("action") is not None:
            # Make resize victims visible to the NEXT batch's actual misses.
            post = self.memory.embedding.policy_observation(reset=True)
            after = self.memory.stats()
            after_time = self._replay_clock()
            self._native(post, after, after_time)  # Check overflow/serial too.
            result["resize_eviction_events"] = self._record_evictions(post, after_time)
            if result.get("resize_interval") is not None:
                result["resize_pause"] = {
                    "start_s": result["resize_interval"]["start_s"],
                    "end_s": self._replay_clock(),
                    "scope": "resize call plus post-resize observation, ledger and eviction drain",
                }
        elif result.get("resize_interval") is not None:
            result["resize_pause"] = {
                **result["resize_interval"],
                "scope": "resize attempt including preflight; no applied action",
            }
        self.batches += 1
        return {**result, "mode": self.mode, "final": bool(final), "window": asdict(window),
                "batch_observation": {"dense_cuda_span_ms": dense_ms, "ps_find_work_ms": ps_ms if complete else None,
                                      "embedding_complete": complete, "kv_complete": kv_complete,
                                      "missing_reasons": reasons, "evictions": events,
                                      "within_batch_eviction_reuse": "not_credited"},
                "observer_ms": (self._wall() - started) * 1000,
                "observer_scope": "inclusive after-batch observation, policy and any resize; do not add action_ms again"}

    def stats(self):
        return {"mode": self.mode, "batches": self.batches, "bounds": self.bounds,
                "resize_cost_mode": self.policy.config.resize_cost_mode,
                "last_window_throughput": self._last_throughput,
                "completion_rate_rps": self._completed / self._last_end if self._last_end else None,
                "arrival_rate_rps": self._arrivals / self._last_end if self._last_end else None,
                "completion_deficit_rps": max(0, self._arrivals - self._completed) / self._last_end
                    if self._last_end else None,
                "throughput_scope": "observed arrival-clock rates through last window; last-window input_limited is an empty-boundary hint, not a service-capacity estimate",
                "resize_cost_ms": self.resize_cost_ms, "proposals": self.policy.proposals,
                "applied_actions": self.policy.applied_actions, "holds": dict(self.policy.holds),
                "safe_rejections": self.safe_rejections, "observation_gaps": self.observation_gaps,
                "miss_work_scope": "dense CUDA-stream span for KV-miss batches and native CPU PS-find walltime; approximate attribution"}

    def close(self):
        if self._closed:
            return
        for owner, name, _, _, hook in self._hooks:
            if owner.__dict__.get(name) is not hook:
                raise RuntimeError("Runtime hook ownership changed: " + name)
        for owner, name, had_slot, old_slot, _ in reversed(self._hooks):
            if had_slot:
                owner.__dict__[name] = old_slot
            else:
                del owner.__dict__[name]
        self._hooks.clear()
        self._closed = True
        self._active = False
