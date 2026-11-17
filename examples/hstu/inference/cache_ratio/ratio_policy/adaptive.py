# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Continuous cache allocation from recent eviction/reuse pressure.

This is a policy, not a resize implementation. Feed nonoverlapping completed
windows, plan a proposed split at a batch boundary, then acknowledge the actual
physical allocation. The fixed benchmark does not call this module yet.

Direction uses a local cost heuristic, not before/after latency or a claimed
counterfactual speedup. Queue growth controls urgency. All observation, resize,
eviction and refill costs must remain on the benchmark's original arrival clock.
"""

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Optional


@dataclass(frozen=True)
class CacheStats:
    """Window counters for one cache; missing instrumentation is None.

    Units are KV tokens or embedding lookup rows. evicted_reuse_units counts
    actual misses overlapping a recently evicted valid prefix/row (GhostHistory
    can count these). Invalidation and first-seen cold misses do not qualify.
    miss_work_ms is measured miss-path work, excluding nested/double-counted
    timers. Using the reuse fraction to attribute that work is an approximation;
    it is not an assertion that attention cost is linear in token count.
    Capacity/residency count usable payload only, not allocator metadata,
    reserved virtual addresses, staging storage or an unallocated budget share.
    """

    capacity_bytes: int
    resident_bytes: int
    miss_units: int
    evicted_reuse_units: Optional[int]
    miss_work_ms: Optional[float]

    def validate(self):
        for value in (self.capacity_bytes, self.resident_bytes, self.miss_units):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid cache count")
        if self.resident_bytes > self.capacity_bytes:
            raise ValueError("Resident payload exceeds cache capacity")
        if self.evicted_reuse_units is not None and (
            type(self.evicted_reuse_units) is not int
            or not 0 <= self.evicted_reuse_units <= self.miss_units
        ):
            raise ValueError("Evicted reuse must be a subset of actual misses")
        if self.miss_work_ms is not None and (
            type(self.miss_work_ms) not in (int, float)
            or not isfinite(self.miss_work_ms)
            or self.miss_work_ms < 0
        ):
            raise ValueError("Invalid miss-path work")

    def reuse_work_ms(self):
        if self.evicted_reuse_units is None or self.miss_work_ms is None:
            return None
        return self.miss_work_ms * self.evicted_reuse_units / max(1, self.miss_units)


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    completed: int
    arrivals: int
    mean_request_tokens: float  # Offered logical input, before KV reuse/batching.
    queue_at_start: int
    queue_at_end: int
    kv: CacheStats
    embedding: CacheStats
    resize_cost_ms: Optional[float]  # Planner/calibration estimate, not default zero.
    payload_budget_bytes: Optional[int] = None  # Fixed ceiling, including slack.

    @property
    def budget_bytes(self):
        return (
            self.kv.capacity_bytes + self.embedding.capacity_bytes
            if self.payload_budget_bytes is None
            else self.payload_budget_bytes
        )

    @property
    def unused_payload_bytes(self):
        return (
            self.budget_bytes - self.kv.capacity_bytes - self.embedding.capacity_bytes
        )

    def throughput(self):
        """Observed rates, not a prediction or a service-capacity estimate.

        input_limited only flags empty queue boundaries with no completion
        deficit. Transient queues within the window can still have occurred.
        """
        duration = self.end_s - self.start_s
        return {
            "completion_rate_rps": self.completed / duration,
            "arrival_rate_rps": self.arrivals / duration,
            "completion_deficit_rps": max(0, self.arrivals - self.completed) / duration,
            "input_limited": self.queue_at_start == self.queue_at_end == 0
            and self.completed >= self.arrivals,
        }

    def validate(self):
        for value in (self.start_s, self.end_s, self.mean_request_tokens):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError("Invalid window time/input size")
        if self.end_s <= self.start_s:
            raise ValueError("Window must have positive duration")
        for value in (
            self.completed,
            self.arrivals,
            self.queue_at_start,
            self.queue_at_end,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid window count")
        if self.resize_cost_ms is not None and (
            type(self.resize_cost_ms) not in (int, float)
            or not isfinite(self.resize_cost_ms)
            or self.resize_cost_ms < 0
        ):
            raise ValueError("Invalid resize cost estimate")
        self.kv.validate()
        self.embedding.validate()
        if type(self.budget_bytes) is not int or self.budget_bytes <= 0:
            raise ValueError("Shared payload budget must be a positive integer")
        if self.unused_payload_bytes < 0:
            raise ValueError("Actual cache payload exceeds its fixed budget")


@dataclass(frozen=True)
class Config:
    # Research starting values, not measured optimal thresholds or an SLA.
    min_ratio: float = 0.0
    max_ratio: float = 1.0
    initial_step: float = 0.025
    min_step: float = 0.005
    max_step: float = 0.10
    min_interval_s: float = 2.0
    max_interval_s: float = 30.0
    fill_timeout_s: float = 30.0
    fullness: float = 0.85
    fill_fraction: float = 0.80
    deadband: float = 0.15
    ewma_alpha: float = 0.5
    drift_factor: float = 1.5
    min_reuse_units: int = 16
    cost_margin: float = 2.0
    resize_cost_mode: str = "amortized"

    def __post_init__(self):
        if self.resize_cost_mode not in ("amortized", "diagnostic"):
            raise ValueError("Invalid resize cost mode")
        for name, value in vars(self).items():
            if name == "resize_cost_mode":
                continue
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError("Invalid policy setting: " + name)
        if not 0 <= self.min_ratio < self.max_ratio <= 1:
            raise ValueError("Invalid backend-supported ratio bounds")
        if not 0 < self.min_step <= self.initial_step <= self.max_step <= 1:
            raise ValueError("Invalid step sizes")
        if not 0 < self.min_interval_s <= self.max_interval_s <= self.fill_timeout_s:
            raise ValueError("Invalid observation/fill intervals")
        if not all(
            0 < x <= 1 for x in (self.fullness, self.fill_fraction, self.ewma_alpha)
        ):
            raise ValueError("Invalid fractions")
        if not 0 <= self.deadband < 1 or self.drift_factor <= 1 or self.cost_margin < 1:
            raise ValueError("Invalid pressure/drift/cost settings")
        if type(self.min_reuse_units) is not int or self.min_reuse_units < 1:
            raise ValueError("Invalid reuse sample threshold")


@dataclass(frozen=True)
class Decision:
    target_ratio: Optional[float]  # None means HOLD; a value is only a proposal.
    reason: str
    next_check_s: float
    pressure_balance: float = 0.0
    estimated_saved_work_ms: float = 0.0  # Local heuristic, NOT queue/E2E benefit.
    resize_cost_mode: str = "amortized"
    completion_rate_rps: Optional[float] = None
    arrival_rate_rps: Optional[float] = None
    completion_deficit_rps: Optional[float] = None
    input_limited: Optional[bool] = None  # Window-boundary hint, not proven capacity.


@dataclass(frozen=True)
class ResizeReceipt:
    """Actual usable payload plus separately measured physical peak.

    Unused payload budget is not storage owned by either cache. The physical
    peak includes both caches and any resize staging/metadata in the declared
    envelope; it must not be replaced by reserved virtual-address size.
    Only complete successful operations qualify. Partial mutations fail closed
    in the executor and must not be acknowledged as pre-execution rejections.
    """

    kv_bytes: int
    embedding_bytes: int
    completed_s: float
    action_ms: float
    preserves_survivors: bool
    peak_cache_bytes: int
    peak_limit_bytes: int  # Separate physical envelope, not the payload budget.
    unused_payload_bytes: int = 0


class AdaptiveRatioPolicy:
    """observe -> proposal -> applied/rejected -> observe, with no lifetime trial cap.

    Nonoverlapping small windows are accumulated until the next check. Supply real NVEmb
    observations, not DynamicEmb counters or zeros for unavailable observations.
    A backend must check both physical capacities, retention and transient HBM
    before applying; no fallback to clearing/rebuilding the caches is permitted.
    Ratio endpoints are candidates only if the backend explicitly supports them.
    """

    def __init__(self, config=Config()):
        self.config = config
        self.step = config.initial_step
        self.interval_s = config.min_interval_s
        self.next_check_s = 0.0
        self.ratio = None
        self.total_bytes = None
        self._kv_bytes = None
        self._embedding_bytes = None
        self.unused_payload_bytes = None
        self._window = None
        self.pending = None
        self._last_end = None
        self._last_evaluation = None
        self._context = None
        self._pressure = None
        self._direction = 0
        self._fill = None
        self.proposals = 0
        self.applied_actions = 0
        self.holds = {}
        self._throughput = {}

    def _decision(self, target, reason, check, balance=0.0, saving=0.0,
                  throughput=None):
        return Decision(
            target, reason, check, balance, saving,
            resize_cost_mode=self.config.resize_cost_mode,
            **(self._throughput if throughput is None else throughput),
        )

    def _hold(self, now, reason, backoff=False, balance=0.0, saving=0.0):
        if backoff:
            self.interval_s = min(self.config.max_interval_s, self.interval_s * 1.5)
        self.next_check_s = now + self.interval_s
        self.holds[reason] = self.holds.get(reason, 0) + 1
        return self._decision(None, reason, self.next_check_s, balance, saving)

    def _accumulate(self, w):
        old = self._window
        if old is None:
            self._window = w
            return w

        def cache(a, b):
            reuse = (
                None
                if a.evicted_reuse_units is None or b.evicted_reuse_units is None
                else a.evicted_reuse_units + b.evicted_reuse_units
            )
            work = (
                None
                if a.miss_work_ms is None or b.miss_work_ms is None
                else a.miss_work_ms + b.miss_work_ms
            )
            return CacheStats(
                b.capacity_bytes,
                b.resident_bytes,
                a.miss_units + b.miss_units,
                reuse,
                work,
            )

        arrivals = old.arrivals + w.arrivals
        tokens = (
            (
                old.mean_request_tokens * old.arrivals
                + w.mean_request_tokens * w.arrivals
            )
            / arrivals
            if arrivals
            else w.mean_request_tokens
        )
        self._window = Window(
            old.start_s,
            w.end_s,
            old.completed + w.completed,
            arrivals,
            tokens,
            old.queue_at_start,
            w.queue_at_end,
            cache(old.kv, w.kv),
            cache(old.embedding, w.embedding),
            w.resize_cost_ms,
            w.payload_budget_bytes,
        )
        return self._window

    def observe(self, window):
        window.validate()
        c, w = self.config, window
        total = w.budget_bytes
        ratio = w.kv.capacity_bytes / total
        if self._last_end is not None and w.start_s < self._last_end:
            raise ValueError("Windows must be chronological and nonoverlapping")
        if self.ratio is not None and (
            total != self.total_bytes
            or w.kv.capacity_bytes != self._kv_bytes
            or w.embedding.capacity_bytes != self._embedding_bytes
        ):
            raise ValueError("Capacity changed without an execution receipt")
        if not c.min_ratio <= ratio <= c.max_ratio:
            raise ValueError("Current ratio is outside supported bounds")
        if self.pending is not None:
            return self._decision(None, "awaiting_resize_receipt", self.next_check_s,
                                  throughput=w.throughput())
        if self._last_end is not None and w.start_s > self._last_end:
            # Do not treat an unobserved gap as a period of zero misses/arrivals.
            self._window = None
            self._pressure = self._context = None
        self.ratio, self.total_bytes, self._last_end = ratio, total, w.end_s
        self._kv_bytes = w.kv.capacity_bytes
        self._embedding_bytes = w.embedding.capacity_bytes
        self.unused_payload_bytes = w.unused_payload_bytes
        w = self._accumulate(w)
        self._throughput = w.throughput()
        duration = w.end_s - w.start_s
        context = (w.arrivals / duration, w.mean_request_tokens)
        drift = self._context is not None and any(
            max(a, b) > c.drift_factor * min(a, b)
            for a, b in zip(context, self._context)
        )
        growing = w.queue_at_end > w.queue_at_start or w.arrivals > w.completed
        urgent = growing or drift
        if urgent:
            self.interval_s = c.min_interval_s
        earliest = (self._last_evaluation or w.start_s) + c.min_interval_s
        if w.end_s < earliest or (not urgent and w.end_s < self.next_check_s):
            check = earliest if urgent else max(earliest, self.next_check_s)
            return self._decision(None, "collecting_window", check)
        self._last_evaluation, self._context = w.end_s, context
        self._window = None
        if not w.completed:
            self._pressure = None
            return self._hold(w.end_s, "no_completed_requests")
        work = (w.kv.reuse_work_ms(), w.embedding.reuse_work_ms())
        if any(value is None for value in work):
            self._pressure = None
            return self._hold(w.end_s, "missing_cache_observation")
        if (
            sum((w.kv.evicted_reuse_units, w.embedding.evicted_reuse_units))
            < c.min_reuse_units
        ):
            self._pressure = None
            return self._hold(w.end_s, "insufficient_evicted_reuse", backoff=not urgent)
        # Cost rate per byte. Endpoint denominator uses one minimum move.
        pressure = tuple(
            cost / duration / max(stats.capacity_bytes, total * c.min_step)
            for cost, stats in zip(work, (w.kv, w.embedding))
        )
        if self._pressure is None or drift:
            self._pressure = pressure
        else:
            self._pressure = tuple(
                c.ewma_alpha * a + (1 - c.ewma_alpha) * b
                for a, b in zip(pressure, self._pressure)
            )
        p_kv, p_emb = self._pressure
        balance = (p_kv - p_emb) / (p_kv + p_emb) if p_kv + p_emb else 0.0
        if abs(balance) <= c.deadband:
            return self._hold(
                w.end_s, "balanced_pressure", backoff=not urgent, balance=balance
            )
        direction = 1 if balance > 0 else -1
        receiver = w.kv if direction > 0 else w.embedding
        if self._fill is not None:
            side, target_resident, deadline, timed_out = self._fill
            stats = w.kv if side > 0 else w.embedding
            if stats.resident_bytes >= target_resident:
                self._fill = None
            elif direction == side:
                if w.end_s >= deadline and not timed_out:
                    self.step = max(c.min_step, self.step / 2)
                    self._fill = (side, target_resident, deadline, True)
                reason = (
                    "growth_not_used" if w.end_s >= deadline else "waiting_for_fill"
                )
                return self._hold(w.end_s, reason, backoff=not urgent, balance=balance)
        if (
            receiver.capacity_bytes
            and receiver.resident_bytes < c.fullness * receiver.capacity_bytes
        ):
            return self._hold(
                w.end_s,
                "receiver_has_free_capacity",
                backoff=not urgent,
                balance=balance,
            )
        if c.resize_cost_mode == "amortized" and w.resize_cost_ms is None:
            return self._hold(w.end_s, "missing_resize_cost", balance=balance)
        step = self.step
        if self._direction:
            step = (
                min(c.max_step, step * 1.25)
                if direction == self._direction
                else max(c.min_step, step / 2)
            )
        delta = max(c.min_step, step * abs(balance))
        target = min(c.max_ratio, max(c.min_ratio, ratio + direction * delta))
        transfer = abs(target - ratio) * total
        if transfer < 1:
            return self._hold(
                w.end_s, "capacity_bound", backoff=not urgent, balance=balance
            )
        # Amortize over one observed window, never an invented long future phase.
        saving = abs(p_kv - p_emb) * transfer * duration
        # Diagnostic mode tests the same evidence-based direction while keeping
        # real resize pauses in measured performance. Missing cost stays None.
        if (c.resize_cost_mode == "amortized"
                and saving <= c.cost_margin * w.resize_cost_ms):
            return self._hold(
                w.end_s,
                "resize_cost_dominates",
                backoff=not urgent,
                balance=balance,
                saving=saving,
            )
        self.step = step
        self.next_check_s = w.end_s + c.min_interval_s
        decision = self._decision(
            target, "eviction_reuse_pressure", self.next_check_s, balance, saving
        )
        self.pending = (decision, w)
        self.proposals += 1
        return decision

    def applied(self, receipt):
        """Record a completed physical move; this method performs no allocation.

        The executor must report actual allocator-rounded sizes, measured pause,
        and a conservative managed-device-pool peak bound. A receipt is a
        caller's assertion, not numerical validation or a total HBM measurement.
        """
        if self.pending is None:
            raise ValueError("No pending proposal")
        decision, window = self.pending
        c = self.config
        for value in (
            receipt.kv_bytes,
            receipt.embedding_bytes,
            receipt.peak_cache_bytes,
            receipt.peak_limit_bytes,
            receipt.unused_payload_bytes,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid physical allocation receipt")
        for value in (receipt.completed_s, receipt.action_ms):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError("Invalid action timing")
        if (
            receipt.completed_s < window.end_s
            or receipt.action_ms > (receipt.completed_s - window.end_s) * 1000 + 1e-6
        ):
            raise ValueError("Action is outside receipt time interval")
        if receipt.preserves_survivors is not True:
            raise ValueError("Cache rebuild is not a preserving move")
        actual_payload = receipt.kv_bytes + receipt.embedding_bytes
        if actual_payload + receipt.unused_payload_bytes != self.total_bytes:
            raise ValueError("Shared payload budget changed")
        before_payload = self._kv_bytes + self._embedding_bytes
        if (
            not max(before_payload, actual_payload)
            <= receipt.peak_cache_bytes
            <= receipt.peak_limit_bytes
        ):
            raise ValueError("Invalid physical peak/envelope")
        ratio = receipt.kv_bytes / self.total_bytes
        intended = decision.target_ratio * self.total_bytes - self._kv_bytes
        actual = receipt.kv_bytes - self._kv_bytes
        embedding_change = receipt.embedding_bytes - self._embedding_bytes
        direction = 1 if intended > 0 else -1
        grown = actual if direction > 0 else embedding_change
        changed = actual != 0 or embedding_change != 0
        if (
            not c.min_ratio <= ratio <= c.max_ratio
            or actual * intended < 0
            or abs(actual) > abs(intended) + 1
            or embedding_change * intended > 0
            or (direction < 0 and grown > abs(intended) + 1)
            or (changed and grown <= 0)
        ):
            raise ValueError("Rounded allocation is outside the proposed move")
        if changed:
            before = window.kv if direction > 0 else window.embedding
            target_resident = before.resident_bytes + min(
                grown, ceil(c.fill_fraction * grown)
            )
            self._fill = (
                direction,
                target_resident,
                receipt.completed_s + c.fill_timeout_s,
                False,
            )
            self._direction = direction
            self.applied_actions += 1
        else:
            # A rounded no-op is not a trial or a bad direction. Try a larger move.
            self.step = min(c.max_step, self.step * 2)
        self.ratio, self.pending = ratio, None
        self._kv_bytes = receipt.kv_bytes
        self._embedding_bytes = receipt.embedding_bytes
        self.unused_payload_bytes = receipt.unused_payload_bytes
        self._window = None
        self._last_evaluation = receipt.completed_s
        self._pressure = None
        self.interval_s = c.min_interval_s
        if c.resize_cost_mode == "amortized":
            self.interval_s = min(
                c.max_interval_s,
                max(c.min_interval_s, c.cost_margin * receipt.action_ms / 1000),
            )
        self.next_check_s = receipt.completed_s + self.interval_s

    def rejected(self, now_s, reason="planner_declined"):
        """Planner declined before any mutation; actual capacities must be unchanged."""
        if (
            self.pending is None
            or type(now_s) not in (int, float)
            or not isfinite(now_s)
            or now_s < self.pending[1].end_s
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("Invalid pre-execution rejection")
        self.pending = None
        self.holds[reason] = self.holds.get(reason, 0) + 1
        self.next_check_s = now_s + self.config.min_interval_s
