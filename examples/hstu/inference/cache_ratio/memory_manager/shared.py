# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Serialize native HSTU/NVEmb forwards and preserving capacity transfers.

The payload ceiling is distinct from the device-pool peak envelope. Unallocated
rounding slack is reported, never counted as an embedding cache. NVEmb plans may
reject unsupported rehashes before either cache changes. After a partial native
failure the model fails closed; a donor shrink is not a successful ratio move.
"""

from contextlib import contextmanager
from dataclasses import asdict
from math import isfinite
from threading import RLock
from time import perf_counter


class SharedCacheMemory:
    def __init__(
        self,
        kv,
        embedding,
        payload_budget_bytes,
        peak_cache_bytes,
        *,
        clock=perf_counter,
    ):
        for value in (payload_budget_bytes, peak_cache_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "Payload and peak cache budgets must be positive integers"
                )
        self.kv, self.embedding = kv, embedding
        self.payload_budget_bytes = payload_budget_bytes
        self.peak_limit_bytes = peak_cache_bytes
        self._clock = clock
        self._lock = RLock()
        self._in_batch = False
        self._state = "open"
        self.last_action = None
        self.actions = 0
        initial = self.stats()
        self._require_open()
        if initial["unused_payload_bytes"] < 0:
            raise ValueError("Actual cache payload exceeds the shared budget")
        if initial["device_pool_bytes"] > peak_cache_bytes:
            raise ValueError("Initial device pools exceed the explicit peak envelope")

    def _require_open(self):
        if self._state != "open":
            raise RuntimeError(f"Shared cache memory is {self._state}; stop model use")

    def stats(self):
        with self._lock:
            return self._stats()

    def _stats(self):
        kv, emb = self.kv.stats(), self.embedding.stats()
        if self._state == "open" and (kv["state"] != "open" or emb["state"] != "ready"):
            self._state = "poisoned"
        used = (
            kv["payload_bytes"] + emb["payload_bytes"]
            if kv["payload_bytes"] is not None and emb["payload_bytes"] is not None
            else None
        )
        pool_bytes = (
            kv["allocated_physical_bytes"] + emb["device_pool_bytes"]
            if kv["allocated_physical_bytes"] is not None
            and emb["device_pool_bytes"] is not None
            else None
        )
        return {
            "state": self._state,
            "kv": kv,
            "embedding": emb,
            "payload_budget_bytes": self.payload_budget_bytes,
            "unused_payload_bytes": self.payload_budget_bytes - used
            if used is not None
            else None,
            "kv_ratio": kv["payload_bytes"] / self.payload_budget_bytes
            if kv["payload_bytes"] is not None
            else None,
            "device_pool_bytes": pool_bytes,
            "peak_limit_bytes": self.peak_limit_bytes,
            "physical_scope": "managed KV backing and NVEmb device pools; excludes model, KV index metadata and allocator reserves",
            "actions": self.actions,
        }

    @contextmanager
    def batch(self):
        with self._lock:
            self._require_open()
            if self._in_batch:
                raise RuntimeError("Nested model forwards are not supported")
            self._in_batch = True
            try:
                with self.kv.batch():
                    yield
            except BaseException:
                self._state = "poisoned"
                raise
            finally:
                self._in_batch = False

    def _plan(self, ratio):
        self._require_open()
        if self._in_batch:
            raise RuntimeError("Resize must run after the completed batch")
        if type(ratio) not in (int, float) or not isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("Native shared cache requires 0 < ratio < 1")
        before = self.stats()
        self._require_open()
        old_kv = before["kv"]["payload_bytes"]
        old_emb = before["embedding"]["payload_bytes"]
        desired_kv = int(ratio * self.payload_budget_bytes)
        growing_kv = desired_kv >= old_kv
        # Quantize toward the current ratio so applying a policy proposal cannot
        # overshoot its requested move by a physical allocation chunk.
        kp = self.kv.plan_resize_bytes(desired_kv, round_up=not growing_kv)
        transfer_limit = abs(desired_kv - old_kv)
        emb_budget = min(
            self.payload_budget_bytes - kp["payload_bytes"],
            old_emb if growing_kv else old_emb + transfer_limit,
        )
        ep = self.embedding.plan_resize(emb_budget)
        new_kv, new_emb = kp["payload_bytes"], ep["payload_bytes"]
        if new_kv + new_emb > self.payload_budget_bytes:
            raise ValueError("Planned native payload exceeds the shared budget")
        if growing_kv:
            if not old_kv <= new_kv <= desired_kv or new_emb > old_emb:
                raise ValueError("Native quantization changes the intended direction")
        elif not desired_kv <= new_kv <= old_kv or new_emb < old_emb:
            raise ValueError("Native quantization changes the intended direction")
        delta_kv, delta_emb = new_kv - old_kv, new_emb - old_emb
        if (growing_kv and delta_emb < 0 and delta_kv == 0) or (
            not growing_kv and delta_kv < 0 and delta_emb == 0
        ):
            raise ValueError("Quantized move would only shrink the donor")
        steps = [("kv", kp, delta_kv), ("embedding", ep, delta_emb)]
        steps.sort(key=lambda step: step[2])  # Release the donor before growing.
        pools = {
            "kv": before["kv"]["allocated_physical_bytes"],
            "embedding": before["embedding"]["device_pool_bytes"],
        }
        peak = sum(pools.values())
        for name, plan, delta in steps:
            if not delta:
                continue
            target_pool = (
                plan["allocated_physical_bytes"]
                if name == "kv"
                else plan["device_pool_bytes"]
            )
            peak = max(
                peak,
                sum(pools.values()) + plan["extra_peak_bytes"],
                sum(pools.values()) - pools[name] + target_pool,
            )
            pools[name] = target_pool
        if peak > self.peak_limit_bytes:
            raise ValueError(
                "Resize plan exceeds the managed-device-pool peak envelope"
            )
        return before, steps, peak

    def resize_ratio(self, ratio, *, at_batch_barrier):
        if at_batch_barrier is not True:
            raise ValueError("An explicit completed-batch boundary is required")
        with self._lock, self.kv.resize_guard():
            started = self._clock()
            before, steps, peak = self._plan(ratio)
            completed = []
            for name, plan, delta in steps:
                if not delta:
                    continue
                # No rollback claim: after donor eviction or release, a failure
                # must stop forwards even if the second cache never changed.
                self._state = "poisoned"
                try:
                    if name == "kv":
                        result = self.kv.resize_pages(
                            plan["target_pages"], at_batch_barrier=True
                        )
                    else:
                        kv_pool = self.kv.stats()["allocated_physical_bytes"]
                        result = self.embedding.resize(
                            plan,
                            at_batch_barrier=True,
                            device_pool_limit_bytes=self.peak_limit_bytes - kv_pool,
                        )
                    if result.get("preserves_survivors") is not True:
                        raise RuntimeError(
                            "Native resize did not acknowledge survivor preservation"
                        )
                    completed.append({"cache": name, "result": result})
                except BaseException:
                    self.last_action = {
                        "status": "failed",
                        "target_ratio": ratio,
                        "completed": completed,
                        "before": before,
                    }
                    raise
            after = self.stats()
            expected = {name: plan["payload_bytes"] for name, plan, _ in steps}
            if any(after[name]["payload_bytes"] != expected[name] for name in expected):
                self._state = "poisoned"
                raise RuntimeError("Native allocation differs from the preflight plan")
            for name, plan, _ in steps:
                field = (
                    "allocated_physical_bytes" if name == "kv" else "device_pool_bytes"
                )
                if after[name][field] != plan[field]:
                    self._state = "poisoned"
                    raise RuntimeError(
                        "Native device pool differs from the preflight plan"
                    )
            if (
                after["unused_payload_bytes"] < 0
                or after["device_pool_bytes"] > self.peak_limit_bytes
            ):
                self._state = "poisoned"
                raise RuntimeError("Native allocation exceeded the shared envelope")
            self._state = "open"
            self.actions += bool(completed)
            finished = self._clock()
            after = self.stats()
            self.last_action = {
                "status": "applied" if completed else "rounded_noop",
                "target_ratio": ratio,
                "before": before,
                "after": after,
                "completed": completed,
                "action_ms": (finished - started) * 1000,
                "peak_cache_bytes_upper_bound": peak,
                "preserves_survivors": True,
            }
            return self.last_action

    def observe(self, policy, window, *, replay_clock):
        """Consume measured native counters supplied by the replay owner.

        Missing miss/reuse/timing instrumentation stays None in Window and makes
        the policy HOLD. This bridge never invents observations from capacity or
        from unmatched pre/post latency. replay_clock uses the arrival origin.
        """
        from ..ratio_policy.adaptive import ResizeReceipt

        with self._lock:
            self._require_open()
            actual = self.stats()
            if (
                window.kv.capacity_bytes != actual["kv"]["payload_bytes"]
                or window.embedding.capacity_bytes
                != actual["embedding"]["payload_bytes"]
                or window.budget_bytes != self.payload_budget_bytes
            ):
                raise ValueError(
                    "Observation does not match the current allocation ledger"
                )
            decision = policy.observe(window)
            if decision.target_ratio is None:
                return {"decision": asdict(decision), "action": None}
            try:
                result = self.resize_ratio(decision.target_ratio, at_batch_barrier=True)
            except Exception:
                if self._state == "open":
                    policy.rejected(replay_clock(), "native_preflight_declined")
                raise
            after = result["after"]
            receipt = ResizeReceipt(
                kv_bytes=after["kv"]["payload_bytes"],
                embedding_bytes=after["embedding"]["payload_bytes"],
                unused_payload_bytes=after["unused_payload_bytes"],
                completed_s=replay_clock(),
                action_ms=result["action_ms"],
                preserves_survivors=True,
                peak_cache_bytes=result["peak_cache_bytes_upper_bound"],
                peak_limit_bytes=self.peak_limit_bytes,
            )
            try:
                policy.applied(receipt)
            except BaseException:
                self._state = "poisoned"
                raise
            return {"decision": asdict(decision), "action": result}

    def close(self):
        with self._lock:
            if self._in_batch:
                raise RuntimeError("Cannot close during a model forward")
            if self._state == "closed":
                return
            self._state = "poisoned"
            self.kv.close(at_batch_barrier=True)
            self._state = "closed"
