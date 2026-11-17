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
from math import isfinite, lcm
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
        try:
            return self._validate_plan(before, kp, ep, desired_kv)
        except BaseException:
            self._cancel_embedding_plan(ep)
            raise

    def _cancel_embedding_plan(self, plan):
        cancel = getattr(self.embedding, "cancel_resize", None)
        if callable(cancel):
            try:
                cancel(plan)
            except BaseException:
                self._state = "poisoned"
                raise

    def _validate_plan(self, before, kp, ep, desired_kv):
        old_kv = before["kv"]["payload_bytes"]
        old_emb = before["embedding"]["payload_bytes"]
        growing_kv = desired_kv >= old_kv
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

    def transfer_quantum_bytes(self):
        """Smallest equal KV/embedding transfer for the current VMM geometry.

        KV allocates a physical chunk in every layer. For example, three
        layers of 64 MiB and 64 MiB embedding chunks require 192 MiB moves.
        """
        with self._lock:
            self._require_open()
            state = self.stats()
            if state["kv"].get("backend") != "gpu_only_vmm":
                raise ValueError("Byte transfers require the whole-chunk KV VMM backend")
            geometry = self.kv.plan_pages(self.kv.minimum_pages)
            kv_quantum = geometry["chunk_pages"] * geometry["page_payload_bytes"]
            tables = state["embedding"].get("tables", {})
            chunks = {table.get("chunk_bytes") for table in tables.values()}
            if (not tables or not all(table.get("vmm") for table in tables.values())
                    or len(chunks) != 1):
                raise ValueError("Byte transfers require common whole-chunk NVEmb VMM pools")
            emb_quantum = chunks.pop()
            if type(emb_quantum) is not int or emb_quantum <= 0:
                raise ValueError("Invalid native embedding chunk geometry")
            return lcm(kv_quantum, emb_quantum)

    def resize_by_bytes(self, kv_delta_bytes, *, at_batch_barrier):
        """Request a byte transfer relative to the current actual allocation.

        Positive grows KV; negative grows embedding. Round toward the current
        allocation without overshooting the requested target; transfer equal
        amounts and preserve existing budget slack. Report any unapplied bytes
        so callers can retain the target. Sub-quantum requests do not prepare or
        mutate caches.
        """
        if at_batch_barrier is not True:
            raise ValueError("An explicit completed-batch boundary is required")
        if type(kv_delta_bytes) is not int:
            raise ValueError("kv_delta_bytes must be an integer")
        with self._lock, self.kv.resize_guard():
            started = self._clock()
            self._require_open()
            if self._in_batch:
                raise RuntimeError("Resize must run after the completed batch")
            before = self.stats()
            self._require_open()
            quantum = self.transfer_quantum_bytes()
            requested_kv = before["kv"]["payload_bytes"] + kv_delta_bytes
            if not 0 < requested_kv < self.payload_budget_bytes:
                raise ValueError("Byte transfer exceeds the shared payload budget")
            actual_delta = (abs(kv_delta_bytes) // quantum) * quantum
            if kv_delta_bytes < 0:
                actual_delta = -actual_delta
            request = {
                "kv_delta_bytes": kv_delta_bytes,
                "requested_delta_bytes": kv_delta_bytes,
                "actual_delta_bytes": actual_delta,
                "unapplied_delta_bytes": kv_delta_bytes - actual_delta,
                "transfer_quantum_bytes": quantum,
                "target_ratio": requested_kv / self.payload_budget_bytes,
            }
            if actual_delta == 0:
                finished = self._clock()
                self.last_action = {
                    "status": "rounded_noop", **request,
                    "reason": "zero_request" if kv_delta_bytes == 0 else "below_transfer_quantum",
                    "before": before, "after": before, "completed": [],
                    "action_ms": (finished - started) * 1000,
                    "prepare_ms": (finished - started) * 1000, "apply_ms": 0.0,
                    "embedding_resize_mode": before["embedding"].get("resize_mode", "legacy_replanned"),
                    "peak_cache_bytes_upper_bound": before["device_pool_bytes"],
                    "preserves_survivors": True,
                }
                return self.last_action
            desired_kv = before["kv"]["payload_bytes"] + actual_delta
            desired_emb = before["embedding"]["payload_bytes"] - actual_delta
            if desired_emb <= 0:
                raise ValueError("Byte transfer exceeds the available embedding donor payload")
            kp = self.kv.plan_resize_bytes(desired_kv)
            if kp["payload_bytes"] != desired_kv:
                raise ValueError("KV allocation does not support the exact byte transfer")
            ep = self.embedding.plan_resize(desired_emb)
            try:
                if ep["payload_bytes"] != desired_emb:
                    raise ValueError("NVEmb allocation does not support the exact byte transfer")
                planned = self._validate_plan(before, kp, ep, desired_kv)
            except BaseException:
                self._cancel_embedding_plan(ep)
                raise
            return self._apply_plan(
                *planned, started=started, **request,
            )

    def resize_ratio(self, ratio, *, at_batch_barrier):
        if at_batch_barrier is not True:
            raise ValueError("An explicit completed-batch boundary is required")
        with self._lock, self.kv.resize_guard():
            started = self._clock()
            return self._apply_plan(*self._plan(ratio), started=started, target_ratio=ratio)

    def _apply_plan(self, before, steps, peak, *, started, **request):
        prepared_at = self._clock()
        ep = next(plan for name, plan, _ in steps if name == "embedding")
        completed = []
        try:
            for name, plan, delta in steps:
                if not delta:
                    continue
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
            # No-op plans still hold native preparation tokens; release them
            # before allowing the next forward, including rounded ratio moves.
            self._cancel_embedding_plan(ep)
            after = self.stats()
            expected = {name: plan["payload_bytes"] for name, plan, _ in steps}
            if any(after[name]["payload_bytes"] != expected[name] for name in expected):
                raise RuntimeError("Native allocation differs from the preflight plan")
            for name, plan, _ in steps:
                field = (
                    "allocated_physical_bytes" if name == "kv" else "device_pool_bytes"
                )
                if after[name][field] != plan[field]:
                    raise RuntimeError("Native device pool differs from the preflight plan")
            if (after["state"] != "open" or after["unused_payload_bytes"] < 0
                    or after["device_pool_bytes"] > self.peak_limit_bytes):
                raise RuntimeError("Native allocation exceeded the shared envelope")
        except BaseException as error:
            # Cancel remaining preparations before inspecting/releasing the
            # runtime. A committed donor cannot be restored by cancellation.
            try:
                self._cancel_embedding_plan(ep)
            except BaseException as cancel_error:
                self._state = "poisoned"
                self.last_action = {"status": "failed", **request,
                                    "completed": completed, "before": before,
                                    "cancel_error": str(cancel_error)}
                raise cancel_error from error
            unchanged = getattr(error, "receipt", {}).get("unchanged") is True
            if completed or not unchanged or self.stats() != before:
                self._state = "poisoned"
            self.last_action = {"status": "rejected" if self._state == "open" else "failed",
                                **request, "completed": completed, "before": before}
            raise
        self.actions += bool(completed)
        finished = self._clock()
        self.last_action = {
            "status": "applied" if completed else "rounded_noop",
            **request,
            "before": before,
            "after": self.stats(),
            "completed": completed,
            "action_ms": (finished - started) * 1000,
            "prepare_ms": (prepared_at - started) * 1000,
            "apply_ms": (finished - prepared_at) * 1000,
            "embedding_resize_mode": ep.get("resize_mode", "legacy_replanned"),
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
            resize_started_s = replay_clock()
            try:
                result = self.resize_ratio(decision.target_ratio, at_batch_barrier=True)
            except Exception as error:
                after = self.stats()
                # A rejection is safe only after verifying the native ledger,
                # including per-table generations, did not change at all.
                if after["state"] == "open" and after == actual:
                    resize_completed_s = replay_clock()
                    policy.rejected(resize_completed_s, "native_preflight_declined")
                    return {"decision": asdict(decision), "action": None,
                            "resize_interval": {"start_s": resize_started_s, "end_s": resize_completed_s,
                                                "scope": "resize_preflight_declined"},
                            "rejection": {"reason": "native_preflight_declined",
                                          "error": str(error), "unchanged": True}}
                self._state = "poisoned"
                raise
            after = result["after"]
            resize_completed_s = replay_clock()
            receipt = ResizeReceipt(
                kv_bytes=after["kv"]["payload_bytes"],
                embedding_bytes=after["embedding"]["payload_bytes"],
                unused_payload_bytes=after["unused_payload_bytes"],
                completed_s=resize_completed_s,
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
            return {"decision": asdict(decision), "action": result,
                    "resize_interval": {"start_s": resize_started_s, "end_s": resize_completed_s,
                                        "scope": "shared_resize_including_prepare_and_apply"}}

    def policy_ratio_bounds(self):
        """Positive KV-batch and NVEmb-chunk bounds from actual reservations."""
        with self._lock:
            self._require_open()
            minimum = self.kv.plan_pages(self.kv.minimum_pages)
            maximum = self.kv.plan_pages(minimum["max_requested_pages"])
            emb = self.embedding.payload_bounds()
            budget = self.payload_budget_bytes
            low = max(minimum["capacity_payload_bytes"], budget - emb["maximum_payload_bytes"])
            high = min(maximum["capacity_payload_bytes"], budget - emb["minimum_payload_bytes"])
            if not 0 < low < high < budget:
                raise ValueError("No positive feasible policy ratio interval")
            return {"min_ratio": low / budget, "max_ratio": high / budget,
                    "minimum_kv_bytes": minimum["capacity_payload_bytes"],
                    "maximum_kv_bytes": maximum["capacity_payload_bytes"], **emb}

    def close(self):
        with self._lock:
            if self._in_batch:
                raise RuntimeError("Cannot close during a model forward")
            if self._state == "closed":
                return
            self._state = "poisoned"
            self.kv.close(at_batch_barrier=True)
            self._state = "closed"
