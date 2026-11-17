# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect native NVEmbedding hierarchical caches to the shared ratio manager.

Requires the accompanying NVEmb SDK patch and an exclusive completed-batch
barrier. This adapter never substitutes erase(), clear(), or a cold new layer
for physical resizing. The native default uses bounded CPU staging and raw
rehash. The opt-in VMM backend keeps NVEmb's lookup/modify kernels and LFU
metadata, and supports factor-two resizing with compatible source bucket ages.
Inspect each binding's capabilities/state to identify the active backend.
"""

import json
import time


class NvEmbResizeError(RuntimeError):
    """A multi-table resize failed; receipt describes the partial state."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class NvEmbMemoryManager:
    def __init__(self, collection, *, host_staging_limit_bytes=512 * 1024**2):
        self.require_capability(collection)
        if type(host_staging_limit_bytes) is not int or host_staging_limit_bytes <= 0:
            raise ValueError("host_staging_limit_bytes must be a positive integer")
        self.collection = collection
        self.host_staging_limit_bytes = host_staging_limit_bytes
        self._bindings = {
            name: embedding.emb_layer
            for name, embedding in collection.embeddings.items()
        }
        configs = {config.name: config for config in collection.embedding_configs()}
        self._vocab_sizes = {name: configs[name].num_embeddings for name in self._bindings}
        self._weights = {
            name: configs[name].num_embeddings * configs[name].embedding_dim
            for name in self._bindings
        }
        self._poisoned = False
        self.last_receipt = None
        self._prepared_support = {}
        for name, binding in self._bindings.items():
            caps = self._decode(binding.cache_resize_capabilities())
            advertised = caps.get("prepared_resize") is True
            methods = ("prepare_cache_resize", "commit_cache_resize", "cancel_cache_resize")
            supported = (advertised and caps.get("prepared_resize_version") == 1
                         and all(callable(getattr(binding, method, None)) for method in methods))
            if advertised and not supported:
                raise RuntimeError(f"NVEmb table {name!r} advertises an incomplete prepared contract")
            self._prepared_support[name] = supported
        self._pending_plan = None
        self._prepared_tokens = {}

    @property
    def supports_prepared_resize(self):
        """Whether every loaded table supports reusable native preparation."""
        return all(self._prepared_support.values())

    @property
    def resize_mode(self):
        if self.supports_prepared_resize:
            return "prepared"
        return "mixed" if any(self._prepared_support.values()) else "legacy_replanned"

    @staticmethod
    def _decode(value):
        return json.loads(value) if isinstance(value, str) else value

    @classmethod
    def require_capability(cls, collection):
        """Check actual loaded bindings before allocating the KV VMM pool."""
        if not collection.embeddings:
            raise ValueError("Dynamic NVEmb collection must contain at least one table")
        methods = (
            "cache_resize_capabilities",
            "cache_resize_state",
            "plan_cache_resize",
            "apply_cache_resize",
        )
        for name, embedding in collection.embeddings.items():
            binding = getattr(embedding, "emb_layer", None)
            if binding is None or any(
                not callable(getattr(binding, m, None)) for m in methods
            ):
                raise RuntimeError(
                    f"NVEmb table {name!r} lacks native preserving resize; build the "
                    "SDK with memory_manager/native/nvemb_preserving_resize.patch"
                )
            capabilities = cls._decode(binding.cache_resize_capabilities())
            if capabilities.get("version") != 1 or not all(
                capabilities.get(field) is True
                for field in (
                    "preserves_values",
                    "preserves_counter_bits",
                    "preserves_pending_inserts",
                )
            ):
                raise RuntimeError(
                    f"NVEmb table {name!r} has an incompatible resize contract"
                )
            state = cls._decode(binding.cache_resize_state())
            if state.get("state") != "ready":
                raise RuntimeError(f"NVEmb table {name!r} is not usable")

    def stats(self):
        tables = {
            name: self._decode(binding.cache_resize_state())
            for name, binding in self._bindings.items()
        }
        unknown = self._poisoned or any(
            s.get("state") != "ready" for s in tables.values()
        )
        result = {"tables": tables, "state": "poisoned" if unknown else "ready",
                  "supports_prepared_resize": self.supports_prepared_resize,
                  "resize_mode": self.resize_mode}
        for field in ("payload_bytes", "device_pool_bytes", "configured_bytes"):
            values = [s.get(field) for s in tables.values()]
            result[field] = None if any(v is None for v in values) else sum(values)
        result[
            "budget_scope"
        ] = "cache_pools_excludes_model_execution_workspaces_and_handler_buffers"
        return result

    def policy_observation(self, *, reset=True):
        """Drain native observations; unavailable or overflowed data stays incomplete.

        Native calls drain their contexts. Their synchronization and JSON/key
        transfer are paid by the replay owner on its unchanged arrival clock.
        """
        tables = {}
        for name, binding in self._bindings.items():
            method = getattr(binding, "cache_policy_observation", None)
            tables[name] = (
                self._decode(method(reset))
                if callable(method)
                else {"enabled": False, "complete": False, "reason": "missing_native_observation"}
            )
        return {"tables": tables}

    def payload_bounds(self):
        """Reservation bounds, not a promise that every interior plan succeeds."""
        states = self.stats()["tables"]
        minimum = maximum = 0
        for state in states.values():
            unit = state["ways"] * state["row_bytes"]
            if not state.get("vmm") or not state.get("reserved_value_bytes"):
                raise ValueError("Policy bounds require explicit NVEmb VMM reservations")
            minimum += max(unit, state["chunk_bytes"])
            maximum += state["reserved_value_bytes"]
        return {"minimum_payload_bytes": minimum, "maximum_payload_bytes": maximum}

    def plan_resize(self, payload_budget_bytes):
        """Prepare every table before the shared manager changes either cache.

        Prepared bindings keep their native staging snapshot and reject forwards
        until commit/cancel. Legacy bindings explicitly report replanning mode.
        Preparation synchronization and validation are part of resize overhead.
        """
        if type(payload_budget_bytes) is not int or payload_budget_bytes <= 0:
            raise ValueError("payload_budget_bytes must be a positive integer")
        if self._pending_plan is not None:
            raise RuntimeError("Commit or cancel the outstanding NVEmb resize plan first")
        before = self.stats()
        if before["state"] != "ready":
            raise RuntimeError("NVEmb manager is poisoned; recreate the runtime")
        quantum = {
            name: state["ways"] * state["row_bytes"]
            for name, state in before["tables"].items()
        }
        remaining = payload_budget_bytes - sum(quantum.values())
        if remaining < 0:
            raise ValueError("Embedding payload must hold one native set per table")
        total_weight = sum(self._weights.values())
        targets = {}
        vmm = [state.get("vmm", False) for state in before["tables"].values()]
        if any(vmm):
            chunks = {state.get("chunk_bytes") for state in before["tables"].values()}
            if not all(vmm) or len(chunks) != 1:
                raise ValueError("Shared VMM planning requires one common chunk size")
            chunk = chunks.pop()
            if type(chunk) is not int or chunk <= 0 or any(chunk % q for q in quantum.values()):
                raise ValueError("VMM chunk must contain whole native sets")
            count = payload_budget_bytes // chunk
            if count < len(quantum):
                raise ValueError("Embedding payload must hold one VMM chunk per table")
            remaining_chunks = count - len(quantum)
            weights = {name: self._vocab_sizes[name] * before["tables"][name]["row_bytes"]
                       for name in self._bindings}
            vmm_weight = sum(weights.values())
            shares = {name: divmod(remaining_chunks * weights[name], vmm_weight)
                      for name in self._bindings}
            spare = remaining_chunks - sum(value[0] for value in shares.values())
            # Stable sorting gives configuration/binding order for exact ties.
            winners = set(sorted(shares, key=lambda name: -shares[name][1])[:spare])
            targets = {name: (1 + shares[name][0] + (name in winners)) * chunk
                       for name in self._bindings}
        plans = {}
        tokens = {}
        started = time.perf_counter()
        host_peak = 0
        try:
            for name, binding in self._bindings.items():
                unit = quantum[name]
                target = targets.get(name) or (
                    unit + remaining * self._weights[name] // (total_weight * unit) * unit
                )
                retained = sum(p.get("prepared_host_bytes", 0) for p in plans.values())
                available_host = self.host_staging_limit_bytes - retained
                prepared = self._prepared_support[name]
                method = binding.prepare_cache_resize if prepared else binding.plan_cache_resize
                native = self._decode(method(target, available_host))
                if native.get("status") != ("prepared" if prepared else "planned"):
                    raise ValueError(
                        f"NVEmb resize rejected for {name!r}: {native.get('reason', native)}"
                    )
                if prepared:
                    token = native["token"]
                    if not isinstance(token, str) or not token:
                        raise RuntimeError("Native preparation returned an invalid one-use token")
                    tokens[name] = token
                native["requested_payload_bytes"] = target
                native["extra_peak_bytes"] = native.get("extra_peak_bytes", 0)
                required = ["payload_bytes", "device_pool_bytes", "generation", "extra_peak_bytes"]
                if prepared:
                    required += ["prepared_host_bytes", "host_staging_peak_bound_bytes",
                                 "commit_host_staging_peak_bound_bytes"]
                if any(type(native.get(key)) is not int or native[key] < 0 for key in required):
                    raise RuntimeError(f"Invalid native resize accounting for {name!r}")
                if not isinstance(native.get("before"), dict) or any(
                    native["before"].get(key) != before["tables"][name][key]
                    for key in ("generation", "payload_bytes", "device_pool_bytes")
                ):
                    raise RuntimeError(f"Invalid native resize snapshot for {name!r}")
                if prepared and (
                    native["prepared_host_bytes"] > native["host_staging_peak_bound_bytes"]
                    or native["host_staging_peak_bound_bytes"] > available_host
                ):
                    raise ValueError("Prepared NVEmb snapshots exceed the global host staging budget")
                host_peak = max(host_peak, retained + native.get(
                    "host_staging_peak_bound_bytes", available_host))
                plans[name] = native
            # All snapshots coexist. Check the same conservative native commit
            # bounds, in commit order, before a KV/embedding donor can change.
            pending_host = sum(p.get("prepared_host_bytes", 0) for p in plans.values())
            order = sorted(plans, key=lambda name: plans[name]["device_pool_bytes"]
                           - before["tables"][name]["device_pool_bytes"])
            for name in order:
                native = plans[name]
                own_host = native.get("prepared_host_bytes", 0)
                bound = native.get("commit_host_staging_peak_bound_bytes", native.get(
                    "host_staging_peak_bound_bytes", self.host_staging_limit_bytes))
                required_host = pending_host - own_host + bound
                if required_host > self.host_staging_limit_bytes:
                    raise ValueError("NVEmb commit exceeds the global host staging budget")
                host_peak = max(host_peak, required_host)
                pending_host -= own_host
            prepare_ms = 1000 * (time.perf_counter() - started)
            plan = {
                "tables": plans,
                "before": before,
                "requested_payload_bytes": payload_budget_bytes,
                "payload_bytes": sum(p["payload_bytes"] for p in plans.values()),
                "device_pool_bytes": sum(p["device_pool_bytes"] for p in plans.values()),
                "extra_peak_bytes": sum(p["extra_peak_bytes"] for p in plans.values()),
                "prepared_host_bytes": sum(p.get("prepared_host_bytes", 0) for p in plans.values()),
                "host_staging_peak_bound_bytes": host_peak,
                "host_staging_limit_bytes": self.host_staging_limit_bytes,
                "plan_ms": prepare_ms,
                "prepare_ms": prepare_ms,
                "resize_mode": self.resize_mode,
                "lifecycle": "planned",
            }
            if tokens:
                self._pending_plan, self._prepared_tokens = plan, tokens
            return plan
        except BaseException:
            self._cancel_tokens(tokens)
            if any(state.get("prepared") for state in self.stats()["tables"].values()):
                self._poisoned = True
            raise

    def _cancel_tokens(self, tokens):
        failures = []
        # Attempt every remaining cancellation, even if an earlier one fails.
        for name, token in list(tokens.items()):
            try:
                result = self._decode(self._bindings[name].cancel_cache_resize(token))
                if result.get("status") != "cancelled":
                    raise RuntimeError(f"Native NVEmb cancellation rejected: {result}")
                del tokens[name]
            except BaseException as error:
                failures.append(error)
        if failures:
            self._poisoned = True
            raise NvEmbResizeError("NVEmb preparation cancellation failed; stop the runtime",
                                   {"status": "failed", "unchanged": False,
                                    "pending_tables": list(tokens)}) from failures[0]

    def cancel_resize(self, plan):
        """Release unused native preparations. Consumed plans are harmless here."""
        if plan.get("lifecycle") != "planned":
            return {"status": "unchanged"}
        if any(p.get("status") == "prepared" for p in plan["tables"].values()):
            if self._pending_plan is not plan:
                raise ValueError("Prepared NVEmb plan does not belong to this manager")
            self._cancel_tokens(self._prepared_tokens)
            self._pending_plan = None
        plan["lifecycle"] = "cancelled"
        return {"status": "cancelled"}

    def resize(
        self,
        plan,
        *,
        at_batch_barrier,
        device_pool_limit_bytes,
        cuda_graphs_in_use=False,
    ):
        """Commit the prepared snapshots, shrinking tables first.

        Known preflight rejections cancel all preparations without poisoning.
        Unknown errors or any partial mutation fail closed; cancellation cannot
        restore an already migrated table or its evicted rows.
        """
        try:
            if at_batch_barrier is not True or cuda_graphs_in_use:
                raise ValueError("NVEmb resize requires exclusive batch barrier and no CUDA graphs")
            if type(device_pool_limit_bytes) is not int or device_pool_limit_bytes <= 0:
                raise ValueError("device_pool_limit_bytes must be a positive integer")
            if plan.get("lifecycle") != "planned":
                raise ValueError("NVEmb resize plan has already been consumed or cancelled")
            before = self.stats()
            if before["state"] != "ready":
                raise RuntimeError("NVEmb manager is poisoned; recreate the runtime")
            if set(plan["tables"]) != set(self._bindings):
                raise ValueError("Resize plan does not match the NVEmb tables")
            if any(p.get("status") == "prepared" for p in plan["tables"].values()):
                if self._pending_plan is not plan:
                    raise ValueError("Prepared NVEmb plan does not belong to this manager")
            for name, planned in plan["tables"].items():
                state, expected = before["tables"][name], planned["before"]
                if any(state[k] != expected[k]
                       for k in ("generation", "payload_bytes", "device_pool_bytes")):
                    raise ValueError(f"Stale NVEmb resize plan for {name!r}")
            target_pool = sum(p["device_pool_bytes"] for p in plan["tables"].values())
            if target_pool > device_pool_limit_bytes:
                raise ValueError("Target NVEmb pools exceed the physical pool envelope")
        except BaseException as error:
            self.cancel_resize(plan)
            if isinstance(error, Exception):
                error.receipt = {"status": "rejected", "unchanged": not self._poisoned}
            raise
        order = sorted(
            self._bindings,
            key=lambda name: plan["tables"][name]["device_pool_bytes"]
            - before["tables"][name]["device_pool_bytes"],
        )
        completed = {}
        current = dict(before["tables"])
        started = time.perf_counter()
        rejected = False
        mutated = False
        try:
            for name in order:
                planned = plan["tables"][name]
                other_pools = sum(state["device_pool_bytes"] for n, state in current.items()
                                  if n != name)
                limit = device_pool_limit_bytes - other_pools
                binding = self._bindings[name]
                other_host = sum(plan["tables"][n]["prepared_host_bytes"]
                                 for n in self._prepared_tokens if n != name)
                host_limit = self.host_staging_limit_bytes - other_host
                if self._prepared_support[name]:
                    receipt = self._decode(binding.commit_cache_resize(
                        self._prepared_tokens[name], limit, host_limit))
                    # Wrong-token rejections must not release another pending
                    # snapshot; only native acknowledgement proves consumption.
                    if receipt.get("token_consumed") is True:
                        self._prepared_tokens.pop(name)
                    elif receipt.get("status") != "rejected":
                        raise RuntimeError("Native commit did not acknowledge token consumption")
                else:
                    receipt = self._decode(binding.apply_cache_resize(
                        planned["requested_payload_bytes"], planned["generation"],
                        limit, host_limit))
                if receipt.get("status") not in ("applied", "unchanged"):
                    rejected = receipt.get("status") == "rejected"
                    raise RuntimeError(f"Native NVEmb apply rejected: {receipt}")
                mutated |= receipt["status"] == "applied"
                current[name] = receipt["after"]
                completed[name] = receipt
                embedding = self.collection.embeddings[name]
                embedding.gpu_cache_size = current[name]["configured_bytes"]
                self.collection.gpu_cache_bytes_by_table[name] = current[name]["configured_bytes"]
        except BaseException as error:
            cancel_error = None
            try:
                self.cancel_resize(plan)
            except BaseException as cleanup_error:
                cancel_error = cleanup_error
            # Only explicit rejection plus an identical native ledger proves
            # that nothing changed. A thrown CUDA/native error is not evidence.
            unchanged = rejected and not mutated and cancel_error is None
            if unchanged:
                after = self.stats()
                unchanged = after == plan["before"]
            if not unchanged:
                self._poisoned = True
            partial = {"status": "rejected" if unchanged else "failed",
                       "unchanged": unchanged, "completed_tables": completed,
                       "elapsed_ms": 1000 * (time.perf_counter() - started),
                       "before": plan["before"], "failed_table": name,
                       "resize_mode": self.resize_mode}
            if cancel_error is not None:
                partial["cancel_error"] = str(cancel_error)
            try:
                partial["after"] = self.stats()
            except BaseException:
                partial["after"] = {"state": "unknown", "device_pool_bytes": None}
            plan["lifecycle"] = "failed"
            self.last_receipt = partial
            raise NvEmbResizeError(
                "NVEmb preflight rejected without mutation" if unchanged
                else "NVEmb migration failed; stop the runtime", partial
            ) from error
        self._pending_plan = None
        plan["lifecycle"] = "applied"
        apply_ms = 1000 * (time.perf_counter() - started)
        result = {
            "status": "applied" if mutated else "unchanged",
            "tables": completed,
            "before": plan["before"],
            "after": self.stats(),
            "action_ms": apply_ms,
            "plan_ms": plan["plan_ms"],
            "prepare_ms": plan["prepare_ms"],
            "apply_ms": apply_ms,
            "resize_mode": self.resize_mode,
            "prepared_host_bytes": plan["prepared_host_bytes"],
            "host_staging_peak_bound_bytes": plan["host_staging_peak_bound_bytes"],
            "preserves_survivors": True,
            "evicted_rows": sum(p.get("evicted_rows", 0) for p in completed.values()),
        }
        self.last_receipt = result
        return result
