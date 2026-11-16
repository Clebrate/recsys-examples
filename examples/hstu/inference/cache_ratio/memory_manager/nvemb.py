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
        self._weights = {
            name: configs[name].num_embeddings * configs[name].embedding_dim
            for name in self._bindings
        }
        self._poisoned = False
        self.last_receipt = None

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
        result = {"tables": tables, "state": "poisoned" if unknown else "ready"}
        for field in ("payload_bytes", "device_pool_bytes", "configured_bytes"):
            values = [s.get(field) for s in tables.values()]
            result[field] = None if any(v is None for v in values) else sum(values)
        result[
            "budget_scope"
        ] = "cache_pools_excludes_model_execution_workspaces_and_handler_buffers"
        return result

    def plan_resize(self, payload_budget_bytes):
        """Read-only native snapshot/rehash plan; caller excludes forward calls.

        The plan cost includes device synchronization and native validation
        (plus CPU staging on the default backend). Callers must charge it to
        adjustment overhead. Ratios address value payload;
        separate device_pool_bytes include NVEmb tags/counters/timestamps.
        """
        if type(payload_budget_bytes) is not int or payload_budget_bytes <= 0:
            raise ValueError("payload_budget_bytes must be a positive integer")
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
        plans = {}
        started = time.perf_counter()
        for name, binding in self._bindings.items():
            unit = quantum[name]
            target = (
                unit + remaining * self._weights[name] // (total_weight * unit) * unit
            )
            native = self._decode(
                binding.plan_cache_resize(target, self.host_staging_limit_bytes)
            )
            if native.get("status") != "planned":
                raise ValueError(
                    f"NVEmb resize rejected for {name!r}: {native.get('reason', native)}"
                )
            plans[name] = native
        return {
            "tables": plans,
            "before": before,
            "requested_payload_bytes": payload_budget_bytes,
            "payload_bytes": sum(p["payload_bytes"] for p in plans.values()),
            "device_pool_bytes": sum(p["device_pool_bytes"] for p in plans.values()),
            "extra_peak_bytes": 0,
            "plan_ms": 1000 * (time.perf_counter() - started),
        }

    def resize(
        self,
        plan,
        *,
        at_batch_barrier,
        device_pool_limit_bytes,
        cuda_graphs_in_use=False,
    ):
        """Execute sequential native migrations, shrinking tables first.

        This operation cannot roll back an already migrated table or restore
        evicted rows. Any exception poisons the manager and carries a partial
        receipt; the enclosing HSTU runtime must stop serving requests.
        """
        if at_batch_barrier is not True or cuda_graphs_in_use:
            raise ValueError(
                "NVEmb resize requires exclusive batch barrier and no CUDA graphs"
            )
        if type(device_pool_limit_bytes) is not int or device_pool_limit_bytes <= 0:
            raise ValueError("device_pool_limit_bytes must be a positive integer")
        before = self.stats()
        if before["state"] != "ready":
            raise RuntimeError("NVEmb manager is poisoned; recreate the runtime")
        if set(plan["tables"]) != set(self._bindings):
            raise ValueError("Resize plan does not match the NVEmb tables")
        # Reject stale/over-budget plans for every table before first mutation.
        for name, planned in plan["tables"].items():
            state = before["tables"][name]
            expected = planned["before"]
            if any(
                state[k] != expected[k]
                for k in ("generation", "payload_bytes", "device_pool_bytes")
            ):
                raise ValueError(f"Stale NVEmb resize plan for {name!r}")
        target_pool = sum(p["device_pool_bytes"] for p in plan["tables"].values())
        if target_pool > device_pool_limit_bytes:
            raise ValueError("Target NVEmb pools exceed the physical pool envelope")
        order = sorted(
            self._bindings,
            key=lambda name: plan["tables"][name]["device_pool_bytes"]
            - before["tables"][name]["device_pool_bytes"],
        )
        completed = {}
        current = dict(before["tables"])
        started = time.perf_counter()
        try:
            for name in order:
                planned = plan["tables"][name]
                other_pools = sum(
                    state["device_pool_bytes"]
                    for n, state in current.items()
                    if n != name
                )
                limit = device_pool_limit_bytes - other_pools
                receipt = self._decode(
                    self._bindings[name].apply_cache_resize(
                        planned["requested_payload_bytes"],
                        planned["generation"],
                        limit,
                        self.host_staging_limit_bytes,
                    )
                )
                if receipt.get("status") not in ("applied", "unchanged"):
                    raise RuntimeError(f"Native NVEmb apply rejected: {receipt}")
                current[name] = receipt["after"]
                completed[name] = receipt
                # Report the real configured native pool size, not value payload.
                embedding = self.collection.embeddings[name]
                embedding.gpu_cache_size = current[name]["configured_bytes"]
                self.collection.gpu_cache_bytes_by_table[name] = current[name][
                    "configured_bytes"
                ]
        except BaseException as error:
            self._poisoned = True
            partial = {
                "status": "failed",
                "completed_tables": completed,
                "elapsed_ms": 1000 * (time.perf_counter() - started),
                "before": before,
                "failed_table": name,
            }
            try:
                partial["after"] = self.stats()
            except BaseException:
                partial["after"] = {"state": "unknown", "device_pool_bytes": None}
            self.last_receipt = partial
            raise NvEmbResizeError(
                "NVEmb migration failed; stop the runtime", partial
            ) from error
        result = {
            "status": "applied",
            "tables": completed,
            "before": before,
            "after": self.stats(),
            "action_ms": 1000 * (time.perf_counter() - started),
            "plan_ms": plan["plan_ms"],
            "preserves_survivors": True,
            "evicted_rows": sum(p.get("evicted_rows", 0) for p in completed.values()),
        }
        self.last_receipt = result
        return result
