# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native correctness check; importing this module launches no CUDA work.

Call run(backend, probe_rows, ratios) after preparing the benchmark HSTU with
cache_memory_backend="vmm", an explicit peak_cache_bytes envelope and
host_staging_limit_bytes. Supply at least four feasible ratios that physically
grow and shrink both caches twice. The caller owns this isolated runtime and
must exclude all other forwards, snapshots and capacity changes until return.
Prepare actual resident NVEmb rows before calling: its normal insertion
threshold can exceed a small probe's request count. Empty caches do not prove
survivor preservation; this check deliberately fails if that coverage is absent.

This uses real native forwards, raw retained KV bytes and native NVEmb receipts.
Snapshot/no-KV validation costs make it unsuitable for latency benchmarking.
No policies, traffic generators, allocations or remote jobs start automatically.
"""

from copy import deepcopy
from math import isfinite


def _forward(backend, rows, *, no_kv):
    # Rebuild input containers: native strip_cached_tokens mutates its HSTUBatch.
    output = backend.forward(deepcopy(rows), no_kv=no_kv)
    backend.synchronize()
    backend.validate(output)
    if not isinstance(output, backend.torch.Tensor) or output.device.type != "cuda":
        raise AssertionError("Expected actual native CUDA model output")
    return output.detach().cpu().clone()


def _kv_snapshot(backend, wanted_uids):
    """Copy only the small supplied probe's live pages, never retain a KV alias."""
    memory, torch = backend.model.cache_memory, backend.torch
    torch.cuda.synchronize(memory.kv.device_idx)
    result = {}
    # get_cache_tables() Tensor aliases stay local and disappear before resize.
    for uid, start, length, pages in memory.kv.impl_._vmm_snapshot():
        if uid not in wanted_uids:
            continue
        layers = tuple(
            table[pages].detach().cpu().contiguous().view(torch.uint8).clone()
            for table in memory.kv.get_cache_tables()
        )
        result[uid] = (start, length, tuple(pages), layers)
    return result


def _check_ledger(state):
    if state["state"] != "open":
        raise AssertionError("Shared manager did not remain usable")
    if state["kv"]["backend"] != "gpu_only_vmm":
        raise AssertionError("Correctness check requires the native VMM backend")
    kv, emb = state["kv"]["payload_bytes"], state["embedding"]["payload_bytes"]
    if kv + emb + state["unused_payload_bytes"] != state["payload_budget_bytes"]:
        raise AssertionError(
            "Actual capacity and unused payload do not conserve budget"
        )
    if state["unused_payload_bytes"] < 0:
        raise AssertionError("Native caches exceeded the payload ceiling")
    if state["device_pool_bytes"] > state["peak_limit_bytes"]:
        raise AssertionError("Native device pools exceeded the separate peak envelope")
    if state["kv"]["mapped_physical_bytes"] != kv:
        raise AssertionError("KV capacity is not backed by mapped physical payload")
    if state["embedding"]["state"] != "ready":
        raise AssertionError("NVEmb native hierarchy is not usable")


def run(backend, probe_rows, ratios):
    """Return correctness evidence or raise; unsupported plans are not passes.

    The probe must contain unique cache UIDs and immutable histories, and fit
    every requested capacity. A native rehash rejection is an uncovered test
    case, never an excuse to clear caches or silently choose a different ratio.
    This checks NVEmb's own survivor-state receipt plus end-to-end output parity;
    it does not independently export its private raw tags/counters/row bytes.
    """
    rows, ratios = deepcopy(list(probe_rows)), list(ratios)
    if not rows or len({row["cache_uid"] for row in rows}) != len(rows):
        raise ValueError("Supply a nonempty probe batch with distinct cache UIDs")
    if len(ratios) < 4 or any(
        type(ratio) not in (int, float) or not isfinite(ratio) or not 0 < ratio < 1
        for ratio in ratios
    ):
        raise ValueError("Supply at least four ratios strictly between zero and one")
    if not backend.torch.cuda.is_available():
        raise RuntimeError("This check requires native CUDA; CPU mocks do not qualify")
    memory = backend.model.cache_memory
    initial = memory.stats()
    _check_ledger(initial)
    collection = backend.model.sparse_module._dynamic_embedding_collection
    layer_ids = {name: id(layer) for name, layer in collection.embeddings.items()}
    binding_ids = {
        name: id(layer.emb_layer) for name, layer in collection.embeddings.items()
    }
    storage_ids = {
        name: id(storage)
        for name, storage in collection._owned_parameter_servers.items()
    }
    wanted = {row["cache_uid"] for row in rows}
    reference = _forward(backend, rows, no_kv=True)
    for _ in range(2):
        actual = _forward(backend, rows, no_kv=False)
        backend.torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.005)
    if set(_kv_snapshot(backend, wanted)) != wanted:
        raise AssertionError("Probe failed to populate every expected KV history")

    moves, direction_counts = [], {
        "kv_grow": 0,
        "kv_shrink": 0,
        "embedding_grow": 0,
        "embedding_shrink": 0,
    }
    for ratio in ratios:
        before = memory.stats()
        before_kv = _kv_snapshot(backend, wanted)
        pointers = tuple(table.data_ptr() for table in memory.kv.get_cache_tables())
        action = memory.resize_ratio(ratio, at_batch_barrier=True)
        after = memory.stats()
        _check_ledger(after)
        if action["status"] != "applied" or action["preserves_survivors"] is not True:
            raise AssertionError(
                "Requested step did not physically resize preserving caches"
            )
        changes = {
            name: after[name]["payload_bytes"] - before[name]["payload_bytes"]
            for name in ("kv", "embedding")
        }
        if (
            not changes["kv"]
            or not changes["embedding"]
            or changes["kv"] * changes["embedding"] >= 0
        ):
            raise AssertionError(
                "This test requires a real transfer between both native caches"
            )
        if {step["cache"] for step in action["completed"]} != {"kv", "embedding"}:
            raise AssertionError("Both native resize implementations must execute")
        for name, delta in changes.items():
            direction_counts[name + ("_grow" if delta > 0 else "_shrink")] += 1
        if (
            tuple(table.data_ptr() for table in memory.kv.get_cache_tables())
            != pointers
        ):
            raise AssertionError("VMM changed a layer's base pointer")
        after_kv = _kv_snapshot(backend, wanted)
        survivors = before_kv.keys() & after_kv.keys()
        if not survivors:
            raise AssertionError("No probe KV survived; retention coverage is empty")
        if changes["kv"] > 0 and before_kv.keys() != after_kv.keys():
            raise AssertionError("KV growth evicted an existing probe history")
        for uid in survivors:
            old, new = before_kv[uid], after_kv[uid]
            if old[:3] != new[:3] or not all(
                backend.torch.equal(a, b) for a, b in zip(old[3], new[3])
            ):
                raise AssertionError(
                    f"Surviving KV changed before forward for UID {uid}"
                )
        embedding_step = next(
            step["result"]
            for step in action["completed"]
            if step["cache"] == "embedding"
        )
        native_survivors = 0
        moved_tables = 0
        for name, native in embedding_step["tables"].items():
            if native["after"]["payload_bytes"] == native["before"]["payload_bytes"]:
                continue
            moved_tables += 1
            if (
                native["status"] != "applied"
                or native["preserves_survivors"] is not True
                or native["preserves_counter_bits_and_ages"] is not True
                or native["after"]["generation"] <= native["before"]["generation"]
            ):
                raise AssertionError(
                    f"Missing native preserving migration evidence for {name}"
                )
            native_survivors += native["survivor_rows"]
        if not moved_tables or not native_survivors:
            raise AssertionError("NVEmb survivor coverage is empty")
        if (
            layer_ids
            != {name: id(layer) for name, layer in collection.embeddings.items()}
            or binding_ids
            != {
                name: id(layer.emb_layer)
                for name, layer in collection.embeddings.items()
            }
            or storage_ids
            != {
                name: id(storage)
                for name, storage in collection._owned_parameter_servers.items()
            }
        ):
            raise AssertionError(
                "Resize replaced the NVEmb layer/binding/CPU parameter server"
            )
        actual = _forward(backend, rows, no_kv=False)
        fresh_reference = _forward(backend, rows, no_kv=True)
        backend.torch.testing.assert_close(
            actual, fresh_reference, rtol=0.02, atol=0.005
        )
        backend.torch.testing.assert_close(
            fresh_reference, reference, rtol=0.02, atol=0.005
        )
        moves.append(
            {
                "target_ratio": ratio,
                "payload_changes": changes,
                "retained_probe_kv_users": len(survivors),
                "native_reported_embedding_survivors": native_survivors,
                "max_abs_output_error": (actual.float() - fresh_reference.float())
                .abs()
                .max()
                .item(),
                "action": action,
            }
        )
    if any(count < 2 for count in direction_counts.values()):
        raise AssertionError("Need at least two real grows and shrinks of each cache")
    return {
        "status": "passed_native_correctness",
        "scope": "manual_ratio_transfer_full_native_forward",
        "initial": initial,
        "final": memory.stats(),
        "moves": moves,
        "direction_counts": direction_counts,
        "all_steps_checked": True,
        "output_tolerance": {"rtol": 0.02, "atol": 0.005},
        "nvemb_raw_survivors_independently_exported": False,
        "nvemb_pending_insert_buffers_independently_verified": False,
        "adaptive_policy_benefit": None,
        "queue_benefit": None,
        "performance_overhead": None,
    }
