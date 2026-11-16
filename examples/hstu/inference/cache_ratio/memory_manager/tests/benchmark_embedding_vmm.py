#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dense native embedding VMM prototype benchmark, serial and resize-only.

This measures the experimental eight-way linear-hash per-row LRU backend,
not the production NVEmb LFU implementation. It uses actual native inserts,
cache-only hit masks and byte-value lookups. Full initial capacity is populated
before timing, making every shrink require eviction. No model forward runs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

GIB = 1024**3
MIB = 1024**2
WAYS = 8


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def geometry(sets):
    base = 1 << (int(sets).bit_length() - 1)
    return base, int(sets) - base


def dense_keys(sets):
    """Exactly eight distinct keys in every active linear-hash bucket."""
    import numpy as np
    base, split = geometry(sets)
    buckets = np.arange(sets, dtype=np.int64)
    modulus = np.where((buckets < split) | (buckets >= base), 2 * base, base)
    keys = np.empty(WAYS * sets, dtype=np.int64)
    for way in range(WAYS):
        keys[way * sets:(way + 1) * sets] = buckets + way * modulus
    return keys


def bucket_ids(keys, sets):
    import numpy as np
    base, split = geometry(sets)
    buckets = np.remainder(keys, base)
    divided = buckets < split
    buckets[divided] = keys[divided] % (2 * base)
    return buckets


def expected_survivors(keys, target_sets):
    """Independent CPU oracle: most recent eight ranks per destination bucket.

    Insertion and the complete pre-resize lookup use the same key order, so
    original array position is the exact relative LRU rank. Ranks are unique.
    A merge of top-eight lists has the same top eight as their original union.
    """
    import numpy as np
    buckets = bucket_ids(keys, target_sets)
    order = np.argsort(buckets, kind="stable")  # Input order is increasing rank.
    sorted_buckets = buckets[order]
    ends = np.r_[np.flatnonzero(sorted_buckets[1:] != sorted_buckets[:-1]) + 1, len(keys)]
    starts = np.r_[0, ends[:-1]]
    if (ends - starts).min() < WAYS:
        raise AssertionError("Dense shrink oracle found an underfilled destination bucket")
    indices = ends[:, None] - np.arange(1, WAYS + 1)
    keep = np.zeros(len(keys), dtype=np.bool_)
    keep[order[indices.ravel()]] = True
    if int(keep.sum()) != target_sets * WAYS:
        raise AssertionError("Oracle did not fill every target row")
    return keep


def value_bytes(torch, keys, row_bytes):
    # Repeated eight-byte key representation gives distinct, exact byte rows;
    # no float16 precision loss occurs at millions of keys.
    return keys.view(torch.uint8).reshape(-1, 8).repeat(1, row_bytes // 8).contiguous()


def insert_all(cache, keys, args, torch):
    for first in range(0, len(keys), args.batch_rows):
        key_batch = torch.as_tensor(keys[first:first + args.batch_rows],
                                    device=f"cuda:{args.device}", dtype=torch.int64)
        values = value_bytes(torch, key_batch, args.row_bytes)
        cache.insert(key_batch, values)
        # The wrapper/native ABI completes before returning. Keep the owners
        # alive until then; no tensors alias the resizable value arena.
        del key_batch, values


def validate_all(cache, keys, expected_hits, args, torch):
    """Check every query's hit bit and every expected survivor's full byte row."""
    checked = 0
    for first in range(0, len(keys), args.batch_rows):
        host = keys[first:first + args.batch_rows]
        key_batch = torch.as_tensor(host, device=f"cuda:{args.device}", dtype=torch.int64)
        wanted = torch.as_tensor(expected_hits[first:first + args.batch_rows],
                                 device=f"cuda:{args.device}", dtype=torch.bool)
        values, hits = cache.lookup(key_batch)
        if not torch.equal(hits, wanted):
            bad = torch.nonzero(hits != wanted).flatten()[:8].cpu().tolist()
            raise AssertionError(f"Native cache hit mask differs at query offsets {[first + i for i in bad]}")
        reference = value_bytes(torch, key_batch, args.row_bytes)
        if not torch.equal(values[wanted], reference[wanted]):
            raise AssertionError(f"Native survivor byte values differ in batch starting at {first}")
        checked += len(host)
        del key_batch, wanted, values, hits, reference
    return {"queried_keys": checked, "expected_hits": int(expected_hits.sum()),
            "expected_misses": int(len(keys) - expected_hits.sum()),
            "all_hit_bits_equal": True, "all_survivor_bytes_equal": True,
            "lookup": "native prototype cache-only lookup; no parameter-server fallback"}


def check_ledger(state, payload, row_bytes):
    if state["state"] != "open":
        raise AssertionError("Cache did not remain open")
    for name in ("payload_bytes", "mapped_physical_bytes", "allocated_physical_bytes"):
        if state[name] != payload:
            raise AssertionError(f"Incorrect {name}: {state[name]} != {payload}")
    if state["capacity_rows"] != payload // row_bytes:
        raise AssertionError("Native logical row capacity differs from mapped value payload")
    if state["device_pool_bytes"] != payload + state["fixed_metadata_bytes"]:
        raise AssertionError("Value backing plus fixed native metadata ledger does not balance")


def functional_check(cache, keys, keep, case, args, torch):
    """Separate untimed growth/reinsertion checks, once per chunk/direction."""
    import numpy as np
    started = time.perf_counter()
    if case["direction"] == "shrink":
        regrow = cache.resize(case["before_payload_bytes"], at_batch_barrier=True)
        if regrow["evicted_rows"]:
            raise AssertionError("Regrowth unexpectedly evicted rows")
        still_present = validate_all(cache, keys, keep, args, torch)
        victims = keys[~keep]
        insert_all(cache, victims, args, torch)
        restored = validate_all(cache, keys, np.ones(len(keys), dtype=np.bool_), args, torch)
        final = cache.stats(include_occupancy=True)
        if final["occupied_rows"] != len(keys):
            raise AssertionError("Reinserted victims did not restore full original occupancy")
        return {"kind": "regrow_then_reinsert_all_victims", "timed_sample": False,
                "regrow_receipt": regrow, "after_regrow": still_present,
                "reinserted_rows": len(victims), "after_reinsertion": restored,
                "wall_s_outside_primary": time.perf_counter() - started}
    # Choose one fresh key in each of 1024 distinct non-full destination sets.
    sets = case["after_payload_bytes"] // (WAYS * args.row_bytes)
    counts = np.bincount(bucket_ids(keys, sets), minlength=sets)
    buckets = np.flatnonzero(counts < WAYS)[:1024].astype(np.int64)
    base, split = geometry(sets)
    modulus = np.where((buckets < split) | (buckets >= base), 2 * base, base)
    probes = buckets + modulus * 32
    if not len(probes) or int(probes.min()) <= int(keys.max()):
        raise AssertionError("Post-grow probes are not demonstrably new keys")
    insert_all(cache, probes, args, torch)
    inserted = validate_all(cache, probes, np.ones(len(probes), dtype=np.bool_), args, torch)
    originals = validate_all(cache, keys, np.ones(len(keys), dtype=np.bool_), args, torch)
    final = cache.stats(include_occupancy=True)
    if final["occupied_rows"] != len(keys) + len(probes):
        raise AssertionError("Post-grow insert changed occupancy unexpectedly")
    return {"kind": "insert_into_new_capacity", "timed_sample": False,
            "inserted_rows": len(probes), "new_keys": inserted, "original_keys": originals,
            "wall_s_outside_primary": time.perf_counter() - started}


def run_case(case, repeat, args, cache_class, torch):
    import numpy as np
    directory = args.output_dir / f"r{repeat}-{case['case']}"
    directory.mkdir()
    result = dict(case, repeat=repeat, status="starting", pid=os.getpid(),
                  backend="native_vmm_linear_hash_8way_per_row_lru_prototype",
                  production_nvemb_backend=False, fixture="full_initial_capacity_all_8_ways")
    cache, phase = None, "setup"
    write_json(directory / "result.json", result)
    try:
        started = time.perf_counter()
        cache = cache_class(row_bytes=args.row_bytes, max_payload_bytes=6 * GIB,
                            initial_payload_bytes=case["before_payload_bytes"],
                            chunk_bytes=case["chunk_bytes"], library_path=args.native_library,
                            device=args.device)
        sets = case["before_payload_bytes"] // (WAYS * args.row_bytes)
        keys = dense_keys(sets)
        keep = (expected_survivors(keys, case["after_payload_bytes"] // (WAYS * args.row_bytes))
                if case["direction"] == "shrink" else np.ones(len(keys), dtype=np.bool_))
        result["fixture"] = {"kind": "dense_all_8_ways", "rows": len(keys),
            "resident_value_bytes": len(keys) * args.row_bytes,
            "keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
            "expected_hits_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
            "expected_survivors": int(keep.sum()), "expected_evictions": int((~keep).sum()),
            "value_pattern": "8-byte key representation repeated across each opaque byte row",
            "LRU_order": "way-major insert then full pre-lookup in identical order; rank is input position",
            "oracle": "CPU stable sort by final bucket, retain eight greatest unique access ranks"}
        phase = "prefill"
        insert_all(cache, keys, args, torch)
        phase = "pre_resize_validation"
        result["before_values"] = validate_all(cache, keys, np.ones(len(keys), dtype=np.bool_), args, torch)
        before = cache.stats(include_occupancy=True)
        check_ledger(before, case["before_payload_bytes"], args.row_bytes)
        if before["occupied_rows"] != len(keys):
            raise AssertionError("Fixture did not completely populate native cache capacity")
        result.update(before=before, setup_prefill_validation_s=time.perf_counter() - started)
        write_json(directory / "result.json", result)
        phase = "resize"
        torch.cuda.synchronize(args.device)
        started = time.perf_counter_ns()
        try:
            receipt = cache.resize(case["after_payload_bytes"], at_batch_barrier=True)
            returned = time.perf_counter_ns()
            torch.cuda.synchronize(args.device)
            finished = time.perf_counter_ns()
            result["adjustment"] = {"resize_call_ms": (returned - started) / 1e6,
                                    "resize_complete_ms": (finished - started) / 1e6,
                                    "completion_wait_ms": (finished - returned) / 1e6,
                                    "receipt": receipt}
        except BaseException:
            result["failed_resize_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
            raise
        phase = "post_resize_validation"
        after = cache.stats(include_occupancy=True)
        result["after"] = after
        check_ledger(after, case["after_payload_bytes"], args.row_bytes)
        if before["fixed_metadata_bytes"] != after["fixed_metadata_bytes"]:
            raise AssertionError("Fixed maximum metadata allocation changed")
        if abs(after["device_pool_bytes"] - before["device_pool_bytes"]) != 3 * GIB:
            raise AssertionError("Device ledger did not change by exactly 3 GiB")
        if before["base_ptr"] != after["base_ptr"] or before["clock"] != after["clock"]:
            raise AssertionError("Resize changed stable value pointer or advanced the LRU clock")
        if after["generation"] != before["generation"] + 1 or receipt.get("preserves_survivor_values") is not True:
            raise AssertionError("Native resize did not acknowledge a preserving capacity change")
        if after["occupied_rows"] != int(keep.sum()) or receipt["evicted_rows"] != int((~keep).sum()):
            raise AssertionError("Native occupancy/eviction count disagrees with independent LRU oracle")
        result["after_values"] = validate_all(cache, keys, keep, args, torch)
        result["correctness"] = {"exact_3GiB_payload_and_device_ledger_delta": True,
            "fixed_metadata_allocation_unchanged": True, "stable_value_address": True,
            "resize_does_not_advance_LRU_clock": True, "all_original_keys_hitmask_checked": True,
            "all_surviving_row_bytes_checked": True, "expected_LRU_victims_absent": True,
            "positive_populated_coverage": True,
            "eviction_required": len(keys) > after["capacity_rows"]}
        if repeat == 1:
            phase = "post_timing_functional_checks"
            result["functional"] = functional_check(cache, keys, keep, case, args, torch)
        else:
            result["functional"] = {"scope": "Separate functional check performed on repeat1 of this chunk/direction"}
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", failed_phase=phase, error=repr(error), traceback=traceback.format_exc())
        if cache is not None:
            try:
                result["failure_state"] = cache.stats()
            except BaseException as state_error:
                result["failure_state_error"] = repr(state_error)
    finally:
        write_json(directory / "result.json", result)
        if cache is not None:
            try:
                cache.close(at_batch_barrier=True)
                result["explicit_close"] = "succeeded"
            except BaseException as error:
                result["explicit_close"] = repr(error)
                if result["status"] == "completed":
                    result["status"] = "failed_cleanup"
        write_json(directory / "result.json", result)
    return result


def cases():
    return [{"case": f"{mib}MiB-{direction}", "direction": direction,
             "chunk_bytes": mib * MIB, "before_payload_bytes": before * GIB,
             "after_payload_bytes": after * GIB, "changed_value_chunks": 3 * GIB // (mib * MIB)}
            for mib in (2, 16, 64) for direction, before, after in (("grow", 3, 6), ("shrink", 6, 3))]


def summarize(runs):
    groups = []
    for case in cases():
        selected = [r for r in runs if r["case"] == case["case"] and r["status"] == "completed"]
        metrics = {}
        for field in ("resize_call_ms", "resize_complete_ms", "completion_wait_ms", "pool_ms", "migration_ms", "total_ms", "barrier_ms"):
            values = [r["adjustment"].get(field, r["adjustment"]["receipt"].get(field)) for r in selected]
            values = [value for value in values if value is not None]
            metrics[field] = {"values": values, "mean": statistics.mean(values) if values else None,
                              "median": statistics.median(values) if values else None,
                              "min": min(values) if values else None, "max": max(values) if values else None,
                              "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        groups.append(dict(case, successful_repeats=len(selected), metrics=metrics))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--row-bytes", type=int, default=2048)
    parser.add_argument("--batch-mib", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.row_bytes <= 0 or args.row_bytes % 8 or 2 * MIB % (WAYS * args.row_bytes):
        parser.error("row-bytes must be a positive multiple of8 and eight rows must divide2MiB")
    if args.batch_mib <= 0 or args.batch_mib > 64 or args.device < 0:
        parser.error("batch-mib must be1..64; device must be nonnegative")
    args.batch_rows = max(1, args.batch_mib * MIB // args.row_bytes)
    if args.describe:
        print(json.dumps({"cases": cases(), "repeats": args.repeats, "row_bytes": args.row_bytes,
                          "batch_rows": args.batch_rows, "fixture": "full initial native cache"}, indent=2))
        return 0
    if args.repo_root is None or args.native_library is None or args.output_dir is None:
        parser.error("Execution needs --repo-root, --native-library, and --output-dir")
    args.repo_root, args.native_library, args.output_dir = (p.resolve() for p in (args.repo_root, args.native_library, args.output_dir))
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing result directory: {args.output_dir}")
    sys.path.insert(0, str(args.repo_root / "examples/hstu/inference"))
    from cache_ratio.memory_manager.embedding_vmm import VmmEmbeddingCache
    import torch
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(args.device)
    source = args.repo_root / "examples/hstu/inference/cache_ratio/memory_manager"
    source_paths = [source / "embedding_vmm.py", source / "pool.py", source / "native/vmm_embedding.cu", args.native_library]
    report = {"status": "running", "complete": False, "all_succeeded": False,
        "backend": VmmEmbeddingCache.backend_name, "production_NVEmb_backend": False,
        "policy": "per-row LRU; different from production NVEmb LFU",
        "host": platform.node(), "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "gpu_total_memory_bytes": properties.total_memory, "torch_version": torch.__version__,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "row_bytes": args.row_bytes, "batch_payload_bytes": args.batch_rows * args.row_bytes,
        "requested_runs": len(cases()) * args.repeats, "requested_repeats": args.repeats,
        "scope": {"primary": "Wall time through completed resize; all preparation/full native hit+value validation outside",
                  "native_phase_times": "Coarse native migration and VMM pool spans from wrapper; inclusive total not additive with child spans",
                  "metadata_accounting": "Fixed maximum native cudaMalloc requested bytes separately reported; excludes allocator rounding/reserves; VMM value backing physical delta is exactly3GiB",
                  "no_model_forward_or_parameter_server": True, "no_per_API_timing_hooks": True,
                  "tails_retained_no_subtraction": True, "functional_checks": "Additional regrow/reinsert or post-growinsert onrepeat1, outsideprimary"},
        "runs": [], "groups": []}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", report)
    for repeat in range(1, args.repeats + 1):
        order = cases()
        if repeat % 2 == 0:
            order.reverse()
        for case in order:
            result = run_case(case, repeat, args, VmmEmbeddingCache, torch)
            report["runs"].append(result)
            report["groups"] = summarize(report["runs"])
            report["complete"] = len(report["runs"]) == report["requested_runs"]
            report["all_succeeded"] = report["complete"] and all(r["status"] == "completed" for r in report["runs"])
            report["status"] = "completed" if report["all_succeeded"] else "running"
            if result["status"] != "completed":
                report["status"] = "failed"
            write_json(args.output_dir / "summary.json", report)
            print(json.dumps({"repeat": repeat, "case": case["case"], "status": result["status"],
                              "adjustment": result.get("adjustment"), "error": result.get("error")}), flush=True)
            if result["status"] != "completed":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
