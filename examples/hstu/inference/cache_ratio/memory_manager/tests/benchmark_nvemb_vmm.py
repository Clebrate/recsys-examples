#!/usr/bin/env python3
"""Original NVEmbedding/LFU VMM experiment; one process per invocation.

No CUDA/import side effects until run(). Values use the public insert/forward
API; exact residency, LFU bits and ages use the read-only metadata diagnostic.
The timer contains planning, apply/context rebuild, and completion, with all
fixture generation and external correctness checks outside it. No timing hooks.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

MIB = 1 << 20
GIB = 1 << 30
WAYS = 8


def decode(value):
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(memoryview(array).cast("B"))
    return result.hexdigest()


def memory_status():
    result = {}
    for filename, prefixes in (("/proc/meminfo", ("MemAvailable:",)),
                               ("/proc/self/status", ("VmRSS:", "VmHWM:"))):
        for line in Path(filename).read_text().splitlines():
            if line.startswith(prefixes):
                key, number, _ = line.split()
                result[key.rstrip(":") + "_bytes"] = int(number) * 1024
    for name in ("memory.max", "memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result["cgroup_" + name] = path.read_text().strip()
    return result


def runtime_identity(root, nve):
    """Reject a mixed installation, including plugins loaded after construction."""
    root = root.resolve()
    paths = {Path(nve.__file__).resolve()}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if ".so" in path.name and (path.name.startswith("libnve") or path.name.startswith("nve.")):
            check(not str(path).endswith(" (deleted)"), "Loaded NVE library was deleted")
            paths.add(path.resolve())
    check(any(p.name.startswith("libnve-common") for p in paths), "NVE common library not found in process maps")
    for path in paths:
        check(path.is_relative_to(root), f"Wrong NVE runtime loaded: {path}; expected under {root}")
    return [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(paths)]


def canonical(np, snapshot):
    valid = np.flatnonzero(snapshot["tags"] >= 0)
    sets = valid // WAYS
    keys = snapshot["tags"][valid] * snapshot["state"]["num_sets"] + sets
    order = np.argsort(keys)
    result = {"keys": keys[order], "bits": snapshot["bits"][valid][order],
              "ages": snapshot["ages"][sets][order]}
    check(not len(keys) or np.all(np.diff(result["keys"]) > 0), "Duplicate resident key")
    counters = result["bits"].view(np.float32)
    check(np.all(np.isfinite(counters) & (counters >= 0)), "Invalid resident LFU counter")
    check(np.all((result["ages"] >= 0) & (result["ages"] <= snapshot["state"]["global_ts"])),
          "Invalid resident set age")
    return result


def snapshot(np, emb):
    binding = emb.emb_layer
    state = decode(binding.cache_resize_state())
    sets = state["num_sets"]
    tags = np.empty(sets * WAYS, dtype=np.int64)
    bits = np.empty(sets * WAYS, dtype=np.uint32)
    ages = np.empty(sets, dtype=np.int64)
    metadata = decode(binding.cache_resize_metadata(sets, tags.ctypes.data, bits.ctypes.data, ages.ctypes.data))
    check(metadata["num_sets"] == sets and metadata["ways"] == WAYS, "Metadata geometry changed")
    result = {"state": metadata, "tags": tags, "bits": bits, "ages": ages}
    result["rows"] = canonical(np, result)
    return result


def brief(np, snap):
    rows = snap["rows"]
    counters = rows["bits"].view(np.float32)
    return {"state": snap["state"], "occupied_rows": len(rows["keys"]),
            "occupied_value_bytes": len(rows["keys"]) * snap["state"]["row_bytes"],
            "raw_metadata_sha256": digest(snap["tags"], snap["bits"], snap["ages"]),
            "canonical_resident_sha256": digest(rows["keys"], rows["bits"], rows["ages"]),
            "distinct_occupied_ages": int(len(np.unique(rows["ages"]))),
            "counter_min": float(counters.min()) if len(counters) else None,
            "counter_max": float(counters.max()) if len(counters) else None}


def policy_evidence(np, snap):
    """Small policy fixtures retain exact counters even if a later check fails."""
    return {**brief(np, snap), "resident_keys": snap["rows"]["keys"].tolist(),
            "resident_counter_bits": snap["rows"]["bits"].tolist(),
            "resident_counters": snap["rows"]["bits"].view(np.float32).tolist(),
            "resident_set_ages": snap["rows"]["ages"].tolist()}


def normalized_priority_match(np, target, max_rows):
    """Match count * float32(1 / batch_size), the original GPU histogram rule."""
    target = np.float32(target)
    totals = np.arange(2, max_rows + 1, dtype=np.int64)
    estimates = np.floor(float(target) * totals).astype(np.int64)
    norm = np.float32(1) / totals.astype(np.float32)
    matches = []
    for offset in (-1, 0, 1):
        count = estimates + offset
        score = count.astype(np.float32) * norm
        found = np.flatnonzero((count > 0) & (count < totals) &
                               (score.view(np.uint32) == target.view(np.uint32)))
        if len(found):
            index = found[0]
            matches.append((int(totals[index]), int(count[index])))
    check(bool(matches), f"No bit-exact normalized priority fixture within {max_rows} rows; target={target!r}")
    total, count = min(matches)
    return count, total


def oracle(np, snap, target_sets):
    """Original preserving-rehash rule: counter descending, key ascending."""
    rows = snap["rows"]
    dest = rows["keys"] % target_sets
    counters = rows["bits"].view(np.float32)
    order = np.lexsort((rows["keys"], -counters, dest))
    sorted_dest = dest[order]
    if len(order):
        adjacent = sorted_dest[1:] == sorted_dest[:-1]
        check(not np.any(adjacent & (rows["ages"][order][1:] != rows["ages"][order][:-1])),
              "mixed_age_merge")
        starts = np.maximum.accumulate(np.where(np.r_[True, ~adjacent], np.arange(len(order)), 0))
        keep = order[np.arange(len(order)) - starts < WAYS]
    else:
        keep = order
    keep = keep[np.argsort(rows["keys"][keep])]
    return {name: array[keep] for name, array in rows.items()}


def compare_rows(np, expected, actual, label):
    for name in ("keys", "bits", "ages"):
        check(np.array_equal(expected[name], actual[name]), f"{label}: {name} differ")


def compare_snapshots(np, left, right, label, raw=False):
    for name in ("num_sets", "row_bytes", "ways", "global_ts", "decay_rate"):
        check(left["state"][name] == right["state"][name], f"{label}: {name} differs")
    compare_rows(np, left["rows"], right["rows"], label)
    if raw:
        check(left["state"] == right["state"], f"{label}: native state changed")
        for name in ("tags", "bits", "ages"):
            check(np.array_equal(left[name], right[name]), f"{label}: raw {name} changed")


def dense_batches(np, sets, low_sets, batch_rows):
    """Each batch contains complete target merge pairs, or complete grow sets."""
    per_bucket = WAYS * (sets // low_sets)
    step = max(1, batch_rows // per_bucket)
    for first in range(0, low_sets, step):
        buckets = np.arange(first, min(first + step, low_sets), dtype=np.int64)
        yield (buckets[:, None] + np.arange(per_bucket, dtype=np.int64)[None, :] * low_sets).reshape(-1)


def values(torch, keys, row_bytes):
    # All 32 key bits are represented exactly in fp16, including every way.
    # Per-column offsets also catch column permutations within repeated blocks.
    cols = torch.arange(row_bytes // 2, device=keys.device, dtype=torch.int64)
    shifts = (cols % 4) * 8
    encoded = ((keys[:, None] >> shifts[None, :]) & 255)
    return (1 + ((encoded + cols[None, :] // 4) % 256)).to(torch.float16).contiguous()


class Experiment:
    def __init__(self, args, report):
        import numpy as np
        import torch
        import pynve.nve as nve
        from pynve.torch import nve_layers, nve_ps
        self.np, self.torch, self.nve = np, torch, nve
        self.layers, self.ps = nve_layers, nve_ps
        self.args, self.report = args, report
        torch.cuda.set_device(args.device)
        self.device = torch.device("cuda", args.device)
        self.batch_rows = args.batch_mib * MIB // args.row_bytes
        report["loaded_libraries"] = runtime_identity(args.runtime_root, nve)
        prop = torch.cuda.get_device_properties(args.device)
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "device": args.device, "gpu_name": prop.name,
                             "gpu_uuid": str(getattr(prop, "uuid", "unavailable")),
                             "total_memory": prop.total_memory,
                             "python": sys.version, "nve_module": nve.__file__}

    def make(self, payload, vmm):
        args, torch = self.args, self.torch
        os.environ["NVE_NVEMB_VMM_MAX_PAYLOAD_BYTES"] = str(args.high_bytes if vmm else 0)
        os.environ["NVE_NVEMB_VMM_CHUNK_BYTES"] = str(args.chunk_mib * MIB)
        os.environ["NVE_NVEMB_VMM_FAST_REHASH"] = str(args.fast_rehash if vmm else 0)
        sets = payload // (WAYS * args.row_bytes)
        ps = self.ps.NVEParameterServer(num_embeddings=0, embedding_size=args.row_bytes // 2,
                                        data_type=torch.float16, initializer=None, initial_size=1024)
        emb = self.layers.NVEmbedding(
            num_embeddings=4 * args.high_bytes // args.row_bytes,
            embedding_size=args.row_bytes // 2, data_type=torch.float16,
            layer_type=self.layers.LayerType.Hierarchical,
            gpu_cache_size=sets * (WAYS * args.row_bytes + 104), host_cache_size=0,
            storage=ps, optimize_for_training=False, device=self.device,
            config={"max_modify_size": max(1024, self.batch_rows)})
        torch.cuda.synchronize()
        snap = snapshot(self.np, emb)
        self.geometry(snap, payload, vmm)
        check(len(snap["rows"]["keys"]) == 0, "Fresh cache is not empty")
        self.report["loaded_libraries"] = runtime_identity(args.runtime_root, self.nve)
        return emb

    def geometry(self, snap, payload, vmm=True):
        state = snap["state"]
        check(state["payload_bytes"] == payload, "Payload differs from requested exact bytes")
        check(state["num_sets"] * WAYS * self.args.row_bytes == payload, "Set capacity mismatch")
        check(state["row_bytes"] == self.args.row_bytes and state["ways"] == WAYS, "Native row/way mismatch")
        check(state.get("vmm", False) == vmm, "Wrong allocation backend")
        if vmm:
            check(state["mapped_value_bytes"] == payload, "Rounded mapping changes controlled payload")
            check(state["allocated_value_bytes"] == payload, "Allocated physical values differ from payload")
            check(state["chunk_bytes"] == self.args.chunk_mib * MIB, "Wrong VMM chunk")
            check(state["device_pool_bytes"] == payload + state["fixed_metadata_bytes"], "Cache pool ledger mismatch")

    def insert(self, emb, cpu_keys):
        torch = self.torch
        keys = torch.as_tensor(cpu_keys, dtype=torch.int64, device=self.device).contiguous()
        val = values(torch, keys, self.args.row_bytes)
        torch.cuda.synchronize()
        emb.insert(keys, val, table_id=0)
        torch.cuda.synchronize()

    def fill(self, emb, payload):
        start = time.perf_counter()
        count = 0
        sets = payload // (WAYS * self.args.row_bytes)
        low_sets = self.args.low_bytes // (WAYS * self.args.row_bytes)
        for keys in dense_batches(self.np, sets, low_sets, self.batch_rows):
            self.insert(emb, keys)
            count += 1
        snap = snapshot(self.np, emb)
        expected = self.np.arange(payload // self.args.row_bytes, dtype=self.np.int64)
        check(self.np.array_equal(snap["rows"]["keys"], expected), "Dense public insert did not retain every intended key")
        return {"public_insert_calls": count, "rows": len(expected),
                "wall_s_outside_resize": time.perf_counter() - start,
                "all_merge_pairs_inserted_in_one_call": True,
                "metadata": brief(self.np, snap)}

    def verify_values(self, emb, keys, label):
        start = time.perf_counter()
        torch = self.torch
        count = 0
        for offset in range(0, len(keys), self.batch_rows):
            batch = torch.as_tensor(keys[offset:offset + self.batch_rows], dtype=torch.int64,
                                    device=self.device).contiguous()
            expected = values(torch, batch, self.args.row_bytes)
            observed = emb(batch)
            torch.cuda.synchronize()
            check(observed.shape == expected.shape and observed.dtype == expected.dtype,
                  f"{label}: public forward shape/dtype differs")
            check(torch.equal(observed.view(torch.uint8), expected.view(torch.uint8)),
                  f"{label}: public forward returned wrong value bytes")
            count += len(batch)
        self.report["loaded_libraries"] = runtime_identity(self.args.runtime_root, self.nve)
        return {"checked_rows": count, "checked_value_bytes": count * self.args.row_bytes,
                "all_value_bytes_equal": True, "public_NVEmbedding_forward": True,
                "keys_sha256": digest(keys), "wall_s_outside_resize": time.perf_counter() - start}

    def resize(self, emb, target, evidence):
        torch = self.torch
        binding = emb.emb_layer
        # Fixture work is already completed. Time the explicit barrier too;
        # never subtract it or native/context tails from the headline number.
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            torch.cuda.synchronize()
            synced = time.perf_counter()
            evidence["pre_sync_ms"] = (synced - started) * 1000
            plan = decode(binding.plan_cache_resize(target, self.args.host_limit_gib * GIB))
            planned = time.perf_counter()
            evidence.update(plan=plan, plan_wall_ms=(planned - synced) * 1000)
            if plan["status"] != "planned":
                raise RuntimeError(f"Resize preflight refused: {plan}")
            applied = decode(binding.apply_cache_resize(target, plan["generation"],
                              self.args.pool_limit_gib * GIB, self.args.host_limit_gib * GIB))
            returned = time.perf_counter()
            evidence.update(apply=applied, apply_wall_ms=(returned - planned) * 1000)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            evidence.update(post_sync_ms=(finished - returned) * 1000,
                            total_wall_ms=(finished - started) * 1000)
            check(applied["status"] == "applied", f"Resize was not applied: {applied}")
            return applied
        except BaseException:
            evidence["wall_ms_until_failure"] = (time.perf_counter() - started) * 1000
            raise

    def checked_resize(self, emb, target, evidence, vmm=True):
        np = self.np
        before = snapshot(np, emb)
        expected = oracle(np, before, target // (WAYS * self.args.row_bytes))
        owners = (id(emb), id(emb.emb_layer), id(emb.storage))
        evidence["before"] = brief(np, before)
        receipt = self.resize(emb, target, evidence.setdefault("timing", {}))
        after = snapshot(np, emb)  # Before lookup can change native LFU metadata.
        self.geometry(after, target, vmm)
        compare_rows(np, expected, after["rows"], "Resize survivor oracle")
        check(before["state"]["global_ts"] == after["state"]["global_ts"], "Resize changed LFU clock")
        check(before["state"]["decay_rate"] == after["state"]["decay_rate"], "Resize changed LFU decay")
        check(after["state"]["generation"] == before["state"]["generation"] + 1, "Wrong generation")
        check(owners == (id(emb), id(emb.emb_layer), id(emb.storage)), "Public hierarchy owner changed")
        if vmm:
            check(before["state"]["value_base_address"] == after["state"]["value_base_address"], "VMM base address changed")
            check(before["state"]["fixed_metadata_bytes"] == after["state"]["fixed_metadata_bytes"], "Fixed metadata allocation changed")
        evicted = len(before["rows"]["keys"]) - len(expected["keys"])
        check(receipt["survivor_rows"] == len(expected["keys"]), "Receipt survivor count differs")
        check(receipt["evicted_rows"] == evicted, "Receipt eviction count differs")
        evidence.update(after=brief(np, after), expected_evicted_rows=evicted,
                        residency={"queried_original_rows": len(before["rows"]["keys"]),
                                   "exact_native_hits": len(expected["keys"]), "exact_native_misses": evicted,
                                   "hit_rate": len(expected["keys"]) / max(1, len(before["rows"]["keys"])),
                                   "source": "read_only_native_tags_before_public_forward; no PS fallback"},
                        all_survivor_keys_counter_bits_and_ages_equal=True)
        evidence["values_after"] = self.verify_values(emb, expected["keys"], "After resize")
        return after

def run_case(exp, report):
    args, np = exp.args, exp.np
    initial, target = ((args.low_bytes, args.high_bytes) if args.direction == "grow"
                       else (args.high_bytes, args.low_bytes))
    emb = exp.make(initial, True)
    report["phase"] = "dense_public_insert"
    report["fixture"] = exp.fill(emb, initial)
    report["phase"] = "full_pre_resize_value_validation"
    before = snapshot(np, emb)
    report["values_before"] = exp.verify_values(emb, before["rows"]["keys"], "Before resize")
    report["memory_before_resize"] = memory_status()
    report["phase"] = "resize_and_full_validation"
    exp.checked_resize(emb, target, report.setdefault("resize", {}))
    check(abs(target - initial) == args.high_bytes - args.low_bytes, "Wrong payload delta")
    report["phase"] = "complete"
    del emb
    gc.collect()
    exp.torch.cuda.synchronize()


def pair_snapshots(exp, embeddings, label):
    snaps = [snapshot(exp.np, emb) for emb in embeddings]
    compare_snapshots(exp.np, *snaps, label)
    return snaps


def run_functional(exp, report):
    np, args = exp.np, exp.args
    stages = report.setdefault("functional_stages", [])
    low_sets = args.low_bytes // (WAYS * args.row_bytes)
    report["phase"] = "functional_public_frequency_decay_strict_admission"
    pair = [exp.make(args.low_bytes, False), exp.make(args.low_bytes, True)]
    policy = {"name": "public_insert_frequency_decay_strict_admission", "passed": False,
              "priority_rule": "float32(frequency) * float32(1 / total_batch_keys)",
              "source": "cuda_ops/gather_keys_data_ptrs.cuh: GatherKeysAndDataPtrs and DefaultGPUHistogram",
              "prior_failed_expectation": "functional-r1 incorrectly expected raw frequency20; original histogram normalizes20/160 to0.125. Native code is unchanged.",
              "observations": {}}
    stages.append(policy)

    def observe(label):
        observations = [snapshot(np, emb) for emb in pair]
        policy["observations"][label] = {backend: policy_evidence(np, snap)
                                        for backend, snap in zip(("original", "vmm"), observations)}
        compare_snapshots(np, *observations, label)
        return observations

    # Each candidate batch includes filler in a different modulo bucket, so
    # its frequency can vary independently of the batch's normalization.
    base = np.arange(WAYS, dtype=np.int64) * low_sets
    candidate = WAYS * low_sets
    filler = 1
    for emb in pair:
        exp.insert(emb, np.repeat(base, 20))
    first = observe("initial_normalized_frequency")
    check(np.array_equal(first[0]["rows"]["keys"], base), "Initial policy fixture keys differ")
    check(np.all(first[0]["rows"]["bits"].view(np.float32) == np.float32(0.125)),
          "Original public histogram did not produce normalized20/160 priority")
    check(first[0]["state"]["decay_rate"] == float(np.float32(0.95)), "Fixture requires original0.95 decay")
    for emb in pair:
        exp.insert(emb, np.r_[np.int64(candidate), np.full(159, filler, dtype=np.int64)])
    below = observe("below_priority_and_observed_native_decay")
    check(np.array_equal(below[0]["rows"]["keys"], np.sort(np.r_[base, np.int64(filler)])),
          "Below-threshold candidate replaced a resident")
    mask = np.isin(below[0]["rows"]["keys"], base)
    decayed = below[0]["rows"]["bits"][mask].view(np.float32)
    check(np.all(decayed == decayed[0]) and 0 < decayed[0] < np.float32(0.125), "Unexpected native lazy decay")
    # Dividing by the exact power-of-two initial score recovers the actual
    # __powf(decay,1) factor without assuming CPU pow and CUDA pow agree.
    factor = np.float32(decayed[0] / np.float32(0.125))
    target = np.float32(decayed[0] * factor)
    equal_count, equal_total = normalized_priority_match(np, target, exp.batch_rows)
    policy.update(initial_frequency=20, initial_batch_rows=160, initial_priority=0.125,
                  observed_native_decay_factor=float(factor),
                  observed_native_decay_factor_bits=int(factor.view(np.uint32)),
                  exact_equal_candidate_count=equal_count, exact_equal_batch_rows=equal_total,
                  exact_equal_priority=float(target), exact_equal_priority_bits=int(target.view(np.uint32)))
    equal_keys = np.r_[np.full(equal_count, candidate, dtype=np.int64),
                      np.full(equal_total - equal_count, filler, dtype=np.int64)]
    for emb in pair:
        exp.insert(emb, equal_keys)
    equal = observe("strict_bit_exact_equal_priority")
    check(np.array_equal(equal[0]["rows"]["keys"], np.sort(np.r_[base, np.int64(filler)])),
          "Bit-exact equal-priority candidate replaced a resident")
    mask = np.isin(equal[0]["rows"]["keys"], base)
    check(np.all(equal[0]["rows"]["bits"][mask] == target.view(np.uint32)),
          "Measured resident priority differs from exact normalized candidate priority")
    for emb in pair:
        exp.insert(emb, np.array([candidate], dtype=np.int64))
    higher = observe("greater_normalized_priority")
    check(candidate in higher[0]["rows"]["keys"], "Priority1 candidate was not admitted")
    policy["values_after"] = {label: exp.verify_values(emb, higher[0]["rows"]["keys"], "Original kernels including filler")
                               for label, emb in zip(("original", "vmm"), pair)}
    policy.update(passed=True, bit_exact_equal_priority_no_replacement=True,
                  below_priority_no_replacement=True, above_priority_admitted=True)
    del emb, pair
    gc.collect()
    exp.torch.cuda.synchronize()

    report["phase"] = "functional_dense_grow"
    pair = [exp.make(args.low_bytes, False), exp.make(args.low_bytes, True)]
    for emb in pair:
        exp.fill(emb, args.low_bytes)
        exp.verify_values(emb, np.arange(args.low_bytes // args.row_bytes, dtype=np.int64), "Functional dense initial")
        # Unequal real counters come from original public duplicate insert.
        hot = np.arange(min(32, low_sets), dtype=np.int64)
        exp.insert(emb, np.repeat(hot, 7))
    pair_snapshots(exp, pair, "Dense pre-grow original/VMM")
    grow = {"name": "dense_grow", "original": {}, "vmm": {}}
    stages.append(grow)
    for emb, label in zip(pair, ("original", "vmm")):
        exp.checked_resize(emb, args.high_bytes, grow[label], vmm=label == "vmm")
    pair_snapshots(exp, pair, "Grow original/VMM")
    grow["passed"] = True
    # Fill all high-capacity slots using matched-age merge pairs. Real counters
    # may differ across ways due to earlier accesses; oracle reads exact bits.
    report["phase"] = "functional_dense_shrink"
    for emb in pair:
        exp.fill(emb, args.high_bytes)
        exp.verify_values(emb, np.arange(args.high_bytes // args.row_bytes, dtype=np.int64), "Functional full before shrink")
    pair_snapshots(exp, pair, "Dense pre-shrink original/VMM")
    shrink = {"name": "dense_shrink_real_LFU_evictions", "original": {}, "vmm": {}}
    stages.append(shrink)
    for emb, label in zip(pair, ("original", "vmm")):
        exp.checked_resize(emb, args.low_bytes, shrink[label], vmm=label == "vmm")
    survived = pair_snapshots(exp, pair, "Shrink original/VMM")
    for emb in pair:
        exp.insert(emb, np.repeat(survived[0]["rows"]["keys"][:32], 3))
        exp.verify_values(emb, survived[0]["rows"]["keys"], "Original kernels after shrink")
    pair_snapshots(exp, pair, "Post-shrink original insert/lookup")
    shrink.update(passed=True, original_kernels_continue_after_resize=True)
    report["phase"] = "functional_regrow_refill"
    regrow = {"name": "regrow_then_refill_all_new_capacity", "original": {}, "vmm": {}}
    stages.append(regrow)
    for emb, label in zip(pair, ("original", "vmm")):
        exp.checked_resize(emb, args.high_bytes, regrow[label], vmm=label == "vmm")
        regrow[label]["refill"] = exp.fill(emb, args.high_bytes)
        regrow[label]["refill_values"] = exp.verify_values(
            emb, np.arange(args.high_bytes // args.row_bytes, dtype=np.int64), "Full regrown capacity")
    pair_snapshots(exp, pair, "Regrow and refill original/VMM")
    regrow["passed"] = True
    del emb, pair
    gc.collect()
    exp.torch.cuda.synchronize()

    report["phase"] = "functional_mixed_age_refusal"
    pair = [exp.make(args.high_bytes, False), exp.make(args.high_bytes, True)]
    rejection = {"name": "mixed_age_merge_refusal_unchanged", "original": {}, "vmm": {}}
    stages.append(rejection)
    keys = np.arange(16, dtype=np.int64) * low_sets
    for emb, label in zip(pair, ("original", "vmm")):
        exp.insert(emb, keys[::2].copy())
        exp.insert(emb, keys[1::2].copy())
        # Values checked before the baseline, then read-only state snapshots.
        exp.verify_values(emb, keys, "Before mixed-age rejection")
        before = snapshot(np, emb)
        check(len(np.unique(before["rows"]["ages"])) == 2, "Mixed-age fixture did not produce two ages")
        plan = decode(emb.emb_layer.plan_cache_resize(args.low_bytes, args.host_limit_gib * GIB))
        check(plan["status"] == "rejected" and plan["reason"] == "mixed_age_merge", "Mixed-age plan did not refuse")
        after = snapshot(np, emb)
        compare_snapshots(np, before, after, "Rejected preflight", raw=True)
        # Apply must independently reject the same unsafe merge without change.
        applied = decode(emb.emb_layer.apply_cache_resize(args.low_bytes, before["state"]["generation"],
                          args.pool_limit_gib * GIB, args.host_limit_gib * GIB))
        check(applied["status"] == "rejected" and applied["reason"] == "mixed_age_merge", "Mixed-age apply did not refuse")
        compare_snapshots(np, before, snapshot(np, emb), "Rejected apply", raw=True)
        value_check = exp.verify_values(emb, keys, "After rejected plan and apply")
        rejection[label] = {"plan": plan, "apply": applied,
                            "all_raw_metadata_bytes_and_native_state_unchanged": True,
                            "all_resident_value_bytes_unchanged": value_check,
                            "before": brief(np, before)}
        exp.insert(emb, np.array([1], dtype=np.int64))
        exp.verify_values(emb, np.r_[keys, np.int64(1)], "Original kernels after rejection")
    pair_snapshots(exp, pair, "Rejected resize then original kernels")
    rejection.update(passed=True, original_kernels_continue_after_refusal=True)
    del emb, pair
    gc.collect()
    exp.torch.cuda.synchronize()
    report["phase"] = "complete"


def parameters(args):
    if args.low_mib is None:
        args.low_mib = 3072 if args.mode == "case" else 2 * args.chunk_mib
    if args.high_mib is None:
        args.high_mib = 2 * args.low_mib
    args.low_bytes, args.high_bytes = args.low_mib * MIB, args.high_mib * MIB
    if args.row_bytes <= 0 or args.row_bytes % 16:
        raise ValueError("row-bytes must be positive and divisible by16")
    if args.low_bytes <= 0 or args.high_bytes != 2 * args.low_bytes:
        raise ValueError("Require exact doubling: high-mib=2*low-mib")
    if any(size % (args.chunk_mib * MIB) or size % (WAYS * args.row_bytes)
           for size in (args.low_bytes, args.high_bytes)):
        raise ValueError("Both payload capacities must align to chunk and native8-way set")
    if args.batch_mib <= 0 or args.batch_mib * MIB < 16 * args.row_bytes:
        raise ValueError("batch-mib must hold at least one full merge pair")
    if args.high_bytes // args.row_bytes >= 2 ** 32:
        raise ValueError("Synthetic value encoding supports keys below2**32")
    if args.repeat_index < 0 or args.host_limit_gib <= 0 or args.pool_limit_gib <= 0:
        raise ValueError("Invalid repeat or memory limit")
    return {"mode": args.mode, "direction": args.direction, "repeat_index": args.repeat_index,
            "chunk_bytes": args.chunk_mib * MIB, "low_payload_bytes": args.low_bytes,
            "high_payload_bytes": args.high_bytes, "payload_delta_bytes": args.high_bytes - args.low_bytes,
            "row_bytes": args.row_bytes, "ways": WAYS, "batch_value_bytes": args.batch_mib * MIB,
            "fast_rehash": args.fast_rehash, "host_staging_limit_bytes": args.host_limit_gib * GIB,
            "device_pool_limit_bytes": args.pool_limit_gib * GIB,
            "expected_low_rows": args.low_bytes // args.row_bytes,
            "expected_high_rows": args.high_bytes // args.row_bytes,
            "runtime_root": str(args.runtime_root) if args.runtime_root else None}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("functional", "case"), required=True)
    result.add_argument("--direction", choices=("grow", "shrink"), default="grow")
    result.add_argument("--chunk-mib", type=int, choices=(2, 16, 64), default=2)
    result.add_argument("--repeat-index", type=int, default=0)
    result.add_argument("--low-mib", type=int)
    result.add_argument("--high-mib", type=int)
    result.add_argument("--row-bytes", type=int, default=2048)
    result.add_argument("--batch-mib", type=int, default=64)
    result.add_argument("--fast-rehash", type=int, choices=(0, 1), default=1)
    result.add_argument("--host-limit-gib", type=int, default=40)
    result.add_argument("--pool-limit-gib", type=int, default=24)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--runtime-root", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--describe", action="store_true", help="Validate parameters and print geometry without imports/CUDA")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        config = parameters(args)
    except ValueError as error:
        cli.error(str(error))
    if args.describe:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if args.runtime_root is None or args.output_dir is None:
        cli.error("--runtime-root and --output-dir are required for execution")
    report = {"schema_version": 1, "status": "starting", "phase": "initialization",
              "configuration": config, "pid": os.getpid(),
              "backend": "actual_NVEmbedding_original_modulo_hash_and_LFU_with_optional_VMM_values",
              "workload": "synthetic_dense_public_insert; no request replay or model forwards",
              "timing_scope": "already_quiescent; includes explicit pre-sync, native plan, apply including context handling, and completion sync; excludes preparation and external validation",
              "instrumentation": "whole-call perf_counter only; no per-CUDA-call hooks",
              "resident_hit_source": "exact native tag metadata; parameter-server fallback never defines hits",
              "value_encoding": "all32 key bits as four exact fp16 bytes1..256, repeated with column-block offset",
              "memory_start": memory_status(), "all_slow_samples_retained": True}
    output = args.output_dir / "result.json"
    write_json(output, report)
    try:
        experiment = Experiment(args, report)
        with experiment.torch.inference_mode():
            (run_case if args.mode == "case" else run_functional)(experiment, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["memory_end"] = memory_status()
        report["cleanup_scope"] = "Normal path releases Python owners and synchronizes; failure-path cleanup is not asserted. Runner must also check process exit status."
        write_json(output, report)
    print(json.dumps({"status": report["status"], "phase": report["phase"], "result": str(output)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
