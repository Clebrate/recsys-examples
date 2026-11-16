#!/usr/bin/env python3
"""Independently audit full request/batch clocks and compare matched modes.

CPU only. Keep all tails, incomplete arms and errors. Complete matched rounds,
not isolated time windows, are the comparison units; no significance claim.
"""
import argparse
from array import array
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))
import cache_ratio_policy_benchmark as bench

MODES = ("fixed", "observe_only", "adaptive")


def close(actual, expected, name):
    bench.require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                  math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-5),
                  f"{name} differs: {actual!r} versus {expected!r}")


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def signature(report, compiler_cache=None):
    initial = report["initial_capacity"]
    return {"scope": report["scope"], "count": report["expected_requests"],
            "source_manifest": report["input"]["manifest"]["sha256"],
            "source_artifacts": report["input"]["artifacts"],
            "arrivals": report["input"]["arrival"], "seed": report["backend"]["seed"],
            "model_parameters": report["model_parameters"],
            "initialized_PS_rows": report["backend"]["initialized_embedding_rows"],
            "startup_probes": report["backend"]["probe_request_ids"],
            "startup_protocol": report["startup_protocol"], "gc_protocol": report["gc"]["protocol"],
            "gc_preexisting_frozen_objects": report["gc"]["preexisting_frozen_objects"],
            "gc_thresholds": report["gc"]["thresholds"], "gpu": report["gpu_identity"],
            "compiler_cache_seed": compiler_cache["seed_identity"] if compiler_cache is not None else None,
            "resize_calibration": report.get("resize_calibration"),
            "host_staging_limit_gib": report["configuration"].get("host_staging_limit_gib"),
            "code": report["code_identity"],
            "libraries": {Path(path).name: item["sha256"] for path, item in report["loaded_libraries"].items()},
            "payload_budget_bytes": initial["payload_budget_bytes"],
            "kv_payload_bytes": initial["kv"]["payload_bytes"],
            "embedding_payload_bytes": initial["embedding"]["payload_bytes"],
            "per_table_payload_bytes": {name: value["payload_bytes"] for name, value in initial["embedding"]["tables"].items()},
            "peak_limit_bytes": initial["peak_limit_bytes"],
            "chunk_bytes": initial["kv"]["chunk_bytes"],
            "round_index": report["configuration"]["round_index"]}


def audit(directory):
    report = json.loads((directory / "summary.json").read_text())
    result = {"directory": str(directory), "mode": report.get("mode"), "scope": report.get("scope"),
              "round_index": report.get("configuration", {}).get("round_index"),
              "status": "failed", "reported_status": report.get("status"), "errors": [], "report": report}
    try:
        cache_path = directory.with_name(directory.name + ".compiler_cache.json")
        compiler_cache = None
        if cache_path.exists():
            compiler_cache = json.loads(cache_path.read_text())
            bench.require(compiler_cache["schema_version"] == 1 and compiler_cache["status"] == "ready" and
                          compiler_cache["verified_copy_before_relocation"] is True,
                          "Compiler-cache preparation did not verify its copy")
            cache_directory = Path(compiler_cache["cache_directory"])
            bench.require(compiler_cache["environment"] == {
                "TRITON_CACHE_DIR": str(cache_directory / "triton"), "CUDA_CACHE_PATH": str(cache_directory / "cuda")},
                "Compiler-cache environment differs from private directories")
            result["compiler_cache_receipt_identity"] = bench.sha_file(cache_path)
        result["compiler_cache"] = compiler_cache
        bench.require(report["status"] == "passed" and report["complete"] and report["all_tails_retained"], "Arm is incomplete or failed")
        bench.require(report["mode"] in MODES and report["scope"] in ("full", "smoke"), "Wrong mode/scope")
        expected = bench.REQUEST_COUNT if report["scope"] == "full" else report["configuration"]["smoke_requests"]
        bench.require(report["scope"] == "full" or 32 <= expected <= 256, "Smoke coverage is outside32..256")
        bench.require(report["expected_requests"] == report["completed_requests"] == expected, "Request coverage differs")
        bench.require(report["cleanup"]["complete"] and not report["cleanup"]["errors"], "Cleanup did not complete")
        bench.require(report["backend_validation"]["passed"] and report["backend_validation"]["cpu_kv_bytes"] == 0, "Backend validation absent")
        bench.require(report["input"]["manifest"]["sha256"] == bench.SOURCE_MANIFEST_SHA and
                      report["input"]["arrival"]["full_effective_arrivals_sha256"] == bench.ARRIVAL_SHA, "Frozen input identity differs")
        bench.require(report["input"]["arrival"]["scale"] == bench.ARRIVAL_SCALE and
                      report["input"]["arrival"]["transforms_applied"] == 1, "Arrival transform differs")
        arrivals, dispatched, ready = array("d"), array("d"), array("d")
        uids, batch_ids, tokens = array("q"), array("Q"), array("Q")
        queue, service, e2e = [], [], []
        digest = hashlib.sha256()
        for index, row in enumerate(records(directory / "requests.jsonl")):
            bench.require(row["request_id"] == index, "Missing, repeated or reordered request ID")
            a, d, r = row["arrival_s"], row["dispatch_s"], row["ready_s"]
            bench.require(all(math.isfinite(t) for t in (a, d, r)) and 0 <= a <= d <= r, "Invalid request clock")
            bench.require(not arrivals or a >= arrivals[-1], "Unordered arrival")
            q, s, total = (d - a) * 1000, (r - d) * 1000, (r - a) * 1000
            for name, value in (("queue_ms", q), ("service_ms", s), ("e2e_ms", total)):
                close(row[name], value, name)
            arrivals.append(a); dispatched.append(d); ready.append(r)
            uids.append(row["cache_uid"]); batch_ids.append(row["batch_index"]); tokens.append(row["tokens"])
            queue.append(q); service.append(s); e2e.append(total)
            digest.update(json.dumps([index, float(a)], separators=(",", ":")).encode() + b"\n")
        bench.require(len(arrivals) == expected, "Raw request coverage differs")
        result["actual_measured_arrival_sha256"] = digest.hexdigest()
        if report["scope"] == "full":
            bench.require(digest.hexdigest() == bench.ARRIVAL_SHA, "Raw full arrivals differ from lock")
        cursor = batches = 0
        for index, row in enumerate(records(directory / "batches.jsonl")):
            bench.require(row["batch_index"] == index and 0 < row["batch_size"] <= 32, "Invalid batch coverage/size")
            end = cursor + row["batch_size"]
            bench.require(row["request_ids"] == list(range(cursor, end)), "Batch source ID mapping differs")
            bench.require(all(i == index for i in batch_ids[cursor:end]), "Request/batch IDs disagree")
            bench.require(len(set(uids[cursor:end])) == end - cursor, "Batch repeats a shared KV UID")
            bench.require(sum(tokens[cursor:end]) == row["tokens"] <= 32768, "Batch token ledger differs")
            for i in range(cursor, end):
                close(dispatched[i], row["dispatch_s"], "Batch dispatch")
                close(ready[i], row["ready_s"], "Batch ready")
            cursor, batches = end, index + 1
        bench.require(cursor == expected and batches == report["completed_batches"], "Raw batch coverage differs")
        calculated = {"queue_ms": bench.fixed.distribution(queue), "service_ms": bench.fixed.distribution(service),
                      "e2e_ms": bench.fixed.distribution(e2e),
                      "queue_population": bench.population_stats(arrivals, dispatched),
                      "system_population": bench.population_stats(arrivals, ready),
                      "arrival_to_drain_s": ready[-1] - arrivals[0],
                      "drain_after_last_arrival_s": ready[-1] - arrivals[-1],
                      "mean_batch_size": expected / batches,
                      "completion_throughput_requests_s": expected / (ready[-1] - arrivals[0])}
        for name, value in calculated.items():
            if isinstance(value, dict):
                for field, number in value.items():
                    close(report[name][field], number, name + "." + field)
            else:
                close(report[name], value, name)
        close(calculated["queue_population"]["area_request_ms"], math.fsum(queue), "Queue integral")
        close(calculated["system_population"]["area_request_ms"], math.fsum(e2e), "System integral")
        observed_windows = list(records(directory / "windows.jsonl"))
        bench.require(observed_windows == list(bench.fixed_windows(arrivals, dispatched, ready)), "Fixed clock windows differ")
        reasons, action_status = Counter(), Counter()
        events = 0
        for row in records(directory / "policy_actions.jsonl"):
            events += 1
            event = row["runtime_event"]
            if event.get("decision"):
                reasons[event["decision"].get("reason", "unknown")] += 1
            if event.get("action"):
                action_status[event["action"].get("status", "unknown")] += 1
                bench.require(report["mode"] == "adaptive", "Nonadaptive mode executed a resize")
        if report["mode"] == "fixed":
            bench.require(events == 0 and report["observer"] == {"enabled": False}, "Fixed mode installed observation")
        if report["mode"] in ("fixed", "observe_only"):
            bench.require(report["final_capacity"]["actions"] == report["initial_capacity"]["actions"], "Control changed capacity")
        result.update(status="passed", actual_requests=expected, actual_batches=batches,
                      metrics=calculated, policy_events=events, decision_reasons=dict(reasons),
                      action_status_counts=dict(action_status), physical_resize_count=report["final_capacity"]["actions"],
                      match_signature=signature(report, compiler_cache),
                      log_identity={name: bench.sha_file(directory / name) for name in
                                    ("requests.jsonl", "batches.jsonl", "policy_actions.jsonl", "windows.jsonl", "summary.json")})
    except (KeyError, ValueError, TypeError, OSError, IndexError) as error:
        result["errors"].append(repr(error))
    return result


def flatten(run):
    metrics = run["metrics"]
    result = {"round": run["round_index"], "mode": run["mode"], "scope": run["scope"],
              "requests": run["actual_requests"], "batches": run["actual_batches"],
              "resize_count": run["physical_resize_count"]}
    for name in ("e2e_ms", "queue_ms", "service_ms"):
        result.update({name + "_" + field: value for field, value in metrics[name].items()})
    for name in ("queue_population", "system_population"):
        result.update({name + "_" + field: value for field, value in metrics[name].items()})
    result.update({name: value for name, value in metrics.items() if not isinstance(value, dict)})
    return result


def summarize(root):
    paths = sorted(root.glob("round*-*/summary.json"))
    runs = []
    for path in paths:
        try:
            runs.append(audit(path.parent))
        except (OSError, ValueError) as error:
            runs.append({"directory": str(path.parent), "status": "failed", "errors": [repr(error)]})
    serial_path = root / "serial_receipt.json"
    serial = json.loads(serial_path.read_text()) if serial_path.exists() else None
    errors, comparisons = [], []
    if serial is None or serial.get("status") != "passed":
        errors.append("Serial receipt missing or failed; clean process exit is unverified")
    else:
        declared_seed = serial.get("compiler_cache", {}).get("seed_identity")
        for run in runs:
            if run["status"] == "passed":
                exits = [x for x in serial["runs"] if x["round"] == run["round_index"] and x["mode"] == run["mode"]]
                if len(exits) != 1 or exits[0].get("returncode") != 0 or exits[0].get("status") != "passed":
                    run["status"] = "failed"
                    run["errors"].append("Clean matching subprocess exit missing")
                elif (run["match_signature"]["compiler_cache_seed"] != declared_seed or
                      exits[0].get("compiler_cache", {}).get("seed_identity") != declared_seed or
                      exits[0].get("compiler_cache", {}).get("receipt_identity") != run.get("compiler_cache_receipt_identity")):
                    run["status"] = "failed"
                    run["errors"].append("Compiler-cache seed/receipt differs from serial suite")
    for round_index in sorted({r.get("round_index") for r in runs if r.get("round_index") is not None}):
        group = [r for r in runs if r.get("round_index") == round_index]
        if len(group) != 3 or {r.get("mode") for r in group} != set(MODES) or any(r["status"] != "passed" for r in group):
            errors.append(f"Round{round_index} is not a complete validated three-arm comparison")
            continue
        reference = group[0]["match_signature"]
        if any(r["match_signature"] != reference for r in group):
            errors.append(f"Round{round_index} identities/settings do not match")
            continue
        by_mode = {r["mode"]: flatten(r) for r in group}
        for treatment, control in (("observe_only", "fixed"), ("adaptive", "fixed"), ("adaptive", "observe_only")):
            left, right = by_mode[treatment], by_mode[control]
            comparisons.append({"round": round_index, "treatment": treatment, "control": control,
                                "scope": reference["scope"],
                                "differences_treatment_minus_control": {key: left[key] - right[key] for key in left
                                    if key not in ("round", "mode", "scope", "requests", "batches")},
                                "e2e_mean_relative_change": left["e2e_ms_mean"] / right["e2e_ms_mean"] - 1})
    if not runs:
        errors.append("No arm summaries found")
    if any(r["status"] != "passed" for r in runs):
        errors.append("One or more arm audits failed")
    signatures = [{key: value for key, value in run["match_signature"].items() if key != "round_index"}
                  for run in runs if run["status"] == "passed"]
    if signatures and any(value != signatures[0] for value in signatures[1:]):
        errors.append("Identity/settings changed across matched rounds")
    return {"schema_version": 1, "status": "passed" if not errors else "failed", "errors": errors,
            "serial_receipt": serial, "runs": runs, "comparisons": comparisons,
            "notes": ["Every raw request and batch was audited; all tails retained, nearest-rank full-request percentiles.",
                      "Difference sign is treatment minus control; each complete matched round is the comparison unit.",
                      "Observe-only minus fixed includes observation, policy-decision and diagnostic logging overhead; adaptive minus observe-only includes resize execution and resulting cache/queue changes.",
                      "Observation, GC, resizing, refusal and refill costs remain on the original arrival clock; no cost subtraction.",
                      "Fixed10s windows are descriptive correlated slices, not independent repeats. No significance or global optimum claim.",
                      "Input-limited200rps completion rate does not establish saturation throughput.",
                      "Zero applied resize means this run cannot establish a benefit from changing the ratio.",
                      "Smoke remains a full-source-initialized prefix at unchanged arrivals and is never reported as full coverage."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_root", type=Path)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    output = args.output_dir or args.results_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    result = summarize(args.results_root.resolve())
    bench.write_json(output / "summary.json", result)
    rows = [flatten(run) for run in result["runs"] if run["status"] == "passed"]
    with (output / "metrics.csv").open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    deltas = [{"round": row["round"], "scope": row["scope"], "treatment": row["treatment"], "control": row["control"],
               **row["differences_treatment_minus_control"], "e2e_mean_relative_change": row["e2e_mean_relative_change"]}
              for row in result["comparisons"]]
    with (output / "comparisons.csv").open("w", newline="") as stream:
        if deltas:
            writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
            writer.writeheader(); writer.writerows(deltas)
    print(json.dumps({"status": result["status"], "arms": len(result["runs"]), "errors": result["errors"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
