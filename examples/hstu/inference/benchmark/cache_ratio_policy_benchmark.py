#!/usr/bin/env python3
"""Full frozen Kuai replay with fixed, observe-only and adaptive cache modes.

Reuses the real fixed benchmark's HSTU model and batching. Input/PS setup always
uses the complete source; smoke selects only a measured prefix without changing
the full-source arrival scale. No checkpoint, simulated model or injected wait.
"""
import argparse
from array import array
from bisect import bisect_right
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

if __package__:
    from . import cache_ratio_benchmark as fixed
else:
    import cache_ratio_benchmark as fixed

GIB = 1 << 30
MIB = 1 << 20
REQUEST_COUNT = 742031
BUDGET = 16 * GIB
CHUNK = 64 * MIB
SEED = 303
SOURCE_MANIFEST_SHA = "dda2b34cf850af66fadd48b678b6f234fbea4e5aef51eebb329202f967d442e1"
ARRIVAL_SHA = "5ccefa25d90cdd4a7e10e731217b922ca532570ba30e73c016e7191f7d335d4b"
ARRIVAL_SCALE = 0.0030611885565087742
SOURCE_BASE_COMMIT = "776cad2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_loaded_inputs():
    require(gc.isenabled(), "Automatic GC must remain enabled")
    record = {"protocol": "preserve image-frozen objects; collect+freeze full loaded inputs before model; subsequent automatic GC stays enabled",
              "preexisting_frozen_objects": gc.get_freeze_count(),
              "thresholds": list(gc.get_threshold()), "collected_before_freeze": gc.collect()}
    gc.freeze()
    record["frozen_objects"] = gc.get_freeze_count()
    record["automatic_gc_enabled_after_freeze"] = gc.isenabled()
    return record


def sha_file(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * MIB), b""):
            digest.update(data)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"Input changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def validate_lock(lock):
    require(lock["source"]["manifest_sha256"] == SOURCE_MANIFEST_SHA, "Wrong frozen source manifest")
    require(lock["requests"]["count"] == REQUEST_COUNT and
            lock["requests"]["source_ids_half_open"] == [0, REQUEST_COUNT], "Wrong full source coverage")
    require(lock["arrivals"]["effective_arrivals_sha256"] == ARRIVAL_SHA, "Wrong frozen arrival identity")
    require(lock["arrivals"]["selection"]["arrival_scale"] == ARRIVAL_SCALE, "Wrong frozen arrival scale")
    require(lock["arrivals"]["last_arrival_s"] == 3710.15, "Wrong last frozen arrival")
    require(lock["batching"] == {"max_batch_size": 32, "max_batch_tokens": 32768, "max_batch_delay_ms": 1.0},
            "Wrong frozen batching contract")
    require(lock["experiment_conditions"]["cache_budget_bytes"] == BUDGET, "Wrong payload budget")


def scale_arrivals(rows, scale, expected_count, expected_hash):
    """Called once on freshly loaded full source; exact historical float order."""
    require(len(rows) == expected_count and bool(rows), "Full source count differs")
    require(all("source_arrival_s" not in row for row in rows), "Arrivals were already transformed")
    origin = rows[0]["arrival_s"]
    span = rows[-1]["arrival_s"] - origin
    require(span > 0, "Invalid source arrival span")
    calculated = (len(rows) - 1) / 200.0 / span
    require(calculated == scale, "Frozen scale does not match the complete source")
    digest = hashlib.sha256()
    previous = -1.0
    for index, row in enumerate(rows):
        require(row["request_id"] == index, "Request IDs are not the full chronological sequence")
        source = row["arrival_s"]
        arrival = (source - origin) * scale
        require(math.isfinite(arrival) and arrival >= previous, "Invalid scaled arrival order")
        row["source_arrival_s"], row["arrival_s"] = source, arrival
        digest.update(json.dumps([index, float(arrival)], separators=(",", ":")).encode() + b"\n")
        previous = arrival
    require(digest.hexdigest() == expected_hash, "Effective arrival hash differs from WORKLOAD_LOCK")
    return {"full_effective_arrivals_sha256": digest.hexdigest(), "scale": scale,
            "source_origin_s": origin, "source_span_s": span,
            "last_effective_arrival_s": rows[-1]["arrival_s"], "transforms_applied": 1}


def load_inputs(args):
    lock = json.loads(args.workload_lock.read_text())
    validate_lock(lock)
    source = args.trace_dir.resolve()
    manifest_receipt = sha_file(source / "manifest.json")
    require(manifest_receipt["sha256"] == SOURCE_MANIFEST_SHA, "Actual manifest SHA differs")
    artifacts = {}
    for name, expected in lock["source"]["artifacts_sha256"].items():
        require(Path(name).name == name, "Unsafe source artifact name")
        artifacts[name] = sha_file(source / name)
        require(artifacts[name]["sha256"] == expected, "Source artifact SHA differs: " + name)
    manifest, sequences, rows = fixed.load_trace(source, 32768)
    require(manifest["request_count"] == REQUEST_COUNT and manifest["seed"] == SEED, "Wrong source count/seed")
    require(manifest["max_history_len"] == 4096 and manifest["num_candidates"] == 100, "Wrong source history/candidates")
    require(manifest["dynamic_table_rows"] == {"user_id": 26575, "video_id": 9988947}, "Wrong full-source tables")
    require(len(sequences) == 26575, "Full source sequences missing")
    arrivals = scale_arrivals(rows, ARRIVAL_SCALE, REQUEST_COUNT, ARRIVAL_SHA)
    require(rows[-1]["arrival_s"] == 3710.15, "Full last arrival differs")
    identity = {"source_dir": str(source), "manifest": manifest_receipt,
                "artifacts": artifacts, "workload_lock": sha_file(args.workload_lock),
                "source_requests": REQUEST_COUNT, "sequence_count": len(sequences),
                "filters": manifest.get("filters"), "arrival": arrivals,
                "scope": "complete frozen derived Kuai source; original history-cap exclusions remain"}
    return manifest, sequences, rows, identity


def arrival_index(rows):
    times, tokens = array("d"), array("Q", [0])
    for row in rows:
        times.append(row["arrival_s"])
        tokens.append(tokens[-1] + row["tokens"])

    def snapshot(now):
        count = bisect_right(times, now)
        return count, tokens[count]
    return times, snapshot


def population_stats(arrivals, departures):
    """Integrate arrived-minus-departed; merge equal-time events before peak."""
    require(len(arrivals) == len(departures), "Population event counts differ")
    require(all(a <= b for a, b in zip(arrivals, departures)), "Departure preceded arrival")
    require(all(a <= b for a, b in zip(departures, departures[1:])), "Departures are not ordered")
    i = j = pending = peak = 0
    previous = arrivals[0] if arrivals else 0.0
    areas = []
    while i < len(arrivals) or j < len(departures):
        now = min(arrivals[i] if i < len(arrivals) else math.inf,
                  departures[j] if j < len(departures) else math.inf)
        areas.append(pending * (now - previous))
        while i < len(arrivals) and arrivals[i] == now:
            i += 1
            pending += 1
        while j < len(departures) and departures[j] == now:
            j += 1
            pending -= 1
        require(pending >= 0, "Negative population")
        peak = max(peak, pending)
        previous = now
    require(pending == 0, "Population did not fully drain")
    return {"peak_requests": peak, "area_request_ms": math.fsum(areas) * 1000,
            "final_pending": pending}


def fixed_windows(arrivals, dispatches, ready, width=10.0):
    """Half-open wall-clock windows; the last window includes final ready."""
    count = max(1, math.floor(ready[-1] / width) + 1)
    for index in range(count):
        start, end = index * width, (index + 1) * width
        # bisect_left gives exact [start,end) arrivals/completions, including
        # all tied events together; terminal extra window handles boundary ready.
        from bisect import bisect_left
        a0, a1 = bisect_left(arrivals, start), bisect_left(arrivals, end)
        d0, d1 = bisect_left(dispatches, start), bisect_left(dispatches, end)
        r0, r1 = bisect_left(ready, start), bisect_left(ready, end)
        yield {"window_index": index, "start_s": start, "end_s": end,
               "arrivals": a1 - a0, "dispatched": d1 - d0, "completed": r1 - r0,
               "completion_rate_requests_s": (r1 - r0) / width,
               "queue_at_start_left": a0 - d0, "queue_at_end_left": a1 - d1,
               "outstanding_at_start_left": a0 - r0, "outstanding_at_end_left": a1 - r1}


def native_identity(runtime_root):
    root = runtime_root.resolve()
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        nve = path.name.startswith("libnve") or path.name.startswith("nve.")
        kv = "kvcache_cpp" in path.name
        if ".so" not in path.name or not (nve or kv):
            continue
        require(not str(path).endswith(" (deleted)"), "Deleted loaded native library")
        path = path.resolve()
        if nve:
            require(path.is_relative_to(root), f"Mixed NVE runtime: {path}")
        libraries[str(path)] = sha_file(path)
    require(any(Path(p).name.startswith("libnve-common") for p in libraries), "NVE common library absent")
    require(any(Path(p).name.startswith("nve.") for p in libraries), "NVE extension absent")
    require(any("kvcache_cpp" in Path(p).name for p in libraries), "Native KV extension absent")
    return libraries


def code_identity():
    hstu = fixed.HSTU_DIR
    paths = [Path(__file__), Path(fixed.__file__), hstu / "model/inference_ranking_gr.py",
             hstu / "inference/inference_gr_ranking.py", hstu / "modules/nve_embeddingcollection.py",
             hstu / "inference/configs/kuairand_1k_inference_ranking.gin"]
    paths += sorted((hstu / "inference/cache_ratio").rglob("*.py"))
    return {str(path.relative_to(hstu)): sha_file(path) for path in paths if "tests" not in path.parts}


def model_parameter_hash(backend):
    digest, count = hashlib.sha256(), 0
    for name, parameter in sorted(backend.model.named_parameters()):
        if "_dynamic_embedding_collection" in name:
            continue
        require(parameter.layout == backend.torch.strided, "Unexpected nonsparse model parameter layout")
        digest.update(name.encode() + b"\0")
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode() + b"\0")
        digest.update(parameter.detach().cpu().contiguous().view(backend.torch.uint8).numpy().tobytes())
        count += parameter.numel()
    require(count > 0, "No dense/static model parameters hashed")
    return {"sha256": digest.hexdigest(), "elements": count, "scope": "dense/static named parameters; dynamic PS values follow seeded initialization contract"}


def validate_backend(backend):
    memory = backend.model.cache_memory
    state = memory.stats()
    require(state["state"] == "open", "Shared VMM backend is not open")
    require(state["payload_budget_bytes"] == BUDGET and state["unused_payload_bytes"] == 0,
            "Initial whole-chunk allocation does not consume exactly16GiB payload")
    require(state["kv"]["payload_bytes"] == 126 * CHUNK, "Initial actual KV must be7.875GiB")
    require(state["embedding"]["payload_bytes"] == 130 * CHUNK, "Initial actual embedding must be8.125GiB")
    require(state["kv"]["payload_bytes"] + state["embedding"]["payload_bytes"] == BUDGET, "Payload ledger mismatch")
    require(state["kv"]["backend"] == "gpu_only_vmm" and state["kv"]["chunk_bytes"] == CHUNK,
            "Native KV VMM/chunk backend differs")
    require(memory.kv.host_kvstorage_manager is None and memory.kv.num_buffer_pages == 0,
            "CPU KV or buffer pages are enabled")
    tables = state["embedding"]["tables"]
    require(set(tables) == {"user_id", "video_id"}, "Wrong dynamic embedding tables")
    for name, table in tables.items():
        require(table.get("vmm") is True and table.get("per_row_ages") is True,
                "Native general VMM storage/row ages not active: " + name)
        require(table.get("rehash_mode") == "native_modulo_general_row_ages_cpu_staging",
                "Native SDK ignored GENERAL mode: " + name)
        require(table["chunk_bytes"] == CHUNK and table["payload_bytes"] % CHUNK == 0 and table["payload_bytes"] >= CHUNK,
                "Embedding chunk geometry differs: " + name)
        require(table["mapped_value_bytes"] == table["allocated_value_bytes"] == table["payload_bytes"],
                "Embedding physical/value ledger differs: " + name)
    return {"passed": True, "cpu_kv_bytes": 0, "kv_vmm": True, "embedding_general_vmm": True,
            "actual_initial_kv_ratio": 126 / 256, "state": state}


def calibration(path, libraries):
    if path is None:
        return None, {"status": "missing", "effect": "policy receives None and must HOLD when resize cost is required"}
    data = json.loads(path.read_text())
    require(data["schema_version"] == 1 and data["backend"] == "nvemb_vmm_general_rehash", "Wrong calibration backend")
    require(data["chunk_bytes"] == CHUNK and data["payload_budget_bytes"] == BUDGET, "Wrong calibration geometry")
    cost = data["resize_cost_ms"]
    require(type(cost) in (int, float) and math.isfinite(cost) and cost > 0, "Calibration cost must be finite and positive")
    actual = {Path(p).name: record["sha256"] for p, record in libraries.items()}
    expected = data["library_sha256"]
    require(bool(expected) and any(name.startswith("libnve-common") for name in expected), "Calibration library identity missing")
    require(all(actual.get(name) == sha for name, sha in expected.items()), "Calibration native library differs")
    require(bool(data.get("method")) and bool(data.get("source_receipts")), "Calibration method/evidence missing")
    return float(cost), {"status": "loaded", "file": sha_file(path), "record": data}


def replay(rows, backend, runtime, args, report):
    arrivals, arrival_snapshot = arrival_index(rows)
    dispatches, ready_times = array("d"), array("d")
    queue, service, e2e = [], [], []
    head = batches = 0
    observer_total_ms = before_observer_ms = 0.0
    started = time.perf_counter()
    clock = lambda: time.perf_counter() - started
    if runtime is not None:
        runtime.start(clock)
    report["replay_started_unix_s"] = time.time()
    report.update(completed_requests=0, completed_batches=0, complete=False)
    last_progress = 0.0
    with (args.output_dir / "requests.jsonl").open("w") as requests_log, \
         (args.output_dir / "batches.jsonl").open("w") as batches_log, \
         (args.output_dir / "policy_actions.jsonl").open("w") as actions_log:
        while head < len(rows):
            report["phase"] = "batch_selection"
            end, reason, wake = fixed.choose_batch(rows, head, clock(), 32, 32768)
            if end == head:
                time.sleep(min(0.25, max(0.0, wake - clock())))
                continue
            batch = rows[head:end]
            before_start = time.perf_counter()
            if runtime is not None:
                runtime.before_batch()
            before_ms = (time.perf_counter() - before_start) * 1000
            before_observer_ms += before_ms
            dispatch = clock()
            report["phase"] = "forward"
            output = backend.forward(batch)
            backend.synchronize()
            ready = clock()
            report["phase"] = "validation_and_observation"
            backend.validate(output)
            for row in batch:
                q, s = (dispatch - row["arrival_s"]) * 1000, (ready - dispatch) * 1000
                require(q >= 0 and s >= 0, "Invalid request clock ordering")
                queue.append(q)
                service.append(s)
                e2e.append(q + s)
                dispatches.append(dispatch)
                ready_times.append(ready)
                emit(requests_log, {"request_id": row["request_id"], "cache_uid": row["cache_uid"],
                                    "batch_index": batches, "arrival_s": row["arrival_s"],
                                    "dispatch_s": dispatch, "ready_s": ready,
                                    "queue_ms": q, "service_ms": s, "e2e_ms": q + s,
                                    "tokens": row["tokens"], "history_len": row["history_len"],
                                    "num_candidates": len(row["candidates"])})
            event = None
            observer_start = time.perf_counter()
            if runtime is not None:
                event = runtime.after_batch(completed_count=end, arrival_snapshot=arrival_snapshot,
                                            final=end == len(rows))
            observer_ms = (time.perf_counter() - observer_start) * 1000 if runtime is not None else 0.0
            observer_total_ms += observer_ms
            if event and (event.get("window") is not None or event.get("decision") is not None or event.get("action") is not None):
                emit(actions_log, {"batch_index": batches, "completed_requests": end,
                                   "event_recorded_s": clock(), "runtime_event": event})
            emit(batches_log, {"batch_index": batches, "request_ids": [r["request_id"] for r in batch],
                               "batch_size": len(batch), "tokens": sum(r["tokens"] for r in batch),
                               "reason": reason, "dispatch_s": dispatch, "ready_s": ready,
                               "before_observer_ms": before_ms if runtime is not None else 0.0,
                               "observer_wall_ms_inclusive": observer_ms,
                               "batch_observation": event.get("batch_observation") if event else None})
            head, batches = end, batches + 1
            report.update(completed_requests=head, completed_batches=batches)
            if batches % 100 == 0:
                for stream in (requests_log, batches_log, actions_log):
                    stream.flush()
            if clock() - last_progress >= 10:
                last_progress = clock()
                write_json(args.output_dir / "progress.json", {"elapsed_s": last_progress,
                           "completed_requests": head, "expected_requests": len(rows),
                           "completed_batches": batches, "complete": False})
        report["last_observation_and_logging_s"] = clock()
    queue_population = population_stats(arrivals, dispatches)
    system_population = population_stats(arrivals, ready_times)
    require(math.isclose(queue_population["area_request_ms"], math.fsum(queue), rel_tol=1e-8, abs_tol=0.001), "Queue area mismatch")
    require(math.isclose(system_population["area_request_ms"], math.fsum(e2e), rel_tol=1e-8, abs_tol=0.001), "System area mismatch")
    with (args.output_dir / "windows.jsonl").open("w") as stream:
        for window in fixed_windows(arrivals, dispatches, ready_times):
            emit(stream, window)
    horizon = ready_times[-1] - arrivals[0]
    report.update(complete=True, all_tails_retained=True, e2e_ms=fixed.distribution(e2e),
                  queue_ms=fixed.distribution(queue), service_ms=fixed.distribution(service),
                  queue_population=queue_population, system_population=system_population,
                  arrival_to_drain_s=horizon, last_arrival_s=arrivals[-1], last_ready_s=ready_times[-1],
                  drain_after_last_arrival_s=ready_times[-1] - arrivals[-1],
                  completion_throughput_requests_s=len(rows) / horizon,
                  mean_batch_size=len(rows) / batches,
                  observer_wall_ms_inclusive=observer_total_ms,
                  before_observer_wall_ms=before_observer_ms if runtime is not None else 0.0,
                  observer=runtime.stats() if runtime is not None else {"enabled": False})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace-dir", type=Path, required=True)
    result.add_argument("--workload-lock", type=Path, required=True)
    result.add_argument("--runtime-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("fixed", "observe_only", "adaptive"), required=True)
    result.add_argument("--scope", choices=("full", "smoke"), default="full")
    result.add_argument("--smoke-requests", type=int, default=128)
    result.add_argument("--resize-calibration", type=Path)
    result.add_argument("--peak-cache-gib", type=int, default=32)
    result.add_argument("--host-staging-limit-gib", type=int, default=40)
    result.add_argument("--round-index", type=int, default=1)
    result.add_argument("--validate-input-only", action="store_true")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not 32 <= args.smoke_requests <= 256 or args.peak_cache_gib < 16 or args.host_staging_limit_gib <= 0:
        cli.error("Require smoke32..256, peak-cache-gib>=16 and positive host staging limit")
    # The fixed HSTU setup consumes these existing argument names.
    args.cache_bytes, args.cache_ratio = BUDGET, 0.5
    args.max_batch_size, args.max_batch_tokens = 32, 32768
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "phase": "input_validation",
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_base_commit": SOURCE_BASE_COMMIT, "mode": args.mode, "scope": args.scope,
              "complete": False, "all_tails_retained": False,
              "comparison_scope": "matched new runner/current SDK; historical results are not matched controls",
              "startup_protocol": "same full-source PS/model setup and retained startup probes in all modes",
              "timing_scope": "one unpaused full-source arrival clock; observation/decisions/resize/validation/logging charge later queues; setup and final analysis/cleanup are separate",
              "weights_scope": "seed303 untrained performance fixture; no quality claim",
              "limits": ["200rps input-limited completion rate is not saturation throughput",
                         "startup numerical probes and all-batch finite checks are not full reference-output equivalence"]}
    backend = runtime = None
    frozen = False
    started = time.perf_counter()
    try:
        manifest, sequences, full_rows, report["input"] = load_inputs(args)
        rows = full_rows if args.scope == "full" else full_rows[:args.smoke_requests]
        report.update(expected_requests=len(rows), source_requests=REQUEST_COUNT,
                      measured_source_ids_half_open=[0, len(rows)], code_identity=code_identity())
        if args.validate_input_only:
            report.update(status="input_validated", phase="input_validation_complete")
        else:
            os.environ.update(NVE_NVEMB_VMM_GENERAL_REHASH="1", NVE_NVEMB_VMM_MAX_PAYLOAD_BYTES=str(BUDGET),
                              NVE_NVEMB_VMM_CHUNK_BYTES=str(CHUNK), NVE_NVEMB_VMM_FAST_REHASH="1",
                              NVE_NVEMB_POLICY_OBSERVATION="0" if args.mode == "fixed" else "1",
                              NVE_NVEMB_POLICY_EVENT_CAPACITY="65536")
            report["native_environment"] = {k: v for k, v in os.environ.items() if k.startswith("NVE_NVEMB_")}
            report["gc"] = freeze_loaded_inputs()
            frozen = True
            report["phase"] = "model_and_full_PS_setup"
            backend = fixed.HSTU(cache_memory_backend="vmm", peak_cache_bytes=args.peak_cache_gib * GIB,
                                 host_staging_limit_bytes=args.host_staging_limit_gib * GIB)
            backend.memory_options["vmm_chunk_bytes"] = CHUNK
            report["backend"] = backend.setup(args, manifest, sequences, full_rows)
            properties = backend.torch.cuda.get_device_properties(backend.torch.cuda.current_device())
            report["gpu_identity"] = {"name": properties.name, "uuid": str(properties.uuid),
                                      "total_memory_bytes": properties.total_memory}
            report["model_parameters"] = model_parameter_hash(backend)
            report["loaded_libraries"] = native_identity(args.runtime_root)
            report["backend_validation"] = validate_backend(backend)
            report["initial_capacity"] = report["backend_validation"]["state"]
            resize_cost, report["resize_calibration"] = calibration(args.resize_calibration, report["loaded_libraries"])
            if args.mode != "fixed":
                from inference.cache_ratio.ratio_policy.runtime import PolicyRuntime
                runtime = PolicyRuntime(backend.model, mode="observe" if args.mode == "observe_only" else "adaptive",
                                        resize_cost_ms=resize_cost)
            report["gc"]["automatic_gc_enabled_before_replay"] = gc.isenabled()
            require(report["gc"]["automatic_gc_enabled_before_replay"], "Automatic GC was disabled during model/runtime setup")
            report["setup_wall_s"] = time.perf_counter() - started
            write_json(args.output_dir / "summary.json", report)
            replay(rows, backend, runtime, args, report)
            report["final_capacity"] = backend.model.cache_memory.stats()
            report["phase"] = "cleanup"
            if runtime is not None:
                runtime.close()
                runtime = None
            backend.close()
            backend = None
            report.update(status="passed", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        for owner in (runtime, backend):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    cleanup_errors.append(repr(error))
        if frozen:
            gc.unfreeze()
            gc.collect()
            report["gc"]["cleanup_frozen_objects"] = gc.get_freeze_count()
        report["cleanup"] = {"errors": cleanup_errors, "complete": not cleanup_errors,
                             "scope": "runtime hooks and model closed; all frozen objects made GC-eligible before exit, without restoring the image's initial frozen set; process exit checked by serial runner"}
        if cleanup_errors:
            report["status"] = "failed"
        report["whole_process_work_wall_s"] = time.perf_counter() - started
        write_json(args.output_dir / "summary.json", report)
    print(json.dumps({"status": report["status"], "scope": args.scope, "mode": args.mode,
                      "completed": report.get("completed_requests", 0), "output": str(args.output_dir)}), flush=True)
    return 0 if report["status"] in ("passed", "input_validated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
