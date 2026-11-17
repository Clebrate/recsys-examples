#!/usr/bin/env python3
"""Real HSTU/general-NVEmb VMM resize calibration, not a serving benchmark.

Use the full frozen PS/model setup and a bounded real-request prefill. Require
both embedding tables to contain actual rows before measuring. Keep one model
for two rounds of equal 192 MiB transfers in both directions. Every tail remains;
the maximum observed non-noop action plus observation drain is the preset.
After every resize, replay the last prefill batch outside the resize timer and
compare with its saved CPU output. Those probes fill caches and touch ages, so
later samples include their real state changes; this is not an untouched loop.
This sparse prefilled fixture does not establish a full-cache worst-case cost.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
import cache_ratio_policy_benchmark as bench

TRANSFER_BYTES = 192 * bench.MIB
TRANSFERS = (-TRANSFER_BYTES, TRANSFER_BYTES, TRANSFER_BYTES, -TRANSFER_BYTES)
PREFILL_REQUESTS = 256
MAX_PREFILL_REQUESTS = 8192
PROBE_RTOL, PROBE_ATOL = 0.02, 0.005


def drain(memory):
    started = time.perf_counter()
    embedding = memory.embedding.policy_observation(reset=True)
    kv = memory.kv.drain_evictions()
    return {"embedding": embedding, "kv_evictions": kv,
            "wall_ms": (time.perf_counter() - started) * 1000}


def populated_embedding(observation):
    """Requests/PS finds alone do not prove that admission filled a cache."""
    tables = observation["embedding"]["tables"]
    if set(tables) != {"user_id", "video_id"}:
        raise ValueError("Population check requires both native embedding tables")
    for name, table in tables.items():
        if table.get("enabled") is not True or table.get("complete") is not True or table.get("overflow"):
            raise ValueError("Incomplete native population observation: " + name)
        resident, capacity = table.get("resident_rows"), table.get("capacity_rows")
        if type(resident) is not int or type(capacity) is not int or not 0 <= resident <= capacity:
            raise ValueError("Invalid native population observation: " + name)
    return all(table["resident_rows"] > 0 for table in tables.values())


def verify_transfer(before, after, delta):
    """Check usable payload, not just a requested ratio or VA reservation."""
    bench.require(after["kv"]["payload_bytes"] - before["kv"]["payload_bytes"] == delta,
                  "KV transfer differs from the requested byte delta")
    bench.require(after["embedding"]["payload_bytes"] - before["embedding"]["payload_bytes"] == -delta,
                  "Embedding transfer is not the equal opposite byte delta")
    bench.require(sum(after[side]["payload_bytes"] for side in ("kv", "embedding")) == bench.BUDGET,
                  "Transfer did not retain the complete shared payload budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_bytes, args.cache_ratio = bench.BUDGET, 0.5
    args.max_batch_size, args.max_batch_tokens = 32, 32768
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = args.output_dir / "resize_receipts.json"
    report = {"schema_version": 1, "status": "running", "phase": "input_validation",
              "configuration": {"payload_budget_bytes": bench.BUDGET, "chunk_bytes": bench.CHUNK,
                                "peak_cache_bytes": 32 * bench.GIB, "host_staging_limit_bytes": 40 * bench.GIB,
                                "seed": 303, "minimum_prefill_requests": PREFILL_REQUESTS,
                                "maximum_prefill_requests": MAX_PREFILL_REQUESTS, "rounds": 2,
                                "initial_nominal_ratio": 0.5, "kv_transfer_bytes_each_round": list(TRANSFERS)},
              "samples": [], "prefill_batches": [], "all_samples_retained": True,
              "scope": "cost calibration with sparse real-request prefill; not arrival-paced performance or a full-cache worst-case bound",
              "timing_scope": "already completed real prefill; explicit pre-sync, shared plan/apply, post-sync, post-resize observer drain and ledger read; no KV generation/model forward in timer",
              "probe_scope": "replay the same last real prefill batch after each resize, outside resize timing; real KV/embedding refill and age touches persist into later samples"}
    backend = None
    frozen = False
    started = time.perf_counter()
    try:
        manifest, sequences, full_rows, report["input"] = bench.load_inputs(args)
        report["code_identity"] = bench.code_identity()
        os.environ.update(NVE_NVEMB_VMM_GENERAL_REHASH="1", NVE_NVEMB_VMM_MAX_PAYLOAD_BYTES=str(bench.BUDGET),
                          NVE_NVEMB_VMM_CHUNK_BYTES=str(bench.CHUNK), NVE_NVEMB_VMM_FAST_REHASH="1",
                          NVE_NVEMB_POLICY_OBSERVATION="1", NVE_NVEMB_POLICY_EVENT_CAPACITY="65536")
        report["gc"] = bench.freeze_loaded_inputs()
        frozen = True
        report["phase"] = "model_and_full_PS_setup"
        backend = bench.fixed.HSTU(cache_memory_backend="vmm", peak_cache_bytes=32 * bench.GIB,
                                   host_staging_limit_bytes=40 * bench.GIB)
        backend.memory_options["vmm_chunk_bytes"] = bench.CHUNK
        report["backend"] = backend.setup(args, manifest, sequences, full_rows)
        report["gc"]["automatic_gc_enabled_after_setup"] = gc.isenabled()
        bench.require(report["gc"]["automatic_gc_enabled_after_setup"], "Automatic GC was disabled during model setup")
        report["backend_validation"] = bench.validate_backend(backend)
        report["model_parameters"] = bench.model_parameter_hash(backend)
        report["loaded_libraries"] = bench.native_identity(args.runtime_root)
        memory = backend.model.cache_memory
        bench.require(memory.embedding.supports_prepared_resize,
                      "Calibration requires prepared-resize native bindings, not the legacy fallback")
        owners = (id(backend.model), id(memory), id(memory.embedding.collection))
        report["startup_observation_drain"] = drain(memory)
        bench.require(all(table.get("enabled") is True for table in
                          report["startup_observation_drain"]["embedding"]["tables"].values()),
                      "Native policy observation is not enabled")
        report["phase"] = "offline_real_prefill"
        prefix = full_rows[:MAX_PREFILL_REQUESTS]
        head = 0
        prefill_started = time.perf_counter()
        population_checks = report["prefill_population_checks"] = []
        populated = False
        while head < len(prefix):
            end, reason, wake = bench.fixed.choose_batch(prefix, head, prefix[-1]["arrival_s"], 32, 32768)
            bench.require(end > head and wake is None, "Offline prefill did not advance")
            batch = prefix[head:end]
            batch_started = time.perf_counter()
            output = backend.forward(batch)
            backend.synchronize()
            backend.validate(output)
            report["prefill_batches"].append({"request_ids": [r["request_id"] for r in batch],
                                             "tokens": sum(r["tokens"] for r in batch), "reason": reason,
                                             "all_output_finite": True,
                                             "wall_ms_outside_resize": (time.perf_counter() - batch_started) * 1000})
            head = end
            if head >= PREFILL_REQUESTS:
                population = drain(memory)
                populated = populated_embedding(population)
                population_checks.append({"completed_requests": head, "observation": population,
                                          "both_tables_populated": populated})
                if populated:
                    break
        bench.require(populated, "Real prefill reached its request bound with an empty embedding table; no resize was measured")
        probe_batch = batch
        reference_started = time.perf_counter()
        reference_cpu = output.detach().cpu().clone()
        output = None
        report["probe_reference"] = {"source_ids": [r["request_id"] for r in probe_batch],
                                     "shape": list(reference_cpu.shape), "dtype": str(reference_cpu.dtype),
                                     "storage": "detached independent CPU clone of final prefill output",
                                     "copy_wall_ms_outside_resize": (time.perf_counter() - reference_started) * 1000,
                                     "rtol": PROBE_RTOL, "atol": PROBE_ATOL}
        report["prefill"] = {"source_ids_half_open": [0, head], "completed_requests": head,
                             "wall_s_outside_resize": time.perf_counter() - prefill_started,
                             "arrival_pacing": False, "full_source_PS_initialization": True,
                             "forward_scope": "real HSTU KV/embedding generation only before calibration timers"}
        backend.torch.cuda.synchronize()
        report["prefill_observation_drain"] = drain(memory)
        report["capacity_after_prefill"] = memory.stats()
        initial_kv_bytes = report["capacity_after_prefill"]["kv"]["payload_bytes"]
        bench.write_json(raw_path, report)
        report["phase"] = "manual_resize_calibration"
        for round_index in (1, 2):
            for delta in TRANSFERS:
                sample = {"round": round_index, "requested_kv_delta_bytes": delta, "status": "running"}
                report["samples"].append(sample)
                backend.torch.cuda.synchronize()
                sample["observation_drain_before_timer"] = drain(memory)
                bench.require(populated_embedding(sample["observation_drain_before_timer"]),
                              "Embedding became empty before resize; no empty-cache sample is valid")
                sample["before"] = memory.stats()
                tick = time.perf_counter()
                try:
                    backend.torch.cuda.synchronize()
                    synced = time.perf_counter()
                    sample["pre_sync_ms"] = (synced - tick) * 1000
                    sample["action"] = memory.resize_by_bytes(delta, at_batch_barrier=True)
                    returned = time.perf_counter()
                    sample["shared_call_wall_ms"] = (returned - synced) * 1000
                    backend.torch.cuda.synchronize()
                    completed = time.perf_counter()
                    sample["post_sync_ms"] = (completed - returned) * 1000
                    sample["resize_wall_ms"] = (completed - tick) * 1000
                    sample["observation_drain_after_resize"] = drain(memory)
                    sample["after"] = memory.stats()
                    sample["observed_wall_ms"] = (time.perf_counter() - tick) * 1000
                    verify_transfer(sample["before"], sample["after"], delta)
                    bench.require(owners == (id(backend.model), id(memory), id(memory.embedding.collection)), "Model/hierarchy was rebuilt")
                    bench.require(sample["after"]["state"] == "open", "Post-calibration cache is poisoned")
                    bench.require(sample["after"]["kv"]["payload_bytes"] + sample["after"]["embedding"]["payload_bytes"] <= bench.BUDGET,
                                  "Calibration exceeded payload ceiling")
                    sample["non_noop"] = any(sample["before"][side]["payload_bytes"] != sample["after"][side]["payload_bytes"]
                                              for side in ("kv", "embedding"))
                    bench.require(sample["action"]["status"] in ("applied", "rounded_noop"), "Unexpected resize status")
                    # observed_wall_ms is already complete. Real refill/age touches
                    # from this validation remain in the model for the next sample.
                    probe = sample["post_resize_probe"] = {
                        "status": "running", "source_ids": report["probe_reference"]["source_ids"],
                        "rtol": PROBE_RTOL, "atol": PROBE_ATOL,
                        "timing_scope": "forward, completion, finite/value checks, CPU copy, observer drain and after-probe ledger; excluded from resize cost"}
                    probe_started = time.perf_counter()
                    try:
                        output = backend.forward(probe_batch)
                        backend.synchronize()
                        backend.validate(output)
                        probe["all_output_finite"] = True
                        actual_cpu = output.detach().cpu().clone()
                        output = None
                        probe["max_abs_err"] = (actual_cpu.to(backend.torch.float64) -
                                                reference_cpu.to(backend.torch.float64)).abs().max().item()
                        probe["observation_drain"] = drain(memory)
                        sample["after_probe"] = memory.stats()
                        backend.torch.testing.assert_close(actual_cpu, reference_cpu,
                                                           rtol=PROBE_RTOL, atol=PROBE_ATOL)
                        bench.require(sample["after_probe"]["state"] == "open", "Post-probe cache is poisoned")
                        bench.require(all(sample["after_probe"][side]["payload_bytes"] == sample["after"][side]["payload_bytes"]
                                          for side in ("kv", "embedding")), "Probe changed configured capacity")
                        probe["status"] = "passed"
                    except BaseException as error:
                        probe.update(status="failed", error=repr(error))
                        raise
                    finally:
                        probe["probe_wall_ms"] = (time.perf_counter() - probe_started) * 1000
                    sample["status"] = "passed"
                except BaseException as error:
                    sample.update(status="failed", error=repr(error), wall_ms_until_failure=(time.perf_counter() - tick) * 1000,
                                  failure_wall_scope="from resize start, including numerical probe if reached; not the resize cost",
                                  last_shared_action=memory.last_action)
                    raise
                finally:
                    bench.write_json(raw_path, report)
            bench.require(memory.stats()["kv"]["payload_bytes"] == initial_kv_bytes,
                          "Bidirectional transfer round did not return to its actual starting capacity")
        measured = [sample["observed_wall_ms"] for sample in report["samples"] if sample["non_noop"]]
        bench.require(len(measured) == 8, "Calibration requires all eight bidirectional capacity-changing samples")
        report["non_noop_samples"] = len(measured)
        report["post_resize_probe_wall_ms_outside_resize"] = sum(sample["post_resize_probe"]["probe_wall_ms"] for sample in report["samples"])
        report["post_resize_probes_passed"] = sum(sample["post_resize_probe"]["status"] == "passed" for sample in report["samples"])
        bench.require(report["post_resize_probes_passed"] == 8, "All eight post-resize numerical probes must pass")
        report["resize_cost_ms"] = max(measured)
        report["phase"] = "cleanup"
        output = memory = None
        backend.close(); backend = None
        report.update(status="passed", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        output = memory = None
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=repr(error))
        if frozen:
            gc.unfreeze(); gc.collect()
            report["gc"]["cleanup_frozen_objects"] = gc.get_freeze_count()
            report["gc"]["cleanup_scope"] = "all frozen objects made GC-eligible before exit; the image's initial frozen set is not restored"
        report["whole_work_wall_s"] = time.perf_counter() - started
        bench.write_json(raw_path, report)
    if report["status"] == "passed":
        preset = {"schema_version": 1, "backend": "nvemb_vmm_general_rehash",
                  "chunk_bytes": bench.CHUNK, "payload_budget_bytes": bench.BUDGET,
                  "resize_cost_ms": report["resize_cost_ms"],
                  "method": "maximum of all completed non-noop external wall samples across 8 scheduled calls, including post-resize observation drain; numerical probes run outside resize timing and their real cache changes persist; both rounds and all tails retained",
                  "library_sha256": {Path(path).name: value["sha256"] for path, value in report["loaded_libraries"].items()},
                  "source_receipts": [{"path": str(raw_path.resolve()), **bench.sha_file(raw_path)}],
                  "sample_count": report["non_noop_samples"], "scheduled_calls": 8, "prefill_requests": report["prefill"]["completed_requests"],
                  "transfer_bytes": TRANSFER_BYTES,
                  "scope_limit": "same model/full PS, bounded real-request prefill with nonempty embedding tables and eight interleaved numerical probes; occupancy recorded, not a full-cache worst-case guarantee or predicted policy gain",
                  "post_resize_probes_passed": report["post_resize_probes_passed"],
                  "post_resize_probe_wall_ms_outside_resize": report["post_resize_probe_wall_ms_outside_resize"],
                  "model_parameters": report["model_parameters"], "all_samples_retained": True}
        bench.write_json(args.output_dir / "calibration.json", preset)
    print(json.dumps({"status": report["status"], "samples": len(report["samples"]),
                      "resize_cost_ms": report.get("resize_cost_ms"), "output": str(args.output_dir)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
