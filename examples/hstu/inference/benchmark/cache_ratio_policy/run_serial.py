#!/usr/bin/env python3
"""Bounded local serial execution inside an already-authorized GPU allocation.

This script neither acquires resources nor connects remotely. The first round
is fixed/observe_only/adaptive; optional round2 reverses that order. Start a
round only when the remaining declared lease covers its entire time budget.
"""
import argparse
import hashlib
import importlib.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MODES = ("fixed", "observe_only", "adaptive")
CACHE_ENV = {"triton": "TRITON_CACHE_DIR", "cuda": "CUDA_CACHE_PATH"}


def fits(lease_epoch, now_epoch, arms, arm_budget, guard):
    return lease_epoch - now_epoch >= arms * arm_budget + guard


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_identity(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Compiler cache changed while hashing: " + str(path))
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def compiler_cache_files(root):
    """Manifest all ordinary files; refuse links or special files in either cache."""
    files = {}
    for name in CACHE_ENV:
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Compiler seed requires ordinary triton/ and cuda/ directories")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("Compiler cache contains a link or special file: " + str(path))
            if path.is_file():
                files[path.relative_to(root).as_posix()] = file_identity(path)
    return files


def verify_compiler_seed(seed, expected=None):
    path = seed / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Compiler seed manifest.json must be an ordinary file")
    manifest_identity = file_identity(path)
    manifest = json.loads(path.read_text())
    if file_identity(path) != manifest_identity or manifest.get("schema_version") != 1:
        raise ValueError("Compiler seed manifest changed or has an unsupported schema")
    roots = manifest.get("source_roots", {})
    if set(roots) != set(CACHE_ENV) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in roots.values()):
        raise ValueError("Compiler seed source_roots must declare absolute triton and cuda origins")
    files = compiler_cache_files(seed)
    if manifest.get("files") != files:
        raise ValueError("Compiler seed file/content manifest differs")
    identity = {"manifest_sha256": manifest_identity["sha256"],
                "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values())}
    if expected is not None and identity != expected:
        raise ValueError("Compiler seed identity changed between arms")
    return manifest, identity


def prepare_compiler_cache(seed, destination, expected):
    """Copy and verify before child startup; only relocate the new arm's indexes."""
    started = time.perf_counter()
    manifest, identity = verify_compiler_seed(seed, expected)
    destination.mkdir(parents=True, exist_ok=False)
    for name in CACHE_ENV:
        shutil.copytree(seed / name, destination / name, copy_function=shutil.copy2)
    if compiler_cache_files(destination) != manifest["files"]:
        raise ValueError("Compiler cache copy differs from frozen seed")
    verify_compiler_seed(seed, expected)
    # A frozen seed may be read-only. Private copies must accept new compiled files.
    for path in [destination, *destination.rglob("*")]:
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    triton = (destination / "triton").resolve()
    origin = Path(manifest["source_roots"]["triton"])
    groups = children = 0
    for path in sorted(triton.rglob("__grp__*")):
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        entries = data.get("child_paths")
        if not isinstance(entries, (dict, list)):
            raise ValueError("Unsupported Triton group index: " + str(path))
        relocated = {}
        for name in entries:
            if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Unsafe Triton group child name")
            target = path.parent / name
            if not target.is_file() or not target.is_relative_to(triton):
                raise ValueError("Triton group references a missing copied file")
            if isinstance(entries, dict):
                old = Path(entries[name])
                if not old.is_absolute() or not old.is_relative_to(origin) or old.relative_to(origin) != target.relative_to(triton):
                    raise ValueError("Triton group references a file outside its declared seed origin")
                relocated[name] = str(target)
        if isinstance(entries, dict):
            data["child_paths"] = relocated
            write(path, data)
            groups += 1
            children += len(relocated)
    return {"schema_version": 1, "status": "ready", "seed_identity": identity,
            "seed_directory": str(seed), "cache_directory": str(destination.resolve()),
            "environment": {variable: str((destination / name).resolve()) for name, variable in CACHE_ENV.items()},
            "copy_method": "shutil.copy2; independent files, no hardlinks or symlinks",
            "verified_copy_before_relocation": True, "relocated_triton_groups": groups,
            "relocated_triton_children": children, "preparation_wall_s": time.perf_counter() - started,
            "scope": "same frozen Triton/CUDA disk-cache seed before each child/setup; new compilation stays in each arm's arrival clock; no torch.compile call in this HSTU replay path"}


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt("Serial runner received termination")
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[1] / "cache_ratio_policy_benchmark.py")
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--workload-lock", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--lease-end-utc", required=True, help="Verified allocation end, ISO8601 with timezone")
    p.add_argument("--resize-calibration", type=Path)
    p.add_argument("--resize-cost-mode", choices=("amortized", "diagnostic"), default="amortized")
    p.add_argument("--compiler-cache-seed", type=Path, help="Frozen directory containing manifest.json, triton/ and cuda/; copied privately before every arm")
    p.add_argument("--scope", choices=("full", "smoke"), default="full")
    p.add_argument("--smoke-requests", type=int, default=128)
    p.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    p.add_argument("--arm-budget-s", type=float, default=5400)
    p.add_argument("--lease-guard-s", type=float, default=600)
    p.add_argument("--peak-cache-gib", type=int, default=32)
    p.add_argument("--host-staging-limit-gib", type=int, default=40)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    try:
        ending = datetime.fromisoformat(args.lease_end_utc.replace("Z", "+00:00"))
        if ending.tzinfo is None:
            raise ValueError("timezone required")
        lease = ending.timestamp()
    except ValueError as error:
        p.error("Invalid lease-end-utc: " + str(error))
    if args.arm_budget_s <= 0 or args.lease_guard_s < 120 or (args.scope == "full" and args.arm_budget_s < 4200):
        p.error("Require positive arm budget (full>=4200s) and lease guard>=120s for bounded child termination")
    if not args.benchmark.is_file():
        p.error("Benchmark file does not exist")
    args.results_root.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "status": "planned", "lease_end_utc": ending.astimezone(timezone.utc).isoformat(),
              "arm_budget_s": args.arm_budget_s, "lease_guard_s": args.lease_guard_s,
              "scope": args.scope, "resize_cost_mode": args.resize_cost_mode, "round1": list(MODES), "round2": list(reversed(MODES)) if args.rounds == 2 else [],
              "required_first_round_arms": 3, "runs": [], "optional_reverse_round": "pending" if args.rounds == 2 else "not_requested",
              "compiler_cache": {"enabled": args.compiler_cache_seed is not None},
              "resource_scope": "caller verified existing allocation and GPU idleness; no allocation or remote access by this script"}
    path = args.results_root / "serial_receipt.json"
    write(path, record)
    if args.plan_only:
        print(json.dumps(record, indent=2))
        return 0
    try:
        seed_identity = None
        if args.compiler_cache_seed is not None:
            args.compiler_cache_seed = args.compiler_cache_seed.resolve()
            _, seed_identity = verify_compiler_seed(args.compiler_cache_seed)
            record["compiler_cache"].update(seed_directory=str(args.compiler_cache_seed), seed_identity=seed_identity)
            write(path, record)
        spec = importlib.util.spec_from_file_location("policy_result_audit", Path(__file__).with_name("summarize_results.py"))
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        reference_signature = None
        for round_index in range(1, args.rounds + 1):
            if not fits(lease, time.time(), 3, args.arm_budget_s, args.lease_guard_s):
                if round_index == 1:
                    raise RuntimeError("Remaining lease cannot cover the required complete first round")
                record["optional_reverse_round"] = "skipped_insufficient_remaining_lease"
                break
            modes = MODES if round_index == 1 else tuple(reversed(MODES))
            record["status"] = "running"
            for mode in modes:
                if not fits(lease, time.time(), 1, args.arm_budget_s, args.lease_guard_s):
                    raise RuntimeError("Lease guard prevents starting the next complete arm")
                directory = args.results_root / f"round{round_index}-{mode}"
                command = [sys.executable, str(args.benchmark.resolve()), "--mode", mode,
                           "--resize-cost-mode", args.resize_cost_mode,
                           "--scope", args.scope, "--smoke-requests", str(args.smoke_requests),
                           "--trace-dir", str(args.trace_dir.resolve()), "--workload-lock", str(args.workload_lock.resolve()),
                           "--runtime-root", str(args.runtime_root.resolve()), "--output-dir", str(directory.resolve()),
                           "--round-index", str(round_index), "--peak-cache-gib", str(args.peak_cache_gib),
                           "--host-staging-limit-gib", str(args.host_staging_limit_gib)]
                if args.resize_calibration:
                    command += ["--resize-calibration", str(args.resize_calibration.resolve())]
                run = {"round": round_index, "mode": mode, "output_dir": str(directory.resolve()),
                       "command": command, "start_unix_s": time.time(), "status": "running"}
                record["runs"].append(run)
                write(path, record)
                child_env = None
                if seed_identity is not None:
                    cache = prepare_compiler_cache(args.compiler_cache_seed,
                                                   args.results_root / "compiler-cache" / directory.name, seed_identity)
                    cache_path = directory.with_name(directory.name + ".compiler_cache.json")
                    write(cache_path, cache)
                    run["compiler_cache"] = {"seed_identity": seed_identity, "receipt": str(cache_path.resolve()),
                                             "receipt_identity": file_identity(cache_path)}
                    child_env = {**os.environ, **cache["environment"]}
                    write(path, record)
                    if not fits(lease, time.time(), 1, args.arm_budget_s, args.lease_guard_s):
                        raise RuntimeError("Lease guard prevents starting arm after compiler-cache preparation")
                timeout = min(args.arm_budget_s, lease - time.time() - args.lease_guard_s)
                with (args.results_root / f"round{round_index}-{mode}.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env)
                    run["pid"] = process.pid
                    write(path, record)
                    try:
                        returncode = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        run["timed_out"] = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            returncode = process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            returncode = process.wait(timeout=30)
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=30)
                        run.update(returncode=process.returncode, end_unix_s=time.time(), status="failed", interrupted=True)
                        raise
                run.update(returncode=returncode, end_unix_s=time.time(), status="failed")
                summary_path = directory / "summary.json"
                result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                expected = 742031 if args.scope == "full" else args.smoke_requests
                if returncode != 0 or result.get("status") != "passed" or not result.get("complete") or result.get("completed_requests") != expected:
                    raise RuntimeError(f"Arm did not complete cleanly: round{round_index}-{mode}, exit={returncode}")
                if seed_identity is not None:
                    verify_compiler_seed(args.compiler_cache_seed, seed_identity)
                checked = verifier.audit(directory)
                write(directory / "raw_audit.json", checked)
                if checked["status"] != "passed":
                    raise RuntimeError("Raw request/batch audit failed: " + repr(checked["errors"]))
                if checked["match_signature"]["compiler_cache_seed"] != seed_identity:
                    raise RuntimeError("Arm compiler-cache seed differs from the declared suite seed")
                signature = {key: value for key, value in checked["match_signature"].items() if key != "round_index"}
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    raise RuntimeError("Input/code/model/library/GPU/capacity identities changed between arms")
                run["raw_audit"] = "passed"
                run["status"] = "passed"
                write(path, record)
            if round_index == 2:
                record["optional_reverse_round"] = "completed"
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
    finally:
        record["completed_arms"] = sum(run["status"] == "passed" for run in record["runs"])
        record["finished_unix_s"] = time.time()
        write(path, record)
    print(json.dumps({"status": record["status"], "completed_arms": record["completed_arms"],
                      "optional_reverse_round": record["optional_reverse_round"]}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
