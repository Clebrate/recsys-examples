#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze KuaiRand arrivals, then replay one fixed cache ratio through real HSTU.

Preparation and reporting use only the standard library. Example, from any cwd:
  python cache_ratio_serving_benchmark.py prepare --sequence-file processed_seqs_inference.csv \
      --batch-file processed_batches.csv --output-dir /tmp/frozen --time-scale 0.001
  python cache_ratio_serving_benchmark.py run --trace-dir /tmp/frozen \
      --output-dir /tmp/ratio-05 --cache-budget-bytes 1073741824 --kv-cache-ratio 0.5 \
      --smoke --max-batch-size 8 --max-batch-tokens 65536 --max-batch-delay-ms 1

Prepared candidates come from checkpoint PS keys, or from observed history keys
which are explicitly populated in the smoke PS. No accuracy claim is made for
smoke weights or sampled candidates. A new process is required for each ratio.
"""

import argparse
import array
from bisect import bisect_right
import csv
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import groupby
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time


CONTEXT_SIZES = {
    "user_active_degree": 10,
    "follow_user_num_range": 9,
    "fans_user_num_range": 10,
    "friend_user_num_range": 8,
    "register_days_range": 8,
}
HSTU_DIR = Path(__file__).resolve().parents[2]


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_ps_keys(path):
    """NVE's .dyn key format is headerless native int64, not a NumPy archive."""
    path = Path(path)
    if path.stat().st_size % 8:
        raise ValueError(f"Invalid int64 PS key file size: {path}")
    values = array.array("q")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // 8)
    return set(values)


def prepare(args):
    start = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Preparation output must be empty: {out}")
    csv.field_size_limit(sys.maxsize)
    counts = {"source_requests": 0, "filtered_date": 0, "filtered_history": 0}

    def arrivals():
        if not args.batch_file:
            with open(args.sequence_file, newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    if args.max_users and index >= args.max_users:
                        break
                    uid = int(row["user_id"])
                    if not 0 <= uid < 2**32:
                        raise ValueError("user_id does not fit the stable int64 KV key")
                    # Source sequence files can span days. Use independent day
                    # prefixes so no request reuses KV from a different history.
                    times = sorted(json.loads(row["time_ms"]))
                    for epoch_day, day_times in groupby(times, key=lambda ts: ts // 86400000):
                        date = int(datetime.fromtimestamp(epoch_day * 86400, timezone.utc).strftime("%Y%m%d"))
                        day_times = list(day_times)
                        endpoints = list(range(args.interactions_per_request, len(day_times) + 1,
                                               args.interactions_per_request))
                        if not endpoints or endpoints[-1] != len(day_times):
                            endpoints.append(len(day_times))
                        for history in endpoints:
                            counts["source_requests"] += 1
                            if args.date is not None and date != args.date:
                                counts["filtered_date"] += 1
                                continue
                            if history > args.max_history_len:
                                counts["filtered_history"] += 1
                                continue
                            yield {
                                "user_id": uid, "date": date, "history_len": history,
                                "timestamp_ms": day_times[history - 1], "source_row": index,
                                "cache_uid": (date << 32) | uid,
                            }
            return
        with open(args.batch_file, newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                counts["source_requests"] += 1
                uid, date = int(row["user_id"]), int(row["date"])
                if args.date is not None and date != args.date:
                    counts["filtered_date"] += 1
                    continue
                history = int(row["interval_indptr"])
                if not 0 < history <= args.max_history_len:
                    counts["filtered_history"] += 1
                    continue
                if not 0 <= uid < 2**32 or not 0 <= date < 2**31:
                    raise ValueError("date and uid do not fit the stable int64 KV key")
                yield {
                    "user_id": uid, "date": date, "history_len": history,
                    "timestamp_ms": int(row["interval_end_ts"]), "source_row": index,
                    "cache_uid": (date << 32) | uid,
                }

    requests = heapq.nsmallest(
        args.max_requests, arrivals(), key=lambda r: (r["timestamp_ms"], r["source_row"])
    )
    if not requests:
        raise ValueError("No requests remain after the date/history filters")
    needed = {}
    for request in requests:
        key = request["cache_uid"]
        if request["history_len"] < needed.get(key, 0):
            raise ValueError(f"History shrinks for user/date {key}; check source ordering")
        needed[key] = request["history_len"]
    batch_csv_s = time.perf_counter() - start
    contexts = [] if args.disable_context else ["user_id", *CONTEXT_SIZES]
    sequences, observed_items = {}, set()
    needed_users = {request["user_id"] for request in requests}
    with open(args.sequence_file, newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if not args.batch_file and args.max_users and index >= args.max_users:
                break
            uid = int(row["user_id"])
            if uid not in needed_users:
                continue
            raw_items, raw_actions = json.loads(row["video_id"]), json.loads(row["action_weights"])
            if len(raw_items) != len(raw_actions):
                raise ValueError(f"Item/action length mismatch for user {uid}")
            if args.batch_file:
                day_sequences = [(int(row["date"]), raw_items, raw_actions)]
            else:
                times = json.loads(row["time_ms"])
                if len(times) != len(raw_items):
                    raise ValueError(f"Time/item length mismatch for user {uid}")
                ordered = sorted(zip(times, raw_items, raw_actions), key=lambda values: values[0])
                day_sequences = []
                for epoch_day, entries in groupby(ordered, key=lambda values: values[0] // 86400000):
                    date = int(datetime.fromtimestamp(epoch_day * 86400, timezone.utc).strftime("%Y%m%d"))
                    entries = list(entries)
                    day_sequences.append((date, [v[1] for v in entries], [v[2] for v in entries]))
            for date, items, actions in day_sequences:
                key = (date << 32) | uid
                if key not in needed:
                    continue
                if key in sequences:
                    raise ValueError(f"Duplicate sequence row for user/date {uid}/{date}")
                length = needed[key]
                items, actions = items[:length], actions[:length]
                if len(items) != length or len(actions) != length:
                    raise ValueError(f"Short sequence for user/date {uid}/{date}")
                if any(not isinstance(k, int) or k < 0 for k in items):
                    raise ValueError("video_id must contain nonnegative integer IDs")
                if any(not isinstance(k, int) or not 0 <= k < 246 for k in actions):
                    raise ValueError("action_weights must fit the 27K training table (246 rows)")
                context = {name: int(row[name]) for name in contexts}
                for name, size in CONTEXT_SIZES.items():
                    if name in context and not 0 <= context[name] < size:
                        raise ValueError(f"Context {name} exceeds its 27K training table")
                sequences[key] = {
                    "cache_uid": key, "context": context,
                    "video_id": items, "action_weights": actions,
                }
                observed_items.update(items)
    if set(sequences) != set(needed):
        raise ValueError(f"Missing {len(set(needed) - set(sequences))} user/date sequences")
    sequence_csv_s = time.perf_counter() - start - batch_csv_s
    users = {request["user_id"] for request in requests}
    ps_files = {}
    if args.checkpoint_dir:
        ps_dir = Path(args.checkpoint_dir) / "ps_module"
        for name in ("video_id", "user_id") if contexts else ("video_id",):
            path = ps_dir / f"{name}_emb_keys.rank_0.world_size_1.dyn"
            ps_files[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
        item_keys = read_ps_keys(ps_files["video_id"]["path"])
        user_keys = read_ps_keys(ps_files["user_id"]["path"]) if contexts else users
        missing_items, missing_users = observed_items - item_keys, users - user_keys
        if missing_items or missing_users:
            raise ValueError(
                f"Checkpoint PS lacks {len(missing_items)} history items and "
                f"{len(missing_users)} users; refusing default-value misses"
            )
        candidate_source = "checkpoint_video_id_ps_keys_uniform_with_replacement"
    else:
        item_keys, user_keys = observed_items, users
        candidate_source = "observed_history_keys_uniform_with_replacement_smoke_fixture"
    if not item_keys:
        raise ValueError("The candidate pool is empty")
    pool, rng = sorted(item_keys), random.Random(args.seed)
    origin = requests[0]["timestamp_ms"]
    with (out / "trace.jsonl").open("w") as handle:
        for index, request in enumerate(requests):
            request.update({
                "request_id": index,
                "arrival_s": (request["timestamp_ms"] - origin) / 1000 * args.time_scale,
                "candidates": rng.choices(pool, k=args.num_candidates),
            })
            handle.write(json.dumps(request, separators=(",", ":")) + "\n")
    with (out / "sequences.jsonl").open("w") as handle:
        for key in sorted(sequences):
            handle.write(json.dumps(sequences[key], separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1, "dataset": "kuairand-27k", "seed": args.seed,
        "request_count": len(requests), "sequence_count": len(sequences),
        "max_history_len": max(r["history_len"] for r in requests),
        "num_candidates": args.num_candidates, "context_features": contexts,
        "time_scale": args.time_scale, "timestamp_origin_ms": origin,
        "request_source": "processed_batches" if args.batch_file else "derived_every_n_interactions",
        "interactions_per_request": args.interactions_per_request if not args.batch_file else None,
        "max_source_users": args.max_users if not args.batch_file else None,
        "candidate_source": candidate_source,
        "dynamic_table_rows": {"video_id": len(item_keys), "user_id": len(user_keys)},
        "checkpoint_ps_key_files": ps_files, "filters": counts,
        "csv_parse_s": {"batch": batch_csv_s, "sequence": sequence_csv_s},
        "prepare_total_s": time.perf_counter() - start,
        "sources": {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                   "mtime_ns": Path(path).stat().st_mtime_ns}
            for name, path in (("sequence_file", args.sequence_file), ("batch_file", args.batch_file)) if path
        },
        "artifacts": {name: sha256(out / name) for name in ("trace.jsonl", "sequences.jsonl")},
        "notes": ["Candidates are frozen sampled IDs, not logged ranking candidates.",
                  "Requests derived from a sequence file are constructed serving arrivals at real interaction timestamps.",
                  "Histories above the configured limit are filtered, not silently truncated.",
                  "KV identity is (date << 32) | user_id; no cross-day prefix reuse."],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def summarize(out):
    out = Path(out)
    requests = [json.loads(line) for line in (out / "request.jsonl").open()]
    batches = [json.loads(line) for line in (out / "batch.jsonl").open()]
    timed = [row for row in requests if row["phase"] == "timed"]
    timed_batches = [row for row in batches if row["phase"] == "timed"]

    def distribution(values):
        values = sorted(values)
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        return {"count": len(values), "mean": statistics.fmean(values),
                **{f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
                   for p in (50, 95, 99)}}

    history = sum(row["cache_stats"].get("history_tokens", 0) for row in timed_batches)
    cached = sum(row["cache_stats"].get("kv_cached_tokens", 0) for row in timed_batches)
    emb_keys = sum(row["cache_stats"].get("embedding_dynamic_keys", 0) for row in timed_batches)
    available = [row["cache_stats"] for row in timed_batches
                 if row["cache_stats"].get("embedding_hit_rate") is not None]
    measured_keys = sum(row.get("embedding_dynamic_keys", 0) for row in available)
    summary = {
        "timed_requests": len(timed), "timed_batches": len(timed_batches),
        "latency_ms": distribution([row["latency_ms"] for row in timed]),
        "queue_ms": distribution([row["queue_ms"] for row in timed]),
        "service_ms": distribution([(row["ready_s"] - row["dispatch_s"]) * 1000 for row in timed]),
        "batch_size": distribution([row["batch_size"] for row in timed_batches]),
        "input_build_ms": distribution([row["input_build_ms"] for row in timed_batches]),
        "forward_wall_ms": distribution([row["forward_wall_ms"] for row in timed_batches]),
        "forward_cuda_span_ms": distribution([row["forward_cuda_span_ms"] for row in timed_batches]),
        "kv_history_tokens": history, "kv_cached_tokens": cached,
        "kv_hit_rate": cached / history if history else None,
        "embedding_dynamic_keys": emb_keys, "embedding_measured_keys": measured_keys,
        "embedding_hit_rate": (sum(row["embedding_hit_rate"] * row["embedding_dynamic_keys"]
                                   for row in available) / measured_keys if measured_keys else None),
        "embedding_hit_rate_scope": "dynamic_tables_gpu_tier",
        "embedding_rate_key_coverage": measured_keys / emb_keys if emb_keys else None,
        "embedding_hit_rate_unavailable_reason": None if measured_keys else "No real backend embedding hit-rate measurement is available",
        "measurement_notes": [
            "Latency is scheduled arrival to result-ready event, including queue and batch wait.",
            "CUDA spans include dependencies/idle gaps and are not pure kernel execution time.",
            "Stage CPU/CUDA spans overlap; do not sum them to reconstruct E2E.",
            "Profiling/export/logging consumes worker time and affects subsequent queueing.",
            "Embedding rates use real dynamic-key-weighted backend rates, without inferred hit counts.",
        ],
    }
    if timed:
        duration = max(r["ready_s"] for r in timed) - min(r["arrival_s"] for r in timed)
        summary["arrival_to_drain_s"] = duration
        summary["throughput_requests_per_s"] = len(timed) / duration if duration else None
    runtime_path = out / "runtime.json"
    if runtime_path.exists():
        summary["runtime"] = json.loads(runtime_path.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def run(args):
    # Imports are deliberately local: prepare/report do not import Torch or initialize CUDA.
    start = time.perf_counter()
    trace_dir, out = Path(args.trace_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Run output must be empty: {out}")
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    for filename, expected in manifest["artifacts"].items():
        if sha256(trace_dir / filename) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    requests = [json.loads(line) for line in (trace_dir / "trace.jsonl").open()]
    profile_request_ids = {int(value) for value in args.profile_request_ids.split(",") if value.strip()}
    if any(value < args.warmup_requests or value >= len(requests) for value in profile_request_ids):
        raise ValueError("profile-request-ids must refer to timed requests in the frozen trace")
    sequences = {row["cache_uid"]: row for row in
                 (json.loads(line) for line in (trace_dir / "sequences.jsonl").open())}
    input_load_s = time.perf_counter() - start
    if args.warmup_requests >= len(requests):
        raise ValueError("Warmup must leave at least one timed request")
    contexts = manifest["context_features"]
    max_tokens = 2 * manifest["max_history_len"] + manifest["num_candidates"] + len(contexts)
    if max_tokens > args.max_batch_tokens:
        raise ValueError(f"max-batch-tokens must accommodate a request ({max_tokens} tokens)")
    if bool(args.checkpoint_dir) == bool(args.smoke):
        raise ValueError("Choose exactly one of --checkpoint-dir and explicit --smoke")
    if args.checkpoint_dir:
        if not manifest["checkpoint_ps_key_files"]:
            raise ValueError("Prepare with the same checkpoint before a checkpoint-backed run")
        for name, expected in manifest["checkpoint_ps_key_files"].items():
            path = Path(args.checkpoint_dir) / "ps_module" / f"{name}_emb_keys.rank_0.world_size_1.dyn"
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Checkpoint PS keys changed for {name}")

    sys.path[:0] = [str(HSTU_DIR), str(HSTU_DIR.parent), str(HSTU_DIR / "model")]
    os.environ.setdefault("HSTU_INFERENCE_ONLY", "1")
    import gin
    import torch
    from commons.datasets.hstu_batch import HSTUBatch
    from configs import EmbeddingBackend, InferenceEmbeddingConfig
    from inference.inference_gr_ranking import get_inference_hstu_model
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    from utils import NetworkArgs

    gin.parse_config_file(args.gin_config_file)
    torch.manual_seed(manifest["seed"])
    dim = NetworkArgs().hidden_size
    # Match 27K static table sizes. Dynamic sizes describe PS row counts, not max ID:
    # NVE's hash tables accept sparse int64 IDs; row counts weight the byte split.
    sizes = {**CONTEXT_SIZES, "action_weights": 246, **manifest["dynamic_table_rows"]}
    emb_configs = [InferenceEmbeddingConfig(
        feature_names=[name], table_name=name, vocab_size=sizes[name], dim=dim,
        use_dynamicemb=name in ("user_id", "video_id"),
    ) for name in [*contexts, "video_id", "action_weights"]]
    model = None
    profiler = None
    try:
        with torch.inference_mode():
            kv_options = json.loads(args.kv_cache_extra_configs)
            if not isinstance(kv_options, dict):
                raise ValueError("kv-cache-extra-configs must be a JSON object")
            model = get_inference_hstu_model(
                emb_configs, args.max_batch_size, len(contexts), max_tokens,
                args.checkpoint_dir, True, embedding_backend=EmbeddingBackend.NVEMB,
                cache_budget_bytes=args.cache_budget_bytes, kv_cache_ratio=args.kv_cache_ratio,
                kv_cache_extra_configs=kv_options,
            )
            if args.smoke:
                # Populate every lookup key with finite, nonzero, seeded values. Empty PS
                # defaults would hide the real miss/fetch path and produce invalid evidence.
                keys_by_table = {"video_id": set(), "user_id": set()}
                for row in sequences.values():
                    keys_by_table["video_id"].update(row["video_id"])
                for row in requests:
                    keys_by_table["video_id"].update(row["candidates"])
                    keys_by_table["user_id"].add(row["user_id"])
                stores = model.sparse_module._dynamic_embedding_collection._owned_parameter_servers
                for name, storage in stores.items():
                    keys = sorted(keys_by_table[name])
                    generator = torch.Generator().manual_seed(manifest["seed"] + len(name))
                    for offset in range(0, len(keys), 16384):
                        chunk = torch.tensor(keys[offset:offset + 16384], dtype=torch.int64)
                        values = torch.randn((len(chunk), dim), generator=generator) * 0.02
                        storage.insert(chunk, values)
                for embedding in model.sparse_module._static_embedding_collection.embeddings.values():
                    embedding.weight.normal_(mean=0, std=0.02)
        model.collect_cache_stats = True
        model.profile_cache = False
        setup_s = time.perf_counter() - start - input_load_s
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        runtime = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": vars(args), "trace_manifest_sha256": sha256(trace_dir / "manifest.json"),
            "weights": "seeded_nonzero_smoke_fixture" if args.smoke else "checkpoint",
            "quality_scope": "Performance smoke only; no model accuracy claim" if args.smoke else "Accuracy not evaluated",
            "input_load_s": input_load_s, "model_setup_s": setup_s,
            "cache_budget": model.cache_budget_report, "torch_version": torch.__version__,
            "embedding_ps_payload_bytes_by_table": {
                name: manifest["dynamic_table_rows"][name] * dim * 4
                for name in ("video_id", "user_id") if name == "video_id" or contexts
            },
            "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(),
            "warmup_protocol": "Unpaced trace prefix; caches preserved into timed replay",
            "trace_timed_origin_s": requests[args.warmup_requests]["arrival_s"],
            "serving_semantics": "Single worker, real HSTU forwards, real wall-clock arrivals; no network/Triton frontend",
            "batching": "FIFO; stop at repeated user/date or full token/batch limit; oldest-arrival deadline",
            "profile_samples": [], "observation_wall_ms": 0.0,
        }
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        device = torch.cuda.current_device()

        def build_batch(rows):
            values, lengths, feature_max = [], [], {}
            for name in [*contexts, "video_id", "action_weights"]:
                feature_lengths = []
                for row in rows:
                    sequence = sequences[row["cache_uid"]]
                    if name in contexts:
                        feature = [sequence["context"][name]]
                    else:
                        feature = sequence[name][:row["history_len"]]
                        if name == "video_id":
                            feature = feature + row["candidates"]
                        else:
                            # The CUDA preprocessor subtracts num_candidates from
                            # both feature lengths; candidate actions are placeholders
                            # and are discarded before forming the HSTU sequence.
                            feature = feature + [0] * len(row["candidates"])
                    values.extend(feature)
                    feature_lengths.append(len(feature))
                lengths.extend(feature_lengths)
                feature_max[name] = max(feature_lengths)
            return HSTUBatch(
                features=KeyedJaggedTensor.from_lengths_sync(
                    keys=[*contexts, "video_id", "action_weights"],
                    values=torch.tensor(values, dtype=torch.int64, device=device),
                    lengths=torch.tensor(lengths, dtype=torch.int64, device=device),
                ),
                batch_size=len(rows), feature_to_max_seqlen=feature_max,
                contextual_feature_names=contexts, item_feature_name="video_id",
                action_feature_name="action_weights", max_num_candidates=manifest["num_candidates"],
                num_candidates=torch.full((len(rows),), manifest["num_candidates"], dtype=torch.int64, device=device),
            )

        cursor, batch_id, profile_count = 0, 0, 0
        arrival_times = [request["arrival_s"] for request in requests]
        phase_origin = time.perf_counter()
        trace_origin = requests[0]["arrival_s"]
        phase = "warmup" if args.warmup_requests else "timed"
        with (out / "request.jsonl").open("w") as request_file, (out / "batch.jsonl").open("w") as batch_file:
            while cursor < len(requests):
                scheduler_start = time.perf_counter()
                if phase == "warmup" and cursor == args.warmup_requests:
                    phase = "timed"
                    phase_origin = time.perf_counter()
                    trace_origin = requests[cursor]["arrival_s"]
                end = args.warmup_requests if phase == "warmup" else len(requests)
                first_arrival = requests[cursor]["arrival_s"] - trace_origin
                deadline = first_arrival + args.max_batch_delay_ms / 1000
                rows, tokens, identities = [], 0, set()
                while cursor + len(rows) < end:
                    row = requests[cursor + len(rows)]
                    row_tokens = 2 * row["history_len"] + len(row["candidates"]) + len(contexts)
                    arrival = row["arrival_s"] - trace_origin
                    now = time.perf_counter() - phase_origin
                    if phase == "timed" and arrival > now:
                        if rows and now >= deadline:
                            break
                        wake = min(arrival, deadline) if rows else arrival
                        time.sleep(max(0, wake - now))
                        if rows and arrival > deadline:
                            break
                    # Inspect the next request's identity/size only after it arrives;
                    # future duplicate IDs must not flush a batch before its deadline.
                    if rows and (row["cache_uid"] in identities or tokens + row_tokens > args.max_batch_tokens):
                        break
                    rows.append(row)
                    identities.add(row["cache_uid"])
                    tokens += row_tokens
                    if len(rows) == args.max_batch_size:
                        break
                if phase == "timed" and cursor + len(rows) == end and len(rows) < args.max_batch_size:
                    time.sleep(max(0, deadline - (time.perf_counter() - phase_origin)))
                dispatch = time.perf_counter() - phase_origin
                queue_depth = (bisect_right(arrival_times, dispatch + trace_origin, lo=cursor, hi=end) - cursor
                               if phase == "timed" else end - cursor)
                scheduler_ms = (time.perf_counter() - scheduler_start) * 1000
                build_start = time.perf_counter()
                batch = build_batch(rows)
                user_ids = torch.tensor([r["cache_uid"] for r in rows], dtype=torch.int64)
                histories = torch.tensor([r["history_len"] * 2 + len(contexts) for r in rows], dtype=torch.int32)
                input_ms = (time.perf_counter() - build_start) * 1000
                sampled = (phase == "timed" and profile_count < args.profile_max_batches
                           and ((args.profile_every > 0 and batch_id % args.profile_every == 0)
                                or any(row["request_id"] in profile_request_ids for row in rows)))
                profiler = None
                if sampled:
                    model.profile_cache = True
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                 torch.profiler.ProfilerActivity.CUDA])
                    profiler.__enter__()
                gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                forward_start = time.perf_counter()
                gpu_start.record()
                with torch.inference_mode():
                    logits = model.forward_with_kvcache(batch, user_ids, histories)
                gpu_end.record()
                # Wait only for work needed for this response, not unrelated device streams.
                gpu_end.synchronize()
                ready = time.perf_counter() - phase_origin
                forward_ms = (time.perf_counter() - forward_start) * 1000
                observation_start = time.perf_counter()
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    completed_profiler, profiler = profiler, None
                    filename = f"profile_batch_{batch_id}.json"
                    completed_profiler.export_chrome_trace(str(out / filename))
                    runtime["profile_samples"].append(filename)
                    profile_count += 1
                    model.profile_cache = False
                if batch_id == 0 and not torch.isfinite(logits).all().item():
                    raise RuntimeError("First output contains nonfinite values")
                cache_stats = getattr(model, "last_cache_stats", None)
                if cache_stats is None:
                    raise RuntimeError("This benchmark requires the collect_cache_stats model instrumentation")
                batch_record = {
                    "batch_id": batch_id, "phase": phase, "batch_size": len(rows),
                    "request_ids": [r["request_id"] for r in rows], "full_input_tokens": tokens,
                    "queue_depth_at_dispatch": queue_depth, "dispatch_s": dispatch, "ready_s": ready,
                    "input_build_ms": input_ms, "forward_wall_ms": forward_ms,
                    "scheduler_wall_including_wait_ms": scheduler_ms,
                    "forward_cuda_span_ms": gpu_start.elapsed_time(gpu_end),
                    "profiled": sampled, "cache_stats": cache_stats,
                }
                batch_file.write(json.dumps(batch_record) + "\n")
                for index, row in enumerate(rows):
                    arrival = row["arrival_s"] - trace_origin if phase == "timed" else dispatch
                    request_file.write(json.dumps({
                        "request_id": row["request_id"], "batch_id": batch_id, "phase": phase,
                        "user_id": row["user_id"], "date": row["date"], "cache_uid": row["cache_uid"],
                        "history_len": row["history_len"], "arrival_s": arrival,
                        "history_tokens": row["history_len"] * 2 + len(contexts),
                        "dispatch_s": dispatch, "ready_s": ready,
                        "queue_ms": (dispatch - arrival) * 1000, "latency_ms": (ready - arrival) * 1000,
                        "kv_cached_tokens": cache_stats.get("kv_cached_lengths", [None] * len(rows))[index],
                        "kv_gpu_cached_tokens": cache_stats.get("gpu_cached_lengths", [None] * len(rows))[index],
                        "kv_host_cached_tokens": cache_stats.get("host_cached_lengths", [None] * len(rows))[index],
                    }) + "\n")
                runtime["observation_wall_ms"] += (time.perf_counter() - observation_start) * 1000
                cursor += len(rows)
                batch_id += 1
                if batch_id % 100 == 0:
                    print(f"{phase}: {cursor}/{len(requests)} requests, {batch_id} batches", flush=True)
            request_file.flush()
            batch_file.flush()
        runtime["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        runtime["finished_utc"] = datetime.now(timezone.utc).isoformat()
        runtime["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        runtime["peak_memory_note"] = "Torch allocator only; NVE/FlexKV external allocations may be excluded"
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        summarize(out)
    finally:
        # Close only the manager created for this run; never signal other
        # processes by name or shut down a global/shared client registry.
        original_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        if profiler is not None:
            try:
                profiler.__exit__(*sys.exc_info())
            except Exception as error:
                cleanup_errors.append(error)
        if model is not None:
            cache = model.dense_module.kvcache
            client = getattr(cache.host_kvstorage_manager, "_client", None)
            try:
                cache.offload_flush()
            except Exception as error:
                cleanup_errors.append(error)
            finally:
                if client is not None:
                    try:
                        client.shutdown()
                    except Exception as error:
                        cleanup_errors.append(error)
        for error in cleanup_errors:
            print(f"Benchmark resource cleanup failed: {error}", file=sys.stderr)
        if cleanup_errors and not original_error:
            raise RuntimeError("Benchmark resource cleanup failed") from cleanup_errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare", help="Freeze arrivals/candidates without importing GPU libraries")
    prep.add_argument("--sequence-file", required=True)
    prep.add_argument("--batch-file", help="Optional; omit to derive arrivals from sequence CSV time_ms")
    prep.add_argument("--max-users", type=int, default=128, help="Raw-sequence mode: scan this many source users; 0 means all")
    prep.add_argument("--interactions-per-request", type=int, default=16, help="Raw-sequence mode: request every N interactions, plus each day's last prefix")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--checkpoint-dir", help="Validate history keys and sample only actual checkpoint PS keys")
    prep.add_argument("--max-requests", type=int, default=10000)
    prep.add_argument("--max-history-len", type=int, default=8000)
    prep.add_argument("--num-candidates", type=int, default=256)
    prep.add_argument("--date", type=int)
    prep.add_argument("--time-scale", type=float, default=1.0, help="Multiply original interarrival times by this factor")
    prep.add_argument("--seed", type=int, default=20261111)
    prep.add_argument("--disable-context", action="store_true")
    replay = stages.add_parser("run", help="Run one fixed ratio in a fresh process")
    replay.add_argument("--trace-dir", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--gin-config-file", default=str(HSTU_DIR / "inference/configs/kuairand_27k_inference_ranking.gin"))
    replay.add_argument("--checkpoint-dir")
    replay.add_argument("--smoke", action="store_true", help="Use explicitly labelled seeded nonzero fixture weights")
    replay.add_argument("--cache-budget-bytes", type=int, required=True)
    replay.add_argument("--kv-cache-ratio", type=float, required=True)
    replay.add_argument("--kv-cache-extra-configs", default="{}", help="JSON object passed to the existing FlexKV backend configuration")
    replay.add_argument("--max-batch-size", type=int, default=8)
    replay.add_argument("--max-batch-tokens", type=int, default=65536)
    replay.add_argument("--max-batch-delay-ms", type=float, default=1.0)
    replay.add_argument("--warmup-requests", type=int, default=0)
    replay.add_argument("--profile-every", type=int, default=0, help="0 disables sampling; use a separate profiling run for clean timings")
    replay.add_argument("--profile-request-ids", default="", help="Comma-separated timed request IDs to profile")
    replay.add_argument("--profile-max-batches", type=int, default=3)
    report = stages.add_parser("report", help="Rebuild the summary from saved measurements")
    report.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for name in ("max_requests", "max_history_len", "num_candidates", "max_batch_size", "max_batch_tokens", "interactions_per_request"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("warmup_requests", "profile_every", "profile_max_batches", "max_batch_delay_ms", "max_users"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    if hasattr(args, "time_scale") and (not math.isfinite(args.time_scale) or args.time_scale <= 0):
        parser.error("time_scale must be finite and positive")
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "run":
        run(args)
    else:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
