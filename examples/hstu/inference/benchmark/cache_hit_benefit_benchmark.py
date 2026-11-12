#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measure conditional KV/embedding hit benefits on frozen real requests.

Batch=1, no arrivals, no wait injection. Four cache states share one model and
one physical layout. State preparation is excluded and separately reported.
This is a service-time calibration, not an online same-budget all-hit result.
Optional P states retain the actual previous request's same-user KV prefix.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

from cache_ratio_serving_benchmark import CONTEXT_SIZES, HSTU_DIR, sha256


def select_requests(trace_dir, history_targets, time_windows, max_samples, previous_prefix=False):
    """Pick the nearest length in each chronological request window, streaming."""
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    count = manifest.get("request_count", manifest.get("requests"))
    if not isinstance(count, int) or count <= 0:
        with (trace_dir / "trace.jsonl").open() as handle:
            count = sum(1 for _ in handle)
    selected, last_prefix = {}, {}
    with (trace_dir / "trace.jsonl").open() as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if previous_prefix:
                previous = last_prefix.get(row["cache_uid"])
                row["previous_history_len"] = previous[0] if previous is not None else None
                row["previous_request_id"] = previous[1] if previous is not None else None
                if previous is not None and not 0 < previous[0] < row["history_len"]:
                    raise ValueError(f"Expected growing same-user prefixes at request {row['request_id']}")
                last_prefix[row["cache_uid"]] = (row["history_len"], row["request_id"])
            window = min(time_windows - 1, index * time_windows // count)
            for target in history_targets:
                key = (window, target)
                distance = abs(row["history_len"] - target)
                if key not in selected or distance < selected[key][0]:
                    selected[key] = (distance, row)
    rows = {}
    for (window, target), (_, row) in selected.items():
        saved = rows.setdefault(row["request_id"], {**row, "selection_targets": []})
        saved["selection_targets"].append({"request_window": window, "history_target": target})
    chosen = sorted(rows.values(), key=lambda row: row["request_id"])
    if len(chosen) > max_samples:
        chosen = [chosen[index * len(chosen) // max_samples] for index in range(max_samples)]
    sequences = {}
    wanted = {row["cache_uid"] for row in chosen}
    with (trace_dir / "sequences.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["cache_uid"] in wanted:
                sequences[row["cache_uid"]] = row
    if set(sequences) != wanted or not rows:
        raise ValueError("Frozen input lacks selected users/requests")
    return manifest, chosen, sequences


def summarize(records):
    """Keep paired differences; do not multiply an average hit rate by a mean cost."""
    timed = [row for row in records if not row["profiled"]]
    paired = {}
    for row in timed:
        if row["state"] not in {"HH", "HM", "MH", "MM", "PH", "PM"}:
            continue
        paired.setdefault((row["request_id"], row["repetition"]), {})[row["state"]] = row
    differences, previous_differences = [], []
    for (request_id, repetition), states in paired.items():
        latency = {name: row["service_ms"] for name, row in states.items()}
        if "HH" in states or "HM" in states:
            if not {"HH", "HM", "MH", "MM"}.issubset(states):
                raise ValueError("Incomplete four-state pair")
            differences.append({
                "request_id": request_id, "repetition": repetition,
                "history_items": states["HH"]["history_items"],
                "kv_benefit_embedding_hit_ms": latency["MH"] - latency["HH"],
                "kv_benefit_embedding_miss_ms": latency["MM"] - latency["HM"],
                "embedding_benefit_kv_hit_ms": latency["HM"] - latency["HH"],
                "embedding_benefit_kv_miss_ms": latency["MM"] - latency["MH"],
                "joint_benefit_ms": latency["MM"] - latency["HH"],
            })
        if "PH" in states or "PM" in states:
            if not {"PH", "PM", "MH", "MM"}.issubset(states):
                raise ValueError("Incomplete previous-prefix pair")
            previous_differences.append({
                "request_id": request_id, "repetition": repetition,
                "history_items": states["PH"]["history_items"],
                "previous_history_items": states["PH"]["previous_history_items"],
                "kv_cached_tokens": states["PH"]["cache_stats"]["kv_cached_tokens"],
                "kv_previous_benefit_embedding_hit_ms": latency["MH"] - latency["PH"],
                "kv_previous_benefit_embedding_miss_ms": latency["MM"] - latency["PM"],
                "embedding_benefit_kv_previous_ms": latency["PM"] - latency["PH"],
                "previous_joint_benefit_ms": latency["MM"] - latency["PH"],
            })
    by_request = []
    for request_id in sorted({row["request_id"] for row in differences}):
        pairs = [row for row in differences if row["request_id"] == request_id]
        means = {key: statistics.fmean(row[key] for row in pairs)
                 for key in pairs[0] if key.endswith("_ms")}
        deviation = {key: statistics.stdev(row[key] for row in pairs) if len(pairs) > 1 else 0
                     for key in means}
        by_request.append({"request_id": request_id, "history_items": pairs[0]["history_items"],
                           "repetitions": len(pairs), "mean": means, "stdev": deviation})
    previous_by_request = []
    for request_id in sorted({row["request_id"] for row in previous_differences}):
        pairs = [row for row in previous_differences if row["request_id"] == request_id]
        means = {key: statistics.fmean(row[key] for row in pairs)
                 for key in pairs[0] if key.endswith("_ms")}
        previous_by_request.append({
            "request_id": request_id, "history_items": pairs[0]["history_items"],
            "previous_history_items": pairs[0]["previous_history_items"],
            "kv_cached_tokens": pairs[0]["kv_cached_tokens"], "repetitions": len(pairs), "mean": means,
            "stdev": {key: statistics.stdev(row[key] for row in pairs) if len(pairs) > 1 else 0. for key in means},
        })
    curves = []
    for request_id in sorted({row["request_id"] for row in timed}):
        measured = [row for row in timed if row["request_id"] == request_id and row["state"].startswith("M")]
        cold_by_rep = {row["repetition"]: row for row in measured if row["state"] == "MM"}
        for state in sorted({row["state"] for row in measured}):
            rows = [row for row in measured if row["state"] == state]
            curves.append({
                "request_id": request_id, "state": state,
                "nominal_embedding_prefill_fraction": rows[0].get("embedding_prefill_fraction", int(state == "MH")),
                "actual_unique_hit_ids": statistics.fmean(
                    sum(table["unique_hit_ids"] for table in row["embedding"].values()) for row in rows),
                "actual_unique_ids": statistics.fmean(
                    sum(table["unique_ids"] for table in row["embedding"].values()) for row in rows),
                "mean_service_ms": statistics.fmean(row["service_ms"] for row in rows),
                "mean_embedding_benefit_vs_cold_ms": statistics.fmean(
                    cold_by_rep[row["repetition"]]["service_ms"] - row["service_ms"] for row in rows),
            })
    return {"scope": "Paired conditional batch=1 service-time benefits; not workload-weighted savings",
            "pairs": differences, "by_request": by_request, "embedding_curves_kv_miss": curves,
            "previous_prefix_pairs": previous_differences, "previous_prefix_by_request": previous_by_request,
            "previous_prefix_skipped_request_ids": sorted({row["request_id"] for row in timed
                if row.get("previous_prefix_status") == "skipped_no_previous_request"})}


def run(args):
    trace_dir, out = Path(args.trace_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    manifest, requests, sequences = select_requests(
        trace_dir, [int(value) for value in args.history_targets.split(",")], args.time_windows, args.samples,
        args.previous_prefix)
    for filename, expected in manifest.get("artifacts", {}).items():
        if sha256(trace_dir / filename) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    (out / "selected_requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    sys.path[:0] = [str(HSTU_DIR), str(HSTU_DIR.parent), str(HSTU_DIR / "model"),
                   str(Path(args.memory_poc_extension).resolve()),
                   str(Path(__file__).with_name("memory_transfer_poc"))]
    os.environ.setdefault("HSTU_INFERENCE_ONLY", "1")
    os.environ.setdefault("FLEXKV_ENABLE_MPS", "0")
    import gin
    import torch
    from banked_embedding import BankedNVEEmbedding, tensor_span
    from commons.datasets.hstu_batch import HSTUBatch
    from configs import EmbeddingBackend, InferenceEmbeddingConfig
    from inference.inference_gr_ranking import get_inference_hstu_model
    from pool import CachePool
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    from utils import NetworkArgs

    gin.parse_config_file(args.gin_config_file)
    network = NetworkArgs()
    if (network.num_layers, network.hidden_size, network.num_attention_heads, network.kv_channels) != (3, 512, 4, 128):
        raise ValueError("This calibration requires the agreed 3-layer/512-wide inference model")
    contexts, dim = manifest["context_features"], network.hidden_size
    max_tokens = 2 * manifest["max_history_len"] + manifest["num_candidates"] + len(contexts)
    torch.manual_seed(manifest["seed"])
    sizes = {**CONTEXT_SIZES, "action_weights": 246, **manifest["dynamic_table_rows"]}
    configs = [InferenceEmbeddingConfig(
        feature_names=[name], table_name=name, vocab_size=sizes[name], dim=dim,
        use_dynamicemb=name in ("user_id", "video_id"),
    ) for name in [*contexts, "video_id", "action_weights"]]
    kv_options = {"flexkv_mode": "direct", "flexkv_num_cpu_blocks": 0,
                  "flexkv_num_local_blocks": 0, "flexkv_num_tmp_cpu_blocks": 1,
                  "flexkv_enable_mps": 0}
    model = pool = user_embedding = None
    records = []
    try:
        with torch.inference_mode():
            model = get_inference_hstu_model(
                configs, 1, len(contexts), max_tokens, args.checkpoint_dir, True,
                embedding_backend=EmbeddingBackend.NVEMB,
                cache_budget_bytes=args.cache_budget_bytes, kv_cache_ratio=.75,
                kv_cache_extra_configs=kv_options)
            model.eval()
            collection = model.sparse_module._dynamic_embedding_collection
            if args.fixture:
                # Only selected-request keys are needed for this conditional probe.
                # Global table row counts still determine the original budget split.
                keys_by_table = {"video_id": set(), "user_id": set()}
                for row in requests:
                    sequence = sequences[row["cache_uid"]]
                    keys_by_table["video_id"].update(sequence["video_id"][:row["history_len"]])
                    keys_by_table["video_id"].update(row["candidates"])
                    keys_by_table["user_id"].add(row["user_id"])
                for name, storage in collection._owned_parameter_servers.items():
                    keys = sorted(keys_by_table[name])
                    generator = torch.Generator().manual_seed(manifest["seed"] + len(name))
                    for offset in range(0, len(keys), 8192):
                        chunk = torch.tensor(keys[offset:offset + 8192], dtype=torch.int64)
                        values = torch.randn((len(chunk), dim), generator=generator) * .02
                        storage.insert(chunk, values)
                for embedding in model.sparse_module._static_embedding_collection.embeddings.values():
                    embedding.weight.normal_(mean=0, std=.02)
            pool = CachePool(model, args.kv_cache_ratio)
            # A rebuildable user table gives both dynamic tables the same controlled
            # cold/hot protocol. This change is reported; it is not stock NVEMB cost.
            if "user_id" in collection.embeddings:
                user_bytes = collection.gpu_cache_bytes_by_table["user_id"]
                del collection.embeddings["user_id"]
                gc.collect()
                user_buffer = torch.empty(user_bytes, dtype=torch.uint8, device="cuda")
                user_embedding = BankedNVEEmbedding(
                    collection._owned_parameter_servers["user_id"], dim,
                    [tensor_span(user_buffer)], max_batch_keys=16384)
                collection.embeddings["user_id"] = user_embedding
                collection.embeddings = torch.nn.ModuleDict({
                    config.name: collection.embeddings[config.name]
                    for config in collection._embedding_configs})
        modules = {"video_id": pool.embedding}
        if user_embedding is not None:
            modules["user_id"] = user_embedding
        fractions = [float(value) for value in args.embedding_fractions.split(",") if value.strip()]
        if fractions and any(not hasattr(module, "probe_unique") for module in modules.values()):
            raise ValueError("Partial embedding calibration requires BankedNVEEmbedding.probe_unique")
        if any(not hasattr(module, "clear_gpu_cache") for module in modules.values()):
            raise ValueError("Calibration requires clear_gpu_cache() that preserves execution contexts")
        model.collect_cache_stats = True
        model.profile_cache = False
        device = torch.cuda.current_device()

        def build_batch(row):
            values, lengths, feature_max = [], [], {}
            sequence = sequences[row["cache_uid"]]
            for name in [*contexts, "video_id", "action_weights"]:
                if name in contexts:
                    feature = [sequence["context"][name]]
                else:
                    feature = sequence[name][:row["history_len"]]
                    feature = feature + (row["candidates"] if name == "video_id"
                                         else [0] * len(row["candidates"]))
                values.extend(feature)
                lengths.append(len(feature))
                feature_max[name] = len(feature)
            return HSTUBatch(
                features=KeyedJaggedTensor.from_lengths_sync(
                    keys=[*contexts, "video_id", "action_weights"],
                    values=torch.tensor(values, dtype=torch.int64, device=device),
                    lengths=torch.tensor(lengths, dtype=torch.int64, device=device)),
                batch_size=1, feature_to_max_seqlen=feature_max,
                contextual_feature_names=contexts, item_feature_name="video_id",
                action_feature_name="action_weights", max_num_candidates=len(row["candidates"]),
                num_candidates=torch.tensor([len(row["candidates"])], dtype=torch.int64, device=device))

        def clear_embedding():
            # Clear GPU entries only. Retire/restore would destroy the warmed
            # lookup, CPU-PS and insert contexts, biasing cold-state measurements.
            for module in modules.values():
                module.clear_gpu_cache()

        def expected_queries(row, cached_tokens):
            batch = build_batch(row)
            model.strip_cached_tokens(batch, torch.tensor([cached_tokens], dtype=torch.int32))
            return {name: batch.features[name].values() for name in modules}

        runtime = {
            "calibration_protocol": "v2_real_full_query_warmup_gpu_clear_preserves_contexts",
            "configuration": vars(args), "trace_manifest_sha256": sha256(trace_dir / "manifest.json"),
            "implementation_sha256": sha256(__file__), "selected_requests": len(requests),
            "model": {"layers": 3, "hidden": dim, "heads": 4, "kv_channels": 128, "dtype": "bfloat16"},
            "cache_budget": pool.capacities(), "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "weights": "seeded_nonzero_fixture" if args.fixture else "checkpoint",
            "request_warmup": [],
            "previous_prefix_skipped_request_ids": [row["request_id"] for row in requests
                if args.previous_prefix and row["previous_history_len"] is None],
            "notes": [
                "No injected wait; real NVE CPU parameter table on misses; FlexKV CPU capacity zero.",
                "Batch=1 service includes input construction and result readiness, excluding queue and state preparation.",
                "Calibration samples cover request-index windows and history lengths, not workload-weighted averages.",
                "Video cache uses the shared banked POC; user cache is replaced by one bank to control cold/hot state.",
                "NVE occurrence hits are rounded from backend float hit rates. Pure-state validation determines unique hits.",
                "Each request first warms its real full KV-miss embedding lookup+insert and KV-hit/previous-prefix paths. State preparation clears GPU entries only, preserving lookup, CPU-PS and insertion scratch contexts; no dummy rows are used.",
                "Bank lookup batches with PS hits are not network MGET calls; unique payload is an estimate, not measured transfer bytes.",
                "Profiled runs are excluded from timing statistics; inclusive stage spans overlap.",
                "Partial embedding samples use per-table unique-key prefill; actual hits are verified by a read-only GPU probe, then the entire state is prepared again before timing.",
                "Optional P state retains the actual immediately preceding same-user history prefix; its measured benefit is not a linear fraction of full KV-hit benefit.",
                "Preparation may need more cached data than an online working set and is not an online full-hit baseline.",
                "Fixture weights and selected-key PS population make no model-accuracy claim." if args.fixture else "Checkpoint accuracy not evaluated.",
            ],
        }
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with (out / "measurements.jsonl").open("w", buffering=1) as handle:
            for row in requests:
                uid = torch.tensor([row["cache_uid"]], dtype=torch.int64)
                histories = torch.tensor([2 * row["history_len"] + len(contexts)], dtype=torch.int32)
                previous_history = row.get("previous_history_len") if args.previous_prefix else None
                cached_tokens = {"M": 0, "H": int(histories[0])}
                previous_row = previous_histories = None
                if previous_history is not None:
                    previous_row = {**row, "history_len": previous_history}
                    previous_histories = torch.tensor([2 * previous_history + len(contexts)], dtype=torch.int32)
                    cached_tokens["P"] = int(previous_histories[0])
                reference = None
                # A forced-cold full request warms the largest real per-bank
                # lookup+insert shape needed by this request's H/M/P states.
                warmup_started = time.perf_counter()
                with torch.inference_mode():
                    pool.device_cache.evict_all()
                    clear_embedding()
                    logits = model.forward_with_kvcache(build_batch(row), uid, histories)
                    if model.last_cache_stats.get("kv_cached_tokens") != 0:
                        raise RuntimeError("Full-shape warmup must begin with empty KV state")
                    full_lookup = {name: dict(module.last_lookup_stats) for name, module in modules.items()}
                    if any(stats["gpu_hits"] != 0 or stats["ps_hits"] != stats["queries"]
                           for stats in full_lookup.values()):
                        raise RuntimeError(f"Full-shape warmup must exercise CPU-PS lookup and insert: {full_lookup}")
                    logits = model.forward_with_kvcache(build_batch(row), uid, histories)
                    reference = logits.detach().clone()
                    # Warm the real partial-prefix shape too. Its retained history
                    # must produce the same current-request outputs as full reuse.
                    if previous_row is not None:
                        pool.device_cache.evict_all()
                        model.forward_with_kvcache(build_batch(previous_row), uid, previous_histories)
                        partial_logits = model.forward_with_kvcache(build_batch(row), uid, histories)
                        if model.last_cache_stats.get("kv_cached_tokens") != cached_tokens["P"]:
                            raise RuntimeError("Previous-prefix warmup did not retain the requested history")
                        torch.testing.assert_close(partial_logits, reference, rtol=.02, atol=.002)
                torch.cuda.synchronize()
                runtime["request_warmup"].append({
                    "request_id": row["request_id"], "wall_ms": (time.perf_counter() - warmup_started) * 1000,
                    "full_embedding_lookup_by_table": full_lookup,
                    "previous_prefix_warmed": previous_row is not None,
                })
                (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
                queries = {mode: expected_queries(row, length) for mode, length in cached_tokens.items()}
                prefill_keys = {}
                for kv_mode in queries:
                    for fraction in [0., 1., *fractions]:
                        selected = {}
                        for name, keys in queries[kv_mode].items():
                            unique = keys.unique()
                            generator = torch.Generator().manual_seed(manifest["seed"] + row["request_id"] + len(name))
                            order = torch.randperm(len(unique), generator=generator, device="cpu").to(device)
                            selected[name] = unique[order[:int(len(unique) * fraction)]]
                        prefill_keys[kv_mode, fraction] = selected

                def prepare_state(kv_mode, fraction):
                    with torch.inference_mode():
                        pool.device_cache.evict_all()
                        if kv_mode == "H":
                            model.forward_with_kvcache(build_batch(row), uid, histories)
                        elif kv_mode == "P":
                            model.forward_with_kvcache(build_batch(previous_row), uid, previous_histories)
                        clear_embedding()
                        for name, keys in prefill_keys[kv_mode, fraction].items():
                            if keys.numel():
                                modules[name](keys)
                    torch.cuda.synchronize()

                for repetition in range(args.repetitions + int(args.profile)):
                    profiled = repetition == args.repetitions
                    states = [("MM", "M", 0.), ("MH", "M", 1.)]
                    if not args.previous_prefix_only:
                        states = [("HH", "H", 1.), ("HM", "H", 0.), *states]
                    states += [(f"M{fraction:g}", "M", fraction) for fraction in fractions]
                    if previous_row is not None:
                        states += [("PH", "P", 1.), ("PM", "P", 0.)]
                    offset = repetition % len(states)
                    states = states[offset:] + states[:offset]
                    if repetition % 2:
                        states.reverse()
                    for state, kv_mode, fraction in states:
                        embedding_hit = fraction == 1
                        prepared = time.perf_counter()
                        prepare_state(kv_mode, fraction)
                        probes = {}
                        if 0 < fraction < 1:
                            for name, keys in queries[kv_mode].items():
                                probes[name] = modules[name].probe_unique(keys)
                            # The probe reads GPU cache payload. Recreate the same
                            # state so this observation cannot warm the timed lookup.
                            prepare_state(kv_mode, fraction)
                        preparation_ms = (time.perf_counter() - prepared) * 1000
                        profile = None
                        if profiled:
                            model.profile_cache = True
                            for module in modules.values():
                                module.profile = True
                            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                         torch.profiler.ProfilerActivity.CUDA])
                            profile.__enter__()
                        cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        began = time.perf_counter()
                        batch = build_batch(row)
                        built = time.perf_counter()
                        cuda_start.record()
                        with torch.inference_mode():
                            logits = model.forward_with_kvcache(batch, uid, histories)
                        cuda_end.record()
                        cuda_end.synchronize()
                        ready = time.perf_counter()
                        stages = []
                        if profile is not None:
                            profile.__exit__(None, None, None)
                            for event in profile.key_averages():
                                if event.key.startswith("recsys."):
                                    stages.append({"name": event.key, "count": event.count,
                                                   "cpu_inclusive_ms": event.cpu_time_total / 1000,
                                                   "cpu_self_ms": event.self_cpu_time_total / 1000,
                                                   "gpu_inclusive_ms": getattr(event, "device_time_total", 0) / 1000})
                            model.profile_cache = False
                            for module in modules.values():
                                module.profile = False
                        stats = dict(model.last_cache_stats)
                        expected_cached = cached_tokens[kv_mode]
                        if stats.get("kv_cached_tokens") != expected_cached:
                            raise RuntimeError(f"KV state mismatch at request {row['request_id']}: {state}, {stats}")
                        embedding_stats = {}
                        for name, module in modules.items():
                            keys = queries[kv_mode][name]
                            measured = dict(module.last_lookup_stats)
                            count, unique = keys.numel(), keys.unique().numel()
                            # Empty features may be skipped by the collection, leaving stale stats.
                            if not count:
                                measured = {"queries": 0, "gpu_hits": 0, "ps_hits": 0, "banks": []}
                            probe = probes.get(name)
                            target_hits = (probe["prelookup_occurrence_gpu_hits"] if probe is not None
                                           else count if embedding_hit else 0)
                            if measured["queries"] != count or measured["gpu_hits"] != target_hits:
                                raise RuntimeError(f"Embedding state mismatch {name}/{state}: {measured}, expected {target_hits}/{count}")
                            unique_hits = (probe["prelookup_unique_gpu_hits"] if probe is not None
                                           else unique if embedding_hit else 0)
                            embedding_stats[name] = {
                                **measured, "unique_ids": unique,
                                "unique_hit_ids": unique_hits,
                                "unique_miss_ids": unique - unique_hits,
                                "unique_miss_payload_bytes": (unique - unique_hits) * dim * 4,
                                "requested_prefill_unique_ids": prefill_keys[kv_mode, fraction][name].numel(),
                                "state_probe": probe,
                                "bank_lookup_batches_with_ps_hits": sum(bank["ps_hits"] > 0 for bank in measured["banks"]),
                            }
                        if not torch.isfinite(logits).all().item():
                            raise RuntimeError(f"Nonfinite output at request {row['request_id']}/{state}")
                        torch.testing.assert_close(logits, reference, rtol=.02, atol=.002)
                        record = {
                            "request_id": row["request_id"], "source_timestamp_ms": row["timestamp_ms"],
                            "history_items": row["history_len"], "candidate_ids": len(row["candidates"]),
                            "previous_history_items": previous_history,
                            "previous_request_id": row.get("previous_request_id"),
                            "prepared_kv_history_items": (expected_cached - len(contexts)) // 2 if expected_cached else 0,
                            "expected_kv_cached_tokens": expected_cached,
                            "previous_prefix_status": ("available" if previous_row is not None else
                                "skipped_no_previous_request" if args.previous_prefix else "disabled"),
                            "selection_targets": row["selection_targets"], "state": state,
                            "embedding_prefill_fraction": fraction,
                            "repetition": repetition, "profiled": profiled, "batch_size": 1,
                            "preparation_ms": preparation_ms, "input_build_ms": (built - began) * 1000,
                            "forward_wall_ms": (ready - built) * 1000,
                            "service_ms": (ready - began) * 1000,
                            "forward_cuda_span_ms": cuda_start.elapsed_time(cuda_end),
                            "cache_stats": stats, "embedding": embedding_stats, "stages": stages,
                            "max_abs_error": (logits.float() - reference.float()).abs().max().item(),
                        }
                        records.append(record)
                        handle.write(json.dumps(record) + "\n")
                print(json.dumps({"completed_request": row["request_id"], "history": row["history_len"]}), flush=True)
                (out / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")
        runtime["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        runtime["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        print(json.dumps({"complete": True, "requests": len(requests), "measurements": len(records)}), flush=True)
    finally:
        if user_embedding is not None:
            user_embedding.close()
        if pool is not None:
            pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-poc-extension", required=True)
    parser.add_argument("--gin-config-file", default=str(HSTU_DIR / "inference/configs/kuairand_1k_inference_ranking.gin"))
    parser.add_argument("--cache-budget-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--kv-cache-ratio", type=float, choices=(.25, .5, .75), default=.5)
    parser.add_argument("--history-targets", default="100,250,500,1000,1500,2000,3000,4000")
    parser.add_argument("--time-windows", type=int, default=3)
    parser.add_argument("--samples", type=int, default=24, help="Cap selected requests, spread across the selected timeline")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--embedding-fractions", default="", help="Optional extra KV-miss states, e.g. .25,.5,.75; requires read-only unique-hit probe")
    parser.add_argument("--previous-prefix", action="store_true", help="Also measure PH/PM using the actual previous same-user history")
    parser.add_argument("--previous-prefix-only", action="store_true", help="Measure only MM/MH/PH/PM; implies --previous-prefix")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--fixture", action="store_true")
    weights.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    args.previous_prefix = args.previous_prefix or args.previous_prefix_only
    if args.repetitions < 2 or args.time_windows < 1 or args.samples < 1 or args.cache_budget_bytes <= 0:
        parser.error("Need at least two repetitions, a positive window count and cache budget")
    if any(int(value) <= 0 for value in args.history_targets.split(",")):
        parser.error("history-targets must contain positive integers")
    fractions = [float(value) for value in args.embedding_fractions.split(",") if value.strip()]
    if len(set(fractions)) != len(fractions) or any(not 0 < fraction < 1 for fraction in fractions):
        parser.error("embedding-fractions must be distinct values strictly between 0 and 1")
    if args.previous_prefix_only and fractions:
        parser.error("previous-prefix-only does not include embedding-fractions states")
    run(args)


if __name__ == "__main__":
    main()
