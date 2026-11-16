# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed cache split, original trace arrivals, FIFO dynamic batches, native HSTU.

Requires the repository's installed HSTU/NVEmb/FlexKV CUDA dependencies.
cache-ratio is the KV share of the cache payload budget; embeddings use the rest.
Model and embedding weights use seeded initialization, without a checkpoint.
Startup probes remain in cache; setup is outside replay. All requests/tails count.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

HSTU_DIR = Path(__file__).resolve().parents[2]
MAX_BATCH_DELAY_S = 0.001
CONTEXT_SIZES = dict(
    user_active_degree=10,
    follow_user_num_range=9,
    fans_user_num_range=10,
    friend_user_num_range=8,
    register_days_range=8,
)


def load_trace(directory, max_tokens):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for value, minimum in (
        (manifest["request_count"], 1),
        (manifest["num_candidates"], 1),
        (manifest["max_history_len"], 0),
        (manifest["seed"], 0),
        (manifest["dynamic_table_rows"]["user_id"], 1),
        (manifest["dynamic_table_rows"]["video_id"], 1),
    ):
        if type(value) is not int or not minimum <= value < 2**63:
            raise ValueError(f"Invalid manifest integer: {value!r}")
    contexts = manifest["context_features"]
    if (
        not isinstance(contexts, list)
        or not contexts
        or contexts[0] != "user_id"
        or len(set(contexts)) != len(contexts)
        or any(x != "user_id" and x not in CONTEXT_SIZES for x in contexts)
    ):
        raise ValueError("Invalid ordered context_features")
    with (root / "sequences.jsonl").open() as stream:
        sequence_rows = [json.loads(line) for line in stream if line.strip()]
    sequences = {}
    for seq in sequence_rows:
        uid = seq["cache_uid"]
        values = [
            uid,
            *(seq["context"][name] for name in contexts),
            *seq["video_id"],
            *seq["action_weights"],
        ]
        if any(type(v) is not int or not 0 <= v < 2**63 for v in values):
            raise ValueError(
                "Sequence IDs and feature values must be nonnegative int64"
            )
        if uid in sequences or len(seq["video_id"]) != len(seq["action_weights"]):
            raise ValueError("Duplicate sequence UID or unequal history arrays")
        for name in contexts:
            value = seq["context"][name]
            if name != "user_id" and value >= CONTEXT_SIZES[name]:
                raise ValueError("Context value exceeds its vocabulary")
        if any(key >= 246 for key in seq["action_weights"]):
            raise ValueError("Action exceeds vocabulary")
        sequences[uid] = seq
    with (root / "trace.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != manifest["request_count"]:
        raise ValueError("Trace count differs from manifest.request_count")
    previous, histories = -1.0, {}
    for i, row in enumerate(rows):
        uid, h = row["cache_uid"], row["history_len"]
        arrival, candidates = row["arrival_s"], row["candidates"]
        if (
            not isinstance(candidates, list)
            or not 0 < len(candidates) <= manifest["num_candidates"]
        ):
            raise ValueError("Invalid candidate count")
        if any(
            type(v) is not int or not 0 <= v < 2**63
            for v in [row["request_id"], uid, h, *candidates]
        ):
            raise ValueError(
                "Request IDs, history length and candidates must be nonnegative int64"
            )
        if row["request_id"] != i:
            raise ValueError("Request IDs must be consecutive from zero")
        if (
            type(arrival) not in (int, float)
            or not math.isfinite(arrival)
            or arrival < 0
            or arrival < previous
        ):
            raise ValueError("Arrival times must be finite, nonnegative and ordered")
        if (
            uid not in sequences
            or h < histories.get(uid, 0)
            or h > min(len(sequences[uid]["video_id"]), manifest["max_history_len"])
        ):
            raise ValueError(
                "History must be a nondecreasing prefix for each cache UID"
            )
        row["tokens"] = len(contexts) + 2 * h + len(candidates)
        if row["tokens"] > max_tokens:
            raise ValueError(
                f"Request {i} exceeds max-batch-tokens; no requests are dropped"
            )
        previous, histories[uid] = arrival, h
    return manifest, sequences, rows


def feature_layout(rows, sequences, contexts, candidate_bound):
    names, values, lengths, bounds = (
        [*contexts, "video_id", "action_weights"],
        [],
        [],
        {},
    )
    for name in names:
        for row in rows:
            seq, h, candidates = (
                sequences[row["cache_uid"]],
                row["history_len"],
                row["candidates"],
            )
            feature = (
                [seq["context"][name]]
                if name in contexts
                else seq[name][:h]
                + (candidates if name == "video_id" else [0] * len(candidates))
            )
            values.extend(feature)
            lengths.append(len(feature))
        # The existing processor subtracts candidate_bound, even for shorter rows.
        bounds[name] = (
            1
            if name in contexts
            else max(r["history_len"] for r in rows) + candidate_bound
        )
    return names, values, lengths, bounds


class HSTU:
    def __init__(
        self,
        *,
        cache_memory_backend="default",
        peak_cache_bytes=None,
        host_staging_limit_bytes=256 * 1024**2,
    ):
        self.model = None
        self.torch = None
        self.memory_options = dict(
            cache_memory_backend=cache_memory_backend,
            peak_cache_bytes=peak_cache_bytes,
            host_staging_limit_bytes=host_staging_limit_bytes,
        )

    def setup(self, args, manifest, sequences, rows):
        for path in (HSTU_DIR, HSTU_DIR.parent, HSTU_DIR / "model"):
            sys.path.insert(0, str(path))
        os.environ.setdefault("HSTU_INFERENCE_ONLY", "1")
        os.environ.setdefault("FLEXKV_ENABLE_MPS", "0")
        import gin
        import torch
        from commons.datasets.hstu_batch import HSTUBatch
        from configs import EmbeddingBackend, InferenceEmbeddingConfig
        from inference.inference_gr_ranking import get_inference_hstu_model
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        from utils import NetworkArgs

        self.torch, self.batch_type, self.kjt = torch, HSTUBatch, KeyedJaggedTensor
        self.manifest, self.sequences = manifest, sequences
        self.contexts = manifest["context_features"]
        gin.clear_config()
        gin.parse_config_file(
            str(HSTU_DIR / "inference/configs/kuairand_1k_inference_ranking.gin")
        )
        torch.manual_seed(manifest["seed"])
        network = NetworkArgs()
        sizes = {
            **CONTEXT_SIZES,
            "action_weights": 246,
            **manifest["dynamic_table_rows"],
        }
        configs = [
            InferenceEmbeddingConfig(
                feature_names=[n],
                table_name=n,
                vocab_size=sizes[n],
                dim=network.hidden_size,
                use_dynamicemb=n in ("user_id", "video_id"),
            )
            for n in [*self.contexts, "video_id", "action_weights"]
        ]
        with torch.inference_mode():
            self.model = get_inference_hstu_model(
                configs,
                args.max_batch_size,
                len(self.contexts),
                len(self.contexts)
                + 2 * manifest["max_history_len"]
                + manifest["num_candidates"],
                None,
                True,
                embedding_backend=EmbeddingBackend.NVEMB,
                cache_budget_bytes=args.cache_bytes,
                kv_cache_ratio=args.cache_ratio,
                **self.memory_options,
            )
            # Populate the CPU parameter servers with actual trace IDs, not 0..N-1.
            history_ends, video_keys = {}, set()
            for row in rows:
                uid = row["cache_uid"]
                history_ends[uid] = max(history_ends.get(uid, 0), row["history_len"])
                video_keys.update(row["candidates"])
            for uid, end in history_ends.items():
                video_keys.update(sequences[uid]["video_id"][:end])
            keys_by_table = {
                "user_id": sorted(
                    {sequences[uid]["context"]["user_id"] for uid in history_ends}
                ),
                "video_id": sorted(video_keys),
            }
            del video_keys
            collection = self.model.sparse_module._dynamic_embedding_collection
            for name, storage in collection._owned_parameter_servers.items():
                keys = keys_by_table[name]
                if len(keys) > sizes[name]:
                    raise ValueError(f"Trace key count exceeds table size: {name}")
                generator = torch.Generator(device="cpu").manual_seed(
                    manifest["seed"] + len(name)
                )
                for offset in range(0, len(keys), 16384):
                    chunk = torch.tensor(
                        keys[offset : offset + 16384], dtype=torch.int64, device="cpu"
                    )
                    values = (
                        torch.randn(
                            (len(chunk), network.hidden_size),
                            generator=generator,
                            dtype=torch.float32,
                            device="cpu",
                        )
                        * 0.02
                    )
                    storage.insert(chunk, values)
            static = self.model.sparse_module._static_embedding_collection.embeddings
            for name, embedding in static.items():
                generator = torch.Generator(device=embedding.weight.device).manual_seed(
                    manifest["seed"] + len(name)
                )
                embedding.weight.normal_(mean=0, std=0.02, generator=generator)
            cache = self.model.dense_module.kvcache.gpu_kvcache_mgr
            for table in cache.gpu_kvcache_tables:
                table.zero_()
        torch.cuda.synchronize()
        self.event = torch.cuda.Event()
        probes, seen = [], set()
        for row in rows:
            if row["cache_uid"] not in seen:
                probes.append(row)
                seen.add(row["cache_uid"])
            if len(probes) == min(8, args.max_batch_size):
                break
        outputs = []
        for no_kv in (False, False, True):
            value = self.forward(probes, no_kv=no_kv)
            self.synchronize()
            self.validate(value)
            outputs.append(value.detach().clone())
        for value in outputs[:2]:
            torch.testing.assert_close(value, outputs[2], rtol=0.02, atol=0.005)
        self.model.dense_module.kvcache.offload_flush()
        torch.cuda.synchronize()
        return dict(
            weights="seeded_untrained",
            seed=manifest["seed"],
            embedding_key_order="sorted IDs used by trace histories and candidates",
            initialized_embedding_rows={
                name: len(keys) for name, keys in keys_by_table.items()
            },
            cache_budget=self.model.cache_budget_report,
            budget_scope="cache payload; model/static tables/allocator overhead excluded",
            host_tier="none (VMM GPU-only)"
            if self.memory_options["cache_memory_backend"] == "vmm"
            else "standard FlexKV",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            initial_cache_state="startup probe retained; not cold",
            probe_request_ids=[r["request_id"] for r in probes],
            validation="startup KV/KV/noKV rtol=.02 atol=.005; replay finite-only",
        )

    def forward(self, rows, no_kv=False):
        if len({r["cache_uid"] for r in rows}) != len(rows):
            raise ValueError("Shared KV state requires distinct cache UIDs per batch")
        t = self.torch
        names, values, lengths, bounds = feature_layout(
            rows, self.sequences, self.contexts, self.manifest["num_candidates"]
        )
        with t.inference_mode():
            batch = self.batch_type(
                features=self.kjt.from_lengths_sync(
                    keys=names,
                    values=t.tensor(values, dtype=t.int64, device="cuda"),
                    lengths=t.tensor(lengths, dtype=t.int64, device="cuda"),
                ),
                batch_size=len(rows),
                feature_to_max_seqlen=bounds,
                contextual_feature_names=self.contexts,
                item_feature_name="video_id",
                action_feature_name="action_weights",
                max_num_candidates=self.manifest["num_candidates"],
                num_candidates=t.tensor(
                    [len(r["candidates"]) for r in rows], dtype=t.int64, device="cuda"
                ),
            )
            result = (
                self.model.forward_nokvcache(batch)
                if no_kv
                else self.model.forward_with_kvcache(
                    batch,
                    t.tensor([r["cache_uid"] for r in rows], dtype=t.int64),
                    t.tensor(
                        [len(self.contexts) + 2 * r["history_len"] for r in rows],
                        dtype=t.int32,
                    ),
                )
            )
            self.event.record()
        return result

    def synchronize(self):
        self.event.synchronize()

    def validate(self, output):
        if not self.torch.isfinite(output).all().item():
            raise ValueError("Nonfinite model output")

    def close(self):
        if self.model is not None:
            if hasattr(self.model, "cache_memory"):
                self.model.cache_memory.close()
            else:
                self.model.dense_module.kvcache.offload_flush()
                self.torch.cuda.synchronize()
            self.model = None
            gc.collect()
            self.torch.cuda.synchronize()


def choose_batch(rows, head, now, max_batch_size, max_batch_tokens):
    if rows[head]["arrival_s"] > now:
        return head, "await_arrival", rows[head]["arrival_s"]
    end, tokens, seen = head, 0, set()
    while end < len(rows) and rows[end]["arrival_s"] <= now:
        row = rows[end]
        if row["cache_uid"] in seen:
            return end, "shared_uid", None
        if tokens + row["tokens"] > max_batch_tokens:
            return end, "token_limit", None
        seen.add(row["cache_uid"])
        tokens += row["tokens"]
        end += 1
        if end - head == max_batch_size or tokens == max_batch_tokens:
            return end, "batch_limit", None
    if end == len(rows):
        return end, "end_of_trace", None
    deadline = rows[head]["arrival_s"] + MAX_BATCH_DELAY_S
    if now >= deadline:
        return end, "batch_wait_limit", None
    return head, "await_fill", min(deadline, rows[end]["arrival_s"])


def distribution(values):
    ordered = sorted(values)
    return {
        "mean": math.fsum(ordered) / len(ordered),
        "max": ordered[-1],
        **{
            k: ordered[max(0, math.ceil(p * len(ordered)) - 1)]
            for k, p in [("p50", 0.5), ("p99", 0.99), ("p99_9", 0.999)]
        },
    }


def replay(rows, backend, args, report, clock=time.perf_counter, sleep=time.sleep):
    queue, service, e2e, dispatches = [], [], [], []
    head = batches = 0
    report.update(completed_requests=0, completed_batches=0, all_tails_retained=False)
    with (args.output_dir / "requests.jsonl").open("w") as requests_log, (
        args.output_dir / "batches.jsonl"
    ).open("w") as batches_log:
        origin = clock()
        while head < len(rows):
            report["phase"] = "batch_selection"
            end, reason, wake = choose_batch(
                rows, head, clock() - origin, args.max_batch_size, args.max_batch_tokens
            )
            if end == head:
                sleep(max(0, wake - (clock() - origin)))
                continue
            batch = rows[head:end]
            dispatch = clock() - origin
            report["phase"] = "forward"
            output = backend.forward(batch)
            backend.synchronize()
            ready = clock() - origin
            report["phase"] = "validate"
            backend.validate(output)  # This cost remains on the arrival clock.
            if ready < dispatch:
                raise RuntimeError("Clock moved backwards")
            report["phase"] = "write_logs"
            for row in batch:
                q, s = (dispatch - row["arrival_s"]) * 1000, (ready - dispatch) * 1000
                if q < 0:
                    raise RuntimeError("Request dispatched before arrival")
                queue.append(q)
                service.append(s)
                e2e.append(q + s)
                dispatches.append(dispatch)
                requests_log.write(
                    json.dumps(
                        dict(
                            request_id=row["request_id"],
                            cache_uid=row["cache_uid"],
                            batch_index=batches,
                            arrival_s=row["arrival_s"],
                            dispatch_s=dispatch,
                            ready_s=ready,
                            queue_ms=q,
                            service_ms=s,
                            e2e_ms=q + s,
                        )
                    )
                    + "\n"
                )
            batches_log.write(
                json.dumps(
                    dict(
                        batch_index=batches,
                        request_ids=[r["request_id"] for r in batch],
                        batch_size=len(batch),
                        tokens=sum(r["tokens"] for r in batch),
                        reason=reason,
                        dispatch_s=dispatch,
                        ready_s=ready,
                    )
                )
                + "\n"
            )
            head, batches = end, batches + 1
            report.update(completed_requests=head, completed_batches=batches)
            requests_log.flush()
            batches_log.flush()
    # Count waiting intervals [arrival, dispatch); immediate dispatch is not backlog.
    waiting = [
        (row["arrival_s"], dispatch)
        for row, dispatch in zip(rows, dispatches)
        if dispatch > row["arrival_s"]
    ]
    departed = peak = 0
    for i, (arrival, _) in enumerate(waiting):
        while departed < i and waiting[departed][1] <= arrival:
            departed += 1
        peak = max(peak, i + 1 - departed)
    horizon = ready - rows[0]["arrival_s"]
    report.update(
        queue_ms=distribution(queue),
        service_ms=distribution(service),
        e2e_ms=distribution(e2e),
        mean_batch_size=head / batches,
        queue_peak_requests=peak,
        queue_area_request_ms=math.fsum(queue),
        arrival_to_drain_s=horizon,
        throughput_requests_per_s=head / horizon if horizon > 0 else None,
        final_pending_requests=0,
        all_tails_retained=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--cache-ratio", type=float, required=True)
    parser.add_argument("--cache-budget-gib", type=float, default=16)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not 0 < args.cache_ratio < 1
        or not math.isfinite(args.cache_budget_gib)
        or args.cache_budget_gib <= 0
    ):
        parser.error("Require 0 < cache-ratio < 1 and a positive finite cache budget")
    args.cache_bytes = int(args.cache_budget_gib * 1024**3)
    if min(args.cache_bytes, args.max_batch_size, args.max_batch_tokens) <= 0:
        parser.error("Budget and batch limits must be positive")
    manifest, sequences, rows = load_trace(args.trace_dir, args.max_batch_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        expected_requests=manifest["request_count"],
        max_batch_delay_ms=MAX_BATCH_DELAY_S * 1000,
        arrival_policy="original trace timestamps; no rescaling or dropping",
        timing="service includes input construction, forward and output sync; validation/logging affect later queues",
        completed_requests=0,
        completed_batches=0,
        all_tails_retained=False,
    )
    backend = HSTU()
    try:
        report["phase"] = "setup"
        started = time.perf_counter()
        report["backend"] = backend.setup(args, manifest, sequences, rows)
        report["setup_wall_s"] = time.perf_counter() - started
        replay(rows, backend, args, report)
        report["phase"] = "cleanup"
        backend.close()
        report.update(status="complete", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            backend.close()
        except BaseException as cleanup_error:
            report["cleanup_error"] = str(cleanup_error)
        raise
    finally:
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    metrics = {
        "cache_ratio": args.cache_ratio,
        "cache_budget_gib": args.cache_budget_gib,
        **{
            k: report[k]
            for k in (
                "completed_requests",
                "completed_batches",
                "mean_batch_size",
                "throughput_requests_per_s",
                "queue_peak_requests",
            )
        },
        **{
            f"{name}_{key}": value
            for name in ("queue_ms", "service_ms", "e2e_ms")
            for key, value in report[name].items()
        },
    }
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
