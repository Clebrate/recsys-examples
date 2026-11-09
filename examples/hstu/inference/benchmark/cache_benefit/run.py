# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure conditional cache benefits and continuous full-request cache behavior."""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HSTU = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HSTU), str(HSTU.parent), str(HERE.parent)]

from costs import Costs
from data import Pool, stream
from metrics import STATES, aggregate, paired_savings


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["paired", "replay", "all-hit"])
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--data", type=Path, help="Processed user_id/video_id/action_weights CSV"
    )
    source.add_argument("--pool", type=Path, help="Existing DYNPOOL1 packed directory")
    p.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--synthetic-users", type=int, default=64)
    p.add_argument("--synthetic-items", type=int, default=8192)
    p.add_argument("--history-cap", type=int, default=4096)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--limit-users", type=int, default=0)
    p.add_argument(
        "--gin",
        type=Path,
        default=HSTU / "inference/configs/kuairand_1k_inference_ranking.gin",
    )
    p.add_argument("--budget-gib", type=float, default=1.0)
    p.add_argument("--fraction", type=float, default=0.55)
    p.add_argument(
        "--fractions",
        help="Replay only: comma-separated ratios, fresh process per ratio",
    )
    p.add_argument(
        "--resize-at",
        default="",
        help="Replay only: request_index:fraction,...; a scripted experiment, not an online policy",
    )
    p.add_argument("--kv-history-items", type=int, default=128)
    p.add_argument("--parameters", choices=["default", "pattern"], default="default")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model-seed", type=int, default=1)
    p.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Distinct requests for paired measurement",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--preflight-requests", type=int, default=8)
    p.add_argument(
        "--warmup", type=int, default=2000, help="Continuous replay warmup requests"
    )
    p.add_argument("--requests", type=int, default=6000)
    p.add_argument("--window", type=int, default=1000)
    p.add_argument(
        "--traffic",
        choices=["original", "steady", "step", "gradual"],
        default="original",
    )
    p.add_argument("--zipf-alpha", type=float, default=0.5)
    p.add_argument("--kv-miss-ms", type=float, default=0.0)
    p.add_argument(
        "--embedding-cost-model", choices=["linear", "mget"], default="linear"
    )
    p.add_argument("--embedding-us-per-id", type=float, default=0.0)
    p.add_argument("--mget-fixed-ms", type=float, default=0.0)
    p.add_argument("--mget-us-per-id", type=float, default=0.0)
    return p


def schedule(value):
    result = {}
    for part in filter(None, value.split(",")):
        index, fraction = part.split(":")
        index, fraction = int(index), float(fraction)
        if index in result or index < 0 or not 0 < fraction < 1:
            raise ValueError(
                "Resize indices must be unique/nonnegative, with 0 < fraction < 1"
            )
        result[index] = fraction
    return result


def report(output, summary):
    lines = [
        "# HSTU cache benefit benchmark",
        "",
        "Full-request E2E includes CPU request preparation, H2D, actual cache handling, executed synthetic waits, HSTU and completion synchronization. One request at a time; no service queue or real Redis measurement.",
        "",
    ]
    if "conditions" in summary:
        lines += [
            "## Same-request four-state measurements",
            "",
            "First character is KV state, second is embedding state. Embedding M means all unique keys initially miss; repeated occurrences within that request can hit after their first fill.",
            "",
            "| State | Mean ms | p50 ms | p95 ms | KV request hit | Embedding key hit | Embedding request hit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for state, values in summary["conditions"].items():
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.1%} | {:.1%} | {:.1%} |".format(
                    state,
                    values["e2e_mean_ms"],
                    values["e2e_p50_ms"],
                    values["e2e_p95_ms"],
                    values["kv_request_hit_rate"],
                    values["embedding_key_hit_rate"],
                    values["embedding_request_hit_rate"],
                )
            )
        lines += [
            "",
            "## Hit savings: miss E2E minus hit E2E",
            "",
            "| Conditional saving | Mean ms/request | Request-cluster bootstrap 95% interval, ms |",
            "| --- | ---: | --- |",
        ]
        for name, value in summary["savings"].items():
            interval = value["request_bootstrap_95ci_ms"]
            text = "{:.4f} to {:.4f}".format(*interval) if interval else "Not estimated"
            lines.append("| {} | {:.4f} | {} |".format(name, value["mean_ms"], text))
        lines += [
            "",
            "Interaction = L(MM) − L(HM) − L(MH) + L(HH). Negative measured savings are retained. Intervals describe sampled requests with repeats clustered together; use independent runs to assess run/hardware variation.",
            "",
            "Cache warmup/invalidation fixtures are outside each timed counterfactual. This isolates conditional benefits and does not simulate natural online eviction/recovery.",
        ]
    else:
        lines += [
            "## Continuous replay",
            "",
            "```json",
            json.dumps(summary, indent=2),
            "```",
            "",
            "Cache state persists through workload phases. Scripted resize cost, induced misses and refill are included in E2E. Natural hit/miss latency groups are observational and are not causal per-hit benefit estimates.",
        ]
    lines += [
        "",
        "Configuration, source/model hashes and cost semantics are in `provenance.json`. No trained checkpoint is loaded. This experimental cache container reuses the reference HSTU kernels; it is not an NVE/FlexKV public resize API.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def ratio_sweep(args):
    if args.mode != "replay" or args.resize_at:
        raise ValueError("--fractions requires replay without a resize schedule")
    fractions = [float(v) for v in args.fractions.split(",")]
    if len(set(fractions)) != len(fractions) or any(not 0 < f < 1 for f in fractions):
        raise ValueError("Need distinct ratios strictly between zero and one")
    args.output.mkdir(parents=True, exist_ok=False)
    base = [args.mode]
    for key, value in vars(args).items():
        if key not in ("mode", "fractions", "fraction", "output") and value is not None:
            base.extend(["--" + key.replace("_", "-"), str(value)])
    results = []
    signature = None
    for i, fraction in enumerate(fractions):
        directory = args.output / ("ratio-{:02d}".format(i))
        with (args.output / ("ratio-{:02d}.log".format(i))).open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-s",
                    str(HERE / "run.py"),
                    *base,
                    "--fraction",
                    str(fraction),
                    "--output",
                    str(directory),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = json.loads((directory / "summary.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        current = [
            provenance["source"],
            provenance["runtime"]["model_state_sha256"],
            provenance["costs"],
            provenance["warmup_index_sha256"],
            provenance["timed_index_sha256"],
        ]
        if signature is not None and current != signature:
            raise RuntimeError(
                "Ratios used different requests, model weights or miss costs"
            )
        signature = current
        results.append({"requested_kv_fraction": fraction, **result})
    write_csv(args.output / "ratios.csv", results)
    write_json(
        args.output / "summary.json", {"mode": "fixed_ratio_sweep", "ratios": results}
    )
    ordered = sorted(results, key=lambda r: r["requested_kv_fraction"])
    deltas = [
        {
            "previous_kv_fraction": before["kv_fraction_start"],
            "next_kv_fraction": after["kv_fraction_start"],
            "kv_hit_change_pp": 100
            * (after["kv_request_hit_rate"] - before["kv_request_hit_rate"]),
            "embedding_key_hit_change_pp": 100
            * (after["embedding_key_hit_rate"] - before["embedding_key_hit_rate"]),
            "embedding_request_hit_change_pp": 100
            * (
                after["embedding_request_hit_rate"]
                - before["embedding_request_hit_rate"]
            ),
            "net_e2e_saving_ms_per_request": before["e2e_mean_ms"]
            - after["e2e_mean_ms"],
        }
        for before, after in zip(ordered, ordered[1:])
    ]
    if deltas:
        write_csv(args.output / "ratio_deltas.csv", deltas)
    lines = [
        "# Cache ratio replay",
        "",
        "| KV ratio | E2E mean ms | KV request hit | Embedding key hit | Embedding request hit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            "| {:.2f} | {:.4f} | {:.2%} | {:.2%} | {:.2%} |".format(
                r["requested_kv_fraction"],
                r["e2e_mean_ms"],
                r["kv_request_hit_rate"],
                r["embedding_key_hit_rate"],
                r["embedding_request_hit_rate"],
            )
        )
    lines += [
        "",
        "Each ratio uses a fresh process, identical source/traffic seed and its own continuous warmup/cache history. This is an offline reference, not an online ratio search.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parser().parse_args()
    for name in ["data", "pool", "output", "gin"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (
        min(
            args.samples,
            args.repeats,
            args.preflight_requests,
            args.requests,
            args.window,
            args.kv_history_items,
        )
        < 1
    ):
        raise ValueError(
            "Sample, repeat, preflight, request, window and KV history counts must be positive"
        )
    if args.warmup < 0 or args.limit_users < 0 or args.budget_gib <= 0:
        raise ValueError("Invalid warmup, user limit or budget")
    if args.fractions:
        ratio_sweep(args)
        return
    changes = schedule(args.resize_at)
    if changes and (args.mode != "replay" or max(changes) >= args.requests):
        raise ValueError("Resize schedule must address timed replay requests")
    args.output.mkdir(parents=True, exist_ok=False)
    pool = (
        Pool.csv(args.data, args.history_cap, args.candidates, limit=args.limit_users)
        if args.data
        else (
            Pool.packed(args.pool)
            if args.pool
            else Pool.synthetic(
                args.synthetic_users,
                args.synthetic_items,
                args.history_cap,
                args.candidates,
                args.seed,
            )
        )
    )
    os.chdir(HSTU)
    from runtime import Runtime

    costs = Costs(
        args.kv_miss_ms,
        args.embedding_cost_model,
        args.embedding_us_per_id,
        args.mget_fixed_ms,
        args.mget_us_per_id,
    )
    runtime = Runtime(
        pool,
        int(args.budget_gib * 2**30),
        args.fraction,
        costs,
        args.gin,
        args.model_seed,
        args.kv_history_items,
        args.parameters,
    )
    for fraction in changes.values():
        runtime.arena.geometry.partition(fraction)
    settings = {
        k: (v.name if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k != "output"
    }
    provenance = {
        "schema_version": 1,
        "python_version": sys.version.split()[0],
        "settings": settings,
        "source": pool.metadata,
        "runtime": runtime.provenance,
        "costs": asdict(costs),
        "cost_semantics": "Executed synthetic delay; MGET fixed cost once per missing-key set/request. No Redis connection.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
    }
    write_json(args.output / "provenance.json", provenance)
    rows = []
    if args.mode == "paired":
        rng = random.Random(args.seed)
        selected = rng.sample(
            range(len(pool.records)), min(args.samples, len(pool.records))
        )
        provenance["selection_sha256"] = hashlib.sha256(
            json.dumps(selected).encode()
        ).hexdigest()
        write_json(args.output / "provenance.json", provenance)
        for index in selected[: args.preflight_requests]:
            for condition in STATES:
                runtime.prepare_conditions(index, condition)
                runtime.forward(index)
        runtime.arena.waits_enabled = True
        for sample, index in enumerate(selected):
            golden = None
            for repeat in range(args.repeats):
                conditions = list(STATES)
                rng.shuffle(conditions)
                for order, condition in enumerate(conditions):
                    runtime.prepare_conditions(index, condition)
                    logits, row = runtime.forward(index)
                    expected_misses = (
                        len(set(pool.records[index].videos.tolist()))
                        if condition[1] == "M"
                        else 0
                    )
                    if (
                        row["kv_full_hit"] != int(condition[0] == "H")
                        or row["embedding_miss_ids"] != expected_misses
                    ):
                        raise RuntimeError(
                            "Forced state was not achieved; measurement rejected"
                        )
                    result = logits.detach().cpu()
                    if not runtime.torch.isfinite(result).all():
                        raise RuntimeError("Non-finite prediction")
                    if golden is None:
                        golden = result
                    else:
                        runtime.torch.testing.assert_close(
                            result, golden, rtol=0.03, atol=0.01
                        )
                    rows.append(
                        {
                            "sample_index": sample,
                            "repeat": repeat,
                            "order": order,
                            "condition": condition,
                            **row,
                        }
                    )
            if (sample + 1) % 8 == 0:
                print(
                    json.dumps({"paired_requests": sample + 1, "total": len(selected)}),
                    flush=True,
                )
        summary = {
            "mode": "paired",
            "conditions": {
                s: aggregate([r for r in rows if r["condition"] == s]) for s in STATES
            },
            "savings": paired_savings(rows, args.seed),
        }
    else:
        warm, timed = stream(
            pool,
            args.traffic,
            args.warmup,
            args.requests,
            args.seed,
            args.zipf_alpha,
            runtime.reference.zipf_stream,
        )
        if args.mode == "all-hit":
            write_json(
                args.output / "all_hit_capacity.json", runtime.all_hit_capacity()
            )
            warm = list(range(len(pool.records)))
        provenance["warmup_index_sha256"] = hashlib.sha256(
            json.dumps(warm).encode()
        ).hexdigest()
        provenance["timed_index_sha256"] = hashlib.sha256(
            json.dumps(timed).encode()
        ).hexdigest()
        write_json(args.output / "provenance.json", provenance)
        for index in warm:
            runtime.forward(index)
        if args.mode == "all-hit":
            # A once-per-user prefill exercises the miss kernels. Warm the
            # actual all-hit path too, so its first compilation/allocation is
            # not misreported as steady all-hit request latency.
            hit_preflight = timed[: args.preflight_requests]
            for index in hit_preflight:
                _, check = runtime.forward(index)
                if not check["kv_full_hit"] or check["embedding_miss_ids"]:
                    raise RuntimeError("All-hit preflight did not actually hit")
            provenance["all_hit_preflight_requests"] = len(hit_preflight)
            write_json(args.output / "provenance.json", provenance)
        runtime.arena.waits_enabled = True
        calls_before = runtime.arena.backend_calls
        for n, index in enumerate(timed):
            _, row = runtime.forward(index, changes.get(n))
            if args.mode == "all-hit" and (
                not row["kv_full_hit"] or row["embedding_miss_ids"]
            ):
                raise RuntimeError("Actual all-hit assertion failed")
            rows.append({"request_index": n, **row})
            if (n + 1) % args.window == 0:
                print(
                    json.dumps(
                        {
                            "requests": n + 1,
                            "total": len(timed),
                            "last_window": aggregate(rows[-args.window :]),
                        }
                    ),
                    flush=True,
                )
        summary = {
            "mode": args.mode,
            **aggregate(rows),
            "kv_fraction_start": rows[0]["actual_kv_fraction"],
            "kv_fraction_end": rows[-1]["actual_kv_fraction"],
            "actual_backend_calls": runtime.arena.backend_calls - calls_before,
        }
        windows = [
            {
                "window": start // args.window,
                "start_request": start,
                **aggregate(rows[start : start + args.window]),
            }
            for start in range(0, len(rows), args.window)
        ]
        write_csv(args.output / "windows.csv", windows)
    write_csv(args.output / "requests.csv", rows)
    summary["cache_payload_bytes"] = runtime.arena.geometry.actual_bytes
    summary[
        "torch_gpu_peak_allocated_bytes"
    ] = runtime.torch.cuda.max_memory_allocated()
    write_json(args.output / "resizes.json", runtime.arena.resize_records)
    write_json(args.output / "summary.json", summary)
    report(args.output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
