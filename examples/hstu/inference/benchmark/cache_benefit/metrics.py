# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-weighted hit rates and same-request conditional latency savings."""
from collections import defaultdict

import numpy as np

STATES = ("HH", "MH", "HM", "MM")  # First character KV; second embedding.


def aggregate(rows):
    latency = np.asarray([r["e2e_ms"] for r in rows])
    if not len(latency):
        raise ValueError("No measurements")
    lookups = sum(r["embedding_lookup_ids"] for r in rows)
    history = sum(r["history_tokens"] for r in rows)
    return {
        "requests": len(rows),
        "e2e_mean_ms": float(latency.mean()),
        "e2e_p50_ms": float(np.quantile(latency, 0.5)),
        "e2e_p95_ms": float(np.quantile(latency, 0.95)),
        "e2e_p99_ms": float(np.quantile(latency, 0.99)),
        "e2e_max_ms": float(latency.max()),
        "kv_request_hit_rate": sum(r["kv_full_hit"] for r in rows) / len(rows),
        "kv_token_hit_rate": sum(r["kv_cached_tokens"] for r in rows) / history
        if history
        else None,
        "embedding_key_hit_rate": 1
        - sum(r["embedding_miss_ids"] for r in rows) / lookups
        if lookups
        else None,
        "embedding_request_hit_rate": sum(r["embedding_miss_ids"] == 0 for r in rows)
        / len(rows),
        "embedding_miss_ids_per_request": sum(r["embedding_miss_ids"] for r in rows)
        / len(rows),
        "embedding_backend_calls": sum(r["embedding_backend_calls"] for r in rows),
        "resize_total_ms": sum(r["resize_ms"] for r in rows),
    }


def paired_savings(rows, seed=1, bootstrap=2000):
    blocks = defaultdict(dict)
    for row in rows:
        key = (row["sample_index"], row["repeat"])
        if row["condition"] in blocks[key]:
            raise ValueError("Duplicate paired condition")
        blocks[key][row["condition"]] = row["e2e_ms"]
    per_request = defaultdict(list)
    for (sample, _), block in blocks.items():
        if set(block) != set(STATES):
            raise ValueError("Incomplete four-state paired measurement")
        hh, mh, hm, mm = [block[s] for s in STATES]
        per_request[sample].append(
            [mh - hh, mm - hm, hm - hh, mm - mh, mm - hm - mh + hh]
        )
    values = np.array([np.mean(repeats, axis=0) for repeats in per_request.values()])
    rng = np.random.default_rng(seed)
    boot = None
    if len(values) >= 2 and bootstrap > 0:
        # Repeat measurements of one request stay together. This interval
        # describes the sampled requests, not cross-run/hardware variation.
        boot = np.array(
            [
                values[rng.integers(0, len(values), len(values))].mean(axis=0)
                for _ in range(bootstrap)
            ]
        )
    names = [
        "kv_saving_given_embedding_hit",
        "kv_saving_given_embedding_miss",
        "embedding_saving_given_kv_hit",
        "embedding_saving_given_kv_miss",
        "interaction",
    ]
    return {
        name: {
            "mean_ms": float(values[:, i].mean()),
            "median_ms": float(np.median(values[:, i])),
            "distinct_requests": len(values),
            "request_bootstrap_95ci_ms": [
                float(v) for v in np.quantile(boot[:, i], [0.025, 0.975])
            ]
            if boot is not None
            else None,
        }
        for i, name in enumerate(names)
    }
