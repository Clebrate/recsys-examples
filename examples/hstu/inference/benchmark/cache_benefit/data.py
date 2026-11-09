# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable real/synthetic requests; source parsing stays outside request timing."""
import bisect
import csv
import hashlib
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Record:
    uid: int
    history: int
    videos: np.ndarray
    actions: np.ndarray


class Pool:
    def __init__(self, records, metadata):
        if not records or len({r.uid for r in records}) != len(records):
            raise ValueError(
                "Need a nonempty pool with one immutable request per unique user key"
            )
        self.records, self.metadata = records, metadata
        self._unique = None

    @classmethod
    def csv(cls, path, history_cap=4096, candidates=100, min_history=32, limit=0):
        if min(history_cap, candidates, min_history) < 1:
            raise ValueError("History/candidate limits must be positive")
        path = Path(path)
        csv.field_size_limit(128 * 2**20)
        records, vmax, amax = [], 0, 0
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                videos, actions = json.loads(row["video_id"]), json.loads(
                    row["action_weights"]
                )
                if min(len(videos), len(actions)) < min_history + candidates:
                    continue
                history = min(len(videos) - candidates, history_cap)
                count = history + candidates
                if len(actions) < count:
                    raise ValueError(
                        "Action history is shorter than the corresponding item history"
                    )
                videos, actions = videos[:count], actions[:count]
                if min(videos + actions) < 0 or max(videos + actions) >= 2**32:
                    raise ValueError("Item/action IDs must fit nonnegative uint32")
                vmax, amax = max(vmax, max(videos)), max(amax, max(actions))
                records.append(
                    Record(
                        int(row["user_id"]),
                        history,
                        np.asarray(videos, dtype=np.uint32),
                        np.asarray(actions, dtype=np.uint32),
                    )
                )
                if limit and len(records) >= limit:
                    break
        return cls(
            records,
            {
                "source": "csv",
                "source_name": path.name,
                "source_sha256": file_sha(path),
                "users": len(records),
                "history_cap": history_cap,
                "candidates": candidates,
                "video_vocab": vmax + 1,
                "action_vocab": max(233, amax + 1),
            },
        )

    @classmethod
    def packed(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        data = np.memmap(directory / "pool.bin", dtype=np.uint8, mode="r")
        if bytes(data[:8]) != b"DYNPOOL1":
            raise ValueError("Unsupported packed pool schema")
        (count,) = struct.unpack_from("<I", data, 8)
        offset, records = 12, []
        for _ in range(count):
            uid, history, size = struct.unpack_from("<qII", data, offset)
            offset += 16
            if history + meta["candidates"] != size:
                raise ValueError("Packed history/candidate sizes disagree")
            records.append(
                Record(
                    uid,
                    history,
                    np.frombuffer(data, dtype="<u4", count=size, offset=offset),
                    np.frombuffer(
                        data, dtype="<u4", count=size, offset=offset + size * 4
                    ),
                )
            )
            offset += size * 8
        if offset != data.size or count != meta["users"]:
            raise ValueError("Truncated or inconsistent packed pool")
        result = cls(
            records,
            {
                "source": "DYNPOOL1",
                "source_name": directory.name,
                "source_sha256": file_sha(directory / "pool.bin"),
                "original_csv_sha256": meta.get("source_csv_sha256"),
                **{
                    k: meta[k]
                    for k in [
                        "users",
                        "history_cap",
                        "candidates",
                        "video_vocab",
                        "action_vocab",
                    ]
                },
            },
        )
        result._mmap = data
        return result

    @classmethod
    def synthetic(cls, users=64, items=8192, history_cap=256, candidates=100, seed=1):
        if users < 1 or items < 1 or history_cap < 32 or candidates < 1:
            raise ValueError("Invalid synthetic dimensions")
        rng = np.random.default_rng(seed)
        records = []
        for uid in range(users):
            history = int(rng.integers(32, history_cap + 1))
            records.append(
                Record(
                    uid,
                    history,
                    rng.integers(0, items, history + candidates, dtype=np.uint32),
                    rng.integers(0, 233, history + candidates, dtype=np.uint32),
                )
            )
        digest = hashlib.sha256()
        for r in records:
            digest.update(struct.pack("<qI", r.uid, r.history))
            digest.update(r.videos.tobytes())
            digest.update(r.actions.tobytes())
        return cls(
            records,
            {
                "source": "synthetic",
                "source_sha256": digest.hexdigest(),
                "seed": seed,
                "users": users,
                "history_cap": history_cap,
                "candidates": candidates,
                "video_vocab": items,
                "action_vocab": 233,
            },
        )

    def unique_keys(self):
        if self._unique is None:
            present = np.zeros(self.metadata["video_vocab"], dtype=bool)
            for record in self.records:
                present[record.videos] = True
            self._unique = np.flatnonzero(present)
        return self._unique


def stream(pool, profile, warmup, requests, seed, alpha=0.5, reference_zipf=None):
    if warmup < 0 or requests < 1:
        raise ValueError("Need nonnegative warmup and positive timed request count")
    count = len(pool.records)
    if profile == "original":
        if reference_zipf is None:
            raise ValueError("Original stream needs the reference zipf_stream function")
        return (
            reference_zipf(count, warmup, alpha, random.Random(seed)),
            reference_zipf(count, requests, alpha, random.Random(seed + 1)),
        )
    order = list(range(count))
    random.Random(seed).shuffle(order)
    if profile not in ("steady", "step", "gradual"):
        raise ValueError("Unknown traffic profile")
    if profile != "steady" and count < 2:
        raise ValueError("Two business groups need at least two users")
    ordered = sorted(order, key=lambda i: pool.records[i].history)
    split = max(1, min(count - 1, int(count * 0.4)))
    groups = [order] if profile == "steady" else [ordered[:split], ordered[split:]]
    specs = []
    for ordinal, group in enumerate(groups):
        group = list(group)
        random.Random(seed + 100 + ordinal).shuffle(group)
        weights = np.arange(1, len(group) + 1, dtype=np.float64) ** -alpha
        cdf = np.cumsum(weights / weights.sum()).tolist()
        cdf[-1] = 1.0
        specs.append((group, cdf))

    def draw(spec, rng):
        group, cdf = spec
        return group[bisect.bisect_left(cdf, rng.random())]

    warm_rng, rng = random.Random(seed + 10), random.Random(seed + 20)
    warm = [draw(specs[0], warm_rng) for _ in range(warmup)]
    timed = []
    for n in range(requests):
        if profile == "steady":
            spec = specs[0]
        else:
            phase = min(2, n * 3 // requests)
            probability_b = (
                float(phase == 1)
                if profile == "step"
                else max(0.0, min(1.0, 3 * n / requests - 1))
            )
            spec = specs[int(rng.random() < probability_b)]
        timed.append(draw(spec, rng))
    return warm, timed
