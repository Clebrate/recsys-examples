# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executed synthetic miss delays; MGET fixed cost applies once per request."""
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    kv_miss_ms: float = 0.0
    embedding_model: str = "linear"
    embedding_us_per_id: float = 0.0
    mget_fixed_ms: float = 0.0
    mget_us_per_id: float = 0.0

    def __post_init__(self):
        if self.embedding_model not in ("linear", "mget"):
            raise ValueError("Embedding cost model must be linear or mget")
        if any(
            not math.isfinite(v) or v < 0
            for v in [
                self.kv_miss_ms,
                self.embedding_us_per_id,
                self.mget_fixed_ms,
                self.mget_us_per_id,
            ]
        ):
            raise ValueError("Miss costs must be finite and nonnegative")

    def embedding_ms(self, missing_ids):
        if not missing_ids:
            return 0.0
        if self.embedding_model == "mget":
            return self.mget_fixed_ms + missing_ids * self.mget_us_per_id / 1000
        return missing_ids * self.embedding_us_per_id / 1000


def wait_ms(duration):
    if duration <= 0:
        return 0.0
    started = time.perf_counter()
    until = started + duration / 1000
    while time.perf_counter() < until:
        pass
    return (time.perf_counter() - started) * 1000
