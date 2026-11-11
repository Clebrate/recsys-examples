# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small online baseline: fit the recently reused users, with two-vote hysteresis.

This observes completed requests only. It does not estimate marginal hit value
or claim an optimum; long reuse intervals and skew can defeat the window rule.
"""

from collections import Counter, deque
import math
import time


class RecentReusePolicy:
    def __init__(self, pool):
        self.pool = pool
        self.history = deque(maxlen=100)
        self.seen = 0
        self.next_decision = 100
        self.candidate = None
        self.votes = 0
        self.decisions = []

    def observe(self, rows, num_context):
        # Warmup is observed too, but ratio changes begin in the timed replay.
        for row in rows:
            pages = math.ceil((2 * row["history_len"] + num_context) / self.pool.device_cache.page_size)
            self.history.append((row["cache_uid"], pages))
            self.seen += 1

    def decide(self, request_id):
        if self.seen < self.next_decision:
            return None
        started = time.perf_counter()
        self.next_decision = (self.seen // 50 + 1) * 50
        counts = Counter(uid for uid, _ in self.history)
        latest = dict(self.history)
        reused = {uid for uid, count in counts.items() if count >= 2}
        required = math.ceil(sum(latest[uid] for uid in reused) * 1.05)
        current = self.pool.ratios[self.pool.level]
        proposed = next((ratio for ratio, pages in zip(self.pool.ratios, self.pool.bounds)
                         if pages >= required), current)
        # An oversized window does not prove thrashing: keep the current ratio.
        if proposed == self.candidate:
            self.votes += 1
        else:
            self.candidate, self.votes = proposed, 1
        target = proposed if self.votes >= 2 and proposed != current else None
        record = {
            "request_id": request_id, "observed_requests": self.seen,
            "window_requests": len(self.history), "reused_users": len(reused),
            "estimated_reused_kv_pages_with_margin": required,
            "current_ratio": current, "proposed_ratio": proposed,
            "votes": self.votes, "target_ratio": target,
            "decision_wall_ms": (time.perf_counter() - started) * 1000,
        }
        self.decisions.append(record)
        return record
