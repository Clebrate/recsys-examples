# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from costs import Costs
from data import Pool, stream
from metadata import KVLRU, EmbeddingLRU, Geometry
from metrics import paired_savings


class BenchmarkTests(unittest.TestCase):
    def test_page_ownership_with_invalidation_and_resize(self):
        cache = EmbeddingLRU(100, 40)
        rng, values = random.Random(8), {}
        for _ in range(1000):
            if rng.random() < 0.15:
                cache.resize(rng.choice([20, 40, 60, 80]))
            elif rng.random() < 0.15:
                cache.drop([rng.randrange(200) for _ in range(10)])
            else:
                keys = [rng.randrange(200) for _ in range(10)]
                slots, missing, dest = cache.touch(keys)
                for key, slot in zip(missing, dest):
                    values[slot] = key
                self.assertEqual([values[s] for s in slots], keys)
            used = set(cache.keys.values())
            free = cache.recycled + [
                s for lo, hi in cache.free_intervals for s in range(lo, hi)
            ]
            self.assertEqual(len(set(free)), len(free))
            self.assertFalse(used & set(free))
            self.assertEqual(used | set(free), set(range(cache.base_slot, 100)))
            for key, slot in cache.keys.items():
                self.assertEqual(cache.reverse[slot], key)
                self.assertEqual(values[slot], key)

    def test_kv_invalidation_releases_each_page_once(self):
        cache = KVLRU(120, 32)
        for uid in range(15):
            cache.allocate(uid, 256)
        cache.drop(7)
        cache.drop(7)
        for size in [80, 100, 40, 120]:
            cache.resize(size)
            used = [p for e in cache.entries.values() for p in e["pages"]]
            self.assertEqual(len(set(used + cache.free)), size)
            self.assertEqual(len(used + cache.free), size)
            self.assertEqual(set(used + cache.free), set(range(size)))
            for uid in range(20, 30):
                cache.allocate(uid, 256)

    def test_mget_is_one_whole_call(self):
        cost = Costs(embedding_model="mget", mget_fixed_ms=7.5, mget_us_per_id=0.87)
        self.assertEqual(cost.embedding_ms(0), 0)
        self.assertAlmostEqual(cost.embedding_ms(100), 7.587)
        self.assertAlmostEqual(
            Costs(embedding_us_per_id=0.410387).embedding_ms(100), 0.0410387
        )
        with self.assertRaises(ValueError):
            Costs(kv_miss_ms=float("nan"))

    def test_counterfactual_savings_and_interaction(self):
        rows = []
        for sample in range(5):
            for repeat in range(3):
                for condition, cost in [("HH", 5), ("MH", 6), ("HM", 7), ("MM", 9)]:
                    rows.append(
                        {
                            "sample_index": sample,
                            "repeat": repeat,
                            "condition": condition,
                            "e2e_ms": cost + sample * 0.3 + repeat * 0.1,
                        }
                    )
        result = paired_savings(rows, bootstrap=50)
        self.assertAlmostEqual(result["kv_saving_given_embedding_hit"]["mean_ms"], 1)
        self.assertAlmostEqual(result["kv_saving_given_embedding_miss"]["mean_ms"], 2)
        self.assertAlmostEqual(result["embedding_saving_given_kv_hit"]["mean_ms"], 2)
        self.assertAlmostEqual(result["embedding_saving_given_kv_miss"]["mean_ms"], 3)
        self.assertAlmostEqual(result["interaction"]["mean_ms"], 1)
        self.assertEqual(result["interaction"]["distinct_requests"], 5)
        with self.assertRaises(ValueError):
            paired_savings(rows[:-1])

    def test_csv_preserves_request_definition_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["user_id", "video_id", "action_weights"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "user_id": 7,
                        "video_id": json.dumps(list(range(150))),
                        "action_weights": json.dumps([2] * 150),
                    }
                )
            pool = Pool.csv(path, history_cap=40, candidates=10)
            self.assertEqual(pool.records[0].history, 40)
            self.assertEqual(pool.records[0].videos.tolist(), list(range(50)))
            self.assertEqual(pool.metadata["candidates"], 10)
            self.assertEqual(len(pool.metadata["source_sha256"]), 64)

    def test_stationary_and_business_streams_are_reproducible(self):
        pool = Pool.synthetic(80, history_cap=512)
        for profile in ["steady", "step", "gradual"]:
            a = stream(pool, profile, 100, 900, 1)
            self.assertEqual(a, stream(pool, profile, 100, 900, 1))
            self.assertEqual(tuple(map(len, a)), (100, 900))
        _, timed = stream(pool, "step", 100, 900, 1)
        histories = [
            sum(pool.records[i].history for i in timed[s : s + 300]) / 300
            for s in [0, 300, 600]
        ]
        self.assertLess(histories[0], histories[1])
        self.assertLess(histories[2], histories[1])

    def test_budget_and_ratio_validation(self):
        g = Geometry(2**30)
        for f in [0.15, 0.35, 0.55, 0.75, 0.85]:
            _, slots, boundary = g.partition(f)
            self.assertEqual(
                (boundary + slots * g.row_bytes) * g.layers, g.actual_bytes
            )
            self.assertLessEqual(g.actual_bytes, g.budget_bytes)
        with self.assertRaises(ValueError):
            g.partition(1)


if __name__ == "__main__":
    unittest.main()
