# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonzero data retention and real HSTU four-state correctness, on one GPU."""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HSTU = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HSTU), str(HSTU.parent), str(HERE.parent)]
import os

os.environ["HSTU_INFERENCE_ONLY"] = "1"
os.chdir(HSTU)

import torch
from arena import Arena, Parameters
from costs import Costs
from data import Pool
from metadata import Geometry
from runtime import Runtime


def main():
    torch.set_num_threads(1)
    parameters = Parameters(512, "pattern")
    arena = Arena(
        Geometry(256 * 2**20),
        0.5,
        parameters,
        Costs(embedding_model="mget", mget_fixed_ms=0.01),
    )
    address = arena.buffer.data_ptr()
    ids = list(range(50000))
    random.Random(3).shuffle(ids)
    for start in range(0, len(ids), 512):
        group = ids[start : start + 512]
        arena.current_ids = group
        result = arena.embedding_lookup(torch.tensor(group, device="cuda"))
        torch.testing.assert_close(
            result.cpu(), parameters.fetch(group), rtol=0, atol=0
        )
    group = ids[:300] + ids[:100]
    assert arena.drop_embedding(group) == 300
    arena.current_ids = group
    arena.waits_enabled = True
    before = arena.backend_calls
    actual = arena.embedding_lookup(torch.tensor(group, device="cuda"))
    torch.testing.assert_close(actual.cpu(), parameters.fetch(group), rtol=0, atol=0)
    assert arena.backend_calls - before == 1
    assert arena.last_embedding["embedding_miss_ids"] == 300
    assert arena.last_embedding["embedding_wait_target_ms"] == 0.01
    arena.embedding_lookup(torch.tensor(group, device="cuda"))
    assert arena.last_embedding["embedding_backend_calls"] == 0
    assert arena.last_embedding["embedding_wait_target_ms"] == 0
    arena.waits_enabled = False
    for uid in range(80):
        entry = arena.kv.allocate(uid, 256)
        for layer, table in enumerate(arena.kv_tables):
            for n, page in enumerate(entry["pages"]):
                table[page].fill_(uid + layer / 4 + n / 32)
    for fraction in [0.7, 0.3, 0.6, 0.5]:
        arena.resize(fraction)
        assert arena.buffer.data_ptr() == address
        for uid, entry in arena.kv.entries.items():
            for layer, table in enumerate(arena.kv_tables):
                for n, page in enumerate(entry["pages"]):
                    torch.testing.assert_close(
                        table[page],
                        torch.full_like(table[page], uid + layer / 4 + n / 32),
                        rtol=0,
                        atol=0,
                    )
        for table, cache in zip(arena.embedding_tables, arena.emb):
            pairs = list(cache.keys.items())
            for start in range(0, len(pairs), 512):
                part = pairs[start : start + 512]
                actual = table.index_select(
                    0, torch.tensor([slot for _, slot in part], device="cuda")
                ).cpu()
                torch.testing.assert_close(
                    actual, parameters.fetch([key for key, _ in part]), rtol=0, atol=0
                )
    del arena
    pool = Pool.synthetic(8, 8192, history_cap=160, seed=8)
    runtime = Runtime(
        pool,
        256 * 2**20,
        0.5,
        Costs(0.02, "mget", mget_fixed_ms=0.01),
        HSTU / "inference/configs/kuairand_1k_inference_ranking.gin",
        parameter_mode="pattern",
    )
    runtime.arena.waits_enabled = True
    for index in range(len(pool.records)):
        expected = None
        for condition in ["HH", "MH", "HM", "MM"]:
            runtime.prepare_conditions(index, condition)
            logits, row = runtime.forward(index)
            assert row["kv_full_hit"] == int(condition[0] == "H")
            misses = (
                len(set(pool.records[index].videos.tolist()))
                if condition[1] == "M"
                else 0
            )
            assert row["embedding_miss_ids"] == misses
            assert row["embedding_backend_calls"] == int(misses > 0)
            if expected is None:
                expected = logits.cpu()
            else:
                torch.testing.assert_close(logits.cpu(), expected, rtol=0.03, atol=0.01)
        runtime.arena.resize(0.7 if index % 2 else 0.3)
    print(
        "PASS: GPU values, ownership, invalidation, whole-call MGET and four-state HSTU predictions"
    )


if __name__ == "__main__":
    main()
