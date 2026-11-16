# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native correctness probe; importing it starts no CUDA work.

Run with ``python -m ...tests.native_embedding_vmm --library /path/library.so``.
Uses at most 8 MiB of mapped values; timings are not benchmark measurements.
An independent full-key LRU oracle checks every inserted key, including misses,
through power-of-two crossings, duplicate inserts, collisions, and regrowth.
"""

import argparse
import json


def bucket(key, sets):
    base = 1 << (sets.bit_length() - 1)
    low = key % base
    return key % (2 * base) if low < sets - base else low


def full_keys(sets):
    base = 1 << (sets.bit_length() - 1)
    split = sets - base
    return [way * (2 * base if s < split or s >= base else base) + s
            for way in range(8) for s in range(sets)]


class Reference:
    def __init__(self, sets):
        self.sets, self.clock = sets, 0
        self.rows, self.buckets = {}, {}

    def insert(self, keys, tokens):
        for key, token in zip(keys, tokens):
            self.clock += 1
            group = self.buckets.setdefault(bucket(key, self.sets), set())
            self.rows[key] = (token, self.clock)
            group.add(key)
            if len(group) > 8:
                victim = min(group, key=lambda k: (self.rows[k][1], -k))
                group.remove(victim)
                del self.rows[victim]

    def lookup(self, keys):
        expected = []
        for key in keys:
            self.clock += 1
            row = self.rows.get(key)
            expected.append(None if row is None else row[0])
            if row is not None:
                self.rows[key] = (row[0], self.clock)
        return expected

    def resize(self, sets):
        groups = {}
        for key in self.rows:
            groups.setdefault(bucket(key, sets), []).append(key)
        before = len(self.rows)
        self.buckets = {}
        for index, keys in groups.items():
            keys.sort(key=lambda key: (-self.rows[key][1], key))
            self.buckets[index] = set(keys[:8])
            for key in keys[8:]:
                del self.rows[key]
        self.sets = sets
        return before - len(self.rows)


def run(library_path, device=0):
    import torch
    from examples.hstu.inference.cache_ratio.memory_manager.embedding_vmm import VmmEmbeddingCache

    mib, row_bytes = 1024 * 1024, 2048
    chunk = 2 * mib
    reference = Reference(chunk // (8 * row_bytes))
    cache = VmmEmbeddingCache(row_bytes, 8 * mib, chunk, chunk,
                              library_path, device=device)
    known, next_token, checks = set(), 1, 0
    initial_pointer = cache.stats()["base_ptr"]
    native_device = f"cuda:{device}"

    def values(tokens):
        ids = torch.tensor(tokens, device=native_device, dtype=torch.int64)
        columns = torch.arange(row_bytes, device=native_device, dtype=torch.int64)
        result = ((ids[:, None] * 37 + columns[None, :] * 13) % 251).to(torch.uint8)
        shifts = torch.arange(0, 64, 8, device=native_device, dtype=torch.int64)
        result[:, :8] = ((ids[:, None] >> shifts[None, :]) & 255).to(torch.uint8)
        return result.contiguous()

    def insert(keys):
        nonlocal next_token
        tokens = list(range(next_token, next_token + len(keys)))
        next_token += len(keys)
        cache.insert(torch.tensor(keys, dtype=torch.int64, device=native_device), values(tokens))
        reference.insert(keys, tokens)
        known.update(keys)

    def check(keys=None):
        nonlocal checks
        keys = sorted(known) + [2**60 + 17, 2**60 + 19] if keys is None else keys
        expected = reference.lookup(keys)
        actual, hits = cache.lookup(torch.tensor(keys, dtype=torch.int64, device=native_device))
        expected_hits = torch.tensor([v is not None for v in expected],
                                     dtype=torch.bool, device=native_device)
        if not torch.equal(hits, expected_hits):
            raise AssertionError("Native hit/miss result disagrees with independent LRU oracle")
        expected_values = values([0 if v is None else v for v in expected])
        expected_values[~expected_hits] = 0
        if not torch.equal(actual, expected_values):
            raise AssertionError("Native lookup returned incorrect survivor bytes or nonzero miss bytes")
        diagnostics = cache.stats(include_occupancy=True)
        if diagnostics["resident_rows"] != len(reference.rows) or diagnostics["clock"] != reference.clock:
            raise AssertionError("Native occupancy or LRU clock disagrees with oracle")
        checks += 1

    actions = []

    def resize(payload):
        expected_evicted = reference.resize(payload // (8 * row_bytes))
        receipt = cache.resize(payload)
        if receipt["evicted_rows"] != expected_evicted:
            raise AssertionError("Resize eviction count disagrees with oracle")
        if receipt["payload_bytes"] != payload or receipt["base_ptr"] != initial_pointer:
            raise AssertionError("Resize changed the base address or mapped unexpected bytes")
        actions.append({"payload_bytes": payload, "evicted_rows": expected_evicted})
        check()

    try:
        insert(full_keys(reference.sets))
        check()
        resize(6 * mib)  # 128 -> 384 sets crosses the 256-set phase boundary.
        insert(full_keys(reference.sets))
        # Same-bucket pressure and duplicate input values exercise schedule-
        # independent selection of the eight newest DISTINCT keys.
        collisions = [3 + 512 * way for way in range(16)]
        insert(collisions + [collisions[1], collisions[1], collisions[0]])
        check()
        check([collisions[0], collisions[0], collisions[-1]])
        resize(4 * mib)
        resize(8 * mib)
        insert(full_keys(reference.sets))
        check()
        resize(2 * mib)  # Two reverse phases, with required collision evictions.
        resize(6 * mib)
        insert([2**40 + 1, 2**40 + 2, 2**40 + 1])
        check()
        return {"status": "passed", "backend": cache.backend_name,
                "native_lookup_checks": checks, "unique_keys_checked": len(known),
                "actions": actions, "final": cache.stats(include_occupancy=True)}
    finally:
        cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.library, arguments.device), sort_keys=True))
