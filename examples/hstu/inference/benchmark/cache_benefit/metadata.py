# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU metadata for real, mutually exclusive KV/embedding GPU page ownership."""
import math
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    budget_bytes: int
    layers: int = 3
    heads: int = 4
    head_dim: int = 128
    page_tokens: int = 32
    embedding_dim: int = 512

    @property
    def page_bytes(self):
        return 2 * self.page_tokens * self.heads * self.head_dim * 2

    @property
    def row_bytes(self):
        return self.embedding_dim * 4

    @property
    def layer_bytes(self):
        return self.budget_bytes // self.layers // self.page_bytes * self.page_bytes

    @property
    def actual_bytes(self):
        return self.layer_bytes * self.layers

    def partition(self, fraction):
        if not 0 < fraction < 1:
            raise ValueError("KV fraction must be strictly between zero and one")
        pages = int(self.layer_bytes * fraction) // self.page_bytes
        boundary = pages * self.page_bytes
        slots = (self.layer_bytes - boundary) // self.row_bytes
        if pages < 16 or slots < 8192:
            raise ValueError("Budget/ratio leaves too few KV pages or embedding rows")
        return pages, slots, boundary


class KVLRU:
    def __init__(self, pages, page_tokens):
        self.capacity, self.page_tokens = pages, page_tokens
        self.entries = OrderedDict()
        self.free = list(reversed(range(pages)))

    def lookup(self, uid):
        entry = self.entries.get(uid)
        if entry is not None:
            self.entries.move_to_end(uid)
        return entry

    def drop(self, uid):
        entry = self.entries.pop(uid, None)
        if entry is not None:
            self.free.extend(entry["pages"])

    def allocate(self, uid, tokens):
        entry = self.lookup(uid)
        if entry is not None:
            if entry["tokens"] != tokens:
                raise ValueError("History changed: invalidate or version the KV key")
            return entry
        needed = math.ceil(tokens / self.page_tokens)
        if needed > self.capacity:
            raise ValueError("Request KV does not fit its pool")
        while len(self.free) < needed:
            _, victim = self.entries.popitem(last=False)
            self.free.extend(victim["pages"])
        entry = {"tokens": tokens, "pages": [self.free.pop() for _ in range(needed)]}
        self.entries[uid] = entry
        return entry

    def resize(self, pages):
        if pages >= self.capacity:
            self.free.extend(reversed(range(self.capacity, pages)))
            evicted = 0
        else:
            victims = [
                uid
                for uid, e in self.entries.items()
                if any(p >= pages for p in e["pages"])
            ]
            for uid in victims:
                self.drop(uid)
            self.free = [p for p in self.free if p < pages]
            evicted = len(victims)
        self.capacity = pages
        return evicted


class EmbeddingLRU:
    def __init__(self, total_slots, base_slot):
        self.total_slots, self.base_slot = total_slots, base_slot
        self.keys = OrderedDict()
        self.reverse = [None] * total_slots
        self.free_intervals = deque([[base_slot, total_slots]])
        self.recycled = []

    @property
    def capacity(self):
        return self.total_slots - self.base_slot

    def touch(self, ids):
        if len(ids) > self.capacity and len(set(ids)) > self.capacity:
            raise ValueError("Unique request keys exceed embedding shard capacity")
        slots, missing, destinations = [], [], []
        for key in ids:
            if key in self.keys:
                slot = self.keys[key]
                self.keys.move_to_end(key)
            else:
                if self.recycled:
                    slot = self.recycled.pop()
                elif self.free_intervals:
                    interval = self.free_intervals[0]
                    slot = interval[0]
                    interval[0] += 1
                    if interval[0] == interval[1]:
                        self.free_intervals.popleft()
                else:
                    _, slot = self.keys.popitem(last=False)
                if not self.base_slot <= slot < self.total_slots:
                    raise RuntimeError("Embedding row overlaps KV ownership")
                self.keys[key] = slot
                self.reverse[slot] = key
                missing.append(key)
                destinations.append(slot)
            slots.append(slot)
        return slots, missing, destinations

    def drop(self, ids):
        removed = 0
        for key in set(ids):
            slot = self.keys.pop(key, None)
            if slot is not None:
                self.reverse[slot] = None
                self.recycled.append(slot)
                removed += 1
        return removed

    def resize(self, base_slot):
        if not 0 <= base_slot < self.total_slots:
            raise ValueError("Invalid embedding boundary")
        evicted = 0
        if base_slot > self.base_slot:
            for slot in range(self.base_slot, base_slot):
                key = self.reverse[slot]
                if key is not None:
                    del self.keys[key]
                    self.reverse[slot] = None
                    evicted += 1
            self.recycled = [s for s in self.recycled if s >= base_slot]
            self.free_intervals = deque(
                [
                    [max(lo, base_slot), hi]
                    for lo, hi in self.free_intervals
                    if hi > base_slot
                ]
            )
        elif base_slot < self.base_slot:
            self.free_intervals.appendleft([base_slot, self.base_slot])
        self.base_slot = base_slot
        return evicted
