# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded signals from observed evictions and subsequent actual cache misses.

Ghost history counts potentially reusable prefix units. It does not estimate GPU
time, certify a benefit from more capacity, or inspect future requests.
"""

from collections import OrderedDict
import math


def _validate_key(key):
    """Accept IDs and explicit table/version namespaces without numeric aliases."""
    if type(key) is int and key >= 0:
        return
    if type(key) is str and key:
        return
    if type(key) is tuple and key:
        for part in key:
            _validate_key(part)
        return
    raise ValueError(
        "key must be a nonnegative integer, nonempty string, or tuple of valid keys"
    )


def _validate_count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_time(value, name, minimum=0, inclusive=True):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value < minimum if inclusive else value <= minimum):
        comparison = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")


class GhostHistory:
    """Remember only actual capacity evictions, with bounded count and age.

    Keys are nonnegative integer IDs, nonempty strings, or nonempty tuples of
    those values; tuples can separate embedding tables and content versions.
    Prefix units are KV tokens, or one unit for an embedding row. The caller
    reports real eviction events and calls ``reused_units`` only for actual
    misses. Cold misses have no ghost evidence. Flushes and content/version
    invalidations must remove history rather than create capacity evictions.

    Records expire when ``now_s - eviction_time >= ttl_s``. Lookups neither
    refresh nor consume them: repeated actual misses can count repeatedly.
    Re-eviction replaces the old prefix and time. When full, the oldest
    eviction record is removed, regardless of recent lookups. The supplied
    event clock is finite, nonnegative and nondecreasing across both recording
    and querying. No clock, future trace, GPU API or latency model is consulted.
    These counts alone do not establish a capacity or performance benefit.
    """

    def __init__(self, max_entries=100000, ttl_s=60):
        _validate_count(max_entries, "max_entries", minimum=1)
        _validate_time(ttl_s, "ttl_s", inclusive=False)
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._entries = OrderedDict()
        self._last_now_s = None

    def _advance(self, now_s):
        _validate_time(now_s, "now_s")
        if self._last_now_s is not None and now_s < self._last_now_s:
            raise ValueError("event clock must be nondecreasing")
        self._last_now_s = now_s
        # Insertion order is eviction-time order because events are monotonic
        # and replacing an eviction moves its record to the end.
        while self._entries:
            _, (_, evicted_s) = next(iter(self._entries.items()))
            if now_s - evicted_s < self._ttl_s:
                break
            self._entries.popitem(last=False)

    def record_eviction(self, key, prefix_units, now_s):
        """Record a real positive-prefix capacity eviction, not a flush/miss.

        A newer eviction of the same key replaces its prefix and timestamp;
        it does not accumulate prefixes or claim that GPU memory was released.
        """
        _validate_key(key)
        _validate_count(prefix_units, "prefix_units", minimum=1)
        self._advance(now_s)
        self._entries.pop(key, None)
        self._entries[key] = (prefix_units, now_s)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def reused_units(self, key, cached_prefix_units, requested_prefix_units, now_s):
        """Count the evicted prefix overlapping this observed actual miss.

        Return ``max(0, min(evicted_prefix, requested_prefix) - cached_prefix)``.
        In particular, an evicted prefix wholly below the current cached prefix
        contributes zero, even when the request has a long missing suffix.
        This query does not consume, refresh or reprioritize the ghost record.
        The count is not a GPU-time estimate or proof that resizing would help.
        """
        _validate_key(key)
        _validate_count(cached_prefix_units, "cached_prefix_units")
        _validate_count(requested_prefix_units, "requested_prefix_units")
        self._advance(now_s)
        record = self._entries.get(key)
        if record is None:
            return 0
        evicted_prefix, _ = record
        return max(0, min(evicted_prefix, requested_prefix_units) - cached_prefix_units)

    def invalidate(self, key):
        """Forget a version/flush invalidation without recording capacity loss.

        Missing keys are harmless. There is no event timestamp or clock advance
        because this operation only removes evidence; it creates no new signal.
        """
        _validate_key(key)
        self._entries.pop(key, None)
