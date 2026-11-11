# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-budget arithmetic for cache construction; no CUDA or Torch dependency."""

import math
from typing import List, Tuple


def validate_bytes(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer number of bytes")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def split_cache_budget(total_bytes: int, kv_ratio: float) -> Tuple[int, int]:
    """Return KV/embedding budgets. Ratio is a fraction of the shared budget."""
    validate_bytes(total_bytes, "cache_budget_bytes")
    if not math.isfinite(kv_ratio) or not 0 < kv_ratio < 1:
        raise ValueError("kv_cache_ratio must be finite and strictly between 0 and 1")
    kv_bytes = int(total_bytes * kv_ratio)
    embedding_bytes = total_bytes - kv_bytes
    validate_bytes(kv_bytes, "kv_cache_bytes")
    validate_bytes(embedding_bytes, "gpu_cache_bytes")
    return kv_bytes, embedding_bytes


def kv_pages_for_budget(
    budget_bytes: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    page_size: int,
    element_size: int,
) -> Tuple[int, int]:
    """Floor to full pages across all layers; return page count and payload bytes."""
    validate_bytes(budget_bytes, "kv_cache_bytes")
    dimensions = (num_layers, num_heads, head_dim, page_size, element_size)
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in dimensions):
        raise ValueError("KV dimensions and element_size must be positive integers")
    bytes_per_page = 2 * math.prod(dimensions)
    pages = budget_bytes // bytes_per_page
    if pages == 0:
        raise ValueError(
            f"kv_cache_bytes must hold at least one page ({bytes_per_page} bytes)"
        )
    return pages, pages * bytes_per_page


def split_table_cache_bytes(
    total_bytes: int, table_bytes: List[int], row_bytes: List[int]
) -> List[int]:
    """Split one collection budget, reserving a row/table, then weighting by size.

    The backend may impose a larger minimum/alignment; these are requested
    payload bytes, not a measurement of total HBM usage.
    """
    validate_bytes(total_bytes, "table cache budget")
    if len(table_bytes) != len(row_bytes):
        raise ValueError("table_bytes and row_bytes must have the same length")
    if any(n <= 0 for n in table_bytes + row_bytes):
        raise ValueError("Table and row sizes must be positive")
    if not table_bytes:
        raise ValueError("Cannot assign a cache budget to an empty collection")
    minimum = sum(row_bytes)
    if total_bytes < minimum:
        raise ValueError(f"Table cache budget must be at least {minimum} bytes")
    remaining = total_bytes - minimum
    weight_sum = sum(table_bytes)
    return [
        row + (remaining * weight // (weight_sum * row)) * row
        for weight, row in zip(table_bytes, row_bytes)
    ]
