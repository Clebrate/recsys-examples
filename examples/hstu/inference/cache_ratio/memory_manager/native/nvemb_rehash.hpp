// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nve {
// Raw is CacheSADeviceModify<int64_t,int64_t>::RawState. Keeping this planner
// independent of CUDA lets CPU tests execute the exact native rehash code.
template<class Raw> struct PreservingRehash {
  Raw target;
  size_t survivor_rows = 0;
  size_t evicted_rows = 0;
  size_t host_peak_bound = 0;
  std::vector<int64_t> evicted_keys;
};

template<class Raw>
PreservingRehash<Raw> plan_preserving_rehash(const Raw& old, size_t sets, size_t host_limit,
                                           bool collect_evictions = false) {
  const auto require = [](bool ok, const char* reason) {
    if (!ok) throw std::invalid_argument(reason);
  };
  const auto product = [&](size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, "geometry_overflow");
    return a * b;
  };
  constexpr size_t ways = 8;
  require(sets && sets <= UINT32_MAX && old.num_sets && old.num_sets <= UINT32_MAX,
          "unsupported_set_count");
  require(old.ways == ways && old.sentinel == -1 && old.row_bytes &&
          old.row_bytes <= UINT32_MAX && old.row_bytes % 16 == 0 && old.global_ts >= 0,
          "unsupported_raw_geometry");
  const size_t old_slots = product(old.num_sets, ways), slots = product(sets, ways);
  const size_t row = old.row_bytes;
  require(old.tags.size() == old_slots && old.data.size() == product(old_slots, row) &&
          old.counters.size() == old_slots && old.set_timestamps.size() == old.num_sets,
          "invalid_raw_shape");
  struct Entry { uint64_t key; size_t slot; size_t set; float score; int64_t age; };
  const auto raw_capacity = [&](const Raw& s) {
    size_t sum = sizeof(Raw);
    for (size_t bytes : {product(s.tags.capacity(), sizeof(int64_t)), s.data.capacity(),
                        product(s.counters.capacity(), sizeof(float)),
                        product(s.set_timestamps.capacity(), sizeof(int64_t))}) {
      require(bytes <= std::numeric_limits<size_t>::max() - sum, "staging_overflow");
      sum += bytes;
    }
    return sum;
  };
  // Include old snapshot, target snapshot plus validation readback, and worst
  // case planning entries. Host allocator bookkeeping is outside this bound.
  const size_t target_values = product(slots, row);
  const size_t target_tags = product(slots, sizeof(int64_t));
  const size_t target_counters = product(slots, sizeof(float));
  const size_t target_ages = product(sets, sizeof(int64_t));
  size_t remaining = host_limit;
  const auto charge = [&](size_t bytes) {
    require(bytes <= remaining, "host_staging_budget"); remaining -= bytes;
  };
  charge(raw_capacity(old));
  for (int i = 0; i < 2; ++i)
    for (size_t bytes : {sizeof(Raw), target_values, target_tags, target_counters, target_ages}) charge(bytes);
  charge(product(old_slots, sizeof(Entry))); charge(3072);
  if (collect_evictions) charge(product(old_slots, sizeof(int64_t)));
  std::vector<Entry> entries;
  entries.reserve(old_slots);
  for (size_t slot = 0; slot < old_slots; ++slot) {
    if (old.tags[slot] == old.sentinel) continue;
    require(old.tags[slot] >= 0 && std::isfinite(old.counters[slot]) && old.counters[slot] >= 0,
            "unsupported_resident_metadata");
    const uint64_t old_set = slot / ways, tag = static_cast<uint64_t>(old.tags[slot]);
    require(old.set_timestamps[old_set] >= 0 && old.set_timestamps[old_set] <= old.global_ts,
            "invalid_resident_age");
    require(tag <= (uint64_t(INT64_MAX) - old_set) / old.num_sets, "key_overflow");
    for (size_t prev = old_set * ways; prev < slot; ++prev)
      require(old.tags[prev] != old.tags[slot], "duplicate_key");
    const uint64_t key = tag * old.num_sets + old_set;
    entries.push_back({key, slot, size_t(key % sets), old.counters[slot], old.set_timestamps[old_set]});
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    if (a.set != b.set) return a.set < b.set;
    return a.score != b.score ? a.score > b.score : a.key < b.key;
  });
  for (size_t i = 0; i < entries.size();) {
    size_t end = i + 1;
    while (end < entries.size() && entries[end].set == entries[i].set) ++end;
    for (size_t j = i; j < end; ++j)
      require(entries[j].age == entries[i].age, "mixed_age_merge");
    require(end - i <= ways || sets < old.num_sets, "growth_rehash_collision");
    i = end;
  }
  PreservingRehash<Raw> plan;
  if (collect_evictions) plan.evicted_keys.reserve(old_slots);
  auto& target = plan.target;
  target.row_bytes = row; target.num_sets = sets; target.ways = ways;
  target.sentinel = old.sentinel; target.decay_rate = old.decay_rate; target.global_ts = old.global_ts;
  target.tags.assign(slots, target.sentinel); target.data.assign(target_values, 0);
  target.counters.assign(slots, 0); target.set_timestamps.assign(sets, 0);
  for (size_t i = 0; i < entries.size();) {
    size_t end = i + 1;
    while (end < entries.size() && entries[end].set == entries[i].set) ++end;
    for (size_t j = i; j < std::min(end, i + ways); ++j) {
      const auto& e = entries[j]; const size_t slot = e.set * ways + j - i;
      target.tags[slot] = static_cast<int64_t>(e.key / sets);
      target.counters[slot] = old.counters[e.slot]; target.set_timestamps[e.set] = e.age;
      std::copy_n(old.data.data() + e.slot * row, row, target.data.data() + slot * row);
      ++plan.survivor_rows;
    }
    if (collect_evictions)
      for (size_t j = std::min(end, i + ways); j < end; ++j)
        plan.evicted_keys.push_back(static_cast<int64_t>(entries[j].key));
    i = end;
  }
  plan.evicted_rows = entries.size() - plan.survivor_rows;
  remaining = host_limit;
  charge(raw_capacity(old)); charge(raw_capacity(target)); charge(raw_capacity(target));
  charge(product(entries.capacity(), sizeof(Entry))); charge(3072);
  charge(product(plan.evicted_keys.capacity(), sizeof(int64_t)));
  plan.host_peak_bound = host_limit - remaining;
  return plan;
}
} // namespace nve
