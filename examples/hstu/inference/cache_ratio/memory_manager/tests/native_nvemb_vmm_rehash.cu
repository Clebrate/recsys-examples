// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// CPU oracle: g++ -x c++ -std=c++17 -O2 -Iinclude this_file.cu -o test && ./test
// Native probe: nvcc -std=c++17 -O2 --use_fast_math -Iinclude this_file.cu -o test && ./test
// The CUDA build executes real kernels. The CPU build makes no CUDA claim.
#include <ecache/ec_vmm_rehash_common.hpp>
#include <nvemb_rehash.hpp>
#ifdef __CUDACC__
#include <ecache/ec_vmm_rehash.cuh>
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <string>

struct Raw {
  size_t row_bytes = 48;
  uint64_t num_sets = 0;
  uint32_t ways = 8;
  int64_t sentinel = -1;
  float decay_rate = .95f;
  int64_t global_ts = 9;
  std::vector<int64_t> tags;
  std::vector<int8_t> data;
  std::vector<float> counters;
  std::vector<int64_t> set_timestamps;
};

Raw empty(uint64_t sets, size_t row_bytes = 48) {
  Raw result;
  result.num_sets = sets;
  result.row_bytes = row_bytes;
  result.tags.assign(sets * 8, -1);
  result.counters.assign(sets * 8, 0);
  result.set_timestamps.assign(sets, 0);
  result.data.assign(sets * 8 * row_bytes, 0);
  return result;
}

Raw make(uint64_t sets, const std::vector<int64_t>& keys, size_t row_bytes = 48) {
  Raw result = empty(sets, row_bytes);
  result.set_timestamps.assign(sets, 7);
  for (int64_t key : keys) {
    size_t slot = (key % sets) * 8;
    while (result.tags[slot] != -1) ++slot;
    assert(slot / 8 == uint64_t(key) % sets);
    result.tags[slot] = key / sets;
    result.counters[slot] = float((key * 7) % 23 + 1) * .125f;
    for (size_t byte = 0; byte < row_bytes; ++byte)
      result.data[slot * row_bytes + byte] = int8_t((key * 17 + byte * 3) % 127);
  }
  return result;
}

std::vector<int64_t> sequence(size_t count) {
  std::vector<int64_t> keys(count);
  std::iota(keys.begin(), keys.end(), 0);
  return keys;
}

// Independently bucket by the NEW modulo geometry and compare the restricted
// factor-two result against the existing, unrestricted published raw planner.
Raw oracle(const Raw& old, uint64_t target) {
  nve::vmm_rehash_direction(old.num_sets, target);
  Raw result = empty(target, old.row_bytes);
  result.global_ts = old.global_ts;
  result.decay_rate = old.decay_rate;
  std::map<uint64_t, std::vector<size_t>> buckets;
  const auto key = [&](size_t slot) { return uint64_t(old.tags[slot]) * old.num_sets + slot / 8; };
  for (size_t slot = 0; slot < old.tags.size(); ++slot)
    if (old.tags[slot] != -1) buckets[key(slot) % target].push_back(slot);
  for (auto& entry : buckets) {
    auto& slots = entry.second;
    const int64_t age = old.set_timestamps[slots.front() / 8];
    for (size_t slot : slots)
      if (old.set_timestamps[slot / 8] != age) throw std::invalid_argument("mixed_age_merge");
    std::sort(slots.begin(), slots.end(), [&](size_t a, size_t b) {
      uint32_t a_bits, b_bits;
      std::memcpy(&a_bits, &old.counters[a], 4);
      std::memcpy(&b_bits, &old.counters[b], 4);
      return nve::vmm_rehash_better_bits(a_bits, key(a), b_bits, key(b));
    });
    for (size_t way = 0; way < std::min<size_t>(8, slots.size()); ++way) {
      const size_t source = slots[way], destination = entry.first * 8 + way;
      result.tags[destination] = key(source) / target;
      std::memcpy(&result.counters[destination], &old.counters[source], sizeof(float));
      std::copy_n(old.data.data() + source * old.row_bytes, old.row_bytes,
                  result.data.data() + destination * old.row_bytes);
      result.set_timestamps[entry.first] = age;
    }
  }
  return result;
}

void equivalent(const Raw& actual, const Raw& expected) {
  assert(actual.num_sets == expected.num_sets && actual.row_bytes == expected.row_bytes);
  assert(actual.global_ts == expected.global_ts && actual.decay_rate == expected.decay_rate);
  assert(actual.tags == expected.tags && actual.set_timestamps == expected.set_timestamps);
  assert(actual.counters.size() == expected.counters.size());
  assert(!std::memcmp(actual.counters.data(), expected.counters.data(), actual.counters.size() * sizeof(float)));
  for (size_t slot = 0; slot < actual.tags.size(); ++slot)
    if (actual.tags[slot] != -1)
      assert(!std::memcmp(actual.data.data() + slot * actual.row_bytes,
                          expected.data.data() + slot * actual.row_bytes, actual.row_bytes));
}

#ifdef __CUDACC__
struct Device {
  int8_t* values = nullptr;
  int64_t* tags = nullptr;
  float* counters = nullptr;
  int64_t* ages = nullptr;
  void* scratch = nullptr;
  static void check(cudaError_t error) { nve::vmm_rehash_detail::checked(error, "native test"); }
  Device(const Raw& old, uint64_t maximum) {
    check(cudaMalloc(reinterpret_cast<void**>(&values), maximum * 8 * old.row_bytes));
    check(cudaMalloc(reinterpret_cast<void**>(&tags), maximum * 8 * sizeof(int64_t)));
    check(cudaMalloc(reinterpret_cast<void**>(&counters), maximum * 8 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&ages), maximum * sizeof(int64_t)));
    check(cudaMalloc(&scratch, 64));
    check(cudaMemset(values, 0x7b, maximum * 8 * old.row_bytes));
    check(cudaMemcpy(values, old.data.data(), old.data.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(tags, old.tags.data(), old.tags.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(counters, old.counters.data(), old.counters.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ages, old.set_timestamps.data(), old.set_timestamps.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  }
  Raw read(const Raw& geometry) {
    Raw out = geometry;
    check(cudaMemcpy(out.data.data(), values, out.data.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(out.tags.data(), tags, out.tags.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(out.counters.data(), counters, out.counters.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(out.set_timestamps.data(), ages, out.set_timestamps.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
    return out;
  }
  ~Device() { cudaFree(scratch); cudaFree(ages); cudaFree(counters); cudaFree(tags); cudaFree(values); }
};
#endif

int main() {
  unsigned cases = 0;
  const auto success = [&](const Raw& old, uint64_t target) {
    const auto planned = nve::plan_preserving_rehash(old, target, 1 << 26);
    equivalent(oracle(old, target), planned.target);
#ifdef __CUDACC__
    Device device(old, std::max(old.num_sets, target));
    const auto result = nve::vmm_factor2_preflight(device.tags, device.counters, device.ages,
        old.num_sets, target, old.global_ts, device.scratch);
    assert(result.occupied_rows == planned.survivor_rows + planned.evicted_rows);
    assert(result.survivor_rows == planned.survivor_rows && result.evicted_rows == planned.evicted_rows);
    nve::vmm_factor2_migrate(device.values, device.tags, device.counters, device.ages,
                            old.row_bytes, old.num_sets, target);
    equivalent(device.read(planned.target), planned.target);
#endif
    ++cases;
  };
  for (uint64_t old_sets : {2, 3, 4, 6}) {
    for (size_t width : {size_t(48), size_t(2048), size_t(4096)}) {
      auto full = make(old_sets, sequence(old_sets * 8), width);
      success(full, old_sets * 2);
      if (!(old_sets % 2)) success(full, old_sets / 2);
    }
  }
  auto tie = make(4, sequence(32));
  std::fill(tie.counters.begin(), tie.counters.end(), 1.0f);
  success(tie, 2);
  auto bits = make(2, sequence(16));
  const uint32_t denormal = 1, negative_zero = 0x80000000;
  std::memcpy(&bits.counters[0], &denormal, 4);
  std::memcpy(&bits.counters[1], &negative_zero, 4);
  success(bits, 4);
  // All surviving candidates are subnormal: FTZ comparison would select the
  // smallest keys instead, and would disagree with the original CPU planner.
  for (size_t slot = 0; slot < bits.counters.size(); ++slot) {
    const uint32_t score_bits = uint32_t(slot);
    std::memcpy(&bits.counters[slot], &score_bits, 4);
  }
  success(bits, 1);
  for (uint32_t valid : {0U, 0x80000000U, 1U, 0x007fffffU, 0x7f7fffffU})
    assert(nve::vmm_rehash_valid_counter(valid));
  for (uint32_t invalid : {0x80000001U, 0xbf800000U, 0x7f800000U, 0x7fc00000U}) {
    assert(!nve::vmm_rehash_valid_counter(invalid));
#ifdef __CUDACC__
    auto bad = bits;
    std::memcpy(&bad.counters[0], &invalid, 4);
    Device rejected_counter(bad, 2);
    try {
      nve::vmm_factor2_preflight(rejected_counter.tags, rejected_counter.counters,
          rejected_counter.ages, 2, 1, bad.global_ts, rejected_counter.scratch);
      assert(false);
    } catch (const std::invalid_argument& e) {
      assert(std::string(e.what()) == "invalid_resident_metadata");
    }
    equivalent(rejected_counter.read(bad), bad);
#endif
  }
  ++cases;
  auto high = make(2, {1, 3, 5, 7}); high.set_timestamps = {1, 8}; success(high, 1);
  auto low = make(2, {0, 2, 4, 6}); low.set_timestamps = {8, 1}; success(low, 1);
  auto vacant = make(2, {}); vacant.set_timestamps = {1, 8}; success(vacant, 1);
  auto mixed = make(2, sequence(16)); mixed.set_timestamps = {6, 8};
  success(mixed, 4);  // Pure split never merges the distinct source ages.
  try { oracle(mixed, 1); assert(false); }
  catch (const std::invalid_argument& e) { assert(std::string(e.what()) == "mixed_age_merge"); }
#ifdef __CUDACC__
  Device rejected(mixed, 2);
  try {
    nve::vmm_factor2_preflight(rejected.tags, rejected.counters, rejected.ages, 2, 1,
                              mixed.global_ts, rejected.scratch);
    assert(false);
  } catch (const std::invalid_argument& e) { assert(std::string(e.what()) == "mixed_age_merge"); }
  auto unchanged = rejected.read(mixed);
  equivalent(unchanged, mixed);
  assert(unchanged.data == mixed.data);  // Rejection changes not even invalid bytes.
#endif
  ++cases;
  try { nve::vmm_rehash_direction(3, 4); assert(false); }
  catch (const std::invalid_argument&) { ++cases; }
  // Reuse the SAME metadata/value allocation. Cleared high metadata must not
  // resurrect evicted keys when a previously retired region becomes active.
  auto cycle = make(4, sequence(32), 4096);
#ifdef __CUDACC__
  Device cycling(cycle, 8);
#endif
  for (uint64_t target : {2, 4, 8, 4, 2}) {
    const auto planned = nve::plan_preserving_rehash(cycle, target, 1 << 26);
    equivalent(oracle(cycle, target), planned.target);
#ifdef __CUDACC__
    const auto result = nve::vmm_factor2_preflight(cycling.tags, cycling.counters, cycling.ages,
        cycle.num_sets, target, cycle.global_ts, cycling.scratch);
    assert(result.survivor_rows == planned.survivor_rows && result.evicted_rows == planned.evicted_rows);
    nve::vmm_factor2_migrate(cycling.values, cycling.tags, cycling.counters, cycling.ages,
                            cycle.row_bytes, cycle.num_sets, target);
    equivalent(cycling.read(planned.target), planned.target);
#endif
    cycle = planned.target;
  }
  ++cases;
#ifdef __CUDACC__
  std::cout << cases << " native GPU factor-two rehash cases passed\n";
#else
  std::cout << cases << " CPU oracle cases passed; no CUDA execution\n";
#endif
}
