// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "../native/nvemb_rehash.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
struct Raw {
  size_t row_bytes = 16;
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
Raw make(size_t sets, const std::vector<int64_t>& keys) {
  Raw s; s.num_sets = sets;
  s.tags.assign(sets * 8, -1); s.data.assign(sets * 8 * 16, 0);
  s.counters.assign(sets * 8, 0); s.set_timestamps.assign(sets, 7);
  for (auto key : keys) {
    size_t slot = key % sets * 8;
    while (s.tags[slot] != -1) ++slot;
    assert(slot / 8 == size_t(key % sets));
    s.tags[slot] = key / sets; s.counters[slot] = float(key + 1) * .125f;
    for (size_t b = 0; b < 16; ++b) s.data[slot * 16 + b] = int8_t((key * 13 + b) % 127);
  }
  return s;
}
using Values = std::map<int64_t, std::pair<std::vector<int8_t>, std::pair<float,int64_t>>>;
Values values(const Raw& s) {
  Values v;
  for (size_t slot=0; slot<s.tags.size(); ++slot) if (s.tags[slot] != -1) {
    int64_t key=s.tags[slot]*s.num_sets+slot/8;
    v.emplace(key, std::make_pair(std::vector<int8_t>(s.data.begin()+slot*16,s.data.begin()+(slot+1)*16),
                                   std::make_pair(s.counters[slot],s.set_timestamps[slot/8])));
  }
  return v;
}
template<class F> void rejects(F f, const std::string& reason) {
  try { f(); assert(false); } catch (const std::invalid_argument& e) { assert(std::string(e.what())==reason); }
}
int main() {
  constexpr size_t cap=1<<20;
  auto old=make(2,{0,1,2,3,4,5,6,7,8,9,10,11});
  auto original=values(old);
  auto grow=nve::plan_preserving_rehash(old,4,cap);
  assert(values(grow.target)==original && grow.survivor_rows==12 && grow.evicted_rows==0);
  assert(grow.target.global_ts==old.global_ts && values(old)==original);
  auto shrink=nve::plan_preserving_rehash(old,1,cap);
  auto survivors=values(shrink.target);
  assert(survivors.size()==8 && shrink.evicted_rows==4);
  for (int key=0;key<4;++key) assert(!survivors.count(key));
  for (auto& entry:survivors) assert(entry.second==original.at(entry.first));
  auto ages=old; ages.set_timestamps[1]=8;
  rejects([&]{nve::plan_preserving_rehash(ages,1,cap);},"mixed_age_merge");
  auto split=nve::plan_preserving_rehash(ages,4,cap);
  assert(values(split.target)==values(ages));
  auto collisions=make(2,{0,3,6,9,12,15,18,21,24});
  rejects([&]{nve::plan_preserving_rehash(collisions,3,cap);},"growth_rehash_collision");
  auto tie=old; for(auto& counter:tie.counters)counter=1.0f;
  auto t=nve::plan_preserving_rehash(tie,1,cap);
  for(int key=0;key<8;++key) assert(values(t.target).count(key));
  rejects([&]{nve::plan_preserving_rehash(old,4,64);},"host_staging_budget");
  auto invalid=old; invalid.counters[0]=std::numeric_limits<float>::quiet_NaN();
  rejects([&]{nve::plan_preserving_rehash(invalid,4,cap);},"unsupported_resident_metadata");
  invalid=old; invalid.tags[1]=invalid.tags[0];
  rejects([&]{nve::plan_preserving_rehash(invalid,4,cap);},"duplicate_key");
  invalid=old; invalid.tags[0]=INT64_MAX;
  rejects([&]{nve::plan_preserving_rehash(invalid,4,cap);},"key_overflow");
  invalid=old; invalid.row_bytes=15;
  rejects([&]{nve::plan_preserving_rehash(invalid,4,cap);},"unsupported_raw_geometry");
  auto empty=make(2,{}); auto empty_resize=nve::plan_preserving_rehash(empty,3,cap);
  assert(empty_resize.survivor_rows==0 && empty_resize.evicted_rows==0);
  // Check exact float object representation, including a denormal counter.
  auto bits=old; uint32_t raw=1; std::memcpy(&bits.counters[0],&raw,sizeof(raw));
  auto preserved=nve::plan_preserving_rehash(bits,4,cap); auto pv=values(preserved.target).at(0).second.first;
  uint32_t actual;std::memcpy(&actual,&pv,sizeof(actual));assert(actual==raw);
  std::cout<<"13 native host planner cases passed\n";
}
