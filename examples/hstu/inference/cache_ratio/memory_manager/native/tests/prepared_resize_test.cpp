// SPDX-License-Identifier: Apache-2.0
// Host-only test of the production plan owner and the unchanged LFU planner.
// g++ -std=c++17 -O2 -I /patched/sdk/include prepared_resize_test.cpp -o test && ./test
#include <nvemb_prepared_resize.hpp>
#include <nvemb_general_rehash.hpp>
#include <cassert>
#include <iostream>
#include <map>
#include <thread>
#include <mutex>
#include <set>
struct Raw {
  size_t row_bytes = 16; uint64_t num_sets = 4; uint32_t ways = 8;
  int64_t sentinel = -1; float decay_rate = .95f; int64_t global_ts = 20;
  std::vector<int64_t> tags; std::vector<int8_t> data; std::vector<float> counters;
  std::vector<int64_t> set_timestamps; bool per_row_ages = true;
  std::vector<int64_t> row_timestamps; std::vector<float> effective_scores;
};
using Plan = nve::GeneralRehashPlan<Raw>;
Raw fixture() {
  Raw old; old.tags.resize(32); old.data.resize(512); old.counters.resize(32);
  old.row_timestamps.resize(32); old.effective_scores.resize(32); old.set_timestamps.assign(4,20);
  for (size_t i = 0; i < 32; ++i) {
    const int64_t key = (i / 8) + (i % 8) * 4;
    old.tags[i] = key / 4; old.counters[i] = (i + 1) / 64.0f;
    old.row_timestamps[i] = i % 21; old.effective_scores[i] = float(i % 9) / 32;
    std::memcpy(old.data.data() + i * 16, &key, 8);
  }
  return old;
}
template<class F> void rejected(F fn) {
  bool failed = false; try { fn(); } catch (const std::invalid_argument&) { failed = true; }
  assert(failed);
}
int main() {
  size_t checks = 0; auto old = fixture();
  for (size_t target : {1, 2, 3, 5, 8}) {
    auto plan = std::make_unique<Plan>(nve::plan_general_rehash(old,target,1ULL<<30));
    auto* address = plan.get(); auto* values = plan->target.data.data();
    auto expected = plan->target.data;
    nve::PreparedResizeSlot<Plan> slot, other;
    const auto token = slot.prepare(std::move(plan),17);
    assert(slot.pending() && slot.generation() == 17); ++checks;
    rejected([&] { slot.prepare(std::make_unique<Plan>(),17); });
    assert(slot.matches(token) && &slot.get() == address); ++checks;
    rejected([&] { slot.consume("000" + token,17); });
    rejected([&] { slot.consume(token,18); });
    assert(slot.pending() && slot.get().target.data == expected); ++checks;
    auto other_token = other.prepare(std::make_unique<Plan>(),17);
    assert(other_token != token); rejected([&] { other.consume(token,17); }); ++checks;
    auto committed = slot.consume(token,17);
    assert(!slot.pending() && committed.get() == address && committed->target.data.data() == values);
    assert(committed->target.data == expected); ++checks;  // No plan/value copy on consume.
    rejected([&] { slot.consume(token,17); }); ++checks;
    assert(committed->target.global_ts == old.global_ts);
    size_t survivors = 0;
    for (size_t s=0; s<committed->target.tags.size(); ++s) {
      if (committed->target.tags[s] < 0) continue;
      const int64_t key = committed->target.tags[s]*target+s/8;
      const size_t original = (key%4)*8+key/4;
      assert(!std::memcmp(&old.counters[original],&committed->target.counters[s],4));
      assert(old.row_timestamps[original] == committed->target.row_timestamps[s]);
      assert(!std::memcmp(old.data.data()+original*16,committed->target.data.data()+s*16,16));
      ++survivors;
    }
    assert(survivors == committed->survivor_rows);
    assert(survivors + committed->evicted_keys.size() == 32); ++checks;
    auto cancelled = other.consume(other_token,17); cancelled.reset();
    assert(!other.pending()); rejected([&] { other.consume(other_token,17); }); ++checks;
  }
  // Token identity must be unique across independent owners and threads.
  std::mutex mutex; std::set<std::string> tokens; std::vector<std::thread> threads;
  for (int t=0;t<8;++t) threads.emplace_back([&] {
    for(int i=0;i<100;++i) {
      nve::PreparedResizeSlot<Plan> slot;
      auto token = slot.prepare(std::make_unique<Plan>(),0);
      std::lock_guard<std::mutex> lock(mutex); assert(tokens.insert(token).second);
    }
  });
  for (auto& thread : threads) thread.join();
  assert(tokens.size() == 800); ++checks;
  // Prepared owner's destruction frees a retained payload even without commit.
  struct Tracked { int* live; explicit Tracked(int* p):live(p){++*live;} ~Tracked(){--*live;} };
  int live=0; { nve::PreparedResizeSlot<Tracked> slot; slot.prepare(std::make_unique<Tracked>(&live),1); assert(live==1); }
  assert(live==0); ++checks;
  std::cout << "PASS " << checks << " prepared-owner/LFU cases; 800 concurrent unique tokens\n";
}
