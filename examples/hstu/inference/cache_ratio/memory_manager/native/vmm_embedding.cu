// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Experimental embedding cache, NOT an integration into NVEmb HierarchicalEmbedding.
// Eight ways, linear hashing, and per-row deterministic LRU replace NVEmb's LFU.
// Values live in an externally owned, stable CUDA VMM address reservation.
// This file never allocates, maps, unmaps, or frees the external value span.
// Metadata is allocated once for max_sets and reported as cudaMalloc request bytes.
//
// Build: nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC vmm_embedding.cu -o libvmm_embedding.so
//
// C ABI: return 0 on success, -1 on a contract violation, -2 on a poisoned handle,
// or a positive CUDA error code. vmm_embedding_last_error() describes the failure.
// All pointer arguments for keys/values/hitmask refer to DEVICE memory; handles,
// integer outputs and create's out_handle refer to HOST memory. Row storage is
// bitwise: any positive row_bytes divisible by four is supported, including
// float32[512]. Input/output buffers must not alias the external cache span or
// each other. Callers own sufficient buffer extents and must keep them alive.
//
// One host thread, no concurrent operations, no CUDA graphs. Calls use the default
// stream and synchronize the device. create selects `device`; subsequent calls
// require that device to be current. A CUDA operation failure poisons the handle.
// destroy ALWAYS consumes a valid non-null handle, including on failure. It
// attempts all frees once; a failed free is quarantined until context teardown.
// Never retry destroy with the consumed handle. Create failures publish no handle.
//
// Lookup touches the LRU clock, including clock positions reserved for misses.
// Insert and lookup reserve one monotonic stamp per input position. Duplicate
// insert keys use the last input position's value. Concurrent insertion CTAs
// retain each bucket's eight newest DISTINCT keys independent of scheduling.
// Resize never advances the clock. Split preserves all rows; merge retains the
// eight greatest stamps, breaking any equal-stamp tie by smaller key.
//
// Grow ownership protocol: map/zero new values BEFORE native resize.
// Shrink ownership protocol: native resize/synchronize BEFORE unmapping the tail.
// A failed migration must stop serving; it does not promise rollback.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
constexpr uint64_t kWays = 8;
constexpr unsigned kThreads = 256;
constexpr uint64_t kMagic = 0x564d4d454d423031ULL;
constexpr int64_t kEmpty = -1;
thread_local std::string last_error;

struct Counters {
  unsigned long long occupied;
  unsigned long long evicted;
  int invalid_key;
};

struct Context {
  uint64_t magic = kMagic;
  uint32_t* values = nullptr;
  int64_t* keys = nullptr;
  unsigned long long* stamps = nullptr;
  int* locks = nullptr;
  Counters* counters = nullptr;
  uint64_t row_bytes = 0;
  uint64_t max_sets = 0;
  uint64_t active_sets = 0;
  uint64_t clock = 0;
  uint64_t metadata_bytes = 0;
  unsigned tile_words = 0;
  int device = -1;
  bool poisoned = false;
};

struct CudaFailure : std::exception {
  cudaError_t code;
  std::string message;
  CudaFailure(cudaError_t value, const char* expression)
      : code(value), message(std::string(expression) + ": " + cudaGetErrorString(value)) {}
  const char* what() const noexcept override { return message.c_str(); }
};

void checked(cudaError_t value, const char* expression) {
  if (value != cudaSuccess) throw CudaFailure(value, expression);
}
#define CUDA_CHECK(expression) checked((expression), #expression)

uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::invalid_argument("Byte extent overflows size_t");
  return a * b;
}

uint64_t floor_power_two(uint64_t value) {
  uint64_t result = 1;
  while (result <= value / 2) result *= 2;
  return result;
}

Context* get_context(void* handle) {
  auto* ctx = static_cast<Context*>(handle);
  if (!ctx || ctx->magic != kMagic) throw std::invalid_argument("Invalid embedding handle");
  int current = -1;
  CUDA_CHECK(cudaGetDevice(&current));
  if (current != ctx->device)
    throw std::invalid_argument("Current CUDA device differs from embedding owner");
  return ctx;
}

void validate_count(Context* ctx, uint64_t count) {
  // One CTA per row. This limit also keeps the grid representable on supported
  // devices without silently truncating the requested operation.
  if (count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Input count exceeds supported CUDA grid extent");
  product(count, ctx->row_bytes);
  product(count, sizeof(int64_t));
  if (count > std::numeric_limits<uint64_t>::max() - ctx->clock)
    throw std::invalid_argument("Embedding LRU clock exhausted");
}

template <class Function>
int operate(void* handle, Function&& function) noexcept {
  last_error.clear();
  Context* ctx = nullptr;
  try {
    ctx = get_context(handle);
    if (ctx->poisoned) {
      last_error = "Embedding handle is poisoned; destroy and recreate it";
      return -2;
    }
    function(ctx);
    return 0;
  } catch (const CudaFailure& error) {
    if (ctx) ctx->poisoned = true;
    last_error = error.what();
    return static_cast<int>(error.code);
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    if (ctx) ctx->poisoned = true;
    last_error = "Unknown native embedding failure";
    return -2;
  }
}

__device__ __forceinline__ uint64_t bucket_for(uint64_t key, uint64_t base, uint64_t split) {
  const uint64_t bucket = key & (base - 1);
  return bucket < split ? key & ((base << 1) - 1) : bucket;
}

__device__ __forceinline__ bool newer(unsigned long long a_stamp, int64_t a_key,
                                      unsigned long long b_stamp, int64_t b_key) {
  return a_stamp > b_stamp || (a_stamp == b_stamp && a_key < b_key);
}

__global__ void validate_keys_kernel(const int64_t* keys, uint64_t count, int* invalid) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && keys[i] < 0) atomicExch(invalid, 1);
}

void validate_keys(Context* ctx, const int64_t* keys, uint64_t count) {
  if (!keys) throw std::invalid_argument("Nonempty operation requires device keys");
  CUDA_CHECK(cudaMemset(&ctx->counters->invalid_key, 0, sizeof(int)));
  validate_keys_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
      keys, count, &ctx->counters->invalid_key);
  CUDA_CHECK(cudaGetLastError());
  int invalid = 0;
  CUDA_CHECK(cudaMemcpy(&invalid, &ctx->counters->invalid_key, sizeof(int), cudaMemcpyDeviceToHost));
  if (invalid) throw std::invalid_argument("Embedding keys must be nonnegative int64 values");
}

__global__ void lookup_kernel(const int64_t* input, uint64_t row_words,
                              uint64_t base, uint64_t split, uint64_t clock_start,
                              const int64_t* keys, unsigned long long* stamps,
                              const uint32_t* values, uint32_t* output, uint8_t* hitmask) {
  const uint64_t i = blockIdx.x;
  __shared__ int found_way;
  __shared__ uint64_t set;
  if (threadIdx.x == 0) {
    const int64_t key = input[i];
    set = bucket_for(static_cast<uint64_t>(key), base, split);
    found_way = -1;
    for (int way = 0; way < 8; ++way) {
      if (keys[set * kWays + way] == key) {
        found_way = way;
        atomicMax(stamps + set * kWays + way,
                  static_cast<unsigned long long>(clock_start + i + 1));
        break;
      }
    }
    hitmask[i] = found_way >= 0 ? 1 : 0;
  }
  __syncthreads();
  for (uint64_t column = threadIdx.x; column < row_words; column += blockDim.x) {
    output[i * row_words + column] = found_way < 0 ? 0 :
        values[(set * kWays + static_cast<uint64_t>(found_way)) * row_words + column];
  }
}

__global__ void insert_kernel(const int64_t* input, const uint32_t* input_values,
                              uint64_t row_words, uint64_t base, uint64_t split,
                              uint64_t clock_start, int64_t* keys,
                              unsigned long long* stamps, uint32_t* values,
                              int* locks, Counters* counters) {
  const uint64_t i = blockIdx.x;
  __shared__ uint64_t set;
  __shared__ int selected_way;
  __shared__ int should_write;
  if (threadIdx.x == 0) {
    const int64_t key = input[i];
    const unsigned long long stamp = clock_start + i + 1;
    set = bucket_for(static_cast<uint64_t>(key), base, split);
    // One lock contender per CTA avoids a warp containing both the owner and
    // spinning contenders. Other CTAs can finish while this one waits.
    while (atomicCAS(locks + set, 0, 1) != 0) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(64);
#endif
    }
    __threadfence();
    int existing = -1, empty = -1, oldest = 0;
    for (int way = 0; way < 8; ++way) {
      const uint64_t slot = set * kWays + way;
      if (keys[slot] == key) existing = way;
      if (keys[slot] == kEmpty && empty < 0) empty = way;
      const uint64_t victim = set * kWays + oldest;
      if (newer(stamps[victim], keys[victim], stamps[slot], keys[slot])) oldest = way;
    }
    selected_way = existing >= 0 ? existing : (empty >= 0 ? empty : oldest);
    const uint64_t selected = set * kWays + selected_way;
    should_write = keys[selected] == kEmpty || newer(stamp, key, stamps[selected], keys[selected]);
    if (should_write && keys[selected] == kEmpty) atomicAdd(&counters->occupied, 1ULL);
  }
  __syncthreads();
  if (should_write) {
    for (uint64_t column = threadIdx.x; column < row_words; column += blockDim.x)
      values[(set * kWays + selected_way) * row_words + column] = input_values[i * row_words + column];
  }
  // Every payload writer publishes before thread 0 releases the bucket lock.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (should_write) {
      const uint64_t slot = set * kWays + selected_way;
      stamps[slot] = clock_start + i + 1;
      keys[slot] = input[i];
    }
    __threadfence();
    atomicExch(locks + set, 0);
  }
}

// Each CTA owns a disjoint low/high bucket pair within one linear-hash phase.
// Only a bounded column tile is staged, so arbitrary row widths do not require
// unbounded shared memory. A tile is fully read before any corresponding writes.
__global__ void split_kernel(uint32_t* values, int64_t* keys, unsigned long long* stamps,
                             uint64_t row_words, uint64_t base, uint64_t first_bucket,
                             unsigned tile_words) {
  const uint64_t low = first_bucket + blockIdx.x;
  const uint64_t high = base + low;
  __shared__ int64_t old_keys[8];
  __shared__ unsigned long long old_stamps[8];
  __shared__ int source_for_dest[16];
  extern __shared__ uint32_t tile[];
  if (threadIdx.x == 0) {
    int low_count = 0, high_count = 0;
    for (int d = 0; d < 16; ++d) source_for_dest[d] = -1;
    for (int s = 0; s < 8; ++s) {
      old_keys[s] = keys[low * kWays + s];
      old_stamps[s] = stamps[low * kWays + s];
      if (old_keys[s] == kEmpty) continue;
      const bool goes_high = (static_cast<uint64_t>(old_keys[s]) & base) != 0;
      const int destination = goes_high ? 8 + high_count++ : low_count++;
      source_for_dest[destination] = s;
    }
  }
  __syncthreads();
  for (uint64_t first = 0; first < row_words; first += tile_words) {
    const unsigned width = static_cast<unsigned>(row_words - first < tile_words ? row_words - first : tile_words);
    for (unsigned item = threadIdx.x; item < 8 * width; item += blockDim.x) {
      const unsigned source = item / width, column = item % width;
      if (old_keys[source] != kEmpty)
        tile[source * tile_words + column] = values[(low * kWays + source) * row_words + first + column];
    }
    __syncthreads();
    for (unsigned item = threadIdx.x; item < 16 * width; item += blockDim.x) {
      const unsigned destination = item / width, column = item % width;
      const int source = source_for_dest[destination];
      if (source >= 0) {
        const uint64_t slot = (destination < 8 ? low : high) * kWays + destination % 8;
        values[slot * row_words + first + column] = tile[source * tile_words + column];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int destination = 0; destination < 16; ++destination) {
      const int source = source_for_dest[destination];
      const uint64_t slot = (destination < 8 ? low : high) * kWays + destination % 8;
      keys[slot] = source < 0 ? kEmpty : old_keys[source];
      stamps[slot] = source < 0 ? 0 : old_stamps[source];
    }
  }
}

__global__ void merge_kernel(uint32_t* values, int64_t* keys, unsigned long long* stamps,
                             Counters* counters, uint64_t row_words, uint64_t base,
                             uint64_t first_bucket, unsigned tile_words) {
  const uint64_t low = first_bucket + blockIdx.x;
  const uint64_t high = base + low;
  __shared__ int64_t old_keys[16];
  __shared__ unsigned long long old_stamps[16];
  __shared__ int order[16];
  __shared__ int valid_count;
  extern __shared__ uint32_t tile[];
  if (threadIdx.x == 0) {
    valid_count = 0;
    for (int source = 0; source < 16; ++source) {
      const uint64_t slot = (source < 8 ? low : high) * kWays + source % 8;
      old_keys[source] = keys[slot];
      old_stamps[source] = stamps[slot];
      if (old_keys[source] == kEmpty) continue;
      int position = valid_count++;
      while (position > 0 && newer(old_stamps[source], old_keys[source],
                                    old_stamps[order[position - 1]], old_keys[order[position - 1]])) {
        order[position] = order[position - 1];
        --position;
      }
      order[position] = source;
    }
  }
  __syncthreads();
  const int retained = valid_count < 8 ? valid_count : 8;
  for (uint64_t first = 0; first < row_words; first += tile_words) {
    const unsigned width = static_cast<unsigned>(row_words - first < tile_words ? row_words - first : tile_words);
    for (unsigned item = threadIdx.x; item < 16 * width; item += blockDim.x) {
      const unsigned source = item / width, column = item % width;
      if (old_keys[source] != kEmpty) {
        const uint64_t slot = (source < 8 ? low : high) * kWays + source % 8;
        tile[source * tile_words + column] = values[slot * row_words + first + column];
      }
    }
    __syncthreads();
    for (unsigned item = threadIdx.x; item < static_cast<unsigned>(retained) * width; item += blockDim.x) {
      const unsigned destination = item / width, column = item % width;
      values[(low * kWays + destination) * row_words + first + column] =
          tile[order[destination] * tile_words + column];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int way = 0; way < 8; ++way) {
      keys[low * kWays + way] = way < retained ? old_keys[order[way]] : kEmpty;
      stamps[low * kWays + way] = way < retained ? old_stamps[order[way]] : 0;
      keys[high * kWays + way] = kEmpty;
      stamps[high * kWays + way] = 0;
    }
    const unsigned long long evicted = static_cast<unsigned long long>(valid_count - retained);
    if (evicted) {
      atomicAdd(&counters->evicted, evicted);
      atomicAdd(&counters->occupied, 0ULL - evicted);
    }
  }
}

// Cleanup checks every call and consumes every pointer once. A CUDA error may
// describe ambiguous completion; failed pointers are deliberately not retried.
int free_metadata(Context* ctx, std::string& detail) noexcept {
  int first_error = 0;
  const auto record = [&](cudaError_t error, const char* action) {
    if (error == cudaSuccess) return;
    if (!first_error) first_error = static_cast<int>(error);
    if (!detail.empty()) detail += "; ";
    detail += std::string(action) + ": " + cudaGetErrorString(error);
  };
  if (ctx->keys) record(cudaFree(ctx->keys), "cudaFree(keys)");
  if (ctx->stamps) record(cudaFree(ctx->stamps), "cudaFree(stamps)");
  if (ctx->locks) record(cudaFree(ctx->locks), "cudaFree(locks)");
  if (ctx->counters) record(cudaFree(ctx->counters), "cudaFree(counters)");
  ctx->keys = nullptr;
  ctx->stamps = nullptr;
  ctx->locks = nullptr;
  ctx->counters = nullptr;
  return first_error;
}
}  // namespace

extern "C" const char* vmm_embedding_last_error() noexcept { return last_error.c_str(); }

extern "C" int vmm_embedding_create(void* value_ptr, uint64_t row_bytes, uint64_t max_sets,
                                     uint64_t initial_sets, int device, void** out_handle,
                                     uint64_t* out_metadata_bytes) noexcept {
  last_error.clear();
  if (out_handle) *out_handle = nullptr;
  if (out_metadata_bytes) *out_metadata_bytes = 0;
  std::unique_ptr<Context> ctx;
  int error_code = -1;
  try {
    if (!out_handle || !out_metadata_bytes || !value_ptr || !row_bytes || row_bytes % 4 ||
        reinterpret_cast<uintptr_t>(value_ptr) % 4 || !initial_sets || initial_sets > max_sets ||
        max_sets > static_cast<uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Invalid output pointer, value address, row width, or set geometry");
    const uint64_t slots = product(max_sets, kWays);
    product(slots, row_bytes);
    const uint64_t key_bytes = product(slots, sizeof(int64_t));
    const uint64_t stamp_bytes = product(slots, sizeof(unsigned long long));
    const uint64_t lock_bytes = product(max_sets, sizeof(int));
    if (key_bytes > std::numeric_limits<uint64_t>::max() - stamp_bytes - lock_bytes - sizeof(Counters))
      throw std::invalid_argument("Metadata size overflow");
    CUDA_CHECK(cudaSetDevice(device));
    int shared_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (shared_limit < 2048) throw std::invalid_argument("Insufficient CUDA shared memory");
    ctx.reset(new Context());
    ctx->values = static_cast<uint32_t*>(value_ptr);
    ctx->row_bytes = row_bytes;
    ctx->max_sets = max_sets;
    ctx->active_sets = initial_sets;
    ctx->device = device;
    ctx->metadata_bytes = key_bytes + stamp_bytes + lock_bytes + sizeof(Counters);
    ctx->tile_words = static_cast<unsigned>(std::min<uint64_t>(row_bytes / 4,
        std::min<uint64_t>(512, static_cast<uint64_t>(shared_limit - 1024) / (16 * sizeof(uint32_t)))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx->keys), key_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx->stamps), stamp_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx->locks), lock_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx->counters), sizeof(Counters)));
    CUDA_CHECK(cudaMemset(ctx->keys, 0xff, key_bytes));
    CUDA_CHECK(cudaMemset(ctx->stamps, 0, stamp_bytes));
    CUDA_CHECK(cudaMemset(ctx->locks, 0, lock_bytes));
    CUDA_CHECK(cudaMemset(ctx->counters, 0, sizeof(Counters)));
    CUDA_CHECK(cudaDeviceSynchronize());
    *out_metadata_bytes = ctx->metadata_bytes;
    *out_handle = ctx.release();
    return 0;
  } catch (const CudaFailure& error) {
    last_error = error.what();
    error_code = static_cast<int>(error.code);
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "Unknown embedding creation failure";
    error_code = -2;
  }
  if (ctx) {
    std::string cleanup;
    free_metadata(ctx.get(), cleanup);
    if (!cleanup.empty()) last_error += "; cleanup: " + cleanup;
  }
  return error_code;
}

extern "C" int vmm_embedding_lookup(void* handle, const int64_t* keys, uint64_t count,
                                     void* values_out, uint8_t* hitmask) noexcept {
  return operate(handle, [&](Context* ctx) {
    validate_count(ctx, count);
    if (!count) return;
    if (!values_out || !hitmask || reinterpret_cast<uintptr_t>(values_out) % 4)
      throw std::invalid_argument("Lookup requires aligned device output and byte hitmask");
    CUDA_CHECK(cudaDeviceSynchronize());
    validate_keys(ctx, keys, count);
    const uint64_t start = ctx->clock;
    ctx->clock += count;
    const uint64_t base = floor_power_two(ctx->active_sets);
    lookup_kernel<<<static_cast<unsigned>(count), kThreads>>>(keys, ctx->row_bytes / 4,
        base, ctx->active_sets - base, start, ctx->keys, ctx->stamps, ctx->values,
        static_cast<uint32_t*>(values_out), hitmask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
}

extern "C" int vmm_embedding_insert(void* handle, const int64_t* keys,
                                     const void* values, uint64_t count) noexcept {
  return operate(handle, [&](Context* ctx) {
    validate_count(ctx, count);
    if (!count) return;
    if (!values || reinterpret_cast<uintptr_t>(values) % 4)
      throw std::invalid_argument("Insert requires aligned device input values");
    CUDA_CHECK(cudaDeviceSynchronize());
    validate_keys(ctx, keys, count);
    const uint64_t start = ctx->clock;
    ctx->clock += count;
    const uint64_t base = floor_power_two(ctx->active_sets);
    insert_kernel<<<static_cast<unsigned>(count), kThreads>>>(keys, static_cast<const uint32_t*>(values),
        ctx->row_bytes / 4, base, ctx->active_sets - base, start, ctx->keys,
        ctx->stamps, ctx->values, ctx->locks, ctx->counters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  });
}

extern "C" int vmm_embedding_resize(void* handle, uint64_t old_sets, uint64_t new_sets,
                                     uint64_t* out_evicted_rows) noexcept {
  if (out_evicted_rows) *out_evicted_rows = 0;
  return operate(handle, [&](Context* ctx) {
    if (!out_evicted_rows || old_sets != ctx->active_sets || !new_sets || new_sets > ctx->max_sets)
      throw std::invalid_argument("Resize has stale source geometry or invalid target/output");
    if (new_sets == old_sets) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(&ctx->counters->evicted, 0, sizeof(unsigned long long)));
    const size_t shared_bytes = 16 * size_t(ctx->tile_words) * sizeof(uint32_t);
    uint64_t current = old_sets;
    while (current < new_sets) {
      const uint64_t base = floor_power_two(current);
      const uint64_t split = current - base;
      const uint64_t count = std::min(new_sets - current, base - split);
      split_kernel<<<static_cast<unsigned>(count), kThreads, shared_bytes>>>(
          ctx->values, ctx->keys, ctx->stamps, ctx->row_bytes / 4, base, split, ctx->tile_words);
      CUDA_CHECK(cudaGetLastError());
      current += count;
    }
    while (current > new_sets) {
      uint64_t base = floor_power_two(current);
      if (current == base) base /= 2;
      const uint64_t split = current - base;
      const uint64_t count = std::min(current - new_sets, split);
      merge_kernel<<<static_cast<unsigned>(count), kThreads, shared_bytes>>>(
          ctx->values, ctx->keys, ctx->stamps, ctx->counters, ctx->row_bytes / 4,
          base, split - count, ctx->tile_words);
      CUDA_CHECK(cudaGetLastError());
      current -= count;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long evicted = 0;
    CUDA_CHECK(cudaMemcpy(&evicted, &ctx->counters->evicted, sizeof(evicted), cudaMemcpyDeviceToHost));
    ctx->active_sets = new_sets;
    *out_evicted_rows = evicted;
  });
}

extern "C" int vmm_embedding_stats(void* handle, uint64_t* out_active_sets,
                                    uint64_t* out_occupied_rows, uint64_t* out_clock) noexcept {
  return operate(handle, [&](Context* ctx) {
    if (!out_active_sets || !out_occupied_rows || !out_clock)
      throw std::invalid_argument("Stats requires three host output pointers");
    CUDA_CHECK(cudaDeviceSynchronize());
    Counters counters{};
    CUDA_CHECK(cudaMemcpy(&counters, ctx->counters, sizeof(counters), cudaMemcpyDeviceToHost));
    *out_active_sets = ctx->active_sets;
    *out_occupied_rows = counters.occupied;
    *out_clock = ctx->clock;
  });
}

extern "C" int vmm_embedding_destroy(void* handle) noexcept {
  last_error.clear();
  auto* ctx = static_cast<Context*>(handle);
  if (!ctx) return 0;
  if (ctx->magic != kMagic) {
    last_error = "Invalid embedding handle passed to destroy";
    return -1;
  }
  int first_error = 0;
  const auto record = [&](cudaError_t error, const char* action) {
    if (error == cudaSuccess) return;
    if (!first_error) first_error = static_cast<int>(error);
    if (!last_error.empty()) last_error += "; ";
    last_error += std::string(action) + ": " + cudaGetErrorString(error);
  };
  // Destruction selects the owning device even if the caller changed it. It
  // deliberately consumes the handle on every path so failed frees cannot be
  // repeated accidentally by a destructor/finalizer.
  const cudaError_t selected = cudaSetDevice(ctx->device);
  record(selected, "cudaSetDevice(owner) during destroy");
  if (selected == cudaSuccess) {
    record(cudaDeviceSynchronize(), "cudaDeviceSynchronize during destroy");
    std::string cleanup;
    const int cleanup_error = free_metadata(ctx, cleanup);
    if (!first_error) first_error = cleanup_error;
    if (!cleanup.empty()) {
      if (!last_error.empty()) last_error += "; ";
      last_error += cleanup;
    }
  } else {
    last_error += "; metadata quarantined because owning device could not be selected";
  }
  ctx->magic = 0;
  delete ctx;
  return first_error;
}
