// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Interface spike only: two independent NVE cache banks, one physical allocation
// transferred to a DLPack tensor. No HSTU execution or FlexKV registration here.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime_api.h>
#include <allocator.hpp>
#include <buffer_wrapper.hpp>
#include <default_allocator.hpp>
#include <gpu_table.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

void bind_banked_embedding(py::module_& module);

namespace {

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Requested bytes tracked by this experiment, not CUDA's rounded allocation
// sizes, driver/context memory, or allocations made by PyTorch itself.
struct Counts {
  size_t live_bytes{0}, peak_bytes{0}, allocate_calls{0}, free_calls{0};
  size_t backing_live_bytes{0}, backing_allocate_calls{0}, backing_free_calls{0};
  void add(size_t bytes, bool backing) {
    live_bytes += bytes;
    peak_bytes = std::max(peak_bytes, live_bytes);
    ++allocate_calls;
    if (backing) { backing_live_bytes += bytes; ++backing_allocate_calls; }
  }
  void remove(size_t bytes, bool backing) {
    live_bytes -= bytes;
    ++free_calls;
    if (backing) { backing_live_bytes -= bytes; ++backing_free_calls; }
  }
};

struct Backing {
  void* ptr{nullptr};
  size_t bytes;
  int device;
  std::shared_ptr<Counts> counts;
  Backing(size_t size, int device_id, std::shared_ptr<Counts> counters)
      : bytes(size), device(device_id), counts(std::move(counters)) {
    checked(cudaSetDevice(device));
    checked(cudaMalloc(&ptr, bytes));
    counts->add(bytes, true);
  }
  ~Backing() {
    // A DLPack consumer can outlive the experiment. Free on the owning device.
    int previous = device;
    const bool known = cudaGetDevice(&previous) == cudaSuccess;
    if (cudaSetDevice(device) == cudaSuccess && cudaFree(ptr) == cudaSuccess)
      counts->remove(bytes, true);
    if (known && previous != device) cudaSetDevice(previous);
  }
};

// Only the constructor's first device allocation is the NVE cache pool.
// Scratch allocations and host allocations remain ordinary NVE allocations.
class LeaseAllocator final : public nve::Allocator {
 public:
  std::shared_ptr<Backing> backing;
  size_t requested_pool_bytes{0};
  bool construction{true}, leased{false};

  explicit LeaseAllocator(std::shared_ptr<Backing> block)
      : backing(std::move(block)), fallback_(nve::GetDefaultAllocator()) {}

  cudaError_t device_allocate(void** ptr, size_t size, int device = -1) noexcept override {
    if (construction && requested_pool_bytes == 0) {
      if (size == 0 || size > backing->bytes ||
          (device >= 0 && device != backing->device)) return cudaErrorInvalidValue;
      requested_pool_bytes = size;
      leased = true;
      *ptr = backing->ptr;
      return cudaSuccess;
    }
    const auto result = fallback_->device_allocate(ptr, size, device);
    if (result != cudaSuccess) return result;
    try {
      device_sizes_.emplace(*ptr, size);
    } catch (...) {
      fallback_->device_free(*ptr, device);
      *ptr = nullptr;
      return cudaErrorMemoryAllocation;
    }
    backing->counts->add(size, false);
    return cudaSuccess;
  }

  cudaError_t device_free(void* ptr, int device = -1) noexcept override {
    if (!ptr) return cudaSuccess;
    if (ptr == backing->ptr) {
      leased = false;  // The backing owner, not NVE, owns cudaFree.
      return cudaSuccess;
    }
    const auto found = device_sizes_.find(ptr);
    if (found == device_sizes_.end()) return cudaErrorInvalidValue;
    const auto result = fallback_->device_free(ptr, device);
    if (result == cudaSuccess) {
      backing->counts->remove(found->second, false);
      device_sizes_.erase(found);
    }
    return result;
  }

  // Use Allocator's synchronous async-method defaults, so every device byte
  // passes through the same bookkeeping and no CUDA memory pool is involved.
  cudaError_t host_allocate(void** ptr, size_t size) noexcept override {
    return fallback_->host_allocate(ptr, size);
  }
  cudaError_t host_free(void* ptr) noexcept override { return fallback_->host_free(ptr); }

 private:
  nve::allocator_ptr_t fallback_;
  std::unordered_map<void*, size_t> device_sizes_;
};

struct TensorOwner {
  DLManagedTensor tensor{};
  std::shared_ptr<Backing> backing;
  std::vector<int64_t> shape;
};

class BankTransfer {
 public:
  BankTransfer(size_t bank_bytes, int64_t row_bytes, int64_t warm_rows, int device)
      : bank_bytes_(bank_bytes), row_bytes_(row_bytes), warm_rows_(warm_rows), device_(device) {
    if (bank_bytes < (1U << 20) || row_bytes < 4 || row_bytes % 4 != 0 ||
        row_bytes > 4096 || warm_rows < 1 || warm_rows > 1024)
      throw std::invalid_argument("Require bank_bytes >= 1 MiB, row_bytes in [4,4096] divisible by 4, warm_rows in [1,1024]");
    checked(cudaSetDevice(device_));
    for (size_t bank = 0; bank < 2; ++bank) {
      allocators_[bank] = std::make_shared<LeaseAllocator>(
          std::make_shared<Backing>(bank_bytes_, device_, counts_));
      nve::GPUTableConfig cfg{};
      cfg.device_id = device_;
      cfg.cache_size = bank_bytes_;  // Immutable for the lifetime of each bank.
      cfg.row_size_in_bytes = row_bytes_;
      cfg.value_dtype = nve::DataType_t::Float32;
      cfg.uvm_table = nullptr;  // Hits cannot be concealed by UVM fallback.
      cfg.modify_on_gpu = true;
      cfg.count_misses = false;
      cfg.max_modify_size = warm_rows_;
      tables_[bank] = std::make_shared<nve::GpuTable<int64_t>>(cfg, allocators_[bank]);
      allocators_[bank]->construction = false;
      if (!allocators_[bank]->leased)
        throw std::runtime_error("NVE did not request the expected constructor backing allocation");
      contexts_[bank] = tables_[bank]->create_execution_context(0, 0, nullptr, allocators_[bank]);
    }
  }

  ~BankTransfer() {
    // Normal callers close explicitly before Python/CUDA interpreter teardown.
    try { close(); } catch (...) {}
  }

  void warm() {
    ensure_open();
    if (warmed_) throw std::runtime_error("Banks may only be warmed once in this spike");
    checked(cudaSetDevice(device_));
    for (size_t bank = 0; bank < 2; ++bank) {
      auto keys = make_keys(bank);
      auto values = make_values(bank);
      auto keys_buffer = std::make_shared<nve::BufferWrapper<const void>>(
          contexts_[bank], "warm_keys", keys.data(), keys.size() * sizeof(int64_t));
      auto values_buffer = std::make_shared<nve::BufferWrapper<const void>>(
          contexts_[bank], "warm_values", values.data(), values.size() * sizeof(float));
      tables_[bank]->insert(contexts_[bank], warm_rows_, keys_buffer,
                            row_bytes_, row_bytes_, values_buffer);
      contexts_[bank]->wait();
      ++insert_calls_[bank];
    }
    warmed_ = true;
  }

  py::dict verify_bank(size_t bank) {
    ensure_open();
    if (bank >= 2 || !tables_[bank] || !warmed_)
      throw std::runtime_error("Verification needs a live, warmed bank");
    checked(cudaSetDevice(device_));
    const auto started = Clock::now();
    auto keys = make_keys(bank);
    const auto expected = make_values(bank);
    std::vector<float> actual(expected.size(), 0);
    std::vector<nve::bitmask64_t> mask((static_cast<size_t>(warm_rows_) + 63) / 64, 0);
    auto keys_buffer = std::make_shared<nve::BufferWrapper<const void>>(
        contexts_[bank], "verify_keys", keys.data(), keys.size() * sizeof(int64_t));
    auto mask_buffer = std::make_shared<nve::BufferWrapper<nve::bitmask64_t>>(
        contexts_[bank], "verify_mask", mask.data(), mask.size() * sizeof(nve::bitmask64_t));
    auto values_buffer = std::make_shared<nve::BufferWrapper<void>>(
        contexts_[bank], "verify_values", actual.data(), actual.size() * sizeof(float));
    tables_[bank]->find(contexts_[bank], warm_rows_, keys_buffer, mask_buffer,
                        row_bytes_, values_buffer, nullptr);
    contexts_[bank]->wait();
    checked(cudaMemcpy(actual.data(), values_buffer->get_buffer(cudaMemoryTypeDevice),
                       actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(mask.data(), mask_buffer->get_buffer(cudaMemoryTypeDevice),
                       mask.size() * sizeof(nve::bitmask64_t), cudaMemcpyDeviceToHost));
    size_t hits = 0, correct = 0;
    const size_t width = static_cast<size_t>(row_bytes_) / sizeof(float);
    for (size_t i = 0; i < static_cast<size_t>(warm_rows_); ++i) {
      const bool hit = (mask[i / 64] & (uint64_t{1} << (i % 64))) != 0;
      hits += hit;
      if (hit && std::equal(expected.begin() + i * width, expected.begin() + (i + 1) * width,
                            actual.begin() + i * width)) ++correct;
    }
    py::dict result;
    result["rows"] = warm_rows_;
    result["hits"] = hits;
    result["correct_rows"] = correct;
    result["lookup_and_validation_ms"] = elapsed_ms(started);
    return result;
  }

  py::dict retire_bank1() {
    ensure_open();
    if (!warmed_ || !tables_[1]) throw std::runtime_error("Bank 1 must be warmed and live");
    checked(cudaSetDevice(device_));
    const auto started = Clock::now();
    contexts_[1]->wait();
    // The first spike deliberately drains the device. This includes all streams;
    // it is not a claim of online or concurrent retirement latency.
    checked(cudaDeviceSynchronize());
    const double drain = elapsed_ms(started);
    const auto retire_start = Clock::now();
    contexts_[1].reset();
    tables_[1].reset();
    if (allocators_[1]->leased) throw std::runtime_error("NVE still holds the retired backing");
    py::dict result;
    result["drain_ms"] = drain;
    result["context_and_table_retire_ms"] = elapsed_ms(retire_start);
    result["retired_backing_ptr"] = reinterpret_cast<uintptr_t>(allocators_[1]->backing->ptr);
    return result;
  }

  py::capsule kv_dlpack(const std::vector<int64_t>& shape) {
    ensure_open();
    if (tables_[1] || allocators_[1]->leased || exported_)
      throw std::runtime_error("Retire bank 1 first; its DLPack tensor can only be exported once");
    if (shape.size() != 5 || shape[1] != 2)
      throw std::invalid_argument("KV shape must be [blocks, 2, page_size, heads, head_dim]");
    size_t bytes = 2;  // FP16
    for (int64_t dim : shape) {
      if (dim <= 0 || static_cast<uint64_t>(dim) > bank_bytes_ / bytes)
        throw std::invalid_argument("KV shape does not fit the retired backing");
      bytes *= static_cast<size_t>(dim);
    }
    auto owner = std::make_unique<TensorOwner>();
    owner->backing = allocators_[1]->backing;
    owner->shape = shape;
    auto& tensor = owner->tensor;
    tensor.dl_tensor.data = owner->backing->ptr;
    tensor.dl_tensor.device = {kDLCUDA, device_};
    tensor.dl_tensor.ndim = static_cast<int>(shape.size());
    tensor.dl_tensor.dtype = {static_cast<uint8_t>(kDLFloat), 16, 1};
    tensor.dl_tensor.shape = owner->shape.data();
    tensor.dl_tensor.strides = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx = owner.get();
    tensor.deleter = [](DLManagedTensor* managed) {
      delete static_cast<TensorOwner*>(managed->manager_ctx);
    };
    py::capsule capsule(&tensor, "dltensor", [](PyObject* capsule_ptr) {
      if (PyCapsule_IsValid(capsule_ptr, "dltensor")) {
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_ptr, "dltensor"));
        managed->deleter(managed);
      }
    });
    owner.release();
    exported_ = true;
    return capsule;
  }

  py::dict snapshot() const {
    checked(cudaSetDevice(device_));
    py::dict result;
    result["configured_bank_bytes"] = bank_bytes_;
    result["tracked_cuda_live_bytes"] = counts_->live_bytes;
    result["tracked_cuda_peak_bytes"] = counts_->peak_bytes;
    result["tracked_cuda_allocate_calls"] = counts_->allocate_calls;
    result["tracked_cuda_free_calls"] = counts_->free_calls;
    result["backing_live_bytes"] = counts_->backing_live_bytes;
    result["backing_allocate_calls"] = counts_->backing_allocate_calls;
    result["backing_free_calls"] = counts_->backing_free_calls;
    py::list banks;
    for (size_t bank = 0; bank < 2; ++bank) {
      py::dict item;
      item["bank"] = bank;
      item["table_ptr"] = reinterpret_cast<uintptr_t>(tables_[bank].get());
      item["backing_ptr"] = allocators_[bank] ? reinterpret_cast<uintptr_t>(allocators_[bank]->backing->ptr) : uintptr_t{0};
      item["nve_requested_pool_bytes"] = allocators_[bank] ? allocators_[bank]->requested_pool_bytes : size_t{0};
      item["insert_calls"] = insert_calls_[bank];
      item["nve_lease_live"] = allocators_[bank] && allocators_[bank]->leased;
      banks.append(item);
    }
    result["banks"] = banks;
    size_t free = 0, total = 0;
    checked(cudaMemGetInfo(&free, &total));
    result["device_free_bytes"] = free;
    result["device_total_bytes"] = total;
    return result;
  }

  void close() {
    if (closed_) return;
    checked(cudaSetDevice(device_));
    checked(cudaDeviceSynchronize());
    // Contexts retain cache references and must be gone before their tables.
    for (auto& context : contexts_) context.reset();
    for (auto& table : tables_) table.reset();
    for (auto& allocator : allocators_) allocator.reset();
    closed_ = true;
  }

 private:
  void ensure_open() const {
    if (closed_) throw std::runtime_error("Experiment is closed");
  }
  std::vector<int64_t> make_keys(size_t bank) const {
    std::vector<int64_t> keys(static_cast<size_t>(warm_rows_));
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = static_cast<int64_t>(2 * i + bank);
    return keys;  // Stable two-way routing; never modulo the active bank count.
  }
  std::vector<float> make_values(size_t bank) const {
    std::vector<float> values(static_cast<size_t>(warm_rows_ * row_bytes_) / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>((bank + 1) * 10000 + i);
    return values;
  }

  size_t bank_bytes_;
  int64_t row_bytes_, warm_rows_;
  int device_;
  bool warmed_{false}, exported_{false}, closed_{false};
  std::shared_ptr<Counts> counts_{std::make_shared<Counts>()};
  // Declaration order also preserves cleanup order if construction throws.
  std::array<std::shared_ptr<LeaseAllocator>, 2> allocators_{};
  std::array<std::shared_ptr<nve::GpuTable<int64_t>>, 2> tables_{};
  std::array<nve::context_ptr_t, 2> contexts_{};
  std::array<size_t, 2> insert_calls_{};
};

}  // namespace

PYBIND11_MODULE(memory_transfer_poc, module) {
  module.doc() = "NVE backing-to-tensor interface spike; no HSTU or FlexKV integration";
  bind_banked_embedding(module);
  py::class_<BankTransfer>(module, "BankTransfer")
      .def(py::init<size_t, int64_t, int64_t, int>(), py::arg("bank_bytes"), py::arg("row_bytes"),
           py::arg("warm_rows"), py::arg("device") = 0)
      .def("warm", &BankTransfer::warm)
      .def("verify_bank", &BankTransfer::verify_bank)
      .def("retire_bank1", &BankTransfer::retire_bank1)
      .def("kv_dlpack", &BankTransfer::kv_dlpack)
      .def("snapshot", &BankTransfer::snapshot)
      .def("close", &BankTransfer::close);
}
