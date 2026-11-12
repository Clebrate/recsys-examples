// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// A small, synchronous-control adapter around real NVE GPU/CPU tables. GPU
// execution uses the caller's stream; only explicit lifecycle boundaries wait.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime_api.h>
#include <allocator.hpp>
#include <buffer_wrapper.hpp>
#include <default_allocator.hpp>
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <insert_heuristic.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Consumes the legacy capsules produced by torch.utils.dlpack.to_dlpack. The
// Python wrapper retains its tensors until work is queued on their own stream.
struct ConsumedTensor {
  DLManagedTensor* managed{nullptr};
  explicit ConsumedTensor(const py::capsule& capsule) {
    if (!PyCapsule_IsValid(capsule.ptr(), "dltensor"))
      throw std::invalid_argument("Expected a fresh legacy DLPack capsule");
    auto* value = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) throw py::error_already_set();
    managed = value;
  }
  ~ConsumedTensor() { if (managed && managed->deleter) managed->deleter(managed); }
  ConsumedTensor(const ConsumedTensor&) = delete;
  ConsumedTensor& operator=(const ConsumedTensor&) = delete;
  const DLTensor& tensor() const { return managed->dl_tensor; }
  void* data() const {
    const auto& t = tensor();
    return static_cast<unsigned char*>(t.data) + t.byte_offset;
  }
};

// The allocation itself belongs to a Python tensor (often an entire KV slab).
// NVE only borrows the constructor's first large device allocation from it.
class SpanAllocator final : public nve::Allocator {
 public:
  void* const ptr;
  const size_t bytes;
  const int device;
  size_t requested_bytes{0};
  bool constructing{true}, leased{false};

  SpanAllocator(void* address, size_t size, int device_id)
      : ptr(address), bytes(size), device(device_id), fallback_(nve::GetDefaultAllocator()) {}

  cudaError_t device_allocate(void** output, size_t size, int device_id = -1) noexcept override {
    if (constructing && requested_bytes == 0) {
      if (!size || size > bytes || (device_id >= 0 && device_id != device))
        return cudaErrorInvalidValue;
      *output = ptr;
      requested_bytes = size;
      leased = true;
      return cudaSuccess;
    }
    return fallback_->device_allocate(output, size, device_id);
  }
  cudaError_t device_free(void* address, int device_id = -1) noexcept override {
    if (address == ptr) { leased = false; return cudaSuccess; }
    return fallback_->device_free(address, device_id);
  }
  cudaError_t host_allocate(void** output, size_t size) noexcept override {
    return fallback_->host_allocate(output, size);
  }
  cudaError_t host_free(void* address) noexcept override { return fallback_->host_free(address); }

 private:
  nve::allocator_ptr_t fallback_;
};

using Layer = nve::HierarchicalEmbeddingLayer<int64_t>;

// Ownership order matters: contexts -> layer -> table -> allocator -> owner.
struct Bank {
  py::object owner;
  uintptr_t ptr;
  size_t bytes;
  size_t last_pool_bytes{0};
  uint64_t generation{0}, retire_count{0}, queries{0}, gpu_hits{0}, ps_hits{0}, insert_batches{0};
  std::shared_ptr<SpanAllocator> allocator;
  std::shared_ptr<nve::GpuTable<int64_t>> gpu;
  std::shared_ptr<Layer> layer;
  std::unordered_map<uintptr_t, nve::context_ptr_t> contexts;
  std::unordered_map<uintptr_t, nve::context_ptr_t> probe_contexts;

  Bank(py::object keep_alive, uintptr_t address, size_t size)
      : owner(std::move(keep_alive)), ptr(address), bytes(size) {}

  void wait() {
    for (auto& entry : contexts) entry.second->wait();
    for (auto& entry : probe_contexts) entry.second->wait();
  }
  void release_layer() {
    wait();
    contexts.clear();
    probe_contexts.clear();
    layer.reset();
    gpu.reset();
    if (allocator && allocator->leased)
      throw std::runtime_error("NVE still owns the bank after context/layer/table release");
    allocator.reset();
  }
};

class BankedNVE {
 public:
  BankedNVE(nve::table_ptr_t parameter_server, int64_t row_elements,
            const py::list& spans, int device, int64_t max_batch_keys)
      : ps_(std::move(parameter_server)), width_(row_elements), device_(device),
        max_batch_keys_(max_batch_keys) {
    if (!ps_ || width_ <= 0 || width_ > std::numeric_limits<int64_t>::max() / 4 ||
        max_batch_keys_ <= 0 || max_batch_keys_ > (1 << 20) || spans.size() == 0)
      throw std::invalid_argument("Need a PS, positive row width/batch limit and at least one bank");
    if (ps_->get_device_id() >= 0 || ps_->get_key_size() != sizeof(int64_t) ||
        ps_->get_max_row_size() != width_ * 4 || ps_->get_value_type() != nve::DataType_t::Float32)
      throw std::invalid_argument("This POC requires a CPU NVE PS with int64 keys and Float32 rows");
    check_cuda(cudaSetDevice(device_));
    for (const auto& item : spans) {
      const auto span = py::cast<py::tuple>(item);
      if (span.size() != 3 || span[2].is_none())
        throw std::invalid_argument("Each bank span must be (device_pointer, bytes, owner)");
      const auto ptr = py::cast<uintptr_t>(span[0]);
      const auto bytes = py::cast<size_t>(span[1]);
      if (!ptr || ptr % 16 || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - ptr)
        throw std::invalid_argument("Bank spans must be nonempty, 16-byte aligned and not overflow");
      cudaPointerAttributes attributes{};
      check_cuda(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(ptr)));
      if (attributes.type != cudaMemoryTypeDevice || attributes.device != device_)
        throw std::invalid_argument("Bank span must refer to device memory on the configured GPU");
      for (const auto& bank : banks_) {
        if (ptr < bank->ptr + bank->bytes && bank->ptr < ptr + bytes)
          throw std::invalid_argument("NVE bank spans must not overlap");
      }
      banks_.push_back(std::make_unique<Bank>(py::reinterpret_borrow<py::object>(span[2]), ptr, bytes));
      build_layer(*banks_.back(), true);
    }
  }

  ~BankedNVE() { try { close(); } catch (...) {} }

  // Diagnostic only: GPU membership before a logical lookup populates any bank.
  // Unlike Layer::lookup's final hitmask, this mask excludes CPU PS resolutions.
  // A separate context keeps probe miss counters out of the measured lookup.
  std::vector<bool> probe_bank(size_t bank_id, const py::capsule& keys_capsule,
                              uintptr_t stream_ptr) {
    ensure_open();
    if (bank_id >= banks_.size()) throw std::out_of_range("Invalid bank ID");
    ConsumedTensor keys(keys_capsule);
    const auto& tensor = keys.tensor();
    if (tensor.device.device_type != kDLCUDA || tensor.device.device_id != device_ ||
        tensor.ndim != 1 || tensor.dtype.code != kDLInt || tensor.dtype.bits != 64 ||
        tensor.dtype.lanes != 1 || (tensor.strides && tensor.strides[0] != 1))
      throw std::invalid_argument("Probe keys must be a contiguous CUDA int64 vector");
    const int64_t count = tensor.shape[0];
    if (count < 0 || count > max_batch_keys_)
      throw std::invalid_argument("Split probe batches at max_batch_keys");
    std::vector<bool> hits(static_cast<size_t>(count), false);
    auto& bank = *banks_[bank_id];
    if (!count || !bank.gpu) return hits;
    if (!keys.data() || reinterpret_cast<uintptr_t>(keys.data()) % 8)
      throw std::invalid_argument("Invalid or unaligned probe keys");
    check_cuda(cudaSetDevice(device_));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto& context = bank.probe_contexts[stream_ptr];
    if (!context) context = bank.gpu->create_execution_context(stream, stream, nullptr, nullptr);
    const size_t mask_elements = (static_cast<size_t>(count) + 63) / 64;
    const size_t mask_bytes = mask_elements * sizeof(nve::bitmask64_t);
    auto* mask = static_cast<nve::bitmask64_t*>(context->get_buffer("probe_mask", mask_bytes, false));
    const size_t value_bytes = static_cast<size_t>(count * width_ * 4);
    auto* values = context->get_buffer("probe_values", value_bytes, false);
    check_cuda(cudaMemsetAsync(mask, 0, mask_bytes, stream));
    auto keys_bw = std::make_shared<nve::BufferWrapper<const void>>(
        context, "probe_keys", keys.data(), static_cast<size_t>(count) * sizeof(int64_t));
    auto mask_bw = std::make_shared<nve::BufferWrapper<nve::bitmask64_t>>(
        context, "probe_mask", mask, mask_bytes);
    // Supply scratch values: the SDK's GPU query kernel writes hit payloads even
    // though Table::find documents nullptr as an exists-only optimization.
    auto values_bw = std::make_shared<nve::BufferWrapper<void>>(
        context, "probe_values", values, value_bytes);
    bank.gpu->reset_lookup_counter(context);
    bank.gpu->find(context, count, keys_bw, mask_bw, width_ * 4, values_bw, nullptr);
    std::vector<nve::bitmask64_t> host_mask(mask_elements);
    check_cuda(cudaMemcpyAsync(host_mask.data(), mask, mask_bytes, cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    for (int64_t i = 0; i < count; ++i)
      hits[static_cast<size_t>(i)] = (host_mask[static_cast<size_t>(i) / 64] >> (i % 64)) & 1;
    return hits;
  }

  py::dict lookup_bank(size_t bank_id, const py::capsule& keys_capsule,
                       const py::capsule& output_capsule, uintptr_t stream_ptr,
                       bool populate) {
    ensure_open();
    if (bank_id >= banks_.size()) throw std::out_of_range("Invalid bank ID");
    ConsumedTensor keys(keys_capsule), output(output_capsule);
    const auto& key_tensor = keys.tensor();
    const auto& out_tensor = output.tensor();
    if (key_tensor.device.device_type != kDLCUDA || key_tensor.device.device_id != device_ ||
        key_tensor.ndim != 1 || key_tensor.dtype.code != kDLInt || key_tensor.dtype.bits != 64 ||
        key_tensor.dtype.lanes != 1 || (key_tensor.strides && key_tensor.strides[0] != 1))
      throw std::invalid_argument("Keys must be a contiguous CUDA int64 vector on the configured GPU");
    const int64_t count = key_tensor.shape[0];
    if (count < 0 || count > max_batch_keys_)
      throw std::invalid_argument("Split lookup batches at max_batch_keys");
    if (out_tensor.device.device_type != kDLCUDA || out_tensor.device.device_id != device_ ||
        out_tensor.ndim != 2 || out_tensor.shape[0] != count || out_tensor.shape[1] != width_ ||
        out_tensor.dtype.code != kDLFloat || out_tensor.dtype.bits != 32 || out_tensor.dtype.lanes != 1 ||
        (out_tensor.strides && (out_tensor.strides[0] != width_ || out_tensor.strides[1] != 1)))
      throw std::invalid_argument("Output must be a contiguous CUDA Float32 [num_keys, row_elements] tensor");
    auto& bank = *banks_[bank_id];
    if (!bank.layer) throw std::runtime_error("Bank has no usable NVE layer");
    int64_t gpu_hits = 0, ps_hits = 0;
    if (count) {
      if (!keys.data() || !output.data() || reinterpret_cast<uintptr_t>(keys.data()) % 8 ||
          reinterpret_cast<uintptr_t>(output.data()) % 4)
        throw std::invalid_argument("Invalid or unaligned lookup buffers");
      check_cuda(cudaSetDevice(device_));
      auto& context = bank.contexts[stream_ptr];
      if (!context) {
        const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        context = bank.layer->create_execution_context(stream, stream, nullptr, nullptr);
      }
      float rates[2]{0.0f, 0.0f};
      bank.layer->lookup(context, count, keys.data(), output.data(), width_ * 4, nullptr, nullptr, rates);
      if (!std::isfinite(rates[0]) || !std::isfinite(rates[1]))
        throw std::runtime_error("NVE returned a nonfinite lookup hit rate");
      gpu_hits = bank.gpu ? std::llround(rates[0] * count) : 0;
      ps_hits = std::llround(rates[bank.gpu ? 1 : 0] * count);
      if (gpu_hits < 0 || ps_hits < 0 || gpu_hits + ps_hits != count)
        throw std::runtime_error("NVE PS did not resolve every requested key; no synthetic fallback is enabled");
      // No background insert jobs: the caller's tensors only need their normal
      // current-stream lifetimes. Warm-up and fixed-ratio controls use this same policy.
      if (populate && bank.gpu && gpu_hits < count) {
        bank.layer->insert(context, count, keys.data(), width_ * 4, width_ * 4, output.data(), 0);
        ++bank.insert_batches;
      }
      bank.queries += static_cast<uint64_t>(count);
      bank.gpu_hits += static_cast<uint64_t>(gpu_hits);
      bank.ps_hits += static_cast<uint64_t>(ps_hits);
    }
    py::dict result;
    result["bank"] = bank_id;
    result["queries"] = count;
    result["gpu_hits"] = gpu_hits;
    result["ps_hits"] = ps_hits;
    result["gpu_hit_rate"] = count ? static_cast<double>(gpu_hits) / count : 0.0;
    result["gpu_bank_live"] = static_cast<bool>(bank.gpu);
    return result;
  }

  void retire(const std::vector<size_t>& ids) {
    validate_ids(ids, true);
    check_cuda(cudaSetDevice(device_));
    for (const auto id : ids) banks_[id]->wait();
    for (const auto id : ids) {
      auto& bank = *banks_[id];
      bank.release_layer();
      build_layer(bank, false);
      ++bank.retire_count;
    }
  }

  void restore(const std::vector<size_t>& ids) {
    validate_ids(ids, false);
    check_cuda(cudaSetDevice(device_));
    for (const auto id : ids) banks_[id]->wait();
    for (const auto id : ids) {
      auto& bank = *banks_[id];
      bank.release_layer();
      build_layer(bank, true);
    }
  }

  void wait() {
    ensure_open();
    check_cuda(cudaSetDevice(device_));
    for (auto& bank : banks_) bank->wait();
  }

  // State preparation only: empty GPU rows without rebuilding table/context
  // scratch buffers, and never clear the shared CPU parameter server.
  void clear_gpu_cache(uintptr_t stream_ptr) {
    ensure_open();
    check_cuda(cudaSetDevice(device_));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    for (auto& bank : banks_) {
      if (!bank->gpu) continue;
      bank->wait();
      auto& context = bank->probe_contexts[stream_ptr];
      if (!context) context = bank->gpu->create_execution_context(stream, stream, nullptr, nullptr);
      bank->gpu->clear(context);
    }
    check_cuda(cudaStreamSynchronize(stream));
  }

  py::dict stats() const {
    py::dict result;
    result["bank_count"] = banks_.size();
    result["row_elements"] = width_;
    result["closed"] = closed_;
    result["population_policy"] = "explicit insert on miss; background auto-insert disabled";
    py::list entries;
    size_t active_bytes = 0, span_bytes = 0;
    for (size_t i = 0; i < banks_.size(); ++i) {
      const auto& bank = *banks_[i];
      py::dict entry;
      entry["bank"] = i;
      entry["span_ptr"] = bank.ptr;
      entry["span_bytes"] = bank.bytes;
      entry["nve_requested_pool_bytes"] = bank.last_pool_bytes;
      entry["table_ptr"] = reinterpret_cast<uintptr_t>(bank.gpu.get());
      entry["live"] = static_cast<bool>(bank.gpu);
      entry["generation"] = bank.generation;
      entry["retire_count"] = bank.retire_count;
      entry["queries"] = bank.queries;
      entry["gpu_hits"] = bank.gpu_hits;
      entry["ps_hits"] = bank.ps_hits;
      entry["insert_batches"] = bank.insert_batches;
      py::dict lookup_context_ptrs;
      for (const auto& context : bank.contexts)
        lookup_context_ptrs[py::int_(context.first)] = reinterpret_cast<uintptr_t>(context.second.get());
      entry["lookup_context_ptrs"] = lookup_context_ptrs;
      entries.append(entry);
      span_bytes += bank.bytes;
      if (bank.gpu) active_bytes += bank.bytes;
    }
    result["banks"] = entries;
    result["active_embedding_span_bytes"] = active_bytes;
    result["total_span_bytes"] = span_bytes;
    return result;
  }

  void close() {
    if (closed_) return;
    check_cuda(cudaSetDevice(device_));
    for (auto& bank : banks_) bank->release_layer();
    for (auto& bank : banks_) bank->owner = py::none();
    ps_.reset();
    closed_ = true;
  }

 private:
  void ensure_open() const {
    if (closed_) throw std::runtime_error("Banked NVE embedding is closed");
  }
  void validate_ids(const std::vector<size_t>& ids, bool expected_live) const {
    ensure_open();
    if (ids.empty()) throw std::invalid_argument("Select at least one bank");
    std::unordered_set<size_t> seen;
    for (const auto id : ids) {
      if (id >= banks_.size() || !seen.insert(id).second)
        throw std::invalid_argument("Invalid or repeated bank ID");
      if (static_cast<bool>(banks_[id]->gpu) != expected_live)
        throw std::runtime_error(expected_live ? "Bank is already retired" : "Bank is already live");
    }
  }
  void build_layer(Bank& bank, bool live) {
    std::vector<nve::table_ptr_t> tables;
    if (live) {
      bank.allocator = std::make_shared<SpanAllocator>(reinterpret_cast<void*>(bank.ptr), bank.bytes, device_);
      nve::GPUTableConfig cfg{};
      cfg.device_id = device_;
      cfg.cache_size = bank.bytes;
      cfg.row_size_in_bytes = width_ * 4;
      cfg.value_dtype = nve::DataType_t::Float32;
      cfg.modify_on_gpu = true;
      cfg.count_misses = true;
      cfg.max_modify_size = max_batch_keys_;
      bank.gpu = std::make_shared<nve::GpuTable<int64_t>>(cfg, bank.allocator);
      bank.allocator->constructing = false;
      if (!bank.allocator->leased)
        throw std::runtime_error("NVE did not acquire the supplied bank span");
      bank.last_pool_bytes = bank.allocator->requested_bytes;
      tables.push_back(bank.gpu);
    }
    tables.push_back(ps_);
    Layer::Config cfg{};
    cfg.layer_name = "memory_transfer_poc";
    cfg.insert_heuristic = std::make_shared<nve::NeverInsertHeuristic>();
    bank.layer = std::make_shared<Layer>(cfg, tables);
    if (live) ++bank.generation;
  }

  nve::table_ptr_t ps_;
  int64_t width_;
  int device_;
  int64_t max_batch_keys_;
  bool closed_{false};
  std::vector<std::unique_ptr<Bank>> banks_;
};

}  // namespace

void bind_banked_embedding(py::module_& module) {
  py::class_<BankedNVE>(module, "BankedNVE")
      .def(py::init<nve::table_ptr_t, int64_t, const py::list&, int, int64_t>(),
           py::arg("parameter_server"), py::arg("row_elements"), py::arg("bank_spans"),
           py::arg("device") = 0, py::arg("max_batch_keys") = 4096)
      .def("lookup_bank", &BankedNVE::lookup_bank, py::arg("bank_id"), py::arg("keys_dlpack"),
           py::arg("output_dlpack"), py::arg("stream_ptr"), py::arg("populate") = true)
      .def("probe_bank", &BankedNVE::probe_bank, py::arg("bank_id"), py::arg("keys_dlpack"),
           py::arg("stream_ptr"))
      .def("clear_gpu_cache", &BankedNVE::clear_gpu_cache, py::arg("stream_ptr"))
      .def("retire", &BankedNVE::retire)
      .def("restore", &BankedNVE::restore)
      .def("wait", &BankedNVE::wait)
      .def("stats", &BankedNVE::stats)
      .def("close", &BankedNVE::close);
}
