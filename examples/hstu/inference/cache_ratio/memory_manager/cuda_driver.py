# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Lazy adapter for NVIDIA's optional ``cuda.bindings.driver`` package.

Python signatures verified against official cuda-python 12/13 binding sources
and the 13.1.1 API reference; no native execution is implied by this check:
https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings_12/cuda/bindings/driver.pyx
https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings_13/cuda/bindings/driver.pyx
https://nvidia.github.io/cuda-python/cuda-bindings/13.1.1/module/driver.html

All calls return (CUresult, *outputs). cuMemSetAccess takes a descriptor list
and count. cuMemsetD8 takes a byte count. This adapter needs a CUDA-compatible
cuda-bindings installation, CUDA driver with VMM support, and an already current
context on the requested device (for example, initialized by the owning runtime).
It never creates/switches contexts, imports torch, or installs dependencies.
"""

import importlib


class CudaDriver:
    """Single-context Driver API operations; construction is the lazy CUDA boundary.

    The caller must prevent concurrent launches and forbid graph use while
    mappings change. Synchronization covers this same context, not other
    processes/contexts. External VMM allocations are not PyTorch allocator bytes.
    """

    def __init__(self, device):
        try:
            self._api = importlib.import_module("cuda.bindings.driver")
        except ImportError as error:
            raise RuntimeError(
                "Real VMM requires NVIDIA cuda-bindings (cuda.bindings.driver)"
            ) from error
        d = self._api
        self._raw("cuInit", 0)
        (context,) = self._raw("cuCtxGetCurrent")
        if int(context) == 0:
            raise RuntimeError("Real VMM requires an initialized current CUDA context")
        (current_device,) = self._raw("cuCtxGetDevice")
        if int(current_device) != device:
            raise RuntimeError("Current CUDA context is on a different device")
        self._context = int(context)
        self._device = device
        (supported,) = self._raw(
            "cuDeviceGetAttribute",
            d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            current_device,
        )
        if not supported:
            raise RuntimeError("Current CUDA device does not support VMM")
        self._prop = d.CUmemAllocationProp()
        self._prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        self._prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._prop.location.id = device
        self._prop.requestedHandleTypes = (
            d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        )
        self._access = d.CUmemAccessDesc()
        self._access.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access.location.id = device
        self._access.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    def _raw(self, name, *args):
        result = getattr(self._api, name)(*args)
        if result[0] != self._api.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{name} failed: {result[0]}")
        return result[1:]

    def _call(self, name, *args):
        (current,) = self._raw("cuCtxGetCurrent")
        if int(current) != self._context:
            raise RuntimeError(
                "CUDA context changed; return to the owning context before using this pool"
            )
        return self._raw(name, *args)

    def granularity(self):
        (size,) = self._call(
            "cuMemGetAllocationGranularity",
            self._prop,
            self._api.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        return int(size)

    def reserve(self, size, alignment):
        (ptr,) = self._call("cuMemAddressReserve", size, alignment, 0, 0)
        return int(ptr)

    def create(self, size):
        (handle,) = self._call("cuMemCreate", size, self._prop, 0)
        return handle

    def map(self, address, size, handle):
        self._call("cuMemMap", address, size, 0, handle, 0)

    def set_access(self, address, size):
        self._call("cuMemSetAccess", address, size, [self._access], 1)

    def zero(self, address, size):
        self._call("cuMemsetD8", address, 0, size)

    def synchronize(self):
        self._call("cuCtxSynchronize")

    def unmap(self, address, size):
        self._call("cuMemUnmap", address, size)

    def release(self, handle):
        self._call("cuMemRelease", handle)

    def free_address(self, address, size):
        self._call("cuMemAddressFree", address, size)
