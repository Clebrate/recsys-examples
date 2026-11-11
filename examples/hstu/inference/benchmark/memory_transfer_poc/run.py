#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise an NVE bank's physical backing transfer to a PyTorch KV tensor.

Build with the NVE headers and libnve-common from the SAME installation:
  cmake -S . -B /tmp/memory-transfer-poc -DNVE_ROOT=/path/to/nve \
      -DNVE_LIB_DIR=/path/to/matching/libraries
  cmake --build /tmp/memory-transfer-poc -j2
  python3 run.py --extension-dir /tmp/memory-transfer-poc --output result.json

NVE_CACHE_LINE_SIZE defaults to `getconf LEVEL1_DCACHE_LINESIZE`, as in NVE's
Linux build. If the library was built with an override, pass the same
-DNVE_CACHE_LINE_SIZE value from its build cache or compile_commands.json.

This is an interface spike, not an HSTU/FlexKV benchmark. Retirement drains
the GPU. The two-bank routing is fixed; retired keys are not redistributed.
"""

import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bank-mib", type=int, default=32)
    parser.add_argument("--row-bytes", type=int, default=512)
    parser.add_argument("--warm-rows", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    for name in ("bank_mib", "row_bytes", "warm_rows", "page_size", "heads", "head_dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))

    import torch

    extension = importlib.import_module("memory_transfer_poc")
    require(torch.cuda.is_available(), "A CUDA GPU is required")
    torch.cuda.set_device(args.device)
    # Complete PyTorch context initialization before the experiment baseline.
    torch.empty(0, device=f"cuda:{args.device}")
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    bank_bytes = args.bank_mib * 1024 * 1024
    kv_block_bytes = 2 * args.page_size * args.heads * args.head_dim * 2
    blocks = bank_bytes // kv_block_bytes
    require(blocks > 0, "One KV block must fit the bank")
    shape = [blocks, 2, args.page_size, args.heads, args.head_dim]
    result = {
        "scope": "NVE external-allocator two-bank to PyTorch DLPack tensor interface spike",
        "hstu_flexkv_integration": False,
        "retirement_mode": "quiescent device drain, bank 1 only",
        "routing": "key % 2 fixed; no remapping or rewarming bank 0",
        "parameters": vars(args) | {"extension_dir": str(args.extension_dir), "output": str(args.output)},
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(args.device),
        "memory_accounting": "tracked_cuda counts external backing plus NVE device scratch requested bytes; torch allocator stats exclude these; cudaMemGetInfo is device-wide",
        "timings_ms": {},
        "snapshots": {},
    }
    spike = None
    kv = None
    try:
        started = time.perf_counter()
        spike = extension.BankTransfer(bank_bytes, args.row_bytes, args.warm_rows, args.device)
        result["timings_ms"]["create_two_banks"] = (time.perf_counter() - started) * 1000
        result["snapshots"]["created"] = spike.snapshot()
        started = time.perf_counter()
        spike.warm()
        result["timings_ms"]["warm_two_banks"] = (time.perf_counter() - started) * 1000
        before = spike.verify_bank(0)
        retired_before = spike.verify_bank(1)
        for check in (before, retired_before):
            require(check["hits"] == args.warm_rows and check["correct_rows"] == args.warm_rows,
                    f"Pre-transfer cache verification failed: {check}")
        result["bank0_before"] = before
        result["bank1_before"] = retired_before
        result["snapshots"]["warmed"] = warmed = spike.snapshot()

        retirement = spike.retire_bank1()
        result["timings_ms"].update({key: value for key, value in retirement.items() if key.endswith("_ms")})
        result["snapshots"]["retired"] = spike.snapshot()
        started = time.perf_counter()
        kv = torch.from_dlpack(spike.kv_dlpack(shape))
        result["timings_ms"]["dlpack_transfer"] = (time.perf_counter() - started) * 1000
        require(kv.data_ptr() == retirement["retired_backing_ptr"], "Transfer allocated a different backing")
        require(kv.dtype == torch.float16 and list(kv.shape) == shape, "Unexpected KV tensor layout")
        require(kv.is_contiguous() and kv.storage_offset() == 0, "KV must use the CUDA allocation base")
        result["kv"] = {
            "shape": shape, "dtype": str(kv.dtype), "data_ptr": kv.data_ptr(),
            "retired_backing_ptr": retirement["retired_backing_ptr"],
            "same_physical_allocation": True, "tensor_bytes": kv.numel() * kv.element_size(),
            "backing_bytes": bank_bytes, "unused_tail_bytes": bank_bytes - kv.numel() * kv.element_size(),
        }
        started = time.perf_counter()
        kv.fill_(0.5)
        torch.cuda.synchronize(args.device)
        result["timings_ms"]["kv_fill_and_sync"] = (time.perf_counter() - started) * 1000
        # Copy for validation rather than allocate another GPU-sized compare tensor.
        require(bool((kv.cpu() == 0.5).all()), "KV fill did not cover the transferred tensor")

        after = spike.verify_bank(0)
        result["bank0_after"] = after
        result["retention_ratio"] = after["correct_rows"] / before["correct_rows"]
        require(after["correct_rows"] == before["correct_rows"] and after["hits"] == before["hits"],
                "Surviving bank lost cache entries or values")
        result["snapshots"]["after_transfer"] = transferred = spike.snapshot()
        for field in ("table_ptr", "backing_ptr", "insert_calls", "nve_requested_pool_bytes"):
            require(warmed["banks"][0][field] == transferred["banks"][0][field],
                    f"Surviving bank changed {field}")
        require(transferred["backing_allocate_calls"] == 2 and transferred["backing_free_calls"] == 0,
                "Physical backing was allocated/freed during ownership transfer")
        require(transferred["backing_live_bytes"] == 2 * bank_bytes, "Physical backing budget changed")
        result["torch_allocator"] = {
            "allocated_bytes": torch.cuda.memory_allocated(args.device),
            "reserved_bytes": torch.cuda.memory_reserved(args.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        }

        started = time.perf_counter()
        spike.close()
        result["timings_ms"]["close_nve"] = (time.perf_counter() - started) * 1000
        result["snapshots"]["nve_closed_tensor_live"] = closed = spike.snapshot()
        require(closed["backing_live_bytes"] == bank_bytes, "DLPack consumer did not retain its backing")
        # The real tensor must remain valid even after every NVE owner is gone.
        kv.fill_(0.25)
        torch.cuda.synchronize(args.device)
        require(bool((kv.cpu() == 0.25).all()), "Tensor lifetime depended on the NVE bank")
        del kv
        kv = None
        gc.collect()
        torch.cuda.synchronize(args.device)
        result["snapshots"]["released"] = released = spike.snapshot()
        require(released["tracked_cuda_live_bytes"] == 0, "Experiment-owned CUDA allocations leaked")
        require(released["backing_free_calls"] == 2, "Backing ownership did not free both allocations")
        result["passed"] = True
    finally:
        if kv is not None:
            del kv
        if spike is not None:
            spike.close()

    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
