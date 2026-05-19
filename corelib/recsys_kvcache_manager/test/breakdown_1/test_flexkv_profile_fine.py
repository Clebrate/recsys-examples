import argparse
from functools import wraps
from typing import Dict

import torch

from test_filex_profiler import (
    CURRENT_NVTX_SCOPE,
    PROFILE_MODE_FINE,
    create_testing_kvcache_manager,
    nvtx_range,
    run_step_1_offload,
    run_step_2_evict_gpu,
    run_step_3_onboard,
)


def _scoped_name(name: str) -> str:
    scope = CURRENT_NVTX_SCOPE.get()
    return f"{scope}::{name}" if scope else name


def wrap_method_with_nvtx_safe(obj, method_name: str, nvtx_name: str) -> None:
    if obj is None or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        with nvtx_range(_scoped_name(nvtx_name)):
            return original(*args, **kwargs)

    wrapped.__nvtx_wrapped__ = True
    try:
        setattr(obj, method_name, wrapped)
    except Exception as e:  # noqa: BLE001
        print(
            f"[WARN] Failed to wrap {obj}.{method_name} with NVTX "
            f"({nvtx_name}): {e}"
        )


def wrap_module_function_with_nvtx_safe(module, func_name: str, nvtx_name: str) -> None:
    if module is None or not hasattr(module, func_name):
        return
    original = getattr(module, func_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        with nvtx_range(_scoped_name(nvtx_name)):
            return original(*args, **kwargs)

    wrapped.__nvtx_wrapped__ = True
    try:
        setattr(module, func_name, wrapped)
    except Exception as e:  # noqa: BLE001
        print(
            f"[WARN] Failed to wrap module function {module}.{func_name} "
            f"with NVTX ({nvtx_name}): {e}"
        )


class _NVTXProxy:
    def __init__(self, target, method_to_nvtx: Dict[str, str]):
        self._target = target
        self._method_to_nvtx = method_to_nvtx
        self._cache = {}

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        if name not in self._method_to_nvtx:
            return attr
        if name in self._cache:
            return self._cache[name]
        nvtx_name = self._method_to_nvtx[name]

        @wraps(attr)
        def wrapped(*args, **kwargs):
            with nvtx_range(_scoped_name(nvtx_name)):
                return attr(*args, **kwargs)

        wrapped.__nvtx_wrapped__ = True
        self._cache[name] = wrapped
        return wrapped


def install_gpu_cpp_kernel_hooks(kvcache_mgr) -> None:
    gpu_mgr = getattr(kvcache_mgr, "gpu_kvcache_mgr", None)
    if gpu_mgr is None:
        return

    # Python-layer entry points in GPU manager.
    wrap_method_with_nvtx_safe(gpu_mgr, "lookup", "gpu.lookup_py")
    wrap_method_with_nvtx_safe(gpu_mgr, "allocate", "gpu.allocate_py")
    wrap_method_with_nvtx_safe(gpu_mgr, "check_for_offload", "gpu.check_for_offload_py")
    wrap_method_with_nvtx_safe(
        gpu_mgr, "acquire_offload_pages", "gpu.acquire_offload_pages_py"
    )
    wrap_method_with_nvtx_safe(
        gpu_mgr, "release_offload_pages", "gpu.release_offload_pages_py"
    )
    wrap_method_with_nvtx_safe(
        gpu_mgr, "revoke_onboard_pages", "gpu.revoke_onboard_pages_py"
    )
    wrap_method_with_nvtx_safe(gpu_mgr, "evict", "gpu.evict_py")
    wrap_method_with_nvtx_safe(gpu_mgr, "evict_all", "gpu.evict_all_py")
    wrap_method_with_nvtx_safe(gpu_mgr, "put", "gpu.put_py")
    wrap_method_with_nvtx_safe(gpu_mgr, "get", "gpu.get_py")

    # C++ pybind entry points in GPU manager implementation.
    # Some pybind objects do not allow setattr on bound methods. We wrap the
    # whole impl_ object with a proxy so GPU manager still calls into C++ via
    # the proxy and emits *_cpp ranges.
    gpu_impl = getattr(gpu_mgr, "impl_", None)
    if gpu_impl is not None:
        gpu_mgr.impl_ = _NVTXProxy(
            gpu_impl,
            {
                "lookup": "gpu.lookup_cpp",
                "allocate": "gpu.allocate_cpp",
                "check_for_offload": "gpu.check_for_offload_cpp",
                "acquire_offload_pages": "gpu.acquire_offload_pages_cpp",
                "release_offload_pages": "gpu.release_offload_pages_cpp",
                "revoke_onboard_pages": "gpu.revoke_onboard_pages_cpp",
                "evict": "gpu.evict_cpp",
                "evict_all": "gpu.evict_all_cpp",
            },
        )

    # Explicit kernel wrapper path used in GPUKVCacheManager.put().
    try:
        import paged_kvcache_ops
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] paged_kvcache_ops import failed, skip kernel hook: {e}")
        return

    wrap_module_function_with_nvtx_safe(
        paged_kvcache_ops, "append_kvcache", "gpu.kernel.append_kvcache"
    )


def install_cpu_cpp_hooks(kvcache_mgr) -> None:
    flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    client = getattr(flexkv_mgr, "_client", None)
    if client is None:
        return

    method_to_nvtx = {
        "get_match": "cpu.get_match_cpp",
        "put_async": "cpu.put_async_cpp",
        "launch": "cpu.launch_cpp",
        "try_wait": "cpu.try_wait_cpp",
        "wait": "cpu.wait_cpp",
        "cancel": "cpu.cancel_cpp",
    }

    # Prefer proxying an internal implementation object if available.
    candidate_impl_attrs = [
        "impl_",
        "_impl",
        "impl",
        "_manager",
        "manager",
        "_kv_manager",
        "kv_manager",
        "_core",
        "core",
    ]
    for attr_name in candidate_impl_attrs:
        inner = getattr(client, attr_name, None)
        if inner is None:
            continue
        if not any(hasattr(inner, m) for m in method_to_nvtx):
            continue
        try:
            setattr(client, attr_name, _NVTXProxy(inner, method_to_nvtx))
            print(f"[INFO] CPU C++ hooks installed on _client.{attr_name}")
            return
        except Exception as e:  # noqa: BLE001
            print(
                f"[WARN] Failed to install CPU C++ hooks on _client.{attr_name}: {e}"
            )

    # Fallback: proxy the client object itself at method-call boundary.
    try:
        flexkv_mgr._client = _NVTXProxy(client, method_to_nvtx)
        print("[INFO] CPU C++ hooks installed on _client boundary")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to install CPU C++ hooks on _client boundary: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-grained FlexKV profiler: "
            "step-level + backend-level + GPU C++/kernel breakdown."
        )
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Max tensor sequence shape used for random input generation",
    )
    parser.add_argument(
        "--len-per-seq",
        type=int,
        default=1024,
        help="Uniform sequence length per request, e.g. 1024/2048/4192",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size, e.g. 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat count for the 3-step flow in one run",
    )
    parser.add_argument(
        "--base-profile-mode",
        choices=[PROFILE_MODE_FINE],
        default=PROFILE_MODE_FINE,
        help="FlexKV fine-grained NVTX hooks (test_filex_profiler) before C++/kernel hooks.",
    )
    parser.add_argument(
        "--mark-input-nvtx",
        action="store_true",
        help="Also mark step1.input and step3.input.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def run_one_case(
    len_per_seq: int,
    max_seq_len: int,
    batch_size: int,
    args: argparse.Namespace,
) -> None:
    with nvtx_range("init.create_kvcache_manager"):
        kvcache_mgr = create_testing_kvcache_manager(
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            profile_mode=args.base_profile_mode,
        )
        install_gpu_cpp_kernel_hooks(kvcache_mgr)
        install_cpu_cpp_hooks(kvcache_mgr)

    try:
        with nvtx_range("init.prepare_inputs"):
            g_keys = [
                torch.randn((3, max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
                for _ in range(batch_size)
            ]
            g_values = [
                torch.randn((3, max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
                for _ in range(batch_size)
            ]

        for repeat_idx in range(args.repeat):
            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step1: input({len_per_seq} x {batch_size}) + lookup/allocate/offload"
            )
            run_step_1_offload(
                kvcache_mgr,
                g_keys,
                g_values,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
                mark_input_nvtx=args.mark_input_nvtx,
            )

            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step2: evict gpu (batch={batch_size})"
            )
            run_step_2_evict_gpu(kvcache_mgr, batch_size=batch_size)

            print(
                f"[RUN] repeat {repeat_idx + 1}/{args.repeat} "
                f"step3: input({len_per_seq} x {batch_size}) + lookup/allocate/onboard"
            )
            run_step_3_onboard(
                kvcache_mgr,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
                mark_input_nvtx=args.mark_input_nvtx,
            )
        print(f"[DONE] fine C++/kernel profile completed. repeat={args.repeat}")
    finally:
        with nvtx_range("init.shutdown"):
            flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
            client = getattr(flexkv_mgr, "_client", None)
            if client is not None and hasattr(client, "shutdown"):
                try:
                    client.shutdown()
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] FlexKV client shutdown failed: {e}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.repeat <= 0:
        raise ValueError("--repeat must be a positive integer")
    if args.len_per_seq <= 0:
        raise ValueError("--len-per-seq must be a positive integer")
    if args.len_per_seq > args.max_seq_len:
        raise ValueError("--max-seq-len must be >= --len-per-seq")

    torch.manual_seed(args.seed)
    print(
        f"[INFO] len_per_seq={args.len_per_seq}, "
        f"batch_size={args.batch_size}, repeat={args.repeat}, "
        f"base_profile_mode={args.base_profile_mode}"
    )
    print(f"[INFO] seqlen={[args.len_per_seq] * args.batch_size}")

    run_one_case(
        len_per_seq=args.len_per_seq,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        args=args,
    )


if __name__ == "__main__":
    main()

