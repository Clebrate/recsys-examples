"""
Shared NVTX hooks, KVCacheManager test setup, and 3-step pipeline helpers.

Used by:
  - test_flexkv_profile_fine.py  (nsys → profiler_result five-level breakdown)
  - single_try_wait_profiling.py  (single offload_try_wait NVTX)

Entry point for full profiling: run_full_profiling.sh → test_flexkv_profile_fine.py
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

from recsys_kvcache_manager.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager
from recsys_kvcache_manager.kvcache_utils import KVLookupResult

PROFILE_MODE_FINE = "flexkv_fine"

CURRENT_NVTX_SCOPE: ContextVar[str] = ContextVar("current_nvtx_scope", default="")


@contextmanager
def nvtx_range(name: str):
    scope_token = None
    if name.startswith("step") or name.startswith("init."):
        scope_token = CURRENT_NVTX_SCOPE.set(name)
    torch.cuda.nvtx.range_push(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.nvtx.range_pop()
        print(f"[NVTX] {name:<44} {elapsed_ms:9.3f} ms")
        if scope_token is not None:
            CURRENT_NVTX_SCOPE.reset(scope_token)


def wrap_method_with_nvtx(obj, method_name: str, nvtx_name: str) -> None:
    if obj is None or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        scope = CURRENT_NVTX_SCOPE.get()
        scoped_name = f"{scope}::{nvtx_name}" if scope else nvtx_name
        with nvtx_range(scoped_name):
            return original(*args, **kwargs)

    wrapped.__nvtx_wrapped__ = True
    setattr(obj, method_name, wrapped)


def _install_precreate_nvtx_hooks() -> None:
    from recsys_kvcache_manager.flex_kvcache_manager import FlexKVStorageManager

    wrap_method_with_nvtx(
        FlexKVStorageManager,
        "register_gpu_cache_tables",
        "flexkv.register_gpu_cache_tables",
    )


def install_nvtx_hooks(kvcache_mgr: KVCacheManager) -> None:
    """flexkv_fine: FlexKV host APIs, adapter/client RPC, recsys glue."""
    flexkv_mgr = kvcache_mgr.host_kvstorage_manager
    wrap_method_with_nvtx(flexkv_mgr, "build_index_meta", "flexkv.build_index_meta")
    wrap_method_with_nvtx(flexkv_mgr, "lookup_kvcache", "flexkv.lookup_kvcache")
    wrap_method_with_nvtx(
        flexkv_mgr, "onboard_kvcache_launch", "flexkv.onboard_kvcache_launch"
    )
    wrap_method_with_nvtx(
        flexkv_mgr, "onboard_kvcache_wait", "flexkv.onboard_kvcache_wait"
    )
    wrap_method_with_nvtx(
        flexkv_mgr, "offload_kvcache_launch", "flexkv.offload_kvcache_launch"
    )
    wrap_method_with_nvtx(
        flexkv_mgr, "offload_kvcache_wait", "flexkv.offload_kvcache_wait"
    )
    wrap_method_with_nvtx(flexkv_mgr, "finish_task", "flexkv.finish_task")
    wrap_method_with_nvtx(flexkv_mgr, "cancel_task", "flexkv.cancel_task")

    adapter = getattr(flexkv_mgr, "_adapter", None)
    wrap_method_with_nvtx(
        adapter, "to_get_match_requests", "flexkv.adapter.to_get_match_requests"
    )
    wrap_method_with_nvtx(
        flexkv_mgr, "_build_slot_mappings", "flexkv._build_slot_mappings"
    )

    client = getattr(flexkv_mgr, "_client", None)
    wrap_method_with_nvtx(client, "get_match", "flexkv.client.get_match")
    wrap_method_with_nvtx(client, "put_async", "flexkv.client.put_async")
    wrap_method_with_nvtx(client, "launch", "flexkv.client.launch")
    wrap_method_with_nvtx(client, "try_wait", "flexkv.client.try_wait")
    wrap_method_with_nvtx(client, "wait", "flexkv.client.wait")

    install_recsys_glue_hooks(kvcache_mgr)


def install_recsys_glue_hooks(kvcache_mgr: KVCacheManager) -> None:
    if getattr(KVLookupResult.merge, "__nvtx_wrapped__", False):
        return
    original_merge = KVLookupResult.merge

    @classmethod
    @wraps(original_merge)
    def merge_with_nvtx(cls, lookup_res1, lookup_res2):
        scope = CURRENT_NVTX_SCOPE.get()
        nvtx_name = "recsys.merge_lookup_results"
        scoped_name = f"{scope}::{nvtx_name}" if scope else nvtx_name
        with nvtx_range(scoped_name):
            return original_merge(lookup_res1, lookup_res2)

    merge_with_nvtx.__nvtx_wrapped__ = True
    KVLookupResult.merge = merge_with_nvtx

    wrap_method_with_nvtx(
        kvcache_mgr, "offload_try_wait", "recsys.offload_try_wait_loop"
    )


def create_testing_kvcache_manager(
    max_batch_size: int,
    max_seq_len: int,
    profile_mode: str = PROFILE_MODE_FINE,
) -> KVCacheManager:
    if profile_mode != PROFILE_MODE_FINE:
        raise ValueError(
            f"Only profile_mode={PROFILE_MODE_FINE!r} is supported; got {profile_mode!r}"
        )
    _install_precreate_nvtx_hooks()
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=max_seq_len * max_batch_size * 32 * 4 * 128 * 2,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs={
            "flexkv_mode": "direct",
            "flexkv_host_kvstorage_fail_policy": "fail_open",
            "flexkv_enable_mps": 0,
        },
    )
    gpu_gib = (
        kvcache_config.num_layers
        * kvcache_config.num_primary_cache_pages
        * kvcache_config.page_size
        * 2
        * kvcache_config.num_heads
        * kvcache_config.head_dim
        * 2
    ) / (1024.0**3)
    host_gib = (
        kvcache_config.num_layers * kvcache_config.host_capacity_per_layer
    ) / (1024.0**3)
    print(f"[DEBUG] KVCache GPU Memory Usage: {gpu_gib:.3f} GiB")
    print(f"[DEBUG] KVCache Host Memory Usage: {host_gib:.3f} GiB")
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    install_nvtx_hooks(kvcache_mgr)
    return kvcache_mgr


def normalize_index_meta(index_meta) -> None:
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]


def build_uniform_batch(all_keys, all_values, len_per_seq: int, batch_size: int):
    seqlen = [len_per_seq] * batch_size
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    keys = [
        all_keys[uid][:, : seqlen[i], ...]
        for i, uid in enumerate(range(batch_size))
    ]
    values = [
        all_values[uid][:, : seqlen[i], ...]
        for i, uid in enumerate(range(batch_size))
    ]
    return user_ids, sequence_lengths, keys, values, seqlen


def build_uniform_request(len_per_seq: int, batch_size: int):
    seqlen = [len_per_seq] * batch_size
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlen, dtype=torch.int32)
    return user_ids, sequence_lengths


def run_step_1_offload(
    kvcache_mgr: KVCacheManager,
    all_keys,
    all_values,
    len_per_seq: int = 1024,
    batch_size: int = 1,
    mark_input_nvtx: bool = False,
) -> None:
    step_name = "step1"
    if mark_input_nvtx:
        with nvtx_range(f"{step_name}.input"):
            user_ids, sequence_lengths, keys, values, _ = build_uniform_batch(
                all_keys=all_keys,
                all_values=all_values,
                len_per_seq=len_per_seq,
                batch_size=batch_size,
            )
    else:
        user_ids, sequence_lengths, keys, values, _ = build_uniform_batch(
            all_keys=all_keys,
            all_values=all_values,
            len_per_seq=len_per_seq,
            batch_size=batch_size,
        )

    with nvtx_range(f"{step_name}.lookup"):
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)
    assert torch.allclose(
        lookup_res.cached_lengths, torch.zeros((batch_size,), dtype=torch.int32)
    )

    with nvtx_range(f"{step_name}.allocate"):
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    assert torch.allclose(kvcache_metadata.total_history_lengths, sequence_lengths.cuda())

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    with nvtx_range(f"{step_name}.offload_launch"):
        task_handle = kvcache_mgr.offload_launch(
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
    if task_handle is None or task_handle.handle is None:
        raise RuntimeError("step1: offload_launch did not return a valid handle")

    with nvtx_range(f"{step_name}.offload_wait"):
        while True:
            kvcache_mgr.offload_try_wait()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break


def run_step_2_evict_gpu(kvcache_mgr: KVCacheManager, batch_size: int = 1) -> None:
    step_name = "step2"
    user_ids = torch.tensor(list(range(batch_size)), dtype=torch.int64)
    with nvtx_range(f"{step_name}.evict_gpu"):
        kvcache_mgr.evict(user_ids, for_gpu=True)


def run_step_3_onboard(
    kvcache_mgr: KVCacheManager,
    len_per_seq: int = 1024,
    batch_size: int = 1,
    mark_input_nvtx: bool = False,
) -> None:
    step_name = "step3"
    if mark_input_nvtx:
        with nvtx_range(f"{step_name}.input"):
            user_ids, sequence_lengths = build_uniform_request(
                len_per_seq=len_per_seq,
                batch_size=batch_size,
            )
    else:
        user_ids, sequence_lengths = build_uniform_request(
            len_per_seq=len_per_seq,
            batch_size=batch_size,
        )

    with nvtx_range(f"{step_name}.lookup"):
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)
    assert torch.allclose(
        lookup_res.gpu_cached_lengths, torch.zeros((batch_size,), dtype=torch.int32)
    ), "step3 requires GPU cache to be evicted before onboard."

    with nvtx_range(f"{step_name}.allocate"):
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        new_history_nnz = int(kvcache_metadata.new_history_nnz_cuda.item())
    else:
        new_history_nnz = int(getattr(kvcache_metadata, "new_history_nnz", -1))
    assert new_history_nnz == 0, (
        "step3 expects same input and full onboard-only recovery; "
        f"new_history_nnz={new_history_nnz}. "
        "Use len_per_seq divisible by 32 (e.g. 1024/2048/4192)."
    )

    with nvtx_range(f"{step_name}.onboard_launch"):
        onboard_handle = kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    assert onboard_handle is not None and onboard_handle.handle is not None, (
        "step3: onboard_launch did not return a valid handle"
    )
    assert onboard_handle.status == HostKVTaskStatus.LAUNCHED, (
        f"step3: onboard status is {onboard_handle.status}, expected LAUNCHED"
    )

    with nvtx_range(f"{step_name}.onboard_wait"):
        deadline = time.time() + 60.0
        onboard_ready = False
        while time.time() < deadline:
            onboard_wait_result = kvcache_mgr.onboard_wait(index_meta, onboard_handle)
            if onboard_wait_result.ready:
                onboard_ready = True
                break
            time.sleep(0.005)
    assert onboard_ready, "step3: onboard_wait did not reach ready=True"
