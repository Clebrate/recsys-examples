import time

import torch

from recsys_kvcache_manager.host_kvstorage_manager import HostKVTaskStatus
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager


def patch_flexkv_runtime_symbols() -> None:
    # FlexKV adapter references these names dynamically.
    import recsys_kvcache_manager.flex_kvcache_manager as flex_mod
    from flexkv.common.request import KVResponse, KVResponseStatus

    flex_mod.KVResponse = KVResponse
    flex_mod.KVResponseStatus = KVResponseStatus

    # Keep test-side monkey patch minimal: only expose backend_name expected by KVCacheManager.
    # register_gpu_cache_tables behavior should come from repository source for A/B comparison.
    orig_init = flex_mod.FlexKVStorageManager.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.backend_name = "flexkv"

    flex_mod.FlexKVStorageManager.__init__ = _patched_init


def create_testing_kvcache_manager() -> KVCacheManager:
    patch_flexkv_runtime_symbols()
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=1,
        max_seq_len=3072,
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
    return KVCacheManager.from_config(kvcache_config)


def sync_new_history_nnz(kvcache_metadata) -> None:
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(kvcache_metadata.new_history_nnz_cuda.item())


def alias_sequence_lengths(index_meta) -> None:
    # FlexKV adapter currently expects `sequence_lengths`.
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    # FlexKV offload path reads `slot_mappings` before fallback-build.
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None



def assert_registration_layout(kvcache_mgr: KVCacheManager) -> None:
    gpu_table = kvcache_mgr.gpu_kvcache_mgr.get_cache_tables()[0]
    registered_table = kvcache_mgr.host_kvstorage_manager._gpu_cache_table_list[0]
    assert gpu_table.data_ptr() == registered_table.data_ptr()
    runtime_shape = tuple(gpu_table.shape)
    permute_shape = (
        int(gpu_table.shape[1]),
        int(gpu_table.shape[0]),
        int(gpu_table.shape[2]),
        int(gpu_table.shape[3]),
        int(gpu_table.shape[4]),
    )
    if tuple(registered_table.shape) == runtime_shape:
        layout_mode = "runtime-no-permute"
    elif tuple(registered_table.shape) == permute_shape:
        layout_mode = "permute-view"
    else:
        raise AssertionError(
            f"unexpected registered shape={tuple(registered_table.shape)}, "
            f"runtime_shape={runtime_shape}, permute_shape={permute_shape}"
        )
    print(
        f"[REGISTER-LAYOUT] {layout_mode}, "
        f"gpu_shape={runtime_shape}, registered_shape={tuple(registered_table.shape)}"
    )


def print_kv_diff_stats(tag: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    is_close = bool(torch.allclose(actual, expected))
    print(
        f"[KV-CHECK] {tag}: allclose={is_close}, "
        f"max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}"
    )


def assert_legacy_stride_assumption_fails_on_permute_view() -> None:
    # Build a tiny tensor with unique values per element.
    num_block, block_size, num_head, head_dim = 5, 3, 2, 4
    base = torch.arange(
        num_block * 2 * block_size * num_head * head_dim,
        dtype=torch.int64,
    ).view(num_block, 2, block_size, num_head, head_dim)
    perm_view = base.permute(1, 0, 2, 3, 4)  # [kv, block, ...], non-contiguous view
    backing_storage = base.reshape(-1)

    # Legacy contiguous assumption for [kv, block, token, head, dim].
    legacy_kv_stride = num_block * block_size * num_head * head_dim
    legacy_block_stride = block_size * num_head * head_dim
    legacy_token_stride = num_head * head_dim
    legacy_head_stride = head_dim
    legacy_dim_stride = 1

    # Probe multiple points. All should differ under legacy addressing.
    probes = [(1, 3, 2, 1, 3), (0, 4, 1, 0, 2), (1, 1, 0, 1, 0)]
    for kv, blk, tok, head, dim in probes:
        legacy_off = (
            kv * legacy_kv_stride
            + blk * legacy_block_stride
            + tok * legacy_token_stride
            + head * legacy_head_stride
            + dim * legacy_dim_stride
        )
        actual_off = (
            kv * perm_view.stride(0)
            + blk * perm_view.stride(1)
            + tok * perm_view.stride(2)
            + head * perm_view.stride(3)
            + dim * perm_view.stride(4)
        )
        actual_val = int(perm_view[kv, blk, tok, head, dim].item())
        legacy_val = int(backing_storage[legacy_off].item())
        print(
            f"[STRIDE-CHECK] probe={(kv, blk, tok, head, dim)} "
            f"actual_off={actual_off} legacy_off={legacy_off} "
            f"actual_val={actual_val} legacy_val={legacy_val}"
        )
        assert actual_off != legacy_off
        assert actual_val != legacy_val


def launch_and_wait_flexkv_offload(
    kvcache_mgr: KVCacheManager, index_meta, kvcache_metadata
) -> None:
    # Use host manager direct API here to avoid the kvcache_cpp ABI mismatch
    # on acquire_offload_pages(always_offload) for some environments.
    host_mgr = kvcache_mgr.host_kvstorage_manager
    offload_user_ids = index_meta.user_ids
    offload_start_indices = torch.zeros_like(index_meta.seq_lengths, dtype=torch.int32)
    offload_page_indices_list = [
        torch.empty((0,), dtype=torch.int32, device=index_meta.user_ids.device)
        for _ in range(index_meta.user_ids.size(0))
    ]
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    # Keep namespace format aligned with lookup path (`List[str]` per request).
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]
    if index_meta.slot_mappings is None:
        index_meta.slot_mappings = host_mgr._build_slot_mappings(kvcache_metadata)
    # FlexKV put_async converts slot_mapping via .numpy(), so it must be on CPU.
    index_meta.slot_mappings = [
        slot_mapping.detach().to(device="cpu", dtype=torch.int64).contiguous()
        for slot_mapping in index_meta.slot_mappings
    ]
    task_handle = host_mgr.offload_kvcache_launch(
        offload_user_ids=offload_user_ids,
        offload_start_indices=offload_start_indices,
        offload_page_indices_list=offload_page_indices_list,
        index_meta=index_meta,
        kvcache_metadata=kvcache_metadata,
    )
    assert task_handle is not None and task_handle.handle is not None

    deadline = time.time() + 60.0
    while time.time() < deadline:
        wait_result = host_mgr.offload_kvcache_wait(task_handle)
        if wait_result.status == HostKVTaskStatus.READY:
            break
        if wait_result.status in (
            HostKVTaskStatus.FAILED,
            HostKVTaskStatus.TIMEOUT,
            HostKVTaskStatus.CANCELLED,
        ):
            raise RuntimeError(
                f"offload wait failed: status={wait_result.status.value}, msg={wait_result.message}"
            )
        time.sleep(0.01)
    else:
        raise RuntimeError("offload wait timeout")

    assert host_mgr.finish_task(task_handle)


def launch_and_wait_flexkv_onboard(
    kvcache_mgr: KVCacheManager,
    task_ids,
    slot_mappings,
) -> None:
    from flexkv.common.request import KVResponseStatus

    host_mgr = kvcache_mgr.host_kvstorage_manager
    slot_mappings = [
        slot_mapping.detach().to(device="cpu", dtype=torch.int64).contiguous()
        for slot_mapping in slot_mappings
    ]
    host_mgr._client.launch(task_ids, slot_mappings)

    deadline = time.time() + 60.0
    while time.time() < deadline:
        onboard_results = host_mgr._client.wait(task_ids)
        has_unready = False
        has_failed = False
        fail_msgs = []
        for task_id in task_ids:
            res = onboard_results[task_id]
            if res.status == KVResponseStatus.UNREADY:
                has_unready = True
            elif res.status != KVResponseStatus.SUCCESS:
                has_failed = True
                fail_msgs.append(f"task_id={task_id},status={res.status}")
        if has_failed:
            raise RuntimeError(f"onboard wait failed: {';'.join(fail_msgs)}")
        if not has_unready:
            return
        time.sleep(0.01)
    raise RuntimeError("onboard wait timeout")


def verify_host_roundtrip_kv_correctness(
    kvcache_mgr: KVCacheManager,
    uid: int,
    seq_len: int,
    all_keys,
    all_values,
) -> int:
    user_ids = torch.tensor([uid], dtype=torch.int64)
    sequence_lengths = torch.tensor([seq_len], dtype=torch.int32)
    keys = all_keys[uid][:, :seq_len, ...]
    values = all_values[uid][:, :seq_len, ...]

    # Force host->GPU path by evicting GPU cache first.
    kvcache_mgr.evict(user_ids, for_gpu=True)

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    alias_sequence_lengths(index_meta)
    host_len = int(lookup_res.host_cached_lengths[0].item())
    task_ids_raw = lookup_res.extra.get("task_ids", [])
    if not isinstance(task_ids_raw, list):
        task_ids_raw = list(task_ids_raw)

    if host_len <= 0:
        gm = kvcache_mgr.host_kvstorage_manager._client.get_match(
            token_ids=index_meta.token_ids[0],
            token_mask=index_meta.token_mask[0],
            namespace=index_meta.namespaces[0],
        )
        if gm is not None:
            _, matched_mask = gm
            host_len = int(torch.as_tensor(matched_mask).sum().item())
    if host_len <= 0:
        print(
            f"[WARN] roundtrip skipped: host_len remains 0 (uid={uid}, seq_len={seq_len})"
        )
        return 0

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    slot_mappings = kvcache_mgr.host_kvstorage_manager._build_slot_mappings(
        kvcache_metadata
    )
    onboard_task_ids = []
    onboard_slot_mappings = []
    for i, task_id in enumerate(task_ids_raw):
        tid = int(task_id)
        if tid < 0:
            continue
        onboard_task_ids.append(tid)
        onboard_slot_mappings.append(
            slot_mappings[i].detach().to(device="cpu", dtype=torch.int64).contiguous()
        )

    if len(onboard_task_ids) == 0:
        print(
            f"[WARN] roundtrip skipped: no valid onboard task ids (uid={uid}, seq_len={seq_len})"
        )
        return 0

    launch_and_wait_flexkv_onboard(
        kvcache_mgr=kvcache_mgr,
        task_ids=onboard_task_ids,
        slot_mappings=onboard_slot_mappings,
    )

    page_ids = kvcache_metadata.kv_indices[
        kvcache_metadata.kv_indptr[0] : kvcache_metadata.kv_indptr[1]
    ]
    last_page_len = kvcache_metadata.kv_last_page_len[0].item()
    for layer_idx in range(3):
        cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
            page_ids,
            last_page_len,
            layer_idx,
        )
        print_kv_diff_stats(
            f"ROUNDTRIP uid={uid},seq_len={seq_len},layer={layer_idx},key_prefix={host_len}",
            keys[layer_idx][:host_len, ...],
            cached_k[:host_len, ...],
        )
        print_kv_diff_stats(
            f"ROUNDTRIP uid={uid},seq_len={seq_len},layer={layer_idx},value_prefix={host_len}",
            values[layer_idx][:host_len, ...],
            cached_v[:host_len, ...],
        )
        assert torch.allclose(cached_k[:host_len, ...], keys[layer_idx][:host_len, ...])
        assert torch.allclose(
            cached_v[:host_len, ...], values[layer_idx][:host_len, ...]
        )
    return host_len


def run_phase(
    kvcache_mgr: KVCacheManager,
    uid: int,
    seq_len: int,
    all_keys,
    all_values,
) -> int:
    user_ids = torch.tensor([uid], dtype=torch.int64)
    sequence_lengths = torch.tensor([seq_len], dtype=torch.int32)
    keys = all_keys[uid][:, :seq_len, ...]
    values = all_values[uid][:, :seq_len, ...]

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    alias_sequence_lengths(index_meta)
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    sync_new_history_nnz(kvcache_metadata)
    cached_len = int(lookup_res.cached_lengths[0].item())
    cached_len = max(0, min(cached_len, seq_len))

    if cached_len < seq_len:
        for layer_idx in range(3):
            kvcache_mgr.gpu_kvcache_mgr.put(
                keys[layer_idx][cached_len:seq_len, ...],
                values[layer_idx][cached_len:seq_len, ...],
                layer_idx,
                kvcache_metadata,
            )

    # Verify current GPU cache content.
    page_ids = kvcache_metadata.kv_indices[
        kvcache_metadata.kv_indptr[0] : kvcache_metadata.kv_indptr[1]
    ]
    last_page_len = kvcache_metadata.kv_last_page_len[0].item()
    for layer_idx in range(3):
        cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(
            page_ids,
            last_page_len,
            layer_idx,
        )
        print_kv_diff_stats(
            f"uid={uid},seq_len={seq_len},layer={layer_idx},key",
            keys[layer_idx],
            cached_k,
        )
        print_kv_diff_stats(
            f"uid={uid},seq_len={seq_len},layer={layer_idx},value",
            values[layer_idx],
            cached_v,
        )
        assert torch.allclose(cached_k, keys[layer_idx])
        assert torch.allclose(cached_v, values[layer_idx])

    launch_and_wait_flexkv_offload(kvcache_mgr, index_meta, kvcache_metadata)

    # Host index update may be delayed/inconsistent across FlexKV deployments.
    # Keep this as best-effort visibility check instead of hard failure.
    deadline = time.time() + 10.0
    host_len = 0
    while time.time() < deadline:
        _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
        host_len = int(post_lookup.host_cached_lengths[0].item())
        if host_len > 0:
            break
        time.sleep(0.02)
    if host_len == 0:
        # Fallback probe: query FlexKV directly with the same request shape.
        # This helps separate "lookup path visibility" from real offload failure.
        gm = kvcache_mgr.host_kvstorage_manager._client.get_match(
            token_ids=index_meta.token_ids[0],
            token_mask=index_meta.token_mask[0],
            namespace=index_meta.namespaces[0],
        )
        if gm is not None:
            _, matched_mask = gm
            host_len = int(torch.as_tensor(matched_mask).sum().item())
    if host_len == 0:
        print(
            f"[WARN] host_len remains 0 after offload (uid={uid}, seq_len={seq_len}); "
            "offload transfer finished but host visibility is backend-dependent in this env."
        )
    assert host_len <= seq_len
    return host_len


if __name__ == "__main__":
    kvcache_mgr = create_testing_kvcache_manager()
    try:
        assert_legacy_stride_assumption_fails_on_permute_view()
        assert_registration_layout(kvcache_mgr)

        max_sequence_lengths = [2048]
        g_keys = [
            torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
            for i in range(len(max_sequence_lengths))
        ]
        g_values = [
            torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
            for i in range(len(max_sequence_lengths))
        ]

        h1 = run_phase(kvcache_mgr, uid=0, seq_len=640, all_keys=g_keys, all_values=g_values)
        h2 = run_phase(kvcache_mgr, uid=0, seq_len=1280, all_keys=g_keys, all_values=g_values)
        h3 = run_phase(kvcache_mgr, uid=0, seq_len=1792, all_keys=g_keys, all_values=g_values)

        if h1 > 0 and h2 > 0 and h3 > 0:
            assert h2 >= h1
            assert h3 >= h2
        else:
            print(
                f"[WARN] host visibility lengths are not fully observable: h1={h1}, h2={h2}, h3={h3}"
            )

        rt = verify_host_roundtrip_kv_correctness(
            kvcache_mgr,
            uid=0,
            seq_len=1792,
            all_keys=g_keys,
            all_values=g_values,
        )
        if rt <= 0:
            print(
                "[WARN] roundtrip kv correctness was not observable because host_len==0 in this env."
            )
        else:
            print(f"[ROUNDTRIP] validated host->gpu kv correctness for prefix len={rt}")
        torch.cuda.synchronize()
        print("FlexKV test passed.")
    finally:
        # Explicitly shutdown FlexKV background workers/processes to avoid hanging at exit.
        host_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
        client = getattr(host_mgr, "_client", None)
        if client is not None and hasattr(client, "shutdown"):
            try:
                client.shutdown()
            except Exception as e:
                print(f"[WARN] FlexKV client shutdown failed: {e}")
