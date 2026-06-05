import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch
from recsys_kvcache_manager.flex_kvcache_manager import FlexKVStorageManager
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

from test_flexkv import (
    run_phase_1,
    run_phase_2,
    run_phase_3,
    run_phase_4,
    run_phase_5,
)

# After each phase, grep FlexKV stdout for these transfer type names.
# Default FLEXKV_LOG_LEVEL=INFO already prints lines like:
#   "H2DISK transfer request: ... finished transfer data size: ... GB"
STORAGE_TIER_LOG_HINTS = """
=== Expected FlexKV transfer logs by storage tier ===
Local SSD (no GDS):
  offload (put):  D2H  ->  H2DISK
  onboard (get):  DISK2H  ->  H2D
Local SSD + GDS:
  offload (put):  D2H  ->  H2DISK          (put still goes via CPU buffer)
  onboard (get):  DISK2D  (+ H2D for CPU-hit prefix)
Shared P2P SSD:
  remote onboard: PEERSSD2H  ->  H2D
  local node may also print: [ssd_handle_loop] Received task_id=...
=== Phase -> what to look for ===
  Phase 1: first offload -> expect H2DISK (SSD tier engaged)
  Phase 2: GPU hit, onboard skipped -> mostly D2H only on new delta offload
  Phase 3: GPU evicted, onboard from host -> expect DISK2H (or DISK2D if GDS)
  Phase 4: fresh users, full offload -> H2DISK again
  Phase 5: partial onboard -> DISK2H/DISK2D on prefix + D2H on delta
Tip: export FLEXKV_LOG_LEVEL=INFO  (default)
     export FLEXKV_ENABLE_TRACE=1 FLEXKV_TRACE_FILE_PATH=./flexkv_trace.log  (optional JSON trace)
"""


def _default_model_kwargs() -> Dict[str, Any]:
    return {
        "num_layers": 3,
        "num_heads": 4,
        "head_dim": 128,
        "page_size": 32,
    }


def _get_flexkv_handles(
    kvcache_mgr: KVCacheManager,
) -> Tuple[FlexKVStorageManager, Any]:
    host_mgr = kvcache_mgr.host_kvstorage_manager
    assert isinstance(host_mgr, FlexKVStorageManager), (
        f"expected FlexKVStorageManager, got {type(host_mgr)}"
    )
    client = host_mgr._client
    assert client is not None, "FlexKV client was not initialized"
    return host_mgr, client


def create_ssd_testing_kvcache_manager(
    ssd_cache_gb: float = 2.0,
    enable_gds: bool = False,
    extra_overrides: Optional[Dict[str, Any]] = None,
    ssd_cache_dir: Optional[str] = None,
) -> Tuple[KVCacheManager, str]:
    if ssd_cache_dir is None:
        ssd_cache_dir = os.getenv("FLEXKV_TEST_SSD_DIR")
    if not ssd_cache_dir:
        ssd_cache_dir = tempfile.mkdtemp(prefix="flexkv_ssd_test_")
    else:
        os.makedirs(ssd_cache_dir, exist_ok=True)

    extra_configs = {
        "flexkv_mode": "direct",
        "flexkv_host_kvstorage_fail_policy": "fail_open",
        "flexkv_enable_mps": 0,
        "flexkv_ssd_cache_gb": float(ssd_cache_gb),
        "flexkv_ssd_cache_dir": ssd_cache_dir,
        "flexkv_enable_gds": 1 if enable_gds else 0,
    }
    if extra_overrides:
        extra_configs.update(extra_overrides)

    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=8,
        max_seq_len=2048,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs=extra_configs,
    )
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    return kvcache_mgr, ssd_cache_dir


def assert_ssd_cache_config(
    kvcache_mgr: KVCacheManager,
    *,
    expected_gds: bool,
    expected_p2p_cpu: bool = False,
    expected_p2p_ssd: bool = False,
    expected_ssd_dir: Optional[str] = None,
) -> None:
    host_mgr, client = _get_flexkv_handles(kvcache_mgr)
    cache_cfg = client.cache_config

    assert host_mgr.ssd_cache_gb > 0
    assert cache_cfg.enable_ssd, "CacheConfig.enable_ssd should be True"
    assert cache_cfg.num_ssd_blocks > 0, "CacheConfig.num_ssd_blocks should be > 0"
    assert cache_cfg.enable_gds == expected_gds
    assert cache_cfg.enable_p2p_cpu == expected_p2p_cpu
    assert cache_cfg.enable_p2p_ssd == expected_p2p_ssd
    assert cache_cfg.enable_kv_sharing == (expected_p2p_cpu or expected_p2p_ssd)

    if expected_ssd_dir is not None:
        assert cache_cfg.ssd_cache_dir == expected_ssd_dir

    cache_engine = client.kv_task_engine.cache_engine
    assert cache_engine.ssd_cache_engine is not None, (
        "GlobalCacheEngine.ssd_cache_engine should be created when SSD is enabled"
    )


def assert_ssd_files_exist(ssd_cache_dir: str) -> None:
    assert os.path.isdir(ssd_cache_dir), f"SSD dir missing: {ssd_cache_dir}"
    files = [
        name
        for name in os.listdir(ssd_cache_dir)
        if os.path.isfile(os.path.join(ssd_cache_dir, name))
    ]
    assert files, f"expected SSD backing files under {ssd_cache_dir}, got none"


def shutdown_flexkv_client(kvcache_mgr: KVCacheManager) -> None:
    host_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    client = getattr(host_mgr, "_client", None)
    if client is not None and hasattr(client, "shutdown"):
        try:
            client.shutdown()
        except Exception as exc:
            print(f"[WARN] FlexKV client shutdown failed: {exc}")


def test_enable_gds_requires_ssd() -> None:
    mgr = FlexKVStorageManager(
        **_default_model_kwargs(),
        ssd_cache_gb=0.0,
        enable_gds=True,
    )
    try:
        mgr._init_client()
        raise AssertionError("expected ValueError when enable_gds=True without SSD")
    except ValueError as exc:
        assert "flexkv_enable_gds requires flexkv_ssd_cache_gb > 0" in str(exc)


def test_enable_p2p_ssd_requires_ssd() -> None:
    mgr = FlexKVStorageManager(
        **_default_model_kwargs(),
        ssd_cache_gb=0.0,
        enable_p2p_ssd=True,
    )
    try:
        mgr._init_client()
        raise AssertionError("expected ValueError when enable_p2p_ssd=True without SSD")
    except ValueError as exc:
        assert "flexkv_enable_p2p_ssd requires flexkv_ssd_cache_gb > 0" in str(exc)


def test_gds_and_p2p_are_mutually_exclusive() -> None:
    mgr = FlexKVStorageManager(
        **_default_model_kwargs(),
        ssd_cache_gb=2.0,
        enable_gds=True,
        enable_p2p_ssd=True,
    )
    try:
        mgr._init_client()
        raise AssertionError("expected ValueError when GDS and P2P are both enabled")
    except ValueError as exc:
        assert "enable_gds and p2p sharing cannot be used together" in str(exc)


def test_local_ssd_config_and_files() -> None:
    kvcache_mgr = None
    ssd_cache_dir = None
    try:
        kvcache_mgr, ssd_cache_dir = create_ssd_testing_kvcache_manager(
            ssd_cache_gb=2.0,
            enable_gds=False,
        )
        assert_ssd_cache_config(
            kvcache_mgr,
            expected_gds=False,
            expected_ssd_dir=ssd_cache_dir,
        )
        assert_ssd_files_exist(ssd_cache_dir)
        print("[TEST] local SSD config + backing files ... Passed.")
    finally:
        if kvcache_mgr is not None:
            shutdown_flexkv_client(kvcache_mgr)
        if ssd_cache_dir and os.path.isdir(ssd_cache_dir):
            shutil.rmtree(ssd_cache_dir, ignore_errors=True)


def _make_phase_test_tensors():
    max_sequence_lengths = [
        2500,
        1024,
        1280,
        2377,
        1854,
        1365,
        2729,
        2049,
        689,
        1417,
        1174,
        1987,
        596,
        520,
        1538,
        1189,
    ]
    g_keys = [
        torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
        for i in range(len(max_sequence_lengths))
    ]
    g_values = [
        torch.randn((3, max_sequence_lengths[i], 4, 128), dtype=torch.bfloat16).cuda()
        for i in range(len(max_sequence_lengths))
    ]
    return g_keys, g_values


def test_local_ssd_functional_phases() -> None:
    """Run test_flexkv phase 1-5 with SSD enabled; inspect FlexKV logs per phase."""
    kvcache_mgr = None
    ssd_cache_dir = None
    cleanup_ssd_dir = os.getenv("FLEXKV_TEST_SSD_DIR") is None
    try:
        kvcache_mgr, ssd_cache_dir = create_ssd_testing_kvcache_manager(
            ssd_cache_gb=2.0,
            enable_gds=False,
        )
        assert_ssd_cache_config(kvcache_mgr, expected_gds=False)
        g_keys, g_values = _make_phase_test_tensors()

        print("Testing Phase 1: allocate + offload (expect H2DISK in log) ...")
        run_phase_1(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")

        print("Testing Phase 2: partial GPU hit, onboard skipped ...")
        run_phase_2(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")

        print("Testing Phase 3: GPU evict + onboard (expect DISK2H in log) ...")
        run_phase_3(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")

        print("Testing Phase 4: new users, full offload (expect H2DISK in log) ...")
        run_phase_4(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")

        print("Testing Phase 5: partial onboard + offload ...")
        run_phase_5(kvcache_mgr, g_keys, g_values)
        print("                 ... Passed.")
        print("[TEST] local SSD functional phases 1-5 ... Passed.")
    finally:
        if kvcache_mgr is not None:
            shutdown_flexkv_client(kvcache_mgr)
        if cleanup_ssd_dir and ssd_cache_dir and os.path.isdir(ssd_cache_dir):
            shutil.rmtree(ssd_cache_dir, ignore_errors=True)


def test_local_ssd_gds_config_optional() -> None:
    try:
        from flexkv.c_ext import transfer_kv_blocks_gds  # noqa: F401
    except Exception:
        print(
            "[SKIP] GDS not available in this FlexKV build "
            "(FLEXKV_ENABLE_GDS=1 required); skipping GDS config test."
        )
        return

    kvcache_mgr = None
    ssd_cache_dir = None
    try:
        kvcache_mgr, ssd_cache_dir = create_ssd_testing_kvcache_manager(
            ssd_cache_gb=2.0,
            enable_gds=True,
        )
        assert_ssd_cache_config(
            kvcache_mgr,
            expected_gds=True,
            expected_ssd_dir=ssd_cache_dir,
        )
        assert_ssd_files_exist(ssd_cache_dir)
        print("[TEST] local SSD + GDS config ... Passed.")
    finally:
        if kvcache_mgr is not None:
            shutdown_flexkv_client(kvcache_mgr)
        if ssd_cache_dir and os.path.isdir(ssd_cache_dir):
            shutil.rmtree(ssd_cache_dir, ignore_errors=True)


def test_p2p_config_wiring_optional() -> None:
    if os.getenv("FLEXKV_TEST_P2P", "0") not in {"1", "true", "yes", "on"}:
        print(
            "[SKIP] P2P integration needs Redis and FLEXKV_ENABLE_P2P=1. "
            "Set FLEXKV_TEST_P2P=1 to run shared SSD wiring test."
        )
        return

    kvcache_mgr = None
    ssd_cache_dir = None
    mooncake_config_path = None
    try:
        mooncake_fd, mooncake_config_path = tempfile.mkstemp(
            suffix=".json",
            prefix="flexkv_mooncake_test_",
        )
        os.close(mooncake_fd)
        with open(mooncake_config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                "{\n"
                f'  "engine_ip": "{os.getenv("FLEXKV_TEST_LOCAL_IP", "127.0.0.1")}",\n'
                '  "engine_port": 5555,\n'
                '  "metadata_backend": "redis",\n'
                f'  "metadata_server": "redis://{os.getenv("FLEXKV_TEST_REDIS_HOST", "127.0.0.1")}:{os.getenv("FLEXKV_TEST_REDIS_PORT", "6379")}",\n'
                '  "metadata_server_auth": "",\n'
                '  "protocol": "tcp",\n'
                '  "device_name": ""\n'
                "}\n"
            )

        kvcache_mgr, ssd_cache_dir = create_ssd_testing_kvcache_manager(
            ssd_cache_gb=2.0,
            enable_gds=False,
            extra_overrides={
                "flexkv_enable_p2p_ssd": 1,
                "flexkv_redis_host": os.getenv("FLEXKV_TEST_REDIS_HOST", "127.0.0.1"),
                "flexkv_redis_port": int(os.getenv("FLEXKV_TEST_REDIS_PORT", "6379")),
                "flexkv_local_ip": os.getenv("FLEXKV_TEST_LOCAL_IP", "127.0.0.1"),
                "flexkv_local_zmq_ip": os.getenv("FLEXKV_TEST_LOCAL_IP", "127.0.0.1"),
                "flexkv_local_zmq_port": int(os.getenv("FLEXKV_TEST_ZMQ_PORT", "5555")),
                "flexkv_mooncake_config_path": mooncake_config_path,
            },
        )
        assert_ssd_cache_config(
            kvcache_mgr,
            expected_gds=False,
            expected_p2p_ssd=True,
            expected_ssd_dir=ssd_cache_dir,
        )
        _, client = _get_flexkv_handles(kvcache_mgr)
        assert client.cache_config.mooncake_config_path == mooncake_config_path
        print("[TEST] shared P2P SSD config wiring ... Passed.")
    finally:
        if kvcache_mgr is not None:
            shutdown_flexkv_client(kvcache_mgr)
        if ssd_cache_dir and os.path.isdir(ssd_cache_dir):
            shutil.rmtree(ssd_cache_dir, ignore_errors=True)
        if mooncake_config_path and os.path.isfile(mooncake_config_path):
            os.remove(mooncake_config_path)


if __name__ == "__main__":
    print(STORAGE_TIER_LOG_HINTS)

    print("Testing SSD config validation ...")
    test_enable_gds_requires_ssd()
    test_enable_p2p_ssd_requires_ssd()
    test_gds_and_p2p_are_mutually_exclusive()
    print("                 ... Passed.")

    print("Testing local SSD config + backing files ...")
    test_local_ssd_config_and_files()

    print("Testing local SSD functional phases 1-5 (check FlexKV logs above) ...")
    test_local_ssd_functional_phases()

    print("Testing local SSD + GDS config (optional) ...")
    test_local_ssd_gds_config_optional()

    print("Testing shared P2P SSD config wiring (optional) ...")
    test_p2p_config_wiring_optional()

    print("All FlexKV SSD tests completed.")
