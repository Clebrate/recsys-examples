#!/usr/bin/env python3
"""Run the FlexKV inference timeline benchmark with cudagraph disabled."""

import argparse
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Set

import torch


class _NoOpNvtxRange:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, fn):
        return fn


def _filter_flexkv_python_nvtx() -> None:
    """Keep transfer direction markers, but hide FlexKV ranges with bad timestamps."""
    try:
        import nvtx
    except ImportError:
        return

    original_annotate = nvtx.annotate
    original_push_range = nvtx.push_range
    original_start_range = nvtx.start_range

    def should_hide(message) -> bool:
        name = str(message or "")
        return (
            name.startswith("transfer scheduler. schedule next ops")
            or name.startswith("transfer scheduler. get new graphs")
            or name.startswith("transfer scheduler. collect finished ops")
            or name.startswith("TransferManagerInter.process_worker.results")
            or name.startswith("schedule D2H op_id")
            or name.startswith("schedule H2DISK op_id")
            or name.startswith("schedule DISK2H op_id")
            or name.startswith("schedule H2D op_id")
        )

    def filtered_annotate(*args, **kwargs):
        message = kwargs.get("message")
        if message is None and args:
            message = args[0]
        if should_hide(message):
            return _NoOpNvtxRange()
        return original_annotate(*args, **kwargs)

    def filtered_push_range(*args, **kwargs):
        message = kwargs.get("message")
        if message is None and args:
            message = args[0]
        if should_hide(message):
            return None
        return original_push_range(*args, **kwargs)

    def filtered_start_range(*args, **kwargs):
        message = kwargs.get("message")
        if message is None and args:
            message = args[0]
        if should_hide(message):
            return None
        return original_start_range(*args, **kwargs)

    nvtx.annotate = filtered_annotate
    nvtx.push_range = filtered_push_range
    nvtx.start_range = filtered_start_range


_filter_flexkv_python_nvtx()

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "examples" / "hstu"))

from inference.benchmark import inference_benchmark_flexkv as bench


def _write_flexkv_config_for_run(scenarios: Set[str]) -> str:
    output_root = Path(
        os.environ.get(
            "INFERENCE_BENCHMARK_TIMELINE_DIR",
            "/home/scratch.noliu_gpu/inference_benchmark_timelines",
        )
    )
    ssd_root = Path(
        os.environ.get(
            "INFERENCE_BENCHMARK_SSD_ROOT",
            "/home/scratch.noliu_gpu/ssd_cache_inference_benchmark",
        )
    )
    run_id = "scenario" + "_".join(sorted(scenarios))
    run_id = f"{run_id}_{int(time.time() * 1000)}_{os.getpid()}"
    ssd_cache_dir = Path(
        os.environ.get("FLEXKV_SSD_CACHE_DIR", str(ssd_root / run_id))
    )
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    ssd_cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}_flexkv_ssd.yml"
    config_path.write_text(
        "\n".join(
            [
                "# Auto-generated FlexKV tier config for one benchmark run.",
                "cpu_cache_gb: 4.0",
                "ssd_cache_gb: 128.0",
                f"ssd_cache_dir: {ssd_cache_dir}",
                "enable_gds: false",
                "enable_p2p_cpu: false",
                "enable_p2p_ssd: false",
                "enable_3rd_remote: false",
                "",
            ]
        )
    )
    return str(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--append-history-len", type=int, default=None)
    parser.add_argument("--ssd-pressure-users", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-size", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-sleep-s", type=float, default=None)
    args, _ = parser.parse_known_args()

    scenarios = {
        s.strip()
        for s in os.environ.get("FLEXKV_SCENARIOS", "1,2,3").split(",")
        if s.strip()
    }
    config_path = os.environ.get("RECSYS_FLEXKV_CONFIG_PATH")
    if not config_path:
        config_path = _write_flexkv_config_for_run(scenarios)
    timed_iters = (
        args.timed_iters
        if args.timed_iters is not None
        else int(os.environ.get("FLEXKV_TIMED_ITERS", str(bench.BENCHMARK_CONFIG.timed_iters)))
    )
    append_history_len = (
        args.append_history_len
        if args.append_history_len is not None
        else int(
            os.environ.get(
                "FLEXKV_APPEND_HISTORY_LEN",
                str(bench.BENCHMARK_CONFIG.append_history_len),
            )
        )
    )
    ssd_pressure_users = (
        args.ssd_pressure_users
        if args.ssd_pressure_users is not None
        else int(
            os.environ.get(
                "FLEXKV_SSD_PRESSURE_USERS",
                str(bench.BENCHMARK_CONFIG.ssd_pressure_users),
            )
        )
    )
    ssd_pressure_batch_size = (
        args.ssd_pressure_batch_size
        if args.ssd_pressure_batch_size is not None
        else int(
            os.environ.get(
                "FLEXKV_SSD_PRESSURE_BATCH_SIZE",
                str(bench.BENCHMARK_CONFIG.ssd_pressure_batch_size),
            )
        )
    )
    ssd_pressure_batch_sleep_s = (
        args.ssd_pressure_batch_sleep_s
        if args.ssd_pressure_batch_sleep_s is not None
        else float(
            os.environ.get(
                "FLEXKV_SSD_PRESSURE_BATCH_SLEEP_S",
                str(bench.BENCHMARK_CONFIG.ssd_pressure_batch_sleep_s),
            )
        )
    )
    bench.BENCHMARK_CONFIG = replace(
        bench.BENCHMARK_CONFIG,
        disable_cudagraph=True,
        flexkv_config_path=config_path,
        timed_iters=timed_iters,
        append_history_len=append_history_len,
        ssd_pressure_users=ssd_pressure_users,
        ssd_pressure_batch_size=ssd_pressure_batch_size,
        ssd_pressure_batch_sleep_s=ssd_pressure_batch_sleep_s,
    )

    cfg = bench.BENCHMARK_CONFIG
    print(
        "[Wrapper] "
        f"disable_cudagraph={cfg.disable_cudagraph}, "
        f"timed_iters={cfg.timed_iters}, "
        f"append_history_len={cfg.append_history_len}, "
        f"ssd_pressure_users={cfg.ssd_pressure_users}, "
        f"ssd_pressure_batch_size={cfg.ssd_pressure_batch_size}, "
        f"ssd_pressure_batch_sleep_s={cfg.ssd_pressure_batch_sleep_s}, "
        f"config_path={cfg.flexkv_config_path}, "
        f"scenarios={','.join(sorted(scenarios))}",
        flush=True,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    print(
        f"[Config] history_len={history_len}, append_history_len={cfg.append_history_len}, "
        f"num_candidates={cfg.num_candidates}"
    )
    model_predict, page_size, max_seqlen = bench.build_model(cfg, history_len)
    print(f"[Config] page_size={page_size}, max_seqlen={max_seqlen}")

    with torch.inference_mode():
        if "1" in scenarios:
            bench.run_scenario_gpu_hit(
                model_predict=model_predict,
                history_len=history_len,
                append_history_len=cfg.append_history_len,
                num_candidates=cfg.num_candidates,
                max_seqlen=max_seqlen,
                warmup_iters=cfg.warmup_iters,
                timed_iters=cfg.timed_iters,
                offload_wait_timeout_s=cfg.offload_wait_timeout_s,
            )
        if "2" in scenarios:
            bench.run_scenario_gpu_miss_host_hit(
                model_predict=model_predict,
                history_len=history_len,
                append_history_len=cfg.append_history_len,
                num_candidates=cfg.num_candidates,
                max_seqlen=max_seqlen,
                warmup_iters=cfg.warmup_iters,
                timed_iters=cfg.timed_iters,
                offload_wait_timeout_s=cfg.offload_wait_timeout_s,
            )
        if "3" in scenarios:
            bench.run_scenario_gpu_cpu_miss_ssd_hit(
                model_predict=model_predict,
                history_len=history_len,
                append_history_len=cfg.append_history_len,
                num_candidates=cfg.num_candidates,
                max_seqlen=max_seqlen,
                page_size=page_size,
                timed_iters=cfg.timed_iters,
                offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                ssd_pressure_users=cfg.ssd_pressure_users,
                ssd_pressure_batch_size=cfg.ssd_pressure_batch_size,
                ssd_pressure_batch_sleep_s=cfg.ssd_pressure_batch_sleep_s,
            )


if __name__ == "__main__":
    main()
