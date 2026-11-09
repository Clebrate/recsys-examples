# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static KV / embedding cache-ratio latency sweep on one GPU.

Edit CONFIG below, then from examples/hstu run:

    python3 -s inference/benchmark/cache_ratio_sweep.py

placement "gpu" splits the budget between GPU KV and GPU embedding.
placement "cpu" splits the same budget between host KV and host embedding.
placement "kv_cpu" puts the KV share on the native host cache and leaves the embedding share on GPU.
placement "kv_flex" puts the KV share on the in-process FlexKV CPU cache and leaves the embedding share on GPU.
placement "gpu_host" splits the GPU budget between GPU KV and GPU embedding, and adds a fixed native host KV cache. Embedding misses still go to the CPU parameter table. A KV miss checks that host cache, then recomputes.
GPU KV in the two host placements is only the in-flight sequence pages. FlexKV SSD stays off.
Each fraction is a new process: NVE cache init fails if a second model is
created in the same process. video_id is the only hierarchical NVE table.
"""

import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import replace

import gin
import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.datasets.hstu_sequence_dataset import HSTUSequenceDataset
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from configs import (
    EmbeddingBackend,
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
)
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from utils import NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr

CONTEXTUAL = []
ITEM = "video_id"
ACTION = "action_weights"
HIDDEN = None
# GPU embedding left in place when the swept embedding cache lives on the host.
# Large enough for NVE to initialize, far below the 4.50 GiB id working set.
CPU_EMB_GPU_FLOOR = 8 * 1024 * 1024


class CONFIG:
    """改这个配置块，然后运行本文件。不要再用单独的 shell 扫测脚本。"""

    # "gpu"：预算在 GPU KV 与 GPU embedding 之间划分。
    # "cpu"：同一份预算在 host KV 与 host embedding 之间划分，GPU 只留当前序列。
    # "kv_cpu"：KV 份额在 native host，embedding 份额留在 GPU，host embedding cache 为 0。
    # "kv_flex"：KV 份额在 FlexKV CPU cache，embedding 份额留在 GPU，SSD 关闭。
    # "kv_flex_gpu"：FlexKV 后端，KV 预算进 GPU pages；CPU 只留 staging（不占 16 GiB 划分），SSD 关闭。
    # miss 不靠 CPU cache，直接重算。embedding 份额留在 GPU，miss 进 CPU 参数表或 Redis。
    # "gpu_host"：GPU 预算仍在 GPU KV 与 GPU embedding 之间划分，另外加一块固定大小的 native host KV。
    # embedding miss 进 CPU 参数表。KV 先查 GPU，再查这块 host cache，都没有才重算。
    # 环境变量 CACHE_RATIO_PLACEMENT 可覆盖。默认保持 gpu，避免改掉正在跑的 native 子进程。
    # CACHE_RATIO_HOST_KV_BYTES 只对 gpu_host 生效，默认 32 GiB。
    # CACHE_RATIO_KV_CACHE_HISTORY 只把最近这么多 item 步写入 native KV；embedding 仍查整段历史。
    placement = os.environ.get("CACHE_RATIO_PLACEMENT", "gpu")
    case = "ordinary"  # ordinary、revisit、replay
    budget_bytes_list = (
        16 * 1024 ** 3,
        32 * 1024 ** 3,
        64 * 1024 ** 3,
    )
    budget_bytes = 16 * 1024 ** 3  # 单档时使用；多档以 budget_bytes_list 为准
    budget_fraction = 0.55
    fractions = tuple(round(i * 0.05, 2) for i in range(21))
    out = ""  # 空字符串则按档写入 cache_ratio_{case}_{placement}_{gib}gib.jsonl
    # 非空时插在 placement 和档位之间，避免覆盖已有 jsonl。
    tag = os.environ.get("CACHE_RATIO_TAG", "")
    # cpu: GPU embedding miss 进本进程 NVHashMap。redis: 进本机 Redis cluster。
    emb_backend = os.environ.get("CACHE_RATIO_EMB_BACKEND", "cpu")
    redis_address = os.environ.get("CACHE_RATIO_REDIS_ADDRESS", "")

    data = "tmp_data/KuaiRand-27K/data/processed_seqs_4_08_to_4_21.csv"
    # 默认仍是 3 层 KuaiRand 推理配置。8 层推理 benchmark 由 CACHE_RATIO_GIN 指定。
    gin_config = os.environ.get(
        "CACHE_RATIO_GIN",
        "inference/configs/kuairand_1k_inference_ranking.gin",
    )
    pool_users = int(os.environ.get("CACHE_RATIO_POOL_USERS", "8000"))
    requests = int(os.environ.get("CACHE_RATIO_REQUESTS", "8000"))
    history_len = int(os.environ.get("CACHE_RATIO_HISTORY", "1024"))
    candidates = int(os.environ.get("CACHE_RATIO_CANDIDATES", "100"))
    min_history = 32
    zipf_alpha = 0.5
    seed = int(os.environ.get("CACHE_RATIO_SEED", "1"))
    # Empty keeps one stream for both passes (running cpuwait_016 jobs).
    timed_seed = os.environ.get("CACHE_RATIO_TIMED_SEED", "").strip()
    cohort_users = 6000
    max_users = 21257
    video_vocab = 0
    action_vocab = 0


def _gib(n):
    return n / (1024 ** 3)


def _page_size():
    major = torch.cuda.get_device_capability()[0]
    return 128 if major >= 10 and major < 12 else 32


def _busy_wait_s(seconds):
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def _kv_wait_s():
    raw = os.environ.get("CACHE_RATIO_KV_WAIT_MS", "").strip()
    if not raw:
        return 0.0
    return max(0.0, float(raw) * 1e-3)


def _split_timing():
    return os.environ.get("CACHE_RATIO_SPLIT_TIMING", "").strip() in (
        "1",
        "true",
        "yes",
    )


def _emb_wait_in_mean():
    return os.environ.get("CACHE_RATIO_EMB_WAIT_IN_MEAN", "").strip() in (
        "1",
        "true",
        "yes",
    )


def _kv_cache_history_items():
    """Item steps written to native KV. 0 keeps the full history."""
    raw = os.environ.get("CACHE_RATIO_KV_CACHE_HISTORY", "").strip()
    if not raw:
        return 0
    return max(0, int(raw))


def _kv_history_tokens(history_items):
    cap = _kv_cache_history_items()
    if cap > 0:
        history_items = min(int(history_items), cap)
    return 2 * int(history_items)


def _kv_token_bytes():
    """bf16 K and V for one token. Item and action are separate tokens."""
    network = NetworkArgs()
    return (
        network.num_layers
        * 2
        * network.num_attention_heads
        * network.kv_channels
        * 2
    )


def _emb_value_bytes():
    return NetworkArgs().hidden_size * 4


def _shape_fields(cfg):
    network = NetworkArgs()
    return {
        "gin_config": cfg.gin_config,
        "num_layers": network.num_layers,
        "num_attention_heads": network.num_attention_heads,
        "hidden_size": network.hidden_size,
        "kv_channels": network.kv_channels,
    }


def _bytes_per_page(num_layers, num_heads, head_dim, page_size):
    per_layer = 2 * page_size * num_heads * head_dim * 2
    return per_layer * num_layers


# FlexKV requires enable_cpu=True. GPU-only placement keeps a tiny CPU pool so
# TransferEngine can start; it is not part of the swept 16 GiB budget.
# T=128 item+action = 256 tokens / page 32 = 8 blocks per request; 8 is the floor.
FLEXKV_GPU_STAGING_CPU_BLOCKS = 8


def _flexkv_staging_cpu_bytes():
    network = NetworkArgs()
    page_size = _page_size()
    return FLEXKV_GPU_STAGING_CPU_BLOCKS * _bytes_per_page(
        network.num_layers,
        network.num_attention_heads,
        network.kv_channels,
        page_size,
    )


def _sequence_kv_bytes(total_max_seqlen):
    """GPU pages for two in-flight sequences. The persistent cache is on the host."""
    network = NetworkArgs()
    page_size = _page_size()
    bytes_per_page = _bytes_per_page(
        network.num_layers,
        network.num_attention_heads,
        network.kv_channels,
        page_size,
    )
    pages = math.ceil(total_max_seqlen / page_size)
    return 2 * pages * bytes_per_page


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _mem_available_bytes():
    available = None
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    return available or 0


def _emb_wait_collection(model):
    sparse = getattr(model, "sparse_module", None)
    return getattr(sparse, "_dynamic_embedding_collection", None)


def _history_lengths(batch):
    item_lengths = batch.features[ITEM].lengths()
    action_lengths = batch.features[ACTION].lengths()
    candidates = batch.num_candidates.to(item_lengths.device)
    history = (item_lengths - candidates) + (action_lengths - candidates)
    history = history.detach().cpu().long()
    cap = _kv_cache_history_items()
    if cap > 0:
        history = torch.clamp(history, max=2 * cap)
    return history


def _suffix_jagged(jt, keep):
    keep = max(0, int(keep))
    values = jt.values()
    total = int(values.shape[0])
    keep = min(keep, total)
    lengths = jt.lengths().clone()
    lengths[0] = keep
    return JaggedTensor(values=values[-keep:] if keep else values[:0], lengths=lengths)


def _keep_last_history(batch, embeddings, item_steps):
    """Keep the last item_steps history tokens plus candidates for HSTU/KV."""
    candidates = int(batch.num_candidates.reshape(-1)[0].item())
    item_jt = batch.features[ITEM]
    action_jt = batch.features[ACTION]
    item_len = int(item_jt.lengths()[0].item())
    action_len = int(action_jt.lengths()[0].item())
    keep_item = min(item_len, item_steps + candidates)
    keep_action = min(action_len, item_steps + candidates)
    item_vals = item_jt.values()[-keep_item:] if keep_item else item_jt.values()[:0]
    action_vals = action_jt.values()[-keep_action:] if keep_action else action_jt.values()[:0]
    new_features = KeyedJaggedTensor(
        values=torch.cat([item_vals, action_vals]),
        lengths=torch.tensor(
            [keep_item, keep_action],
            device=item_jt.lengths().device,
            dtype=item_jt.lengths().dtype,
        ),
        keys=[ITEM, ACTION],
    )
    new_batch = replace(
        batch,
        features=new_features,
        feature_to_max_seqlen={ITEM: keep_item, ACTION: keep_action},
    )
    new_emb = dict(embeddings)
    if ITEM in embeddings:
        new_emb[ITEM] = _suffix_jagged(embeddings[ITEM], keep_item)
    if ACTION in embeddings:
        new_emb[ACTION] = _suffix_jagged(embeddings[ACTION], keep_action)
    return new_batch, new_emb


def _strip_embeddings(embeddings, batch_after):
    """Drop the KV-cached prefix from embeddings already looked up.

    strip_cached_tokens mutates batch.features in place, so do not compare a
    before/after batch. Each embedding still has the unstripped length.
    """
    out = {}
    for name, jt in embeddings.items():
        if name not in batch_after.features.keys():
            out[name] = jt
            continue
        old_len = int(jt.lengths()[0].item())
        new_len = int(batch_after.features[name].lengths()[0].item())
        if new_len > old_len:
            raise RuntimeError(
                f"embedding {name} shorter than striped ids: "
                f"emb_len={old_len} id_len={new_len}"
            )
        n_strip = old_len - new_len
        values = jt.values()
        if int(values.shape[0]) < n_strip + new_len:
            raise RuntimeError(
                f"embedding {name} rows={values.shape[0]} cannot strip "
                f"{n_strip} and keep {new_len}"
            )
        lengths = jt.lengths().clone()
        lengths[0] = new_len
        out[name] = JaggedTensor(
            values=values[n_strip:] if new_len else values[:0],
            lengths=lengths,
        )
    return out


_STRIP_ASSERTS_LEFT = 8


def _assert_embeddings_match_batch(embeddings, batch, where=""):
    global _STRIP_ASSERTS_LEFT
    if _STRIP_ASSERTS_LEFT <= 0:
        return
    _STRIP_ASSERTS_LEFT -= 1
    for name in batch.features.keys():
        if name not in embeddings:
            continue
        feat_sum = int(batch.features[name].lengths().sum().item())
        emb_rows = int(embeddings[name].values().shape[0])
        emb_sum = int(embeddings[name].lengths().sum().item())
        if not (feat_sum == emb_rows == emb_sum):
            raise RuntimeError(
                f"embedding/id length mismatch {where} {name}: "
                f"feat_sum={feat_sum} emb_rows={emb_rows} emb_sum={emb_sum}"
            )


def _self_check_strip_embeddings():
    """Catch the n_strip=0 bug: ids already 100, embeddings still 228."""
    old_len = 228
    new_len = 100
    dim = 8
    values = torch.arange(old_len, dtype=torch.float32).unsqueeze(1).expand(old_len, dim).contiguous()

    class _LenOnly:
        def __init__(self, n):
            self._n = n

        def lengths(self):
            return torch.tensor([self._n], dtype=torch.int64)

    class _Batch:
        def __init__(self, n):
            self.features = {ITEM: _LenOnly(n), ACTION: _LenOnly(n)}

    emb = {
        ITEM: JaggedTensor(
            values=values,
            lengths=torch.tensor([old_len], dtype=torch.int64),
        ),
        ACTION: JaggedTensor(
            values=values.clone(),
            lengths=torch.tensor([old_len], dtype=torch.int64),
        ),
    }
    after = _Batch(new_len)
    out = _strip_embeddings(emb, after)
    _assert_embeddings_match_batch(out, after, where="self_check")
    kept = out[ITEM].values()
    first_kept = float(old_len - new_len)
    if int(kept.shape[0]) != new_len or float(kept[0, 0]) != first_kept:
        raise RuntimeError(
            f"strip self-check failed: rows={tuple(kept.shape)} first={float(kept[0, 0])}"
        )
    print(
        f"strip self-check ok: dropped {old_len - new_len} prefix rows, kept {new_len}",
        flush=True,
    )


def _forward_with_kv_window(model, batch, user_ids, history, item_steps, split_events=None):
    """Embed the full 4K window, then run HSTU/KV on the last item_steps."""
    if split_events is None:
        embeddings = model.sparse_module(batch.features)
        work_batch, work_emb = _keep_last_history(batch, embeddings, item_steps)
        kvcache = model.dense_module.kvcache
        index_meta, lookup_res = kvcache.lookup_kvcache(user_ids, history)
        kvcache_metadata = kvcache.allocate_kvcache(index_meta, lookup_res)
        kvcache.onboard_launch(index_meta, lookup_res, kvcache_metadata)
        striped_batch = model.strip_cached_tokens(work_batch, lookup_res.cached_lengths)
        striped_emb = _strip_embeddings(work_emb, striped_batch)
        _assert_embeddings_match_batch(striped_emb, striped_batch, where="kv_window")
        return model.dense_module.forward_with_kvcache(
            striped_batch,
            striped_emb,
            user_ids,
            history,
            (index_meta, lookup_res, kvcache_metadata),
        )

    ev = split_events
    torch.cuda.nvtx.range_push("cache_ratio.embedding")
    ev[0].record()
    embeddings = model.sparse_module(batch.features)
    ev[1].record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("cache_ratio.kv_lookup")
    kvcache = model.dense_module.kvcache
    index_meta, lookup_res = kvcache.lookup_kvcache(user_ids, history)
    kvcache_metadata = kvcache.allocate_kvcache(index_meta, lookup_res)
    kvcache.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    ev[2].record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("cache_ratio.strip")
    work_batch, work_emb = _keep_last_history(batch, embeddings, item_steps)
    striped_batch = model.strip_cached_tokens(work_batch, lookup_res.cached_lengths)
    striped_emb = _strip_embeddings(work_emb, striped_batch)
    _assert_embeddings_match_batch(striped_emb, striped_batch, where="kv_window_split")
    ev[3].record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("cache_ratio.hstu")
    logits = model.dense_module.forward_with_kvcache(
        striped_batch,
        striped_emb,
        user_ids,
        history,
        (index_meta, lookup_res, kvcache_metadata),
    )
    ev[4].record()
    torch.cuda.nvtx.range_pop()
    return logits


_PLACEMENTS = ("gpu", "cpu", "kv_cpu", "kv_flex", "kv_flex_gpu", "gpu_host")


def _install_flexkv_lookup_replay(model):
    """Replay one FlexKV get_match into the following forward.

    The timed loop looks up once to record hit lengths, then forward looks up
    again. FlexKV get_match allocates a task id, so the second call must reuse
    that result instead of opening another match.
    """
    if getattr(model, "_flexkv_lookup_replay", False):
        return
    kvcache = model.dense_module.kvcache
    host = kvcache.backend.host_kvstorage_manager
    if getattr(host, "backend_name", "") != "flexkv":
        return
    original = kvcache.lookup_kvcache
    model._flexkv_lookup_replay = True
    pending = []

    def lookup_once(user_ids, sequence_lengths):
        if pending:
            return pending.pop()
        result = original(user_ids, sequence_lengths)
        pending.append(result)
        return result

    kvcache.lookup_kvcache = lookup_once


def _mean_ms(values):
    if not values:
        return None
    return sum(values) / len(values)


def _install_kv_cpu_split(model, buckets, phase):
    """Wall-clock KV path. FlexKV get_match is CPU; CUDA events miss it.

    `_kv_counts` does the real lookup before the timed CUDA events, then
    FlexKV replay returns that result in forward. `phase` is "counts" or
    "forward". onboard_wait sits inside dense HSTU, not in split_kv.
    """
    kvcache = model.dense_module.kvcache
    if getattr(kvcache, "_cpu_split_installed", False):
        return

    def wrap(fn, key, timed_only=False):
        def inner(*args, **kwargs):
            if timed_only and phase[0] != "forward":
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                buckets[key].append((time.perf_counter() - start) * 1000.0)

        return inner

    def wrap_lookup(fn):
        def inner(*args, **kwargs):
            key = (
                "lookup_untimed_ms"
                if phase[0] == "counts"
                else "lookup_replay_ms"
            )
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                buckets[key].append((time.perf_counter() - start) * 1000.0)

        return inner

    kvcache.lookup_kvcache = wrap_lookup(kvcache.lookup_kvcache)
    kvcache.allocate_kvcache = wrap(kvcache.allocate_kvcache, "alloc_ms", True)
    kvcache.onboard_launch = wrap(kvcache.onboard_launch, "onboard_launch_ms", True)
    kvcache.onboard_wait = wrap(kvcache.onboard_wait, "onboard_wait_ms", True)
    kvcache._cpu_split_installed = True


def _flexkv_runtime_blocks(model):
    host = model.dense_module.kvcache.backend.host_kvstorage_manager
    manager = getattr(host, "_client", None)
    cache_cfg = None
    for name in ("cache_config", "_cache_config"):
        cache_cfg = getattr(manager, name, None)
        if cache_cfg is not None:
            break
    if cache_cfg is None and manager is not None:
        print(
            "flexkv manager attrs "
            + ",".join(name for name in dir(manager) if "cache" in name.lower() or "ssd" in name.lower()),
            flush=True,
        )
        return None, None, None, None
    cpu_blocks = getattr(cache_cfg, "num_cpu_blocks", None)
    ssd_blocks = getattr(cache_cfg, "num_ssd_blocks", None)
    enable_ssd = getattr(cache_cfg, "enable_ssd", None)
    enable_cpu = getattr(cache_cfg, "enable_cpu", None)
    return cpu_blocks, ssd_blocks, enable_ssd, enable_cpu


def _drain_flexkv_offload(model, timeout_s=180.0):
    kvcache = model.dense_module.kvcache
    host = kvcache.backend.host_kvstorage_manager
    if getattr(host, "backend_name", "") != "flexkv":
        return
    deadline = time.time() + timeout_s
    while len(kvcache.backend.ongoing_offload_tasks) > 0:
        kvcache.offload_try_wait()
        pending = len(kvcache.backend.ongoing_offload_tasks)
        if pending == 0:
            break
        if time.time() > deadline:
            print(
                f"flexkv offload drain timeout pending={pending}",
                flush=True,
            )
            break
        time.sleep(0.001)


def _kv_counts(model, user_ids, history_lengths):
    _, lookup = model.dense_module.kvcache.lookup_kvcache(user_ids, history_lengths)
    denom = int(history_lengths.sum().item())
    gpu = int(lookup.gpu_cached_lengths.sum().item())
    host_tensor = lookup.host_cached_lengths
    host = int(host_tensor.sum().item()) if host_tensor is not None else 0
    merged_tensor = lookup.cached_lengths
    merged = int(merged_tensor.sum().item()) if merged_tensor is not None else max(gpu, host)
    return gpu, host, merged, denom


def build_model(
    emb_configs,
    kv_bytes,
    host_kv_bytes,
    emb_gpu_bytes,
    emb_host_bytes,
    num_contextual,
    total_max_seqlen,
    use_flexkv=False,
    flexkv_cpu_bytes=0,
    flexkv_enable_cpu=True,
):
    network = NetworkArgs()
    ranking = RankingArgs()
    page_size = _page_size()
    bytes_per_page = _bytes_per_page(
        network.num_layers,
        network.num_attention_heads,
        network.kv_channels,
        page_size,
    )
    num_pages = int(kv_bytes // bytes_per_page)
    actual_kv = num_pages * bytes_per_page
    per_layer_page = bytes_per_page // network.num_layers
    host_pages = int(host_kv_bytes // bytes_per_page) if host_kv_bytes else 0
    host_capacity_per_layer = host_pages * per_layer_page
    max_seq = math.ceil(total_max_seqlen / 32) * 32
    kv_max_seq = math.ceil(total_max_seqlen / page_size) * page_size
    hstu_config = get_inference_hstu_config(
        hidden_size=network.hidden_size,
        num_layers=network.num_layers,
        num_attention_heads=network.num_attention_heads,
        head_dim=network.kv_channels,
        max_batch_size=1,
        max_seq_len=max_seq,
        scaling_seqlen=total_max_seqlen,
        dtype=torch.bfloat16,
        position_encoding_config=PositionEncodingConfig(
            num_position_buckets=8192,
            num_time_buckets=2048,
            use_time_encoding=False,
            static_max_seq_len=max_seq,
        ),
        contextual_max_seqlen=num_contextual,
    )
    flexkv_blocks = 0
    flexkv_bytes = 0
    extra_configs = None
    backend = "native"
    offload_timeout_ms = 100.0
    if use_flexkv:
        # 一个 block 覆盖全部 layer，字节数等于一页 GPU KV。
        # CPU 层打开时 0 会落回 FlexKVStorage 的 4096 默认值，所以至少留 1 个 block。
        # GPU-only 时显式 0 并关掉 enable_cpu，避免 miss 从 CPU onboard。
        if flexkv_enable_cpu:
            flexkv_blocks = max(1, int(flexkv_cpu_bytes) // bytes_per_page)
        else:
            flexkv_blocks = 0
        flexkv_bytes = flexkv_blocks * bytes_per_page
        backend = "flexkv"
        offload_timeout_ms = 5000.0
        extra_configs = {
            "flexkv_mode": "direct",
            "flexkv_host_kvstorage_fail_policy": "fail_open",
            "flexkv_enable_mps": 0,
            "flexkv_as_batch": 1,
            "flexkv_num_cpu_blocks": flexkv_blocks,
            "flexkv_num_local_blocks": max(1, num_pages),
            "flexkv_num_tmp_cpu_blocks": FLEXKV_GPU_STAGING_CPU_BLOCKS,
            "flexkv_enable_ssd": False,
            "flexkv_enable_cpu": bool(flexkv_enable_cpu),
        }
    kv_config = get_kvcache_config(
        num_layers=network.num_layers,
        num_heads=network.num_attention_heads,
        head_dim=network.kv_channels,
        page_size=page_size,
        offload_chunksize=1024 if 1024 % page_size == 0 else page_size,
        num_primary_cache_pages=num_pages,
        num_buffer_pages=0,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=1,
        max_seq_len=kv_max_seq,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend=backend,
        offload_timeout_ms=offload_timeout_ms,
        offload_mode="lazy",
        extra_configs=extra_configs,
    )
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking.prediction_head_arch,
        prediction_head_act_type=ranking.prediction_head_act_type,
        prediction_head_bias=ranking.prediction_head_bias,
        num_tasks=ranking.num_tasks,
        eval_metrics=ranking.eval_metrics,
    )
    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_config,
        task_config=task_config,
        use_cudagraph=False,
        embedding_backend=EmbeddingBackend.NVEMB,
        gpu_cache_bytes=int(emb_gpu_bytes),
        host_cache_bytes=int(emb_host_bytes),
    )
    model.bfloat16()
    model.eval()
    runtime_cpu_blocks, runtime_ssd_blocks, runtime_enable_ssd, runtime_enable_cpu = (
        None,
        None,
        None,
        None,
    )
    if use_flexkv:
        runtime_cpu_blocks, runtime_ssd_blocks, runtime_enable_ssd, runtime_enable_cpu = (
            _flexkv_runtime_blocks(model)
        )
        print(
            f"flexkv requested_cpu_blocks={flexkv_blocks} "
            f"runtime_cpu_blocks={runtime_cpu_blocks} runtime_ssd_blocks={runtime_ssd_blocks} "
            f"enable_cpu={runtime_enable_cpu} enable_ssd={runtime_enable_ssd} "
            f"gpu_pages={num_pages}",
            flush=True,
        )
        if runtime_enable_ssd:
            raise RuntimeError(
                "FlexKV SSD cache is enabled; this sweep keeps enable_ssd false"
            )
        if not flexkv_enable_cpu:
            raise RuntimeError("FlexKV requires enable_cpu=True")
    return model, {
        "page_size": page_size,
        "kv_pages": num_pages,
        "kv_bytes": actual_kv,
        "kv_host_bytes": flexkv_bytes if use_flexkv else host_pages * bytes_per_page,
        "kv_backend": backend,
        "flexkv_cpu_blocks": flexkv_blocks,
        "flexkv_cpu_bytes": flexkv_bytes,
        "flexkv_ssd_blocks": int(runtime_ssd_blocks or 0),
        "flexkv_enable_ssd": bool(runtime_enable_ssd),
        "flexkv_enable_cpu": bool(flexkv_enable_cpu) if use_flexkv else None,
        "emb_gpu_bytes": int(emb_gpu_bytes),
        "emb_host_bytes": int(emb_host_bytes),
    }


def _configs(video_vocab, action_vocab, hidden):
    return [
        InferenceEmbeddingConfig(
            feature_names=[ITEM],
            table_name=ITEM,
            vocab_size=video_vocab,
            dim=hidden,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=[ACTION],
            table_name=ACTION,
            vocab_size=action_vocab,
            dim=hidden,
            use_dynamicemb=False,
        ),
    ]


def _release(model):
    try:
        host = model.dense_module.kvcache.backend.host_kvstorage_manager
        client = getattr(host, "_client", None)
        shutdown = getattr(client, "shutdown", None)
        if shutdown is not None:
            shutdown()
    except Exception as exc:
        print(f"kv backend shutdown: {exc}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_pass(model, dataset, timed):
    latencies = []
    gpu_tokens = 0
    host_tokens = 0
    merged_tokens = 0
    hist_tokens = 0
    full_hit = 0
    zero_hit = 0
    partial_hit = 0
    page = _page_size()
    requests = 0
    kv_wait_s = _kv_wait_s()
    kv_wait_anchor = None
    if timed and kv_wait_s > 0:
        kv_wait_anchor = torch.zeros(1, device=torch.cuda.current_device())
    _install_flexkv_lookup_replay(model)
    wait_coll = _emb_wait_collection(model)
    split_on = timed and _split_timing()
    split_emb = []
    split_kv = []
    split_strip = []
    split_hstu = []
    split_events = None
    cpu_split = {
        "lookup_untimed_ms": [],
        "lookup_replay_ms": [],
        "alloc_ms": [],
        "onboard_launch_ms": [],
        "onboard_wait_ms": [],
    }
    cpu_phase = ["counts"]
    if split_on:
        split_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(5)
        ]
        _install_kv_cpu_split(model, cpu_split, cpu_phase)
    wall_start = time.perf_counter()
    if getattr(dataset, "user_ids", None) is not None:
        user_id_values = dataset.user_ids
    else:
        user_id_values = torch.tensor(
            dataset._seq_logs_frame["user_id"].to_numpy(), dtype=torch.long
        )
    for batch in dataset:
        batch = batch.to(device=torch.cuda.current_device())
        user_ids = user_id_values[requests : requests + 1]
        history = _history_lengths(batch)
        cpu_phase[0] = "counts"
        gpu_n, host_n, merged_n, denom_n = _kv_counts(model, user_ids, history)
        gpu_tokens += gpu_n
        host_tokens += host_n
        merged_tokens += merged_n
        hist_tokens += denom_n
        # Host lengths are page-aligned, so a full user can sit just under denom.
        if denom_n <= 0 or merged_n <= 0:
            zero_hit += 1
            kv_miss = True
        elif merged_n >= denom_n or denom_n - merged_n < page:
            full_hit += 1
            kv_miss = False
        else:
            partial_hit += 1
            kv_miss = True
        if timed:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if kv_wait_s > 0 and kv_miss:
                kv_wait_anchor.add_(1)
                torch.cuda.synchronize()
                _busy_wait_s(kv_wait_s)
        item_steps = _kv_cache_history_items()
        cpu_phase[0] = "forward"
        if item_steps > 0:
            _forward_with_kv_window(
                model,
                batch,
                user_ids,
                history,
                item_steps,
                split_events=split_events if timed else None,
            )
        else:
            model.forward_with_kvcache(batch, user_ids, history)
        n_miss = 0
        if timed:
            end.record()
            torch.cuda.synchronize()
            gpu_ms = start.elapsed_time(end)
            if split_events is not None:
                split_emb.append(split_events[0].elapsed_time(split_events[1]))
                split_kv.append(split_events[1].elapsed_time(split_events[2]))
                split_strip.append(split_events[2].elapsed_time(split_events[3]))
                split_hstu.append(split_events[3].elapsed_time(split_events[4]))
            if wait_coll is not None and hasattr(wait_coll, "flush_emb_wait"):
                n_miss = wait_coll.flush_emb_wait() or 0
            extra_ms = 0.0
            if _emb_wait_in_mean():
                wait_us = float(os.environ.get("CACHE_RATIO_EMB_WAIT_US", "0") or 0)
                extra_ms = float(n_miss) * wait_us * 1e-3
            latencies.append(gpu_ms + extra_ms)
        elif wait_coll is not None and hasattr(wait_coll, "flush_emb_wait"):
            wait_coll.flush_emb_wait()
        requests += 1
        if requests % 200 == 0:
            print(
                f"  {'bench' if timed else 'warmup'} {requests}",
                flush=True,
            )
    if timed:
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    denom = max(hist_tokens, 1)
    return {
        "requests": requests,
        "wall_s": wall,
        "latencies_ms": latencies,
        "kv_gpu_hit": gpu_tokens / denom if requests else None,
        "kv_host_hit": host_tokens / denom if requests else None,
        "kv_hit": merged_tokens / denom if requests else None,
        "req_full_hit": full_hit / requests if requests else None,
        "req_zero_hit": zero_hit / requests if requests else None,
        "req_partial_hit": partial_hit / requests if requests else None,
        "split_emb_ms": _mean_ms(split_emb),
        "split_kv_ms": _mean_ms(split_kv),
        "split_strip_ms": _mean_ms(split_strip),
        "split_hstu_ms": _mean_ms(split_hstu),
        "split_lookup_cpu_ms": _mean_ms(cpu_split["lookup_untimed_ms"]),
        "split_lookup_replay_cpu_ms": _mean_ms(cpu_split["lookup_replay_ms"]),
        "split_alloc_cpu_ms": _mean_ms(cpu_split["alloc_ms"]),
        "split_onboard_launch_cpu_ms": _mean_ms(cpu_split["onboard_launch_ms"]),
        "split_onboard_wait_cpu_ms": _mean_ms(cpu_split["onboard_wait_ms"]),
    }


def summarize_bench(bench):
    latencies = bench["latencies_ms"]
    requests = bench["requests"]
    return {
        "requests": requests,
        "wall_s": bench["wall_s"],
        "mean_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "p50_ms": _percentile(latencies, 0.50),
        "p99_ms": _percentile(latencies, 0.99),
        "throughput_rps": (1000.0 / (sum(latencies) / len(latencies)))
        if latencies
        else None,
        "kv_gpu_hit_last": bench["kv_gpu_hit"],
        "kv_host_hit_last": bench["kv_host_hit"],
        "kv_hit_last": bench["kv_hit"],
        "req_full_hit": bench["req_full_hit"],
        "req_zero_hit": bench["req_zero_hit"],
        "req_partial_hit": bench["req_partial_hit"],
        "split_emb_ms": bench.get("split_emb_ms"),
        "split_kv_ms": bench.get("split_kv_ms"),
        "split_strip_ms": bench.get("split_strip_ms"),
        "split_hstu_ms": bench.get("split_hstu_ms"),
        "split_lookup_cpu_ms": bench.get("split_lookup_cpu_ms"),
        "split_lookup_replay_cpu_ms": bench.get("split_lookup_replay_cpu_ms"),
        "split_alloc_cpu_ms": bench.get("split_alloc_cpu_ms"),
        "split_onboard_launch_cpu_ms": bench.get("split_onboard_launch_cpu_ms"),
        "split_onboard_wait_cpu_ms": bench.get("split_onboard_wait_cpu_ms"),
    }


class RevisitCohort:
    """One round per iteration. A second iteration is the same users again."""

    def __init__(self, batches, user_ids, vocabs, meta):
        self.batches = batches
        self.user_ids = user_ids
        self.vocabs = vocabs
        self.meta = meta
        self._num_samples = len(batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


def _one_revisit_batch(videos, actions, history_len, candidates):
    item = videos[: history_len + candidates]
    action = actions[: history_len + candidates]
    item_tensor = torch.tensor(item, dtype=torch.int64)
    action_tensor = torch.tensor(action, dtype=torch.int64)
    features = KeyedJaggedTensor.from_lengths_sync(
        keys=[ITEM, ACTION],
        values=torch.cat([item_tensor, action_tensor]),
        lengths=torch.tensor([item_tensor.numel(), action_tensor.numel()]),
    )
    num_candidates = torch.tensor([candidates], dtype=torch.int64)
    labels = KeyedJaggedTensor.from_lengths_sync(
        keys=["label"],
        values=action_tensor[-candidates:],
        lengths=num_candidates,
    )
    window = history_len + candidates
    return HSTUBatch(
        features=features,
        batch_size=1,
        feature_to_max_seqlen={ITEM: window, ACTION: window},
        contextual_feature_names=[],
        item_feature_name=ITEM,
        action_feature_name=ACTION,
        max_num_candidates=candidates,
        num_candidates=num_candidates,
        labels=labels,
        actual_batch_size=1,
    )


def load_revisit_cohort(path, cohort_users, history_len, candidates):
    csv.field_size_limit(128 * 1024 * 1024)
    need = history_len + candidates
    batches = []
    user_ids = []
    unique_videos = set()
    video_max = 0
    action_max = 0
    scanned = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned += 1
            videos = json.loads(row[ITEM])
            actions = json.loads(row[ACTION])
            if len(videos) < need or len(actions) < need:
                continue
            window = videos[:need]
            action_window = actions[:need]
            batches.append(
                _one_revisit_batch(window, action_window, history_len, candidates)
            )
            user_ids.append(int(row["user_id"]))
            unique_videos.update(window)
            video_max = max(video_max, max(window))
            action_max = max(action_max, max(action_window))
            if len(batches) >= cohort_users:
                break
    if not batches:
        raise RuntimeError("revisit cohort is empty")
    kv_bytes = len(batches) * 2 * history_len * _kv_token_bytes()
    meta = {
        "case": "revisit",
        "history_len": history_len,
        "candidates": candidates,
        "users": len(batches),
        "rows_scanned": scanned,
        "unique_videos": len(unique_videos),
        "emb_working_set_bytes": len(unique_videos) * _emb_value_bytes(),
        "kv_working_set_bytes": kv_bytes,
    }
    print(
        f"revisit users={meta['users']} rows_scanned={scanned} "
        f"unique_videos={meta['unique_videos']} "
        f"kv={_gib(kv_bytes):.2f}GiB emb={_gib(meta['emb_working_set_bytes']):.2f}GiB",
        flush=True,
    )
    return RevisitCohort(
        batches,
        torch.tensor(user_ids, dtype=torch.long),
        (video_max + 1, max(action_max + 1, 233)),
        meta,
    )


class OrdinaryTraffic:
    """Zipf request stream. Timed pass defaults to the warmup draws.

    Set timed_batches to use a different seed on the timed pass so embedding
    miss is not a hard zero after warmup.
    """

    def __init__(
        self,
        batches,
        user_ids,
        vocabs,
        meta,
        timed_batches=None,
        timed_user_ids=None,
    ):
        self.warmup_batches = batches
        self.warmup_user_ids = user_ids
        self.timed_batches = batches if timed_batches is None else timed_batches
        self.timed_user_ids = user_ids if timed_user_ids is None else timed_user_ids
        self.vocabs = vocabs
        self.meta = meta
        self.set_pass("warmup")

    def set_pass(self, name):
        if name == "timed":
            self.batches = self.timed_batches
            self.user_ids = self.timed_user_ids
        else:
            self.batches = self.warmup_batches
            self.user_ids = self.warmup_user_ids
        self._num_samples = len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


def zipf_stream(pool, requests, alpha, rng):
    order = list(range(pool))
    rng.shuffle(order)
    weights = [(rank + 1) ** (-alpha) for rank in range(pool)]
    total = sum(weights)
    wheel = []
    acc = 0.0
    for weight in weights:
        acc += weight / total
        wheel.append(acc)
    draws = []
    for _ in range(requests):
        u = rng.random()
        lo, hi = 0, pool - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= wheel[mid]:
                hi = mid
            else:
                lo = mid + 1
        draws.append(order[lo])
    return draws


def load_ordinary_traffic(
    path,
    pool_users,
    requests,
    max_history,
    min_history,
    candidates,
    alpha,
    seed,
    timed_seed=None,
):
    csv.field_size_limit(128 * 1024 * 1024)
    need = min_history + candidates
    pool = []
    user_ids = []
    histories = []
    unique_videos = set()
    video_max = 0
    action_max = 0
    scanned = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned += 1
            videos = json.loads(row[ITEM])
            actions = json.loads(row[ACTION])
            if len(videos) < need or len(actions) < need:
                continue
            history_len = min(len(videos) - candidates, max_history)
            window = videos[: history_len + candidates]
            action_window = actions[: history_len + candidates]
            pool.append(_one_revisit_batch(window, action_window, history_len, candidates))
            user_ids.append(int(row["user_id"]))
            histories.append(history_len)
            unique_videos.update(window)
            video_max = max(video_max, max(window))
            action_max = max(action_max, max(action_window))
            if pool_users > 0 and len(pool) >= pool_users:
                break
    if not pool:
        raise RuntimeError("ordinary pool is empty")
    stream = zipf_stream(len(pool), requests, alpha, random.Random(seed))
    seen = set(stream)
    token_bytes = _kv_token_bytes()
    kv_pool = sum(_kv_history_tokens(h) * token_bytes for h in histories)
    kv_stream = sum(_kv_history_tokens(histories[i]) * token_bytes for i in seen)
    batches = [pool[i] for i in stream]
    request_user_ids = torch.tensor([user_ids[i] for i in stream], dtype=torch.long)
    timed_batches = None
    timed_user_ids = None
    timed_seen = seen
    if timed_seed is not None and int(timed_seed) != int(seed):
        timed_stream = zipf_stream(
            len(pool), requests, alpha, random.Random(int(timed_seed))
        )
        timed_seen = set(timed_stream)
        timed_batches = [pool[i] for i in timed_stream]
        timed_user_ids = torch.tensor(
            [user_ids[i] for i in timed_stream], dtype=torch.long
        )
        kv_stream = sum(
            _kv_history_tokens(histories[i]) * token_bytes for i in (seen | timed_seen)
        )
    meta = {
        "case": "ordinary",
        "history_cap": max_history,
        "kv_cache_history": _kv_cache_history_items(),
        "min_history": min_history,
        "candidates": candidates,
        "pool_users": len(pool),
        "requests": len(stream),
        "zipf_alpha": alpha,
        "seed": seed,
        "timed_seed": int(timed_seed) if timed_seed is not None else seed,
        "distinct_in_stream": len(seen),
        "distinct_in_timed": len(timed_seen),
        "rows_scanned": scanned,
        "unique_videos": len(unique_videos),
        "emb_working_set_bytes": len(unique_videos) * _emb_value_bytes(),
        "kv_working_set_bytes": kv_pool,
        "kv_stream_bytes": kv_stream,
        "history_p50": sorted(histories)[len(histories) // 2],
    }
    print(
        f"ordinary pool={meta['pool_users']} requests={meta['requests']} "
        f"distinct={meta['distinct_in_stream']} timed_distinct={meta['distinct_in_timed']} "
        f"rows_scanned={scanned} unique_videos={meta['unique_videos']} "
        f"kv_pool={_gib(kv_pool):.2f}GiB kv_stream={_gib(kv_stream):.2f}GiB "
        f"emb={_gib(meta['emb_working_set_bytes']):.2f}GiB timed_seed={meta['timed_seed']}",
        flush=True,
    )
    return OrdinaryTraffic(
        batches,
        request_user_ids,
        (video_max + 1, max(action_max + 1, 233)),
        meta,
        timed_batches=timed_batches,
        timed_user_ids=timed_user_ids,
    )


def make_dataset(path, nrows, max_history, max_candidates, num_tasks):
    return HSTUSequenceDataset(
        seq_logs_file=path,
        batch_size=1,
        max_history_seqlen=max_history,
        item_feature_name=ITEM,
        contextual_feature_names=CONTEXTUAL,
        action_feature_name=ACTION,
        max_num_candidates=max_candidates,
        num_tasks=num_tasks,
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=0,
        is_train_dataset=True,
        nrows=nrows,
        load_candidate_action=True,
    )


def vocab_from_dataset(dataset):
    frame = dataset._seq_logs_frame
    video_max = 0
    action_max = 0
    limit = dataset._max_history_seqlen + dataset._max_num_candidates
    for raw_video, raw_action in zip(frame[ITEM], frame[ACTION]):
        videos = json.loads(raw_video)[:limit]
        actions = json.loads(raw_action)[:limit]
        if videos:
            video_max = max(video_max, max(videos))
        if actions:
            action_max = max(action_max, max(actions))
    return video_max + 1, max(action_max + 1, 233)


def sweep_fractions(
    dataset,
    vocabs,
    fractions,
    budget_bytes,
    host_tier,
    num_contextual,
    total_max_seqlen,
    hidden,
    emb_host_bytes,
    always_emb_host=False,
    placement="gpu",
):
    video_vocab, action_vocab = vocabs
    configs = _configs(video_vocab, action_vocab, hidden)
    host_kv_bytes = 0
    if host_tier and placement != "cpu":
        available = _mem_available_bytes()
        host_kv_bytes = int(available * 0.35)
    rows = []
    for fraction in fractions:
        kv_budget = int(budget_bytes * fraction)
        emb_budget = budget_bytes - kv_budget
        use_flexkv = False
        flexkv_cpu_bytes = 0
        flexkv_enable_cpu = True
        if placement == "cpu":
            kv_gpu_bytes = _sequence_kv_bytes(total_max_seqlen)
            emb_gpu_bytes = CPU_EMB_GPU_FLOOR
            this_host_kv = kv_budget
            this_emb_host = emb_budget
            tier_name = "cpu"
        elif placement == "kv_cpu":
            # GPU 上只留当前序列的 KV 页，避免 0 页驱逐循环。
            # embedding 预算留在 GPU cache，miss 直接进 CPU 参数表。
            kv_gpu_bytes = _sequence_kv_bytes(total_max_seqlen)
            emb_gpu_bytes = emb_budget
            this_host_kv = kv_budget
            this_emb_host = 0
            tier_name = "kv_cpu"
        elif placement == "gpu_host":
            # GPU 预算照常拆成 GPU KV 和 GPU embedding。
            # host KV 不占这份预算，大小由 CACHE_RATIO_HOST_KV_BYTES 固定。
            kv_gpu_bytes = kv_budget
            emb_gpu_bytes = emb_budget
            this_host_kv = int(
                os.environ.get("CACHE_RATIO_HOST_KV_BYTES", str(32 * 1024 ** 3))
            )
            this_emb_host = 0
            tier_name = "gpu_host"
        elif placement == "kv_flex":
            # 与 kv_cpu 同一刀法：KV 预算进 FlexKV CPU cache，embedding 留在 GPU。
            # native host 容量保持 0，避免再分配一份 pinned KV。
            kv_gpu_bytes = _sequence_kv_bytes(total_max_seqlen)
            emb_gpu_bytes = emb_budget
            this_host_kv = 0
            this_emb_host = 0
            use_flexkv = True
            flexkv_cpu_bytes = kv_budget
            flexkv_enable_cpu = True
            tier_name = "kv_flex"
        elif placement == "kv_flex_gpu":
            # FlexKV 后端：KV 预算进 GPU pages。CPU 只留 staging，SSD 关闭。
            # miss 不靠 CPU cache，直接重算。embedding 预算留在 GPU。
            kv_gpu_bytes = kv_budget
            emb_gpu_bytes = emb_budget
            this_host_kv = 0
            this_emb_host = 0
            use_flexkv = True
            flexkv_cpu_bytes = _flexkv_staging_cpu_bytes()
            flexkv_enable_cpu = True
            tier_name = "kv_flex_gpu"
        else:
            kv_gpu_bytes = kv_budget
            emb_gpu_bytes = emb_budget
            this_host_kv = host_kv_bytes
            this_emb_host = (
                emb_host_bytes if (host_tier or always_emb_host) else 0
            )
            tier_name = "host" if host_tier else "gpu"
        print(
            f"fraction {fraction} placement={placement} "
            f"kv_gpu={_gib(kv_gpu_bytes):.3f}GiB kv_host={_gib(this_host_kv):.2f}GiB "
            f"flexkv_cpu={_gib(flexkv_cpu_bytes):.2f}GiB flexkv_enable_cpu={flexkv_enable_cpu} "
            f"emb_gpu={_gib(emb_gpu_bytes):.3f}GiB emb_host={_gib(this_emb_host):.2f}GiB",
            flush=True,
        )
        model, sizes = build_model(
            configs,
            kv_gpu_bytes,
            this_host_kv,
            emb_gpu_bytes,
            this_emb_host,
            num_contextual,
            total_max_seqlen,
            use_flexkv=use_flexkv,
            flexkv_cpu_bytes=flexkv_cpu_bytes,
            flexkv_enable_cpu=flexkv_enable_cpu,
        )
        wait_coll = None
        with torch.inference_mode():
            if hasattr(dataset, "set_pass"):
                dataset.set_pass("warmup")
            warmup = run_pass(model, dataset, timed=False)
            _drain_flexkv_offload(model)
            wait_coll = _emb_wait_collection(model)
            if wait_coll is not None and hasattr(wait_coll, "reset_emb_wait_stats"):
                wait_coll.reset_emb_wait_stats()
            if hasattr(dataset, "set_pass"):
                dataset.set_pass("timed")
            bench = run_pass(model, dataset, timed=True)
        row = {
            "tier": tier_name,
            "placement": placement,
            "emb_miss_backend": os.environ.get("CACHE_RATIO_EMB_BACKEND", "cpu"),
            "redis_address": os.environ.get("CACHE_RATIO_REDIS_ADDRESS", ""),
            "kv_fraction": fraction,
            "kv_budget_bytes": kv_budget,
            "emb_budget_bytes": emb_budget,
            "kv_cache_history": _kv_cache_history_items(),
            **sizes,
            "warmup_requests": warmup["requests"],
            "warmup_wall_s": warmup["wall_s"],
            **summarize_bench(bench),
        }
        wait_us = os.environ.get("CACHE_RATIO_EMB_WAIT_US", "").strip()
        if wait_us:
            row["emb_wait_us"] = float(wait_us)
            if wait_coll is not None and hasattr(wait_coll, "emb_wait_stats"):
                row.update(wait_coll.emb_wait_stats())
        kv_wait_ms = os.environ.get("CACHE_RATIO_KV_WAIT_MS", "").strip()
        if kv_wait_ms:
            row["kv_wait_ms"] = float(kv_wait_ms)
        print(json.dumps(row), flush=True)
        rows.append(row)
        _release(model)
    return rows


def _output_path(cfg, budget_bytes):
    if cfg.out:
        return cfg.out
    gib = int(budget_bytes / (1024 ** 3))
    tag = f"_{cfg.tag}" if cfg.tag else ""
    return f"inference/benchmark/cache_ratio_{cfg.case}_{cfg.placement}{tag}_{gib}gib.jsonl"


def _append_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"appended {path}", flush=True)


def run_fraction(cfg, fraction, budget_bytes=None):
    if cfg.placement not in _PLACEMENTS:
        raise SystemExit(f"CONFIG.placement must be one of {_PLACEMENTS}, got {cfg.placement}")
    if cfg.case not in ("ordinary", "revisit", "replay"):
        raise SystemExit(f"CONFIG.case must be ordinary, revisit, or replay, got {cfg.case}")

    gin.parse_config_file(cfg.gin_config)
    network = NetworkArgs()
    ranking = RankingArgs()
    hidden = network.hidden_size
    max_candidates = cfg.candidates
    max_history = cfg.history_len
    total_max_seqlen = (max_history + max_candidates) * 2
    device_name = torch.cuda.get_device_name(0)
    total_gpu = torch.cuda.get_device_properties(0).total_memory
    budget = (
        int(budget_bytes)
        if budget_bytes is not None
        else cfg.budget_bytes
        if cfg.budget_bytes > 0
        else int(total_gpu * cfg.budget_fraction)
    )
    out = _output_path(cfg, budget)
    print(
        f"device {device_name} gpu={_gib(total_gpu):.2f}GiB budget={_gib(budget):.2f}GiB "
        f"placement={cfg.placement} emb_miss={cfg.emb_backend} case={cfg.case} fraction={fraction} "
        f"capability={torch.cuda.get_device_capability()}",
        flush=True,
    )

    if cfg.case == "revisit":
        dataset = load_revisit_cohort(
            cfg.data, cfg.cohort_users, max_history, max_candidates
        )
        rows = sweep_fractions(
            dataset,
            dataset.vocabs,
            (fraction,),
            budget,
            False,
            0,
            total_max_seqlen,
            hidden,
            0,
            placement=cfg.placement,
        )
        row = rows[0]
        row["device"] = device_name
        row["budget_bytes"] = budget
        row["placement"] = cfg.placement
        row.update(_shape_fields(cfg))
        row.update(dataset.meta)
        _append_row(out, row)
        return

    if cfg.case == "ordinary":
        timed_seed = int(cfg.timed_seed) if cfg.timed_seed else None
        dataset = load_ordinary_traffic(
            cfg.data,
            cfg.pool_users,
            cfg.requests,
            max_history,
            cfg.min_history,
            max_candidates,
            cfg.zipf_alpha,
            cfg.seed,
            timed_seed=timed_seed,
        )
        rows = sweep_fractions(
            dataset,
            dataset.vocabs,
            (fraction,),
            budget,
            False,
            0,
            total_max_seqlen,
            hidden,
            0,
            placement=cfg.placement,
        )
        row = rows[0]
        row["device"] = device_name
        row["budget_bytes"] = budget
        row["placement"] = cfg.placement
        row.update(_shape_fields(cfg))
        row.update(dataset.meta)
        both = row["kv_stream_bytes"] + row["emb_working_set_bytes"]
        row["both_working_set_bytes"] = both
        row["budget_lt_both"] = budget < both
        emb_cache_bytes = (
            row["emb_host_bytes"] if cfg.placement == "cpu" else row["emb_gpu_bytes"]
        )
        row["emb_cache_lt_working_set"] = emb_cache_bytes < row["emb_working_set_bytes"]
        _append_row(out, row)
        return

    nrows = cfg.max_users if cfg.max_users > 0 else None
    dataset = make_dataset(
        cfg.data, nrows, max_history, max_candidates, ranking.num_tasks
    )
    if cfg.video_vocab > 0 and cfg.action_vocab > 0:
        vocabs = (cfg.video_vocab, cfg.action_vocab)
    else:
        vocabs = vocab_from_dataset(dataset)
    print(f"users {dataset._num_samples} vocabs video,action {vocabs}", flush=True)
    rows = sweep_fractions(
        dataset,
        vocabs,
        (fraction,),
        budget,
        False,
        0,
        total_max_seqlen,
        hidden,
        0,
        placement=cfg.placement,
    )
    row = rows[0]
    row["device"] = device_name
    row["users"] = dataset._num_samples
    row["budget_bytes"] = budget
    row["placement"] = cfg.placement
    row.update(_shape_fields(cfg))
    _append_row(out, row)


def launch(cfg):
    if cfg.placement not in _PLACEMENTS:
        raise SystemExit(f"CONFIG.placement must be one of {_PLACEMENTS}, got {cfg.placement}")
    if cfg.case not in ("ordinary", "revisit", "replay"):
        raise SystemExit(f"CONFIG.case must be ordinary, revisit, or replay, got {cfg.case}")
    script = os.path.abspath(__file__)
    budget_env = os.environ.get("CACHE_RATIO_BUDGETS", "")
    if budget_env:
        budgets = tuple(int(part) * 1024 ** 3 for part in budget_env.split(",") if part)
    else:
        budgets = tuple(cfg.budget_bytes_list) if cfg.budget_bytes_list else (cfg.budget_bytes,)
    for budget in budgets:
        out = _output_path(cfg, budget)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w"):
            pass
        fractions = cfg.fractions
        if cfg.placement in ("gpu", "kv_flex_gpu"):
            # 0 页 GPU KV 会在驱逐循环里空转，整次扫测停在第一个点。
            fractions = tuple(fraction for fraction in fractions if fraction != 0.0)
        elif cfg.placement == "gpu_host":
            # 0 页 GPU KV 会空转。比例 1 把 GPU embedding 打成 0，NVE 初始化失败。
            fractions = tuple(
                fraction for fraction in fractions if fraction not in (0.0, 1.0)
            )
        frac_env = os.environ.get("CACHE_RATIO_FRACTIONS", "").strip()
        if frac_env:
            fractions = tuple(
                float(part) for part in frac_env.split(",") if part.strip()
            )
        print(
            f"case={cfg.case} placement={cfg.placement} budget_bytes={budget} "
            f"kv_cache_history={_kv_cache_history_items()} "
            f"fractions={fractions} out={out}",
            flush=True,
        )
        for fraction in fractions:
            print(f"=== budget {budget} fraction {fraction} ===", flush=True)
            child_env = os.environ.copy()
            ident = "{0}_{1}_{2}".format(os.getpid(), str(fraction).replace(".", "p"), int(time.time()))
            recv = "ipc:///tmp/flexkv_recv_{0}".format(ident)
            # KVManager always binds gpu_register at `{server_recv}_gpu_register`.
            # KVTPClient reads FLEXKV_GPU_REGISTER_PORT separately; they must match.
            child_env["FLEXKV_SERVER_RECV_PORT"] = recv
            child_env["FLEXKV_GPU_REGISTER_PORT"] = recv + "_gpu_register"
            rc = subprocess.call(
                [
                    sys.executable,
                    "-s",
                    script,
                    "--fraction",
                    str(fraction),
                    "--budget-bytes",
                    str(int(budget)),
                ],
                env=child_env,
            )
            if rc != 0:
                print(f"budget {budget} fraction {fraction} failed", flush=True)
                if os.environ.get("CACHE_RATIO_ABORT_ON_FAIL") == "1":
                    raise SystemExit(rc)


def main():
    _self_check_strip_embeddings()
    if len(sys.argv) >= 3 and sys.argv[1] == "--fraction":
        fraction = float(sys.argv[2])
        budget_bytes = None
        if len(sys.argv) == 5 and sys.argv[3] == "--budget-bytes":
            budget_bytes = int(sys.argv[4])
        elif len(sys.argv) != 3:
            raise SystemExit("usage: cache_ratio_sweep.py --fraction F [--budget-bytes N]")
        run_fraction(CONFIG, fraction, budget_bytes)
        return
    if len(sys.argv) != 1:
        raise SystemExit(
            "Edit CONFIG in this file, then run it with no arguments."
        )
    launch(CONFIG)


if __name__ == "__main__":
    main()
