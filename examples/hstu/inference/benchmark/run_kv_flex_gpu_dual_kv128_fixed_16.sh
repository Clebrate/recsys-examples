#!/bin/bash
# FlexKV backend, GPU KV tier, dual T=128, 16 GiB, after strip fix.
# KV share on FlexKV GPU pages: hit stays on GPU, miss recomputes.
# FlexKV CPU is staging only (128 blocks, not in the 16 GiB split); SSD off.
# Embedding share on GPU.
# 1) CPU parameter table, no wait
# 2) real Redis miss path
# 3) oldwait 权衡: KV_WAIT_MS=1.329810, EMB_WAIT_US=0.410387
# New placement kv_flex_gpu; do not overwrite kv_flex CPU-KV or native gpu jsonl.
set -euo pipefail
REPO=/home/scratch.noliu_gpu/recsys-examples-cache-ratio
HSTU="${REPO}/examples/hstu"
BENCH="${HSTU}/inference/benchmark"
LOG="${BENCH}/task_queue/logs/kv_flex_gpu_dual_kv128_fixed_16.out"
mkdir -p "${BENCH}/task_queue/logs" "${BENCH}/figures"
exec > >(tee -a "${LOG}") 2>&1
echo "hostname $(hostname)"
echo "slurm_job ${SLURM_JOB_ID:-unknown}"
date -Is
nvidia-smi -L
cd "${HSTU}"
export NVE_INSTALL_ROOT=/opt/nve
if [[ -d /opt/nve/default/python/pynve ]]; then
  NVE_PY=/opt/nve/default/python
elif [[ -d /opt/nve/26.05/python/pynve ]]; then
  NVE_PY=/opt/nve/26.05/python
  export NVE_VERSION=26.05
else
  echo "pynve prefix missing under /opt/nve" >&2
  ls -la /opt/nve || true
  exit 1
fi
export PYTHONPATH="${NVE_PY}:${REPO}/examples:${HSTU}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CACHE_RATIO_ABORT_ON_FAIL=1

export CACHE_RATIO_PLACEMENT=kv_flex_gpu
export CACHE_RATIO_BUDGETS=16
export CACHE_RATIO_FRACTIONS=0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95
export CACHE_RATIO_SEED=1
export CACHE_RATIO_CANDIDATES=100
export CACHE_RATIO_HISTORY=4096
export CACHE_RATIO_POOL_USERS=0
export CACHE_RATIO_REQUESTS=26000
export CACHE_RATIO_TIMED_SEED=2
export CACHE_RATIO_KV_CACHE_HISTORY=128

assert_fresh() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "REFUSE overwrite existing ${path}" >&2
    exit 1
  fi
}

jsonl_for() {
  echo "${BENCH}/cache_ratio_ordinary_kv_flex_gpu_${1}_16gib.jsonl"
}

stop_redis() {
  bash "${BENCH}/start_redis_cluster.sh" stop || true
}
trap stop_redis EXIT

run_sweep() {
  echo "=== placement=${CACHE_RATIO_PLACEMENT} backend=${CACHE_RATIO_EMB_BACKEND} tag=${CACHE_RATIO_TAG} fractions=${CACHE_RATIO_FRACTIONS} wait_kv=${CACHE_RATIO_KV_WAIT_MS:-} wait_emb=${CACHE_RATIO_EMB_WAIT_US:-} redis=${CACHE_RATIO_REDIS_ADDRESS:-} $(date -Is) ==="
  python3 -s inference/benchmark/cache_ratio_sweep.py
}

plot_one() {
  python3 "${BENCH}/plot_dual_latency_hit.py" "$@" || echo "plot failed $*"
}

NOSIDE=$(jsonl_for dual_kv128_fixed_noside)
REDIS=$(jsonl_for dual_kv128_fixed_redis)
MATCH=$(jsonl_for dual_kv128_fixed_match_oldwait)
assert_fresh "${NOSIDE}"
assert_fresh "${REDIS}"
assert_fresh "${MATCH}"

# --- 1. KV GPU + 重算；embedding GPU + CPU 表 ---
export CACHE_RATIO_EMB_BACKEND=cpu
export CACHE_RATIO_TAG=dual_kv128_fixed_noside
unset CACHE_RATIO_EMB_WAIT_US CACHE_RATIO_KV_WAIT_MS CACHE_RATIO_EMB_WAIT_IN_MEAN CACHE_RATIO_REDIS_ADDRESS CACHE_RATIO_SPLIT_TIMING || true
run_sweep
plot_one "${NOSIDE}" "${BENCH}/figures/kv_flex_gpu_dual_kv128_fixed_noside_16_min.png" "" auto 最低点 0.70

# --- 2. KV GPU + 重算；embedding GPU + 真 Redis ---
export CACHE_RATIO_EMB_BACKEND=redis
export CACHE_RATIO_REDIS_ADDRESS=127.0.0.1:7000
export CACHE_RATIO_TAG=dual_kv128_fixed_redis
unset CACHE_RATIO_EMB_WAIT_US CACHE_RATIO_KV_WAIT_MS CACHE_RATIO_EMB_WAIT_IN_MEAN CACHE_RATIO_SPLIT_TIMING || true
bash "${BENCH}/start_redis_cluster.sh" start
python3 -s "${BENCH}/preload_redis_embeddings.py" \
  tmp_data/KuaiRand-27K/data/processed_seqs_4_08_to_4_21.csv
run_sweep
plot_one "${REDIS}" "${BENCH}/figures/kv_flex_gpu_dual_kv128_fixed_redis_16_min.png" "" auto 最低点 0.70
stop_redis
unset CACHE_RATIO_REDIS_ADDRESS || true
export CACHE_RATIO_EMB_BACKEND=cpu

# --- 3. 同上 CPU 表，加上对齐 wait ---
export CACHE_RATIO_TAG=dual_kv128_fixed_match_oldwait
export CACHE_RATIO_EMB_WAIT_US=0.410387
export CACHE_RATIO_KV_WAIT_MS=1.329810
export CACHE_RATIO_EMB_WAIT_IN_MEAN=1
unset CACHE_RATIO_REDIS_ADDRESS CACHE_RATIO_SPLIT_TIMING || true
run_sweep
plot_one "${MATCH}" "${BENCH}/figures/kv_flex_gpu_dual_kv128_fixed_match_oldwait_16_min.png" "" auto 最低点 0.70

echo "done $(date -Is)"
echo "jsonl ${NOSIDE} ${REDIS} ${MATCH}"
