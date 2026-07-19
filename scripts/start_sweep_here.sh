#!/usr/bin/env bash
# Paste this inside the already-running inference container (root@...).
set -euo pipefail
ROOT=/home/scratch.noliu_gpu/pr429-verify
OUT=/home/scratch.noliu_gpu/nsys_pr429_timed_sweep
# In this docker, /tmp is local overlay on host NVMe (not NFS).
export FLEXKV_SSD_CACHE_BASE="${FLEXKV_SSD_CACHE_BASE:-/tmp/nvidia-mp}"
mkdir -p "${OUT}" "${FLEXKV_SSD_CACHE_BASE}"
LOG_TMP=/tmp/pr429_timed_sweep.log
: > "${LOG_TMP}"
echo "[start] SSD base=${FLEXKV_SSD_CACHE_BASE}"
echo "[start] logging to ${LOG_TMP}"
nohup bash "${ROOT}/scripts/run_sweep_in_container.sh" > "${LOG_TMP}" 2>&1 &
echo "[start] pid=$!"
tail -f "${LOG_TMP}"
