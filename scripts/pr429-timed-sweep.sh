#!/usr/bin/env bash
set -uo pipefail
# Note: do NOT use `set -e` — a failed scenario must not abort the whole sweep.

ROOT=/home/scratch.noliu_gpu/pr429-verify
OUT_DIR="${OUT_DIR:-/home/scratch.noliu_gpu/nsys_pr429_timed_sweep}"
TIMED_ITERS=10

HISTORY_LENS=(512 1024 2048 4096)
BATCH_SIZES=(1 2 4 8)

# Resume support: only run missing tags (empty = all).
# Example: ONLY_HISTORY_LENS="4096" ONLY_BATCH_SIZES="1 2 4 8" bash scripts/pr429-timed-sweep.sh
if [[ -n "${ONLY_HISTORY_LENS:-}" ]]; then
  # shellcheck disable=SC2206
  HISTORY_LENS=(${ONLY_HISTORY_LENS})
fi
if [[ -n "${ONLY_BATCH_SIZES:-}" ]]; then
  # shellcheck disable=SC2206
  BATCH_SIZES=(${ONLY_BATCH_SIZES})
fi

cd "${ROOT}"

export ROOT
bash "${ROOT}/scripts/write_flexkv_sweep_configs.sh"

SSD_BASE="$("${ROOT}/scripts/detect_flexkv_ssd_base.sh")"
SSD_CACHE_DIR="${SSD_BASE}/flexkv_pr429_ssd_cache"
CONFIG_CPUHIT="${ROOT}/corelib/recsys_kvcache_manager/configs/flexkv_ssd_cpuhit.yml"
CONFIG_SSDHIT="${ROOT}/corelib/recsys_kvcache_manager/configs/flexkv_ssd_local.yml"

echo "[sweep] SSD cache dir=${SSD_CACHE_DIR}"
mkdir -p "${SSD_CACHE_DIR}" "${OUT_DIR}"

export FLEXKV_BUILD_ROOT="${FLEXKV_BUILD_ROOT:-/tmp/flexkv_pr429_build}"
export PYTHONPATH=${FLEXKV_BUILD_ROOT}/src:${ROOT}/corelib/recsys_kvcache_manager:${ROOT}/examples/hstu:${ROOT}/examples:${ROOT}:${PYTHONPATH:-}
if [[ -x "${FLEXKV_BUILD_ROOT}/venv/bin/python" ]]; then
  export PATH="${FLEXKV_BUILD_ROOT}/venv/bin:${PATH}"
fi
export LD_LIBRARY_PATH=${FLEXKV_BUILD_ROOT}/build/lib:${FLEXKV_BUILD_ROOT}/src/flexkv/lib:${LD_LIBRARY_PATH:-}
export HSTU_INFERENCE_ONLY=1
export FLEXKV_ENABLE_GDS=0
export FLEXKV_USE_CE_TRANSFER_D2H=1
export FLEXKV_USE_CE_TRANSFER_H2D=0
export FLEXKV_TRANSFER_NUM_CTA_H2D=4
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;9.0;8.9}"
export FLEXKV_SSD_CACHE_BASE="${FLEXKV_SSD_CACHE_BASE:-/tmp/nvidia-mp}"

if ! python3 -c "import flexkv.c_ext" >/dev/null 2>&1; then
  echo "[sweep] FlexKV import failed; building into ${FLEXKV_BUILD_ROOT} ..."
  if [[ ! -f "${ROOT}/flexkv_source_for_container.tar.gz" ]]; then
    tar --exclude='.git' --exclude='build' --exclude='*.egg-info' \
      -czf "${ROOT}/flexkv_source_for_container.tar.gz" -C "${ROOT}/third_party/FlexKV" .
  fi
  bash "${ROOT}/scripts/build_flexkv_local_tmp.sh"
  export PYTHONPATH=${FLEXKV_BUILD_ROOT}/src:${ROOT}/corelib/recsys_kvcache_manager:${ROOT}/examples/hstu:${ROOT}/examples:${ROOT}:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${FLEXKV_BUILD_ROOT}/build/lib:${FLEXKV_BUILD_ROOT}/src/flexkv/lib:${LD_LIBRARY_PATH:-}
else
  echo "[sweep] FlexKV already importable; skip build"
fi

python3 - <<'PY'
import torch
import flexkv.c_ext
print("[sweep] torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[sweep] flexkv.c_ext OK:", flexkv.c_ext.__file__)
PY

cd "${ROOT}/examples/hstu"

clean_ssd_cache() {
  echo "[cleanup] reset FlexKV SSD cache: ${SSD_CACHE_DIR}"
  rm -rf "${SSD_CACHE_DIR:?}"/*
  rm -rf "${SSD_CACHE_DIR}"
  mkdir -p "${SSD_CACHE_DIR}"
}

run_one() {
  local name="$1"
  shift
  local out="${OUT_DIR}/${name}"
  local log="${OUT_DIR}/${name}.log"
  local sqlite="${OUT_DIR}/${name}.sqlite"

  if [[ -f "${sqlite}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] ${name} (sqlite exists)"
    return 0
  fi

  echo
  echo "========== ${name} =========="
  clean_ssd_cache
  rm -f "${out}.nsys-rep" "${sqlite}" "${log}"

  nsys profile --wait=primary -f true -t cuda,nvtx,osrt \
    -o "${out}" \
    "$@" 2>&1 | tee "${log}"
  local profile_rc=${PIPESTATUS[0]}

  if [[ ! -f "${out}.nsys-rep" ]]; then
    echo "[WARN] ${name}: no nsys-rep (profile_rc=${profile_rc}); skip export"
    return 0
  fi

  nsys export --type sqlite --force-overwrite=true \
    -o "${sqlite}" \
    "${out}.nsys-rep"
  local export_rc=$?

  if [[ ${profile_rc} -ne 0 ]]; then
    echo "[WARN] ${name}: profile exited ${profile_rc} (continuing sweep)"
  fi
  if [[ ${export_rc} -ne 0 ]]; then
    echo "[WARN] ${name}: export exited ${export_rc} (continuing sweep)"
  fi
  sleep 2
  return 0
}

common_args() {
  local history_len="$1"
  local batch_size="$2"
  echo --history-len "${history_len}" --batch-size "${batch_size}" \
    --timed-iters "${TIMED_ITERS}" --disable-cudagraph
}

for history_len in "${HISTORY_LENS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    tag="hl${history_len}_bs${batch_size}"

    run_one "no_cache_only_onboard_${tag}" \
      python3 inference/benchmark/inference_benchmark_flexkv.py \
        --only-onboard $(common_args "${history_len}" "${batch_size}") \
        --scenarios no_cache

    run_one "only_onboard_s1_gpu_${tag}" \
      python3 inference/benchmark/inference_benchmark_flexkv.py \
        --only-onboard $(common_args "${history_len}" "${batch_size}") \
        --scenarios gpu_hit --flexkv-config-path "${CONFIG_CPUHIT}"

    run_one "only_onboard_s2_cpu_${tag}" \
      python3 inference/benchmark/inference_benchmark_flexkv.py \
        --only-onboard $(common_args "${history_len}" "${batch_size}") \
        --scenarios cpu_hit --flexkv-config-path "${CONFIG_CPUHIT}"

    run_one "only_onboard_s3_ssd_${tag}" \
      python3 inference/benchmark/inference_benchmark_flexkv.py \
        --only-onboard $(common_args "${history_len}" "${batch_size}") \
        --scenarios ssd_hit --flexkv-config-path "${CONFIG_SSDHIT}"
  done
done

python3 "${ROOT}/scripts/analyze_pr429_timed_sweep.py" \
  --root "${OUT_DIR}" \
  --out "${ROOT}/docs/PR429_TIMED_SWEEP_ANALYSIS.md" || true

echo "Sweep done. Results in ${OUT_DIR}"
