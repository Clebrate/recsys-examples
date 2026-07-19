#!/usr/bin/env bash
# Run PR429 timed sweep inside the already-running inference GPU container.
set -euo pipefail

ROOT=/home/scratch.noliu_gpu/pr429-verify
cd "${ROOT}"

export ROOT
export FLEXKV_SSD_CACHE_BASE="${FLEXKV_SSD_CACHE_BASE:-/tmp/nvidia-mp}"
mkdir -p "${FLEXKV_SSD_CACHE_BASE}"
export HSTU_INFERENCE_ONLY=1
export FLEXKV_ENABLE_GDS=0
export FLEXKV_USE_CE_TRANSFER_D2H=1
export FLEXKV_USE_CE_TRANSFER_H2D=0
export FLEXKV_TRANSFER_NUM_CTA_H2D=4

# Blackwell (sm_120) if present; keep 8.9/9.0 as fallbacks.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;9.0;8.9}"

bash "${ROOT}/scripts/write_flexkv_sweep_configs.sh"

export FLEXKV_BUILD_ROOT="${FLEXKV_BUILD_ROOT:-/tmp/flexkv_pr429_build}"
# Do NOT put third_party/FlexKV on PYTHONPATH — it shadows a working install
# with an unbuilt source tree (no c_ext.so).
export PYTHONPATH=${FLEXKV_BUILD_ROOT}/src:${ROOT}/corelib/recsys_kvcache_manager:${ROOT}/examples/hstu:${ROOT}/examples:${ROOT}:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${FLEXKV_BUILD_ROOT}/build/lib:${FLEXKV_BUILD_ROOT}/src/flexkv/lib:${LD_LIBRARY_PATH:-}
if [[ -x "${FLEXKV_BUILD_ROOT}/venv/bin/python" ]]; then
  export PATH="${FLEXKV_BUILD_ROOT}/venv/bin:${PATH}"
fi

if ! python3 -c "import flexkv.c_ext" >/dev/null 2>&1; then
  echo "[prep] building FlexKV into ${FLEXKV_BUILD_ROOT} ..."
  if [[ ! -f "${ROOT}/flexkv_source_for_container.tar.gz" ]]; then
    tar --exclude='.git' --exclude='build' --exclude='*.egg-info' \
      -czf "${ROOT}/flexkv_source_for_container.tar.gz" -C "${ROOT}/third_party/FlexKV" .
  fi
  bash "${ROOT}/scripts/build_flexkv_local_tmp.sh"
  export PYTHONPATH=${FLEXKV_BUILD_ROOT}/src:${ROOT}/corelib/recsys_kvcache_manager:${ROOT}/examples/hstu:${ROOT}/examples:${ROOT}:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${FLEXKV_BUILD_ROOT}/build/lib:${FLEXKV_BUILD_ROOT}/src/flexkv/lib:${LD_LIBRARY_PATH:-}
fi

python3 - <<'PY'
import torch, flexkv.c_ext
print("[prep] torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[prep] flexkv", flexkv.c_ext.__file__)
PY

command -v nsys >/dev/null

exec bash "${ROOT}/scripts/pr429-timed-sweep.sh"
