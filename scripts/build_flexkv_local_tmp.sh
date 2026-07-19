#!/usr/bin/env bash
# Build FlexKV into /tmp (container-local), avoiding NFS root_squash write issues.
# Run inside the inference docker container on the GPU node.
set -euo pipefail

TARBALL=${FLEXKV_SOURCE_TARBALL:-/home/scratch.noliu_gpu/pr429-verify/flexkv_source_for_container.tar.gz}
STAGE=/tmp/flexkv_pr429_build
BUILD_DIR="${STAGE}/build"
INSTALL_DIR="${STAGE}/install"

export FLEXKV_ENABLE_METRICS=0
export FLEXKV_ENABLE_GDS=0
export FLEXKV_ENABLE_P2P=0
export FLEXKV_DEBUG=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;9.0;8.9}"

echo "=== stage FlexKV source tarball (read-only NFS -> /tmp) ==="
rm -rf "${STAGE}/src"
mkdir -p "${STAGE}/src"
if [[ ! -r "${TARBALL}" ]]; then
  echo "ERROR: source tarball is not readable: ${TARBALL}"
  exit 1
fi
tar -xzf "${TARBALL}" -C "${STAGE}/src"

# Ensure nested xxHash exists (skip git submodule)
if [[ ! -f "${STAGE}/src/third_party/xxHash/xxhash.c" ]]; then
  echo "ERROR: missing xxhash.c in staged source"
  exit 1
fi

echo "=== cmake build ==="
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${STAGE}/src" -B "${BUILD_DIR}" -DFLEXKV_ENABLE_MONITORING=OFF
cmake --build "${BUILD_DIR}" -j"$(nproc)"

mkdir -p "${STAGE}/src/flexkv/lib"
cp -f "${BUILD_DIR}"/lib/*.so* "${STAGE}/src/flexkv/lib/" 2>/dev/null || true

if python3 -c "import torch" >/dev/null 2>&1; then
  echo "=== pip install (system python, inference container) ==="
  cd "${STAGE}/src"
  FLEXKV_DEBUG=1 pip install -v --no-build-isolation . --break-system-packages 2>/dev/null \
    || FLEXKV_DEBUG=1 pip install -v --no-build-isolation .
else
  VENV="${STAGE}/venv"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install -U pip setuptools wheel scikit-build cmake ninja
  echo "=== pip install in venv ==="
  cd "${STAGE}/src"
  FLEXKV_DEBUG=1 pip install -v --no-build-isolation .
fi

echo "=== verify ==="
export PYTHONPATH="${STAGE}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${STAGE}/src/flexkv/lib:${LD_LIBRARY_PATH}"
python3 -c "import flexkv.c_ext; print('flexkv.c_ext OK:', flexkv.c_ext.__file__)"

cat <<EOF

=== Build OK ===
Add to your shell before running benchmark:

export FLEXKV_BUILD_ROOT=${STAGE}
export PYTHONPATH=\${FLEXKV_BUILD_ROOT}/src:\$ROOT/corelib/recsys_kvcache_manager:\$PWD:\$PYTHONPATH
export LD_LIBRARY_PATH=\${FLEXKV_BUILD_ROOT}/build/lib:\${FLEXKV_BUILD_ROOT}/src/flexkv/lib:\$LD_LIBRARY_PATH

EOF
