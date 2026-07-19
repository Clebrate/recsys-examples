#!/usr/bin/env bash
# Generate FlexKV yml configs for PR429 timed sweep with correct cache sizes.
set -euo pipefail

ROOT="${ROOT:-/home/scratch.noliu_gpu/pr429-verify}"
SSD_BASE="$("${ROOT}/scripts/detect_flexkv_ssd_base.sh")"
SSD_CACHE_DIR="${SSD_BASE}/flexkv_pr429_ssd_cache"

CONFIG_DIR="${ROOT}/corelib/recsys_kvcache_manager/configs"
mkdir -p "${CONFIG_DIR}" "${SSD_CACHE_DIR}"

# Block size ~= 1 MiB for HSTU benchmark (8 layers, 4 heads, 256 head_dim, page_size=32, bf16).
# Worst-case target set: history_len=4096, batch_size=8, timed_iters=10
# -> 80 users x 256 MiB/user = 20 GiB. Use 32 GiB for targets,
# tmp blocks, pinned-transfer staging, allocator headroom, and clean reruns.
# Scenario3 pressure scales with CPU capacity; 32 GiB CPU plus the 20 GiB
# target set requires about 55.5 GiB SSD, so 200 GiB remains ample.
cat > "${CONFIG_DIR}/flexkv_ssd_cpuhit.yml" <<EOF
# GPU-hit / CPU-hit scenarios: keep prefixes in CPU (not SSD).
cpu_cache_gb: 32.0
ssd_cache_gb: 64.0
ssd_cache_dir: ${SSD_CACHE_DIR}
enable_gds: false
enable_p2p_cpu: false
enable_p2p_ssd: false
enable_3rd_remote: false
EOF

cat > "${CONFIG_DIR}/flexkv_ssd_local.yml" <<EOF
# SSD-hit: CPU needs headroom for D2H PUT; SSD > CPU for spill.
cpu_cache_gb: 32.0
ssd_cache_gb: 200.0
ssd_cache_dir: ${SSD_CACHE_DIR}
enable_gds: false
enable_p2p_cpu: false
enable_p2p_ssd: false
enable_3rd_remote: false
EOF

echo "[flexkv-config] SSD base=${SSD_BASE}"
echo "[flexkv-config] SSD cache dir=${SSD_CACHE_DIR}"
echo "[flexkv-config] cpuhit yml=${CONFIG_DIR}/flexkv_ssd_cpuhit.yml (cpu=32G ssd=64G)"
echo "[flexkv-config] ssdhit yml=${CONFIG_DIR}/flexkv_ssd_local.yml (cpu=32G ssd=200G)"
