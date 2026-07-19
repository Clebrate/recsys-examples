#!/usr/bin/env bash
# Pick a writable local (non-NFS) path for FlexKV SSD cache files.
# Inside docker, /tmp is usually overlay on host NVMe and will NOT show "nvme"
# in findmnt SOURCE — treat writable non-NFS /tmp as local.
set -euo pipefail

if [[ -n "${FLEXKV_SSD_CACHE_BASE:-}" ]]; then
  mkdir -p "${FLEXKV_SSD_CACHE_BASE}"
  echo "${FLEXKV_SSD_CACHE_BASE}"
  exit 0
fi

is_nfs_like() {
  local path="$1"
  local fstype
  fstype="$(findmnt -n -o FSTYPE -T "${path}" 2>/dev/null || true)"
  case "${fstype}" in
    nfs|nfs4|ceph|cifs|fuse.sshfs|fuse)
      return 0
      ;;
  esac
  return 1
}

# Prefer explicit NVMe bind mounts if present.
for candidate in /tmp/nvidia-mp /local_nvme /mnt/nvme0 /mnt/nvme1 /mnt/nvme /local/ssd; do
  if [[ -d "${candidate}" ]] && ! is_nfs_like "${candidate}"; then
    echo "${candidate}"
    exit 0
  fi
done

# Docker / crun: /tmp is local (overlay or host tmp), not NFS.
if [[ -d /tmp ]] && ! is_nfs_like /tmp; then
  mkdir -p /tmp/nvidia-mp
  echo /tmp/nvidia-mp
  exit 0
fi

echo "ERROR: no local SSD path found. Set FLEXKV_SSD_CACHE_BASE." >&2
findmnt -rn -o TARGET,FSTYPE,SOURCE | head -20 >&2 || true
exit 1
