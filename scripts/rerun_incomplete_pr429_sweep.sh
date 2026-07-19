#!/usr/bin/env bash
# Remove missing/corrupt PR429 results, then rerun only the affected hl2048/4096
# matrix entries. Valid sqlite files are preserved and skipped by the sweep.
set -euo pipefail

ROOT="${ROOT:-/home/scratch.noliu_gpu/pr429-verify}"
OUT_DIR="${OUT_DIR:-/home/scratch.noliu_gpu/nsys_pr429_timed_sweep}"

export ROOT OUT_DIR

python3 - "${OUT_DIR}" <<'PY' |
import os
import sqlite3
import sys

root = sys.argv[1]
for name in sorted(os.listdir(root)):
    if not name.endswith(".sqlite"):
        continue
    if "_hl2048_" not in name and "_hl4096_" not in name:
        continue
    path = os.path.join(root, name)
    try:
        con = sqlite3.connect(path)
        row = con.execute(
            """
            SELECT COUNT(*) FROM NVTX_EVENTS
            WHERE eventType=59 AND (
              text LIKE 'scenario%_timed_run_%'
              OR text LIKE 'no_cache_timed_run_%'
            )
            """
        ).fetchone()
        con.close()
        valid = bool(row and row[0] == 10)
    except sqlite3.Error:
        valid = False
    if not valid:
        print(os.path.splitext(path)[0])
PY
while IFS= read -r stem; do
  echo "[rerun] remove invalid result: ${stem}"
  rm -f "${stem}.sqlite" "${stem}.nsys-rep" "${stem}.log"
done

# This result is known to be missing after the duplicate-process failure.
rm -f \
  "${OUT_DIR}/only_onboard_s3_ssd_hl2048_bs8.sqlite" \
  "${OUT_DIR}/only_onboard_s3_ssd_hl2048_bs8.nsys-rep" \
  "${OUT_DIR}/only_onboard_s3_ssd_hl2048_bs8.log"

export ONLY_HISTORY_LENS="2048 4096"
exec bash "${ROOT}/scripts/run_sweep_in_container.sh"
