#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc


STAGES: List[Tuple[str, str, List[str]]] = [
    ("get_total_cache_length", "get_total_cache_length", ["PKA: get_total_cache_length"]),
    ("compute_new_tokens", "compute_new_tokens", ["PKA: compute_new_tokens"]),
    ("alloc_offload_uids_buffer", "alloc_offload_uids_buffer", ["PKA: alloc_offload_uids_buffer"]),
    ("alloc_metadata_host_buffer", "alloc_metadata_host_buffer", ["PKA: alloc_metadata_host_buffer"]),
    ("submit_prepare_kvcache", "submit_prepare_kvcache", ["PKA: submit_prepare_kvcache"]),
    ("reset_onload_handle", "reset_onload_handle", ["PKA: reset_onload_handle"]),
    ("submit_onload_kvcache", "submit_onload_kvcache", ["PKA: submit_onload_kvcache"]),
    ("wait_prepare_future", "wait_prepare_future", ["PKW: wait_prepare_future"]),
    ("build_metadata_from_buffer", "build_metadata_from_buffer", ["PKW: build_metadata_from_buffer"]),
    ("worker_prepare_kvcache_op", "worker_prepare_kvcache_op", ["PKA_WORKER: prepare_kvcache_op"]),
    ("worker_onload_kvcache_op", "worker_onload_kvcache_op", ["PKA_WORKER: onload_kvcache_op"]),
]


def _normalize_name(name: str) -> str:
    n = name.strip()
    while n.startswith(":"):
        n = n[1:].strip()
    return n


def _pick_col(columns: List[str], keys: List[str]) -> str:
    for c in columns:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    raise RuntimeError(f"Cannot find column with keys={keys} in {columns}")


def parse_nsys_nvtx_avg_ms(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    cols = list(rows[0].keys())
    name_col = _pick_col(cols, ["range"])
    avg_col = _pick_col(cols, ["avg", "ns"])

    out: Dict[str, float] = {}
    for row in rows:
        name = _normalize_name(row.get(name_col, ""))
        if not name:
            continue
        avg_ns = float(str(row.get(avg_col, "0")).replace(",", "") or 0.0)
        out[name] = avg_ns / 1e6
    return out


def build_stage_values(nsys_avg_map: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, _, aliases in STAGES:
        value = 0.0
        for alias in aliases:
            alias_norm = _normalize_name(alias)
            if alias_norm in nsys_avg_map:
                value = nsys_avg_map[alias_norm]
                break
        out[key] = value
    return out


def save_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["batch_size"] + [k for k, _, _ in STAGES] + ["total_prepare_ms"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = {k: f"{float(row[k]):.6f}" for k in fields if k != "batch_size"}
            data["batch_size"] = int(row["batch_size"])
            writer.writerow(data)


def plot_one(row: Dict[str, float], out_png: Path) -> None:
    stage_labels = [label for _, label, _ in STAGES]
    values = [float(row[k]) for k, _, _ in STAGES]
    y = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(y, values, color="#4E79A7")
    ax.set_yticks(y)
    ax.set_yticklabels(stage_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Time (ms)")
    ax.set_title(f"Prepare Fine Breakdown (batch_size={int(row['batch_size'])})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    vmax = max(values) if values else 1.0
    for i, b in enumerate(bars):
        v = values[i]
        ax.text(
            b.get_width() + max(0.01, vmax * 0.02),
            b.get_y() + b.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
        )
    ax.set_xlim(0, vmax * 1.25 if vmax > 0 else 1.0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot fine-grained prepare_kvcache_async breakdown from nsys CSV."
    )
    parser.add_argument(
        "--nsys_dir",
        type=Path,
        default=Path(
            "/workspace/recsys-examples/worktrees/main-baseline/examples/hstu/inference/benchmark/benchmark_with_FlexKV/L20"
        ),
        help="Directory containing *_nvtx_pushpop_sum.csv files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="e2e_nvtx_main_kvcache_prepare_deatil",
        help="Prefix used in nsys csv names, e.g. <prefix>_bs1_nvtx_pushpop_sum.csv",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="1,8",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "plot",
        help="Output directory for prepare breakdown figures/csv.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="l20_nvtx_prepare_fine",
        help="Output file prefix.",
    )
    args = parser.parse_args()

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []
    for bs in batches:
        csv_path = args.nsys_dir / f"{args.prefix}_bs{bs}_nvtx_pushpop_sum.csv"
        avg_map = parse_nsys_nvtx_avg_ms(csv_path)
        stage_vals = build_stage_values(avg_map)
        total = sum(stage_vals.values())
        row: Dict[str, float] = {"batch_size": float(bs), **stage_vals, "total_prepare_ms": total}
        rows.append(row)

        out_png = args.out_dir / f"{args.out_prefix}_bs{bs}.png"
        plot_one(row, out_png)
        print(f"[OK] {out_png}")

    out_csv = args.out_dir / f"{args.out_prefix}.csv"
    save_csv(rows, out_csv)
    print(f"[OK] {out_csv}")


if __name__ == "__main__":
    main()
