#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc


CASE_RE = re.compile(r"test case\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")
TIME_RE = re.compile(r"time\(ms\)\s*([0-9]*\.?[0-9]+)")


def parse_paged_log(log_path: Path) -> List[Dict[str, float]]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: List[Dict[str, float]] = []
    pending: Optional[Tuple[int, int, int, int]] = None

    for line in lines:
        m_case = CASE_RE.search(line)
        if m_case:
            pending = tuple(map(int, m_case.groups()))  # bs, total, new, targets
            continue

        m_time = TIME_RE.search(line)
        if m_time and pending is not None:
            bs, total, new, targets = pending
            rows.append(
                {
                    "bs": float(bs),
                    "total": float(total),
                    "new": float(new),
                    "targets": float(targets),
                    "cached": float(total - new),
                    "time_ms": float(m_time.group(1)),
                }
            )
            pending = None

    return rows


def _build_grouped_positions(
    rows: List[Dict[str, float]], totals: List[int], bar_width: float, group_gap: float
) -> Tuple[List[float], List[float], List[str], List[Tuple[float, str]], List[float]]:
    by_total: Dict[int, List[Dict[str, float]]] = {}
    for t in totals:
        by_total[t] = sorted(
            [r for r in rows if int(r["total"]) == t],
            key=lambda z: int(z["cached"]),
        )

    max_bars = max((len(v) for v in by_total.values()), default=1)
    group_span = max_bars * bar_width

    xs: List[float] = []
    ys: List[float] = []
    xlabels: List[str] = []
    group_labels: List[Tuple[float, str]] = []
    separators: List[float] = []

    cursor = 0.0
    for idx, total in enumerate(totals):
        group = by_total[total]
        start = cursor
        for j, r in enumerate(group):
            x = start + j * bar_width
            xs.append(x)
            ys.append(r["time_ms"])
            xlabels.append(f"cached={int(r['cached'])}")

        if group:
            center = start + (len(group) - 1) * bar_width / 2
            group_labels.append((center, f"total_len={total}"))
        else:
            group_labels.append((start + group_span / 2, f"total_len={total}"))

        cursor += group_span + group_gap
        if idx < len(totals) - 1:
            separators.append(cursor - group_gap / 2)

    return xs, ys, xlabels, group_labels, separators


def plot_for_batch(
    all_rows: List[Dict[str, float]],
    batch_size: int,
    targets: int,
    out_path: Path,
) -> None:
    rows = [
        r
        for r in all_rows
        if int(r["bs"]) == batch_size and int(r["targets"]) == targets
    ]
    if not rows:
        raise RuntimeError(
            f"No rows found for batch_size={batch_size}, targets={targets}."
        )

    totals = sorted({int(r["total"]) for r in rows})
    bar_width = 0.22
    group_gap = 0.7
    xs, ys, xlabels, group_labels, separators = _build_grouped_positions(
        rows, totals, bar_width=bar_width, group_gap=group_gap
    )

    fig, ax = plt.subplots(figsize=(22, 8))
    bars = ax.bar(xs, ys, width=bar_width * 0.92, color="#3f8f2f")

    # Add numeric value on each bar top.
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + max(0.03, 0.01 * max(ys)),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    ax.set_title(
        f"Performance of HSTU block with KV cache on L20 (Batch Size = {batch_size})",
        fontsize=18,
    )
    ax.set_ylabel("Time (ms)", fontsize=14)
    ax.set_xlabel(
        f"Cached length under each total sequence length (targets={targets})",
        fontsize=14,
        labelpad=50,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=12)

    for sep in separators:
        ax.axvline(sep, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    y_min, y_max = ax.get_ylim()
    extra = (y_max - y_min) * 0.22
    ax.set_ylim(y_min, y_max + extra)
    # Move per-group "total_len=..." labels downward in axis coordinates.
    # This adjusts the group labels themselves, instead of changing plot title spacing.
    group_text_y = -0.36
    for center, glabel in group_labels:
        ax.text(
            center,
            group_text_y,
            glabel,
            ha="center",
            va="top",
            fontsize=12,
            clip_on=False,
            transform=ax.get_xaxis_transform(),
        )

    fig.subplots_adjust(bottom=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped HSTU paged benchmark figures from paged_hstu_with_kvcache.log"
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--log_file",
        type=Path,
        default=root / "L20" / "paged_hstu_with_kvcache.log",
        help="Path to paged_hstu_with_kvcache.log",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root,
        help="Output directory for generated png files",
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=256,
        help="Filter rows by num_targets",
    )
    args = parser.parse_args()

    rows = parse_paged_log(args.log_file)
    if not rows:
        raise RuntimeError(
            f"No benchmark rows parsed from {args.log_file}. "
            "Please ensure the log is complete."
        )

    out1 = args.out_dir / "hstu_inference_l20_batch1.png"
    out8 = args.out_dir / "hstu_inference_l20_batch8.png"

    plot_for_batch(rows, batch_size=1, targets=args.targets, out_path=out1)
    plot_for_batch(rows, batch_size=8, targets=args.targets, out_path=out8)

    print(f"[OK] {out1}")
    print(f"[OK] {out8}")


if __name__ == "__main__":
    main()
