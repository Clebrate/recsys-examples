#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc


LOG_NAME_RE = re.compile(r"^(?P<mode>.+)_bs(?P<bs>\d+)\.log$")
TOTAL_TIME_RE = re.compile(r"Total time\(ms\):\s*([0-9]*\.?[0-9]+)")


def _mode_display_name(mode: str) -> str:
    mapping = {
        "baseline_nokvcache": "baseline_nokvcache",
        "kvcache_nop": "kvcache_nop",
        "kvcache_flexkv_direct": "kvcache_flexkv_direct",
        "kvcache_flexkv_sc": "kvcache_flexkv_sc",
    }
    return mapping.get(mode, mode)


def _mode_sort_key(mode: str) -> int:
    priority = {
        "baseline_nokvcache": 0,
        "kvcache_nop": 1,
        "kvcache_flexkv_direct": 2,
        "kvcache_flexkv_sc": 3,
    }
    return priority.get(mode, 99)


def parse_total_time_ms(log_path: Path) -> Optional[float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = TOTAL_TIME_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def collect_results(log_dir: Path) -> Tuple[Dict[str, Dict[int, float]], List[Path]]:
    results: Dict[str, Dict[int, float]] = {}
    skipped: List[Path] = []

    for log_path in sorted(log_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log_path.name)
        if not m:
            skipped.append(log_path)
            continue
        mode = m.group("mode")
        bs = int(m.group("bs"))
        total_time_ms = parse_total_time_ms(log_path)
        if total_time_ms is None:
            skipped.append(log_path)
            continue

        if mode not in results:
            results[mode] = {}
        results[mode][bs] = total_time_ms

    return results, skipped


def save_summary_csv(
    results: Dict[str, Dict[int, float]],
    out_csv: Path,
) -> None:
    baseline = results.get("baseline_nokvcache", {})
    all_bs = sorted({bs for m in results.values() for bs in m.keys()})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "batch_size", "total_time_ms", "speedup_vs_baseline"])
        for mode in sorted(results.keys(), key=_mode_sort_key):
            for bs in all_bs:
                t = results[mode].get(bs)
                if t is None:
                    continue
                b = baseline.get(bs)
                speedup = (b / t) if (b is not None and t > 0) else ""
                writer.writerow([mode, bs, f"{t:.6f}", speedup if speedup == "" else f"{speedup:.6f}"])


def plot_total_time(
    results: Dict[str, Dict[int, float]],
    out_png: Path,
    title: str,
) -> None:
    modes = sorted(results.keys(), key=_mode_sort_key)
    all_bs = sorted({bs for m in results.values() for bs in m.keys()})
    if not modes or not all_bs:
        raise RuntimeError("No plottable data found for total-time figure.")

    x = list(range(len(all_bs)))
    group_width = 0.82
    bar_width = group_width / max(len(modes), 1)
    offset_base = -group_width / 2 + bar_width / 2

    plt.figure(figsize=(11, 6))
    for i, mode in enumerate(modes):
        y = [results[mode].get(bs, float("nan")) for bs in all_bs]
        offsets = [xx + offset_base + i * bar_width for xx in x]
        plt.bar(offsets, y, width=bar_width, label=_mode_display_name(mode))

    plt.xticks(x, [str(bs) for bs in all_bs])
    plt.xlabel("Batch Size")
    plt.ylabel("Total Time (ms)")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_speedup_vs_baseline(
    results: Dict[str, Dict[int, float]],
    out_png: Path,
    title: str,
) -> None:
    baseline = results.get("baseline_nokvcache")
    if not baseline:
        raise RuntimeError("Cannot plot speedup: baseline_nokvcache is missing.")

    all_bs = sorted({bs for m in results.values() for bs in m.keys()})
    modes = [m for m in sorted(results.keys(), key=_mode_sort_key) if m != "baseline_nokvcache"]
    if not modes:
        raise RuntimeError("No non-baseline modes found for speedup figure.")

    plt.figure(figsize=(10, 6))
    for mode in modes:
        xs: List[int] = []
        ys: List[float] = []
        for bs in all_bs:
            t_base = baseline.get(bs)
            t_mode = results[mode].get(bs)
            if t_base is None or t_mode is None or t_mode <= 0:
                continue
            xs.append(bs)
            ys.append(t_base / t_mode)
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=_mode_display_name(mode))

    plt.xlabel("Batch Size")
    plt.ylabel("Speedup vs baseline_nokvcache (x)")
    plt.title(title)
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse L20 benchmark logs and generate plots."
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=default_root / "L20",
        help="Directory containing *.log benchmark files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_root / "plot",
        help="Directory for generated PNG/CSV outputs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="l20",
        help="Output file prefix.",
    )
    args = parser.parse_args()

    results, skipped = collect_results(args.log_dir)
    if not results:
        raise RuntimeError(f"No valid benchmark results found under: {args.log_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    total_time_png = args.out_dir / f"{args.prefix}_total_time_ms.png"
    speedup_png = args.out_dir / f"{args.prefix}_speedup_vs_baseline.png"

    save_summary_csv(results, summary_csv)
    plot_total_time(
        results,
        total_time_png,
        title="HSTU Inference Benchmark (L20): Total Time",
    )
    plot_speedup_vs_baseline(
        results,
        speedup_png,
        title="HSTU Inference Benchmark (L20): Speedup vs Baseline",
    )

    print(f"[OK] Parsed modes: {', '.join(sorted(results.keys(), key=_mode_sort_key))}")
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Total-time plot: {total_time_png}")
    print(f"[OK] Speedup plot: {speedup_png}")
    if skipped:
        print("[WARN] Skipped files:")
        for p in skipped:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
