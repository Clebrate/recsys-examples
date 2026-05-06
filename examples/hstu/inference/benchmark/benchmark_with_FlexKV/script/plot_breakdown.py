#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


@dataclass
class StageRow:
    batch_size: int
    variant: str
    build_lookup_tokens_ms: float = 0.0
    lookup_ms: float = 0.0
    prepare_kvcache_async_ms: float = 0.0
    strip_ms: float = 0.0
    allocate_kvcache_ms: float = 0.0
    embedding_ms: float = 0.0
    preprocess_ms: float = 0.0
    hstublock_ms: float = 0.0
    forward_ms: float = 0.0


def _to_float(raw: Optional[str]) -> float:
    if raw is None:
        return 0.0
    text = raw.strip()
    if not text:
        return 0.0
    return float(text)


def _normalize_range_name(name: str) -> str:
    out = name.strip()
    while out.startswith(":"):
        out = out[1:].strip()
    return out


def _variant_display_name(name: str) -> str:
    mapping = {
        "main": "main",
        "baseline": "main",
        "flexkv": "flexkv",
        "flexkv_direct": "flexkv_direct",
    }
    return mapping.get(name.strip().lower(), name.strip())


def _variant_sort_key(name: str) -> int:
    key = name.strip().lower()
    if key in {"main", "baseline"}:
        return 0
    if key in {"flexkv", "flexkv_direct"}:
        return 1
    return 99


def _resolve_preprocess_ms(row: StageRow) -> float:
    if row.preprocess_ms > 0:
        return row.preprocess_ms
    if row.forward_ms > 0 and row.hstublock_ms > 0:
        return max(row.forward_ms - row.hstublock_ms, 0.0)
    return 0.0


def _resolve_hstublock_ms(row: StageRow, preprocess_ms: float) -> float:
    if row.hstublock_ms > 0:
        return row.hstublock_ms
    if row.forward_ms > 0:
        # If hstublock is not separately marked, fallback to forward stage.
        return max(row.forward_ms - preprocess_ms, 0.0)
    return 0.0


def _pick_col(columns: List[str], keys: List[str]) -> Optional[str]:
    for c in columns:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    return None


def load_nsys_nvtx_avg_ms(nsys_csv: Path) -> Dict[str, float]:
    if not nsys_csv.exists():
        raise FileNotFoundError(f"NSYS CSV not found: {nsys_csv}")

    with nsys_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    columns = list(rows[0].keys())
    range_col = _pick_col(columns, ["range"]) or _pick_col(columns, ["name"])
    avg_ns_col = _pick_col(columns, ["avg", "ns"])
    total_ns_col = _pick_col(columns, ["total", "ns"])
    inst_col = _pick_col(columns, ["instance"]) or _pick_col(columns, ["count"])
    if range_col is None:
        raise RuntimeError(f"Cannot find range/name column in: {nsys_csv}")

    out: Dict[str, float] = {}
    for row in rows:
        name = _normalize_range_name(row.get(range_col, ""))
        if not name:
            continue

        avg_ns: Optional[float] = None
        if avg_ns_col and row.get(avg_ns_col, "").strip():
            avg_ns = float(str(row[avg_ns_col]).replace(",", ""))
        elif (
            total_ns_col
            and inst_col
            and row.get(total_ns_col, "").strip()
            and row.get(inst_col, "").strip()
        ):
            total_ns = float(str(row[total_ns_col]).replace(",", ""))
            inst = float(str(row[inst_col]).replace(",", ""))
            if inst > 0:
                avg_ns = total_ns / inst

        if avg_ns is None:
            continue
        out[name] = avg_ns / 1e6
    return out


def _lookup_ms(values: Dict[str, float], aliases: List[str]) -> float:
    for key in aliases:
        norm = _normalize_range_name(key)
        if norm in values:
            return values[norm]
    return 0.0


def _infer_batches_from_nsys(main_dir: Path, flex_dir: Path) -> List[int]:
    batch_re = re.compile(r"bs(\d+)_nvtx_pushpop_sum\.csv$")
    bs_set = set()
    for base in (main_dir, flex_dir):
        for p in base.glob("*_nvtx_pushpop_sum.csv"):
            m = batch_re.search(p.name)
            if m:
                bs_set.add(int(m.group(1)))
    return sorted(bs_set)


def build_stage_rows_from_nsys(
    main_nsys_dir: Path,
    flex_nsys_dir: Path,
    batches: List[int],
) -> List[StageRow]:
    rows: List[StageRow] = []
    for bs in batches:
        main_csv = main_nsys_dir / f"e2e_nvtx_main_kvcache_bs{bs}_nvtx_pushpop_sum.csv"
        flex_csv = (
            flex_nsys_dir
            / f"e2e_nvtx_flexkv_direct_kvcache_bs{bs}_nvtx_pushpop_sum.csv"
        )
        main_map = load_nsys_nvtx_avg_ms(main_csv)
        flex_map = load_nsys_nvtx_avg_ms(flex_csv)

        rows.append(
            StageRow(
                batch_size=bs,
                variant="main",
                prepare_kvcache_async_ms=_lookup_ms(
                    main_map, ["E2E: prepare_kvcache_async"]
                ),
                strip_ms=_lookup_ms(main_map, ["E2E: strip cached tokens"]),
                embedding_ms=_lookup_ms(main_map, ["E2E: embedding"]),
                preprocess_ms=_lookup_ms(
                    main_map, ["E2E: preprocess", "HSTUBlock preprocess forward"]
                ),
                hstublock_ms=_lookup_ms(main_map, ["E2E: hstublock"]),
                forward_ms=_lookup_ms(main_map, ["E2E: forward with kvcache"]),
            )
        )
        rows.append(
            StageRow(
                batch_size=bs,
                variant="flexkv",
                build_lookup_tokens_ms=_lookup_ms(
                    flex_map, ["E2E: build lookup tokens"]
                ),
                lookup_ms=_lookup_ms(flex_map, ["E2E: lookup kvcache"]),
                strip_ms=_lookup_ms(flex_map, ["E2E: strip cached tokens"]),
                allocate_kvcache_ms=_lookup_ms(flex_map, ["E2E: allocate kvcache"]),
                embedding_ms=_lookup_ms(flex_map, ["E2E: embedding"]),
                preprocess_ms=_lookup_ms(
                    flex_map, ["E2E: preprocess", "HSTUBlock preprocess forward"]
                ),
                hstublock_ms=_lookup_ms(flex_map, ["E2E: hstublock"]),
                forward_ms=_lookup_ms(flex_map, ["E2E: forward with kvcache"]),
            )
        )
    return rows


def load_input_rows(input_csv: Path) -> List[StageRow]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Use --write_template to generate a template first."
        )

    rows: List[StageRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("Input CSV has no header.")
        required = {"batch_size", "variant"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Missing required columns in input CSV: {sorted(missing)}"
            )

        for item in reader:
            if not item.get("batch_size", "").strip():
                continue
            rows.append(
                StageRow(
                    batch_size=int(item["batch_size"]),
                    variant=item["variant"].strip(),
                    build_lookup_tokens_ms=_to_float(item.get("build_lookup_tokens_ms")),
                    lookup_ms=_to_float(item.get("lookup_ms")),
                    prepare_kvcache_async_ms=_to_float(
                        item.get("prepare_kvcache_async_ms")
                    ),
                    strip_ms=_to_float(item.get("strip_ms")),
                    allocate_kvcache_ms=_to_float(item.get("allocate_kvcache_ms")),
                    embedding_ms=_to_float(item.get("embedding_ms")),
                    preprocess_ms=_to_float(item.get("preprocess_ms")),
                    hstublock_ms=_to_float(item.get("hstublock_ms")),
                    forward_ms=_to_float(item.get("forward_ms")),
                )
            )
    if not rows:
        raise RuntimeError(f"No valid data rows found in {input_csv}")
    return rows


def build_breakdown_record(row: StageRow) -> Dict[str, float]:
    preprocess_ms = _resolve_preprocess_ms(row)
    hstublock_ms = _resolve_hstublock_ms(row, preprocess_ms)

    lookup_component_ms = (
        row.build_lookup_tokens_ms + row.lookup_ms + row.prepare_kvcache_async_ms
    )
    a_lookup_strip_ms = lookup_component_ms + row.strip_ms
    b_pre_hstublock_ms = row.allocate_kvcache_ms + row.embedding_ms + preprocess_ms
    c_hstublock_ms = hstublock_ms
    total_ms = a_lookup_strip_ms + b_pre_hstublock_ms + c_hstublock_ms

    return {
        "batch_size": row.batch_size,
        "variant": _variant_display_name(row.variant),
        "lookup_component_ms": lookup_component_ms,
        "strip_ms": row.strip_ms,
        "allocate_ms": row.allocate_kvcache_ms,
        "embedding_ms": row.embedding_ms,
        "preprocess_ms": preprocess_ms,
        "hstublock_ms": hstublock_ms,
        "a_lookup_strip_ms": a_lookup_strip_ms,
        "b_pre_hstublock_ms": b_pre_hstublock_ms,
        "c_hstublock_ms": c_hstublock_ms,
        "total_ms": total_ms,
    }


def build_detailed_record(row: StageRow) -> Dict[str, float]:
    preprocess_ms = _resolve_preprocess_ms(row)
    hstublock_ms = _resolve_hstublock_ms(row, preprocess_ms)
    out = {
        "batch_size": row.batch_size,
        "variant": _variant_display_name(row.variant),
        "prepare_kvcache_async_ms": row.prepare_kvcache_async_ms,
        "build_lookup_tokens_ms": row.build_lookup_tokens_ms,
        "lookup_ms": row.lookup_ms,
        "allocate_kvcache_ms": row.allocate_kvcache_ms,
        "strip_ms": row.strip_ms,
        "embedding_ms": row.embedding_ms,
        "preprocess_ms": preprocess_ms,
        "hstublock_ms": hstublock_ms,
    }
    out["total_ms"] = (
        out["prepare_kvcache_async_ms"]
        + out["build_lookup_tokens_ms"]
        + out["lookup_ms"]
        + out["allocate_kvcache_ms"]
        + out["strip_ms"]
        + out["embedding_ms"]
        + out["preprocess_ms"]
        + out["hstublock_ms"]
    )
    return out


def save_breakdown_csv(records: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_size",
        "variant",
        "a_lookup_strip_ms",
        "b_pre_hstublock_ms",
        "c_hstublock_ms",
        "total_ms",
        "lookup_component_ms",
        "strip_ms",
        "allocate_ms",
        "embedding_ms",
        "preprocess_ms",
        "hstublock_ms",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (int(x["batch_size"]), _variant_sort_key(str(x["variant"])))):
            writer.writerow(
                {
                    k: (
                        f"{float(r[k]):.6f}"
                        if isinstance(r[k], float)
                        else r[k]
                    )
                    for k in fields
                }
            )


def save_detailed_csv(records: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_size",
        "variant",
        "prepare_kvcache_async_ms",
        "build_lookup_tokens_ms",
        "lookup_ms",
        "allocate_kvcache_ms",
        "strip_ms",
        "embedding_ms",
        "preprocess_ms",
        "hstublock_ms",
        "total_ms",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(
            records,
            key=lambda x: (int(x["batch_size"]), _variant_sort_key(str(x["variant"]))),
        ):
            writer.writerow(
                {
                    k: (f"{float(r[k]):.6f}" if isinstance(r[k], float) else r[k])
                    for k in fields
                }
            )


def plot_one_batch(records: List[Dict[str, float]], batch_size: int, out_png: Path) -> None:
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )
    target = [r for r in records if int(r["batch_size"]) == batch_size]
    if not target:
        raise RuntimeError(f"No records found for batch_size={batch_size}")
    target.sort(key=lambda x: _variant_sort_key(str(x["variant"])))

    x_labels = [str(r["variant"]) for r in target]
    xs = list(range(len(target)))
    a_vals = [float(r["a_lookup_strip_ms"]) for r in target]
    b_vals = [float(r["b_pre_hstublock_ms"]) for r in target]
    c_vals = [float(r["c_hstublock_ms"]) for r in target]
    totals = [float(r["total_ms"]) for r in target]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bar_width = 0.58

    bars_a = ax.bar(xs, a_vals, width=bar_width, label="(a) lookup + strip", color="#4E79A7")
    bars_b = ax.bar(
        xs,
        b_vals,
        width=bar_width,
        bottom=a_vals,
        label="(b) allocate + embedding + preprocess",
        color="#F28E2B",
    )
    bottoms_c = [a_vals[i] + b_vals[i] for i in range(len(xs))]
    bars_c = ax.bar(
        xs,
        c_vals,
        width=bar_width,
        bottom=bottoms_c,
        label="(c) hstublock",
        color="#59A14F",
    )

    max_total = max(totals) if totals else 1.0
    seg_text_threshold = max(0.06, max_total * 0.02)

    def _annotate_segment(bars, vals, bottoms, small_slots):
        for i, b in enumerate(bars):
            v = vals[i]
            if v <= 0:
                continue
            x = b.get_x() + b.get_width() / 2
            y_center = bottoms[i] + v / 2
            if v >= seg_text_threshold:
                ax.text(
                    x,
                    y_center,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )
                continue

            # For tiny segments, place labels outside with arrow.
            slot = small_slots[i]
            small_slots[i] += 1
            x_text = x + 0.50 + 0.10 * (slot // 2)
            y_text = bottoms[i] + v + max_total * (0.015 + 0.02 * (slot % 2))
            ax.annotate(
                f"{v:.3f}",
                xy=(x, y_center),
                xytext=(x_text, y_text),
                ha="left",
                va="center",
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"),
            )

    small_slots = [0 for _ in xs]
    _annotate_segment(bars_a, a_vals, [0.0] * len(xs), small_slots)
    _annotate_segment(bars_b, b_vals, a_vals, small_slots)
    _annotate_segment(bars_c, c_vals, bottoms_c, small_slots)

    for i, total in enumerate(totals):
        ax.text(
            xs[i],
            total + max_total * 0.02,
            f"{total:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Time Breakdown (batch_size={batch_size})", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.set_ylim(0, max_total * 1.15)
    ax.set_xlim(-0.6, len(xs) - 0.4 + 0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_detailed_one_batch(
    records: List[Dict[str, float]], batch_size: int, out_png: Path
) -> None:
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )
    target = [r for r in records if int(r["batch_size"]) == batch_size]
    if not target:
        raise RuntimeError(f"No detailed records found for batch_size={batch_size}")
    target.sort(key=lambda x: _variant_sort_key(str(x["variant"])))

    x_labels = [str(r["variant"]) for r in target]
    xs = list(range(len(target)))
    stack_items = [
        ("prepare_kvcache_async_ms", "prepare_kvcache_async", "#76B7B2"),
        ("build_lookup_tokens_ms", "build_lookup_tokens", "#4E79A7"),
        ("lookup_ms", "lookup", "#A0CBE8"),
        ("allocate_kvcache_ms", "allocate_kvcache", "#F28E2B"),
        ("strip_ms", "strip", "#FFBE7D"),
        ("embedding_ms", "embedding", "#59A14F"),
        ("preprocess_ms", "preprocess", "#8CD17D"),
        ("hstublock_ms", "hstublock", "#E15759"),
    ]
    totals = [float(r["total_ms"]) for r in target]
    max_total = max(totals) if totals else 1.0

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bar_width = 0.58
    bottoms = [0.0 for _ in target]
    seg_text_threshold = max(0.08, max_total * 0.03)
    small_slots = [0 for _ in xs]

    for key, label, color in stack_items:
        vals = [float(r.get(key, 0.0)) for r in target]
        bars = ax.bar(xs, vals, width=bar_width, bottom=bottoms, label=label, color=color)
        for i, b in enumerate(bars):
            v = vals[i]
            if v <= 0:
                continue
            x = b.get_x() + b.get_width() / 2
            y_center = bottoms[i] + v / 2
            if v >= seg_text_threshold:
                ax.text(
                    x,
                    y_center,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                )
            else:
                slot = small_slots[i]
                small_slots[i] += 1
                x_text = x + 0.52 + 0.10 * (slot // 2)
                y_text = bottoms[i] + v + max_total * (0.012 + 0.02 * (slot % 2))
                ax.annotate(
                    f"{v:.3f}",
                    xy=(x, y_center),
                    xytext=(x_text, y_text),
                    ha="left",
                    va="center",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"),
                )
        bottoms = [bottoms[i] + vals[i] for i in range(len(bottoms))]

    for i, total in enumerate(totals):
        ax.text(
            xs[i],
            total + max_total * 0.02,
            f"{total:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Detailed Breakdown (batch_size={batch_size})", pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    ax.set_ylim(0, max_total * 1.17)
    ax.set_xlim(-0.6, len(xs) - 0.4 + 1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "variant",
        "build_lookup_tokens_ms",
        "lookup_ms",
        "prepare_kvcache_async_ms",
        "strip_ms",
        "allocate_kvcache_ms",
        "embedding_ms",
        "preprocess_ms",
        "hstublock_ms",
        "forward_ms",
    ]
    rows = [
        {
            "batch_size": 4,
            "variant": "main",
            "prepare_kvcache_async_ms": 0.061180,
            "strip_ms": 0.683633,
            "embedding_ms": 1.560000,
            "forward_ms": 6.603000,
        },
        {
            "batch_size": 4,
            "variant": "flexkv",
            "build_lookup_tokens_ms": 0.977532,
            "lookup_ms": 1.328000,
            "allocate_kvcache_ms": 1.838000,
            "strip_ms": 0.756747,
            "embedding_ms": 1.692000,
            "forward_ms": 5.047000,
        },
        {"batch_size": 1, "variant": "main"},
        {"batch_size": 1, "variant": "flexkv"},
        {"batch_size": 8, "variant": "main"},
        {"batch_size": 8, "variant": "flexkv"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_batch_filter(raw: str) -> Optional[List[int]]:
    text = raw.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot 3-part NVTX breakdown: (a) lookup+strip, (b) pre-hstublock, (c) hstublock."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=root / "plot" / "nvtx_breakdown_input.csv",
        help="Input CSV with stage times. One row per (batch_size, variant).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "plot",
        help="Directory to save output CSV/PNG.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="nvtx",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="",
        help="Optional batch filter, e.g. '1,8'. Empty means all batches in input.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Only generate breakdown CSV, do not generate PNG plots.",
    )
    parser.add_argument(
        "--write_template",
        action="store_true",
        help="Write a template CSV to --input_csv and exit.",
    )
    parser.add_argument(
        "--from_nsys",
        action="store_true",
        help="Build input rows directly from nsys nvtx_pushpop_sum.csv files.",
    )
    parser.add_argument(
        "--main_nsys_dir",
        type=Path,
        default=Path(
            "/workspace/recsys-examples/worktrees/main-baseline/examples/hstu/inference/benchmark/benchmark_with_FlexKV/L20"
        ),
        help="Directory containing main-baseline *_nvtx_pushpop_sum.csv files.",
    )
    parser.add_argument(
        "--flex_nsys_dir",
        type=Path,
        default=root / "L20",
        help="Directory containing flexkv *_nvtx_pushpop_sum.csv files.",
    )
    parser.add_argument(
        "--dump_input_csv",
        action="store_true",
        help="Write auto-generated stage input CSV to --input_csv.",
    )
    parser.add_argument(
        "--with_detailed",
        action="store_true",
        help="Also generate detailed breakdown CSV/PNG with all split stages.",
    )
    args = parser.parse_args()

    if args.write_template:
        write_template_csv(args.input_csv)
        print(f"[OK] Template written: {args.input_csv}")
        return

    if args.from_nsys:
        selected_batches = _parse_batch_filter(args.batches)
        if selected_batches is None:
            selected_batches = _infer_batches_from_nsys(
                args.main_nsys_dir, args.flex_nsys_dir
            )
        if not selected_batches:
            raise RuntimeError(
                "No batch ids found. Use --batches or check nsys csv file names."
            )
        stage_rows = build_stage_rows_from_nsys(
            args.main_nsys_dir, args.flex_nsys_dir, selected_batches
        )
        if args.dump_input_csv:
            args.input_csv.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "batch_size",
                "variant",
                "build_lookup_tokens_ms",
                "lookup_ms",
                "prepare_kvcache_async_ms",
                "strip_ms",
                "allocate_kvcache_ms",
                "embedding_ms",
                "preprocess_ms",
                "hstublock_ms",
                "forward_ms",
            ]
            with args.input_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in stage_rows:
                    writer.writerow(
                        {
                            "batch_size": r.batch_size,
                            "variant": r.variant,
                            "build_lookup_tokens_ms": f"{r.build_lookup_tokens_ms:.6f}",
                            "lookup_ms": f"{r.lookup_ms:.6f}",
                            "prepare_kvcache_async_ms": f"{r.prepare_kvcache_async_ms:.6f}",
                            "strip_ms": f"{r.strip_ms:.6f}",
                            "allocate_kvcache_ms": f"{r.allocate_kvcache_ms:.6f}",
                            "embedding_ms": f"{r.embedding_ms:.6f}",
                            "preprocess_ms": f"{r.preprocess_ms:.6f}",
                            "hstublock_ms": f"{r.hstublock_ms:.6f}",
                            "forward_ms": f"{r.forward_ms:.6f}",
                        }
                    )
            print(f"[OK] Auto input CSV: {args.input_csv}")
    else:
        stage_rows = load_input_rows(args.input_csv)
    records = [build_breakdown_record(r) for r in stage_rows]
    detailed_records = [build_detailed_record(r) for r in stage_rows]

    selected_batches = _parse_batch_filter(args.batches)
    if selected_batches is not None:
        records = [r for r in records if int(r["batch_size"]) in set(selected_batches)]
        detailed_records = [
            r for r in detailed_records if int(r["batch_size"]) in set(selected_batches)
        ]
    if not records:
        raise RuntimeError("No data left after batch filtering.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{args.prefix}_breakdown.csv"
    save_breakdown_csv(records, out_csv)
    print(f"[OK] Breakdown CSV: {out_csv}")
    if args.with_detailed:
        detailed_csv = args.out_dir / f"{args.prefix}_breakdown_detailed.csv"
        save_detailed_csv(detailed_records, detailed_csv)
        print(f"[OK] Detailed breakdown CSV: {detailed_csv}")

    if args.no_plot:
        print("[OK] --no_plot enabled, skipped figure generation.")
        return

    all_batches = sorted({int(r["batch_size"]) for r in records})
    for bs in all_batches:
        out_png = args.out_dir / f"{args.prefix}_breakdown_bs{bs}.png"
        plot_one_batch(records, batch_size=bs, out_png=out_png)
        print(f"[OK] Plot: {out_png}")
        if args.with_detailed:
            out_png_detailed = (
                args.out_dir / f"{args.prefix}_breakdown_detailed_bs{bs}.png"
            )
            plot_detailed_one_batch(
                detailed_records, batch_size=bs, out_png=out_png_detailed
            )
            print(f"[OK] Plot: {out_png_detailed}")


if __name__ == "__main__":
    main()
