import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


def parse_int_list(spec: str, arg_name: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{arg_name} cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"{arg_name} values must be positive integers")
    return values


def find_nvtx_csv(csv_root: str, mode_dir: str, case_tag: str) -> str:
    mode_path = os.path.join(csv_root, mode_dir)
    pattern = os.path.join(mode_path, f"{case_tag}*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        available = sorted(glob.glob(os.path.join(mode_path, "*.csv")))
        available_short = [os.path.basename(p) for p in available[:10]]
        raise FileNotFoundError(
            "NVTX CSV not found. "
            f"pattern={pattern}, mode_dir={mode_dir}, case_tag={case_tag}, "
            f"available_examples={available_short}"
        )

    # Compatible with multiple nsys report names:
    # nvtxsum / nvtxppsum / nvtx_pushpop_sum / other nvtx*.csv variants.
    def priority(path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        if "nvtxsum" in name:
            return (0, name)
        if "nvtxppsum" in name:
            return (1, name)
        if "nvtx_pushpop_sum" in name:
            return (2, name)
        if "nvtx" in name:
            return (3, name)
        return (4, name)

    matches = sorted(matches, key=priority)
    return matches[0]


def detect_label_and_time_columns(df: pd.DataFrame) -> Tuple[str, str, float]:
    label_candidates = ["Range", "Name", "NVTX Range"]
    time_candidates = [
        ("Total Time (ns)", 1e-6),
        ("Total Time (us)", 1e-3),
        ("Total Time (ms)", 1.0),
        ("Total Time", 1.0),
    ]

    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find NVTX label column in {list(df.columns)}")

    time_col = None
    scale_to_ms = None
    for c, scale in time_candidates:
        if c in df.columns:
            time_col = c
            scale_to_ms = scale
            break
    if time_col is None:
        raise ValueError(f"Cannot find NVTX time column in {list(df.columns)}")
    return label_col, time_col, scale_to_ms


def default_prefixes_for_mode(mode: str) -> List[str]:
    if mode == "pipeline_coarse":
        return ["pipeline."]
    if mode == "flexkv_coarse":
        return ["flexkv."]
    if mode == "flexkv_fine":
        return ["flexkv.adapter.", "flexkv.client.", "flexkv._"]
    return [""]


def default_prefixes_for_step_flow() -> List[str]:
    return ["step1.", "step2.", "step3."]


def matches_any_prefix(label: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return True
    for prefix in prefixes:
        if label.startswith(prefix):
            return True
        if "::" in label:
            suffix = label.split("::", 1)[1]
            if suffix.startswith(prefix):
                return True
    return False


def load_mode_breakdown(
    csv_root: str,
    mode_dir: str,
    x_values: List[int],
    case_pattern: str,
    include_prefixes: List[str],
    group_by_step: bool = False,
    group_by_step_op: bool = False,
) -> pd.DataFrame:
    records: Dict[int, Dict[str, float]] = {}
    for x in x_values:
        case_tag = case_pattern.format(value=x)
        csv_path = find_nvtx_csv(csv_root, mode_dir, case_tag)
        df = pd.read_csv(csv_path, comment="#")
        label_col, time_col, scale_to_ms = detect_label_and_time_columns(df)
        df = df[[label_col, time_col]].copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)
        df["time_ms"] = df[time_col] * scale_to_ms
        # nsys CSV often prefixes range names with ":" (e.g. ":step1.lookup").
        # Normalize once so prefix filtering/grouping works for both forms.
        df["label_norm"] = (
            df[label_col].astype(str).str.strip().str.lstrip(":")
        )
        if include_prefixes:
            mask = df["label_norm"].apply(lambda x: matches_any_prefix(x, include_prefixes))
            df = df[mask]
        if group_by_step:
            # For step-flow wall-clock view, keep only top-level step labels.
            # Exclude nested internal labels, e.g. step1.lookup::flexkv.lookup_kvcache.
            df = df[~df["label_norm"].str.contains("::", regex=False)]
            df["group_label"] = (
                df["label_norm"]
                .apply(
                    lambda s: re.match(r"(step\d+)\.", s).group(1)
                    if re.match(r"(step\d+)\.", s)
                    else s
                )
            )
        elif group_by_step_op:
            # Keep only top-level step operations, e.g. step1.lookup.
            # Exclude nested internal labels, e.g. step1.lookup::flexkv.lookup_kvcache.
            df = df[~df["label_norm"].str.contains("::", regex=False)]
            df["group_label"] = (
                df["label_norm"]
                .apply(
                    lambda s: re.match(r"(step\d+\.[^.:\s]+)", s).group(1)
                    if re.match(r"(step\d+\.[^.:\s]+)", s)
                    else s
                )
            )
        else:
            df["group_label"] = df["label_norm"]
        grouped = df.groupby("group_label")["time_ms"].sum().to_dict()
        records[x] = grouped

    breakdown_df = pd.DataFrame.from_dict(records, orient="index").fillna(0.0)
    breakdown_df.index.name = "x_value"
    return breakdown_df.sort_index()


def aggregate_step_op_avg7(df: pd.DataFrame) -> pd.DataFrame:
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["lookup_avg(step1,step3)"] = (col("step1.lookup") + col("step3.lookup")) / 2.0
    out["allocate_avg(step1,step3)"] = (col("step1.allocate") + col("step3.allocate")) / 2.0
    out["step1.offload_launch"] = col("step1.offload_launch")
    out["step1.offload_wait"] = col("step1.offload_wait")
    out["step2.evict_gpu"] = col("step2.evict_gpu")
    out["step3.onboard_launch"] = col("step3.onboard_launch")
    out["step3.onboard_wait"] = col("step3.onboard_wait")
    out.index.name = df.index.name
    return out


def add_value_labels_with_arrows(
    ax: plt.Axes,
    x_positions: List[float],
    heights_by_component: List[List[float]],
    totals: List[float],
    small_ratio: float = 0.07,
) -> None:
    max_total = max(totals) if totals else 0.0
    if max_total <= 0:
        return

    # Prepare renderer and pixel-scale metrics to decide whether text can fit
    # inside each stacked segment. If it can fit, we place it inside; otherwise
    # we fallback to arrow annotations.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    font_size = 8
    font_prop = FontProperties(size=font_size)
    x0_px = ax.transData.transform((0.0, 0.0))[0]
    x1_px = ax.transData.transform((0.65, 0.0))[0]
    bar_width_px = abs(x1_px - x0_px)
    small_abs = max(max_total * small_ratio, 0.1)
    per_bar_arrow_count = [0 for _ in x_positions]

    bottoms = [0.0 for _ in x_positions]
    for comp_vals in heights_by_component:
        for i, v in enumerate(comp_vals):
            if v <= 0:
                bottoms[i] += v
                continue
            x = x_positions[i]
            y_center = bottoms[i] + v / 2.0
            # Keep a stable fixed precision for chart labels.
            label = f"{v:.5f}"
            text_w_px, text_h_px, _ = renderer.get_text_width_height_descent(
                label, font_prop, ismath=False
            )
            yb0 = ax.transData.transform((0.0, bottoms[i]))[1]
            yb1 = ax.transData.transform((0.0, bottoms[i] + v))[1]
            seg_h_px = abs(yb1 - yb0)
            can_fit_inside = (
                v >= small_abs
                and text_w_px <= bar_width_px * 0.9
                and text_h_px <= seg_h_px * 0.8
            )
            if can_fit_inside:
                ax.text(
                    x,
                    y_center,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                    clip_on=True,
                )
            else:
                per_bar_arrow_count[i] += 1
                y_text = totals[i] + per_bar_arrow_count[i] * (max_total * 0.04)
                ax.annotate(
                    label,
                    xy=(x, y_center),
                    xytext=(x + 0.15, y_text),
                    textcoords="data",
                    ha="left",
                    va="bottom",
                    fontsize=font_size,
                    arrowprops=dict(arrowstyle="->", lw=0.8),
                )
            bottoms[i] += v


def plot_stacked_breakdown(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    output_png: str,
    topk: int,
) -> None:
    if df.empty:
        raise ValueError("Breakdown dataframe is empty, nothing to plot.")

    totals_by_component = df.sum(axis=0).sort_values(ascending=False)
    keep_components = list(totals_by_component.head(topk).index)
    other_components = [c for c in df.columns if c not in keep_components]

    plot_df = df[keep_components].copy()
    if other_components:
        plot_df["others"] = df[other_components].sum(axis=1)

    x_values = list(plot_df.index)
    x_positions = list(range(len(x_values)))
    bottoms = [0.0 for _ in x_values]
    heights_by_component: List[List[float]] = []

    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap("tab20")
    for i, comp in enumerate(plot_df.columns):
        vals = plot_df[comp].tolist()
        heights_by_component.append(vals)
        plt.bar(
            x_positions,
            vals,
            bottom=bottoms,
            label=comp,
            color=cmap(i % 20),
            width=0.65,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    add_value_labels_with_arrows(
        plt.gca(),
        x_positions=x_positions,
        heights_by_component=heights_by_component,
        totals=bottoms,
        small_ratio=0.07,
    )

    plt.xticks(x_positions, [str(s) for s in x_values])
    plt.xlabel(x_label)
    plt.ylabel("Total Time (ms)")
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked breakdown across sequence lengths."
    )
    parser.add_argument("--csv-root", required=True, help="CSV root directory")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pipeline_coarse", "flexkv_coarse", "flexkv_fine"],
        help="Profiling mode",
    )
    parser.add_argument(
        "--mode-dir",
        default="",
        help="Optional subdirectory name under csv-root (default: same as --mode)",
    )
    parser.add_argument("--seq-lens", default="256,512,1024,2048")
    parser.add_argument(
        "--batch-sizes",
        default="",
        help="Comma-separated batch sizes, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--x-kind",
        choices=["seq_len", "batch_size"],
        default="seq_len",
        help="X-axis dimension and case selection dimension",
    )
    parser.add_argument(
        "--case-pattern",
        default="",
        help=(
            "Case tag pattern used to match CSV filename (supports {value}), "
            "e.g. seq{value}, bs{value}, len1024_bs{value}"
        ),
    )
    parser.add_argument(
        "--view",
        choices=["mode_breakdown", "step_flow", "step_op_flow", "step_op_avg7"],
        default="mode_breakdown",
        help=(
            "mode_breakdown: plot selected NVTX labels directly; "
            "step_flow: aggregate labels into step1/step2/step3; "
            "step_op_flow: aggregate into top-level step ops (step1.lookup etc.); "
            "step_op_avg7: average step1/step3 lookup+allocate into 7 ops"
        ),
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-K ranges to keep")
    parser.add_argument(
        "--include-prefixes",
        default="",
        help="Comma-separated NVTX range prefixes to include; empty uses mode defaults",
    )
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    if args.x_kind == "seq_len":
        x_values = parse_int_list(args.seq_lens, "--seq-lens")
        default_case_pattern = "seq{value}"
        x_label = "Sequence Length"
    else:
        x_values = parse_int_list(args.batch_sizes, "--batch-sizes")
        default_case_pattern = "bs{value}"
        x_label = "Batch Size"

    mode_dir = args.mode_dir.strip() if args.mode_dir.strip() else args.mode
    case_pattern = args.case_pattern.strip() if args.case_pattern.strip() else default_case_pattern
    if "{value}" not in case_pattern:
        raise ValueError("--case-pattern must contain '{value}' placeholder")

    if args.include_prefixes.strip():
        prefixes = [p.strip() for p in args.include_prefixes.split(",") if p.strip()]
    elif args.view in ("step_flow", "step_op_flow", "step_op_avg7"):
        prefixes = default_prefixes_for_step_flow()
    else:
        prefixes = default_prefixes_for_mode(args.mode)

    df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=mode_dir,
        x_values=x_values,
        case_pattern=case_pattern,
        include_prefixes=prefixes,
        group_by_step=(args.view == "step_flow"),
        group_by_step_op=(args.view in ("step_op_flow", "step_op_avg7")),
    )
    if args.view == "step_op_avg7":
        df = aggregate_step_op_avg7(df)
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv)

    if args.view == "step_flow":
        title = f"{args.mode} Step Flow Breakdown"
    elif args.view == "step_op_flow":
        title = f"{args.mode} Step-Op Breakdown"
    elif args.view == "step_op_avg7":
        title = f"{args.mode} Step-Op Avg7 Breakdown"
    else:
        title = f"{args.mode} Time Breakdown"

    effective_topk = args.topk
    if args.view == "step_op_flow":
        effective_topk = max(args.topk, 9)
    if args.view == "step_op_avg7":
        effective_topk = max(args.topk, 7)
    plot_stacked_breakdown(
        df=df,
        title=title,
        x_label=x_label,
        output_png=args.output_png,
        topk=effective_topk,
    )
    print(f"Saved plot: {args.output_png}")
    if args.output_csv:
        print(f"Saved table: {args.output_csv}")


if __name__ == "__main__":
    main()
