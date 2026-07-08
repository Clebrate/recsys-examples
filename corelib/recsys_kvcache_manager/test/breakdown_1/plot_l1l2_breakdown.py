#!/usr/bin/env python3
"""Draw L1/L2 nested latency breakdown donuts from explicit timing data.

The inner ring is the L1 step-op percentage of total time. The outer ring is
the L2 function percentage of total time inside each corresponding L1 sector.
If the labeled L2 functions do not add up to the L1 time, the remainder is
shown as an unlabeled "unattributed" segment so the geometry stays correct.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
from typing import List, Tuple


OUTPUT = (
    Path(__file__).resolve().parent
    / "H100_result"
    / "L1L2_latency_breakdown_latency_bs8_len2048.png"
)

BATCH_SIZE = 8
SEQ_LEN = 2048

# seq_len=2048 timings from FLEXKV_CPU_BREAKDOWN.md.
#
# Denominator:
#   Step1 lookup + Step1 allocate + offload_launch + offload_wait
#   + onboard_launch + onboard_wait.
#
# This intentionally excludes Step2 evict_gpu and Step3 lookup/allocate.
GROUPS = [
    (
        "offload_launch",
        3.793,
        [
            ("gpu.acquire_offload_pages", 0.093),
            ("flexkv._build_slot_mappings", 0.489),
            ("flexkv.client.put_async", 2.743),
        ],
    ),
    (
        "offload_wait",
        30.655,
        [
            ("flexkv.client.try_wait", 15.180),
            ("flexkv.finish_task", 0.010),
            ("gpu.release_offload_pages", 0.074),
        ],
    ),
    ("onboard_wait", 21.244, [("flexkv.client.wait", 21.180)]),
    (
        "onboard_launch",
        1.380,
        [
            ("flexkv._build_slot_mappings", 0.623),
            ("flexkv.client.launch", 0.311),
        ],
    ),
    (
        "lookup",
        3.523,
        [
            ("gpu.lookup", 0.135),
            ("flexkv.build_index_meta", 0.428),
            ("flexkv.adapter.to_get_match_requests", 0.090),
            ("flexkv.client.get_match", 2.214),
            ("recsys.merge_lookup_results", 0.247),
        ],
    ),
    ("allocate", 1.140, [("gpu.allocate", 1.126)]),
]

COLORS = {
    "offload_launch": "#1f77b4",
    "offload_wait": "#ff7f0e",
    "lookup": "#2ca02c",
    "allocate": "#d62728",
    "onboard_launch": "#9467bd",
    "onboard_wait": "#8c564b",
}


def pct(value: float, total: float) -> str:
    return f"{value / total * 100.0:.2f}%"


def polar_point(degrees: float, radius: float) -> Tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians) * radius, math.sin(radians) * radius


def add_unattributed(
    l1_value: float, children: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    child_total = sum(value for _, value in children)
    remainder = max(l1_value - child_total, 0.0)
    if remainder > 1e-9:
        return children + [("unattributed", remainder)]
    return children


def draw() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = sum(value for _, value, _ in GROUPS)
    fig, ax = plt.subplots(figsize=(10.24, 7.68))

    inner_radius = 0.70
    inner_width = 0.22
    outer_radius = 1.08
    outer_width = 0.28
    # Rotate the chart so lookup sits near the bottom and its labels do not
    # need to cross the large transfer-wait sectors.
    start_angle = 240.0
    callouts = []
    legend_handles = []

    for group_name, l1_value, raw_children in GROUPS:
        color = COLORS[group_name]
        span = l1_value / total * 360.0
        theta1 = start_angle - span
        theta2 = start_angle

        ax.add_patch(
            Wedge(
                (0, 0),
                inner_radius,
                theta1,
                theta2,
                width=inner_width,
                facecolor=color,
                edgecolor="white",
                linewidth=2.0,
            )
        )

        mid = (theta1 + theta2) / 2.0
        if l1_value / total >= 0.015:
            x, y = polar_point(mid, inner_radius - inner_width / 2.0)
            ax.text(
                x,
                y,
                pct(l1_value, total),
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            )

        legend_handles.append(Patch(color=color, label=group_name))

        child_cursor = theta2
        for child_idx, (child_name, child_value) in enumerate(
            add_unattributed(l1_value, raw_children)
        ):
            child_span = child_value / l1_value * span if l1_value > 0 else 0.0
            child_theta1 = child_cursor - child_span
            child_theta2 = child_cursor
            is_unattributed = child_name == "unattributed"

            ax.add_patch(
                Wedge(
                    (0, 0),
                    outer_radius,
                    child_theta1,
                    child_theta2,
                    width=outer_width,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.30 if is_unattributed else 0.92,
                )
            )

            if not is_unattributed and child_value / total >= 0.00015:
                child_mid = (child_theta1 + child_theta2) / 2.0
                anchor = polar_point(child_mid, outer_radius - outer_width / 2.0)
                label_x, label_y = polar_point(child_mid, outer_radius + 0.16)
                side = 1 if label_x >= 0 else -1
                callouts.append(
                    {
                        "anchor": anchor,
                        "x": label_x,
                        "y": label_y,
                        "side": side,
                        "label": f"{child_name} {pct(child_value, total)}",
                        "color": color,
                    }
                )

            child_cursor = child_theta1

        start_angle = theta1

    # Reduce vertical label collisions independently on each side.
    min_gap = 0.075
    for side in (-1, 1):
        side_callouts = sorted(
            [callout for callout in callouts if callout["side"] == side],
            key=lambda callout: callout["y"],
        )
        previous_y = None
        for callout in side_callouts:
            if previous_y is not None and callout["y"] - previous_y < min_gap:
                callout["y"] = previous_y + min_gap
            previous_y = callout["y"]

    for callout in callouts:
        ax.annotate(
            callout["label"],
            xy=callout["anchor"],
            xytext=(callout["x"], callout["y"]),
            ha="left" if callout["side"] > 0 else "right",
            va="center",
                fontsize=5.6,
            arrowprops=dict(arrowstyle="-", color=callout["color"], lw=0.7),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78),
        )

    ax.set_title(
        f"L1/L2_latency_breakdown percentage\nbs={BATCH_SIZE}, seq_len={SEQ_LEN}",
        fontsize=14,
    )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.94, 0.5),
        frameon=False,
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.38, 1.50)
    ax.set_ylim(-1.25, 1.16)
    ax.axis("off")
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    draw()
