#!/usr/bin/env python3
"""One dual-axis latency / hit plot from a cache-ratio jsonl."""
from __future__ import print_function

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
HERE = os.path.dirname(os.path.abspath(__file__))
# CPU/Redis dual jsonl has no emb_wait_* unless CACHE_RATIO_EMB_WAIT_US > 0.
# Hit rate only depends on GPU embedding slots and the lookup stream, so the
# wait sidecar at the same GPU budget is a valid replay.
SIDECARS = {
    16: [
        os.path.join(HERE, "cache_ratio_ordinary_gpu_dual_kv128_16gib.jsonl"),
        os.path.join(HERE, "cache_ratio_ordinary_gpu_case3_match_16gib.jsonl"),
        os.path.join(HERE, "cache_ratio_ordinary_gpu_dual_wait2_16gib.jsonl"),
    ],
    32: [
        os.path.join(HERE, "cache_ratio_ordinary_gpu_dual_kv128_32gib.jsonl"),
        os.path.join(HERE, "cache_ratio_ordinary_gpu_dual_wait14_32gib.jsonl"),
    ],
    64: [
        os.path.join(HERE, "cache_ratio_ordinary_gpu_dual_wait14_64gib.jsonl"),
    ],
}


def load_rows(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows.sort(key=lambda row: row["kv_fraction"])
    return rows


def emb_hit(row):
    lookups = row.get("emb_wait_lookups")
    miss = row.get("emb_wait_miss_ids")
    if lookups:
        return 1.0 - float(miss) / float(lookups)
    return None


def sidecar_emb_by_fraction(budget_bytes):
    gib = budget_bytes / (1024.0 ** 3)
    key = min(SIDECARS, key=lambda size: abs(size - gib))
    if abs(key - gib) > 0.2:
        return {}
    for path in SIDECARS[key]:
        if not os.path.isfile(path):
            continue
        hits = {row["kv_fraction"]: emb_hit(row) for row in load_rows(path)}
        if hits and all(value is not None for value in hits.values()):
            return hits
    return {}


def plot_jsonl(
    jsonl,
    png,
    title,
    trade=None,
    skip_trade=False,
    mark_label="折中",
    emb_knee=None,
):
    rows = load_rows(jsonl)
    if not rows:
        raise SystemExit("no rows in {0}".format(jsonl))
    xs = [row["kv_fraction"] for row in rows]
    means = [row["mean_ms"] for row in rows]
    kv = [
        row["kv_hit_last"]
        if row.get("kv_hit_last") is not None
        else row.get("kv_gpu_hit_last")
        for row in rows
    ]
    kv_host = max((row.get("kv_host_hit_last") or 0.0) for row in rows)
    kv_label = (
        "KV 命中率"
        if rows[0].get("kv_backend") == "flexkv" or kv_host > 0
        else "GPU KV 命中率"
    )
    emb = [emb_hit(row) for row in rows]
    if any(value is None for value in emb):
        fallback = sidecar_emb_by_fraction(rows[0].get("budget_bytes", 0))
        emb = [fallback.get(frac, None) for frac in xs]
    if skip_trade:
        trade = None
    elif trade is None:
        trade = min(rows, key=lambda row: row["mean_ms"])["kv_fraction"]
    i_trade = xs.index(trade) if trade is not None else None
    ymin = min(means) - 0.08
    ymax = max(means) + 0.08
    if ymax - ymin < 0.25:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - 0.125, mid + 0.125

    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    ax2 = ax.twinx()
    (l1,) = ax.plot(xs, means, "-o", color="#2e7d32", markersize=5, zorder=6)
    handles = [l1]
    labels = ["平均延迟"]
    if all(value is not None for value in emb):
        (l_emb,) = ax2.plot(xs, emb, "-o", color="#ef6c00", markersize=5, zorder=6)
        handles.append(l_emb)
        labels.append("embedding 命中率")
    if trade is not None:
        ax.scatter(
            [trade],
            [means[i_trade]],
            s=42,
            color="#d32f2f",
            zorder=8,
            edgecolors="white",
            linewidths=0.6,
        )
        if mark_label == "最低点":
            text_xy = (0.12, ymin + 0.60 * (ymax - ymin))
        else:
            text_xy = (
                max(0.08, trade - 0.28),
                means[i_trade] + 0.12 * (ymax - ymin),
            )
        ax.annotate(
            "{0} {1:.2f}，{2:.3f} ms".format(
                mark_label, trade, means[i_trade]
            ),
            xy=(trade, means[i_trade]),
            xytext=text_xy,
            color="#d32f2f",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#d32f2f", lw=0.8),
        )
        trade_handle = Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d32f2f",
            markeredgecolor="white",
            markersize=7,
        )
        handles.append(trade_handle)
        labels.append(mark_label)
    if emb_knee is not None and all(value is not None for value in emb):
        ax.axvline(emb_knee, color="#ef6c00", ls="--", lw=0.8, zorder=3)
        i_knee = min(range(len(xs)), key=lambda i: abs(xs[i] - emb_knee))
        ax2.annotate(
            "{0:.2f} embedding 掉速加快".format(xs[i_knee]),
            xy=(xs[i_knee], emb[i_knee]),
            xytext=(
                max(0.08, xs[i_knee] - 0.28),
                min(1.02, emb[i_knee] + 0.14),
            ),
            color="#ef6c00",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#ef6c00", lw=0.8),
        )
    (l_kv,) = ax2.plot(xs, kv, "-o", color="#1565c0", markersize=5, zorder=6)
    handles.append(l_kv)
    labels.append(kv_label)
    ax.set_xlabel("KV 占预算的比例")
    ax.set_ylabel("平均延迟 (ms)")
    ax2.set_ylabel("命中率")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin, ymax)
    ax2.set_ylim(0.0, 1.05)
    if title:
        ax.set_title(title)
    ax.grid(True, color="#d9d9d9")
    ax.set_axisbelow(True)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    if trade is None:
        i_min = means.index(min(means))
        print(
            "wrote {0} n={1} min={2:.3f} at {3:.2f} (no trade marker)".format(
                png, len(xs), means[i_min], xs[i_min]
            )
        )
    else:
        print(
            "wrote {0} n={1} min={2:.3f} at {3:.2f}".format(
                png, len(xs), means[i_trade], trade
            )
        )


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: plot_dual_latency_hit.py JSONL PNG TITLE [TRADE_F|none] [MARK] [KNEE]"
        )
    skip_trade = False
    trade = None
    if len(sys.argv) >= 5:
        raw = sys.argv[4]
        if raw.lower() in ("none", "no", "-"):
            skip_trade = True
        elif raw.lower() != "auto":
            trade = float(raw)
    mark_label = sys.argv[5] if len(sys.argv) >= 6 else "折中"
    emb_knee = float(sys.argv[6]) if len(sys.argv) >= 7 else None
    plot_jsonl(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        trade,
        skip_trade,
        mark_label=mark_label,
        emb_knee=emb_knee,
    )


if __name__ == "__main__":
    main()
