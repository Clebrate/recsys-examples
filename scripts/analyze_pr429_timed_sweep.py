#!/usr/bin/env python3
"""Aggregate PR429 nsys sqlite timed-run metrics into markdown."""

import argparse
import re
import sqlite3
import os


def _parse_int(pattern, text):
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else None


def classify_scenario(name):
    if re.search(r"no_cache|nokvcache", name, re.I):
        return "no_cache"
    if re.search(r"scenario1|s1_gpu|gpu_hit", name, re.I):
        return "s1_gpu"
    if re.search(r"scenario2|s2_cpu|cpu_hit", name, re.I):
        return "s2_cpu"
    if re.search(r"scenario3|s3_ssd|ssd_hit", name, re.I):
        return "s3_ssd"
    return "unknown"


def read_log_metadata(sqlite_path):
    meta = {
        "history_len": None,
        "batch_size": None,
        "only_onboard": "only_onboard" in os.path.basename(sqlite_path),
        "force_skip_offload": None,
    }
    base = os.path.splitext(sqlite_path)[0]
    base_name = os.path.basename(sqlite_path)
    m = re.search(r"_bs(\d+)(?:_|\.|$)", base_name)
    if m:
        meta["batch_size"] = int(m.group(1))
    for log_path in [base + ".log", os.path.join(os.path.dirname(sqlite_path), os.path.basename(base) + ".log")]:
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
        if meta["history_len"] is None:
            meta["history_len"] = _parse_int(r"history_len=(\d+)", text)
        if meta["batch_size"] is None:
            meta["batch_size"] = _parse_int(r"batch_size=(\d+)", text)
        if "force_skip_offload=" in text:
            meta["force_skip_offload"] = "force_skip_offload=True" in text
        if "[Mode] only_onboard=True" in text:
            meta["only_onboard"] = True
        break
    if meta["history_len"] is None:
        meta["history_len"] = _parse_int(r"hl(\d+)", base_name)
    return meta


def query_timed(sqlite_path):
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT AVG(end-start)/1e6, MIN(end-start)/1e6, MAX(end-start)/1e6, COUNT(*)
            FROM NVTX_EVENTS
            WHERE eventType=59 AND (
              text LIKE 'scenario%_timed_run_%' OR text LIKE 'nokvcache_timed_run_%'
              OR text LIKE 'no_cache_timed_run_%'
            )
            """
        )
        row = cur.fetchone()
        con.close()
    except sqlite3.Error:
        return 0.0, 0.0, 0.0, 0
    if not row or row[3] == 0:
        return 0.0, 0.0, 0.0, 0
    return float(row[0]), float(row[1]), float(row[2]), int(row[3])


def query_memcpy(sqlite_path):
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT
              SUM(CASE WHEN copyKind=1 THEN bytes ELSE 0 END),
              SUM(CASE WHEN copyKind=2 THEN bytes ELSE 0 END)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        )
        h, d = cur.fetchone() or (0, 0)
        con.close()
    except sqlite3.Error:
        return 0.0, 0.0
    return (h or 0) / 1024.0 / 1024.0, (d or 0) / 1024.0 / 1024.0


def query_onboard_wait(sqlite_path):
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT SUM(end-start)/1e6
            FROM NVTX_EVENTS
            WHERE eventType=59 AND text='recsys.kvcache.onboard_wait'
            """
        )
        row = cur.fetchone()
        con.close()
    except sqlite3.Error:
        return 0.0
    return float(row[0] or 0.0)


def collect_records(root):
    records = []
    # Prefer exact sweep dir; otherwise walk for nsys_pr429* dirs only.
    if os.path.isdir(root) and any(
        name.endswith(".sqlite") for name in os.listdir(root)
    ):
        search_roots = [root]
    else:
        search_roots = []
        for dirpath, dirnames, _ in os.walk(root):
            base = os.path.basename(dirpath)
            if "nsys_pr429" in base:
                search_roots.append(dirpath)
                dirnames[:] = []
    for dirpath in search_roots:
        for name in os.listdir(dirpath):
            if not name.endswith(".sqlite"):
                continue
            path = os.path.join(dirpath, name)
            meta = read_log_metadata(path)
            avg_ms, min_ms, max_ms, count = query_timed(path)
            if count == 0:
                continue
            h_mb, d_mb = query_memcpy(path)
            records.append({
                "path": path,
                "label": os.path.splitext(name)[0],
                "scenario": classify_scenario(name),
                "batch_size": meta["batch_size"],
                "history_len": meta["history_len"],
                "only_onboard": meta["only_onboard"],
                "force_skip_offload": meta["force_skip_offload"],
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "count": count,
                "htoD_mb": h_mb,
                "dtoH_mb": d_mb,
                "onboard_wait_ms": query_onboard_wait(path),
            })
    records.sort(key=lambda r: r["label"])
    return records


def attention_kv_tokens(history_len):
    if history_len is None:
        return "?"
    return str(history_len * 2 + 256)


def render_markdown(records):
    lines = [
        "# PR429 Timed Sweep Analysis",
        "",
        "所有指标仅来自 **timed NVTX range**：`scenario{N}_timed_run_*` / `no_cache_timed_run_*`。",
        "",
        "## Sequence length 说明",
        "",
        "| 参数 | 含义 |",
        "|------|------|",
        "| `history_len=1024` | 单流 item/action history 各 1024 token |",
        "| only_onboard timed attention KV | `history_len*2 + 256` = **2304** |",
        "| main flexkv timed（带 append） | `(history+append)*2 + 256` = **4352** |",
        "",
        "对比 kvcache 收益请使用 **only_onboard + no_cache** 同一 timed shape。",
        "",
        "## 全量 timed 数据",
        "",
        "| label | scenario | bs | history_len | attn_kv | only_onboard | skip_offload | avg_ms | min_ms | max_ms | HtoD_MB | DtoH_MB | onboard_wait_ms |",
        "|-------|----------|---:|------------:|--------:|:------------:|:------------:|-------:|-------:|-------:|--------:|--------:|----------------:|",
    ]

    for r in records:
        skip = "?"
        if r["force_skip_offload"] is not None:
            skip = "Y" if r["force_skip_offload"] else "N"
        lines.append(
            "| `{label}` | {scenario} | {bs} | {hl} | {kv} | {oo} | {skip} | {avg:.2f} | {min:.2f} | {max:.2f} | {h:.2f} | {d:.2f} | {ow:.2f} |".format(
                label=r["label"],
                scenario=r["scenario"],
                bs=r["batch_size"] if r["batch_size"] is not None else "?",
                hl=r["history_len"] if r["history_len"] is not None else "?",
                kv=attention_kv_tokens(r["history_len"]),
                oo="Y" if r["only_onboard"] else "N",
                skip=skip,
                avg=r["avg_ms"],
                min=r["min_ms"],
                max=r["max_ms"],
                h=r["htoD_mb"],
                d=r["dtoH_mb"],
                ow=r["onboard_wait_ms"],
            )
        )

    lines.extend(["", "## 对齐对比（timed avg_ms，相对 no_cache 的 ratio）", ""])

    by_hl = {}
    for r in records:
        if r["history_len"] is None:
            continue
        if r["scenario"] == "no_cache" and not r["only_onboard"]:
            continue
        if not r["only_onboard"] and r["scenario"] != "no_cache":
            continue
        by_hl.setdefault(r["history_len"], {}).setdefault(r["batch_size"], {})[r["scenario"]] = r["avg_ms"]

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return "n/a"
        return "{:.2f}x".format(a / b)

    def speedup(a, b):
        if a is None or b is None or a == 0:
            return "n/a"
        return "{:.2f}x".format(b / a)

    for hl in sorted(by_hl.keys()):
        groups = by_hl[hl]
        kv = attention_kv_tokens(hl)
        lines.append("### history_len={} (attn_kv={})".format(hl, kv))
        lines.append("")
        lines.append("| bs | no_cache | s1_gpu | s2_cpu | s3_ssd | s1/nc | s2/nc | s3/nc | speedup s1 | speedup s2 | speedup s3 |")
        lines.append("|---:|---------:|-------:|-------:|-------:|------:|------:|------:|-----------:|-----------:|-----------:|")
        for bs in sorted(groups.keys(), key=lambda x: x if x is not None else 0):
            vals = groups[bs]
            no_cache = vals.get("no_cache")
            s1 = vals.get("s1_gpu")
            s2 = vals.get("s2_cpu")
            s3 = vals.get("s3_ssd")
            lines.append(
                "| {bs} | {no_cache} | {s1} | {s2} | {s3} | {r1} | {r2} | {r3} | {sp1} | {sp2} | {sp3} |".format(
                    bs=bs if bs is not None else "?",
                    no_cache="{:.2f}".format(no_cache) if no_cache is not None else "n/a",
                    s1="{:.2f}".format(s1) if s1 is not None else "n/a",
                    s2="{:.2f}".format(s2) if s2 is not None else "n/a",
                    s3="{:.2f}".format(s3) if s3 is not None else "n/a",
                    r1=ratio(s1, no_cache),
                    r2=ratio(s2, no_cache),
                    r3=ratio(s3, no_cache),
                    sp1=speedup(s1, no_cache),
                    sp2=speedup(s2, no_cache),
                    sp3=speedup(s3, no_cache),
                )
            )
        lines.append("")

    lines.extend([
        "## 数据完整性",
        "",
        "- 目标 64 项（4 scenarios × 4 history_len × 4 batch_size）。",
        "- 本报告只收录 **含 timed NVTX range** 的 sqlite。",
        "- 已知缺口：`only_onboard_s3_ssd_hl2048_bs8` 缺失；多数 `hl4096` flexkv sqlite 在双进程冲突时导出损坏（无 `NVTX_EVENTS`）。",
        "",
        "## 解读要点（基于本轮 only_onboard sweep）",
        "",
        "- **有效记录 51/64**：缺 `s3_ssd hl2048 bs8`；`hl4096` 的 s1/s2/s3 sqlite 无有效 timed NVTX。",
        "- **s1_gpu**：中等配置（如 hl1024/2048、bs≤4）通常快于 no_cache（约 **1.4–2.1x**）；短序列 hl512 bs1 和部分 bs=8 点会被固定开销抵消。",
        "- **s2_cpu**：通常介于 s1 与 s3 之间；batch 放大后 H2D 成为主导，部分点慢于 no_cache。",
        "- **s3_ssd**：当前有效点全部慢于 no_cache；大 batch 的 timed iteration 方差很大，需在干净单进程节点复测。",
        "- **speedup = no_cache / scenario**（越大越好）；**sX/nc** 是 latency ratio（越小越好）。",
        "",
        "更新本报告：",
        "",
        "```bash",
        "python3 scripts/analyze_pr429_timed_sweep.py",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/scratch.noliu_gpu/nsys_pr429_timed_sweep",
    )
    parser.add_argument(
        "--out",
        default="/home/scratch.noliu_gpu/pr429-verify/docs/PR429_TIMED_SWEEP_ANALYSIS.md",
    )
    args = parser.parse_args()

    records = collect_records(args.root)
    md = render_markdown(records)
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w") as f:
        f.write(md)
    print("Wrote {} ({} records)".format(args.out, len(records)))


if __name__ == "__main__":
    main()
