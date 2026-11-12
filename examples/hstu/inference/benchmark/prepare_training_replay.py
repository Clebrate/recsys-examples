"""Freeze chronological inference requests from KuaiRand training sequences.

This is a replay derived from exposure logs, not recorded online candidate sets.
Histories are stable, cross-day prefixes. No training shuffle or sliding window
is used. NumPy keeps key accounting compact; a temporary local SQLite database
sorts requests without retaining all candidate lists as Python objects.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np


CONTEXT_SIZES = {
    "user_active_degree": 10,
    "follow_user_num_range": 9,
    "fans_user_num_range": 10,
    "friend_user_num_range": 8,
    "register_days_range": 8,
}
CONTEXTS = ["user_id", "user_active_degree", "follow_user_num_range",
            "fans_user_num_range", "friend_user_num_range", "register_days_range"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    if not values:
        return {"count": 0}
    values = np.asarray(values)
    result = {"count": int(values.size), "mean": float(values.mean())}
    result.update(zip(("min", "p50", "p90", "p95", "p99", "max"),
                      map(float, np.percentile(values, [0, 50, 90, 95, 99, 100]))))
    return result


def prepare(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    csv.field_size_limit(sys.maxsize)
    counts = dict(source_users=0, source_interactions=0, users_without_request=0,
                  users_with_history_cap_omissions=0, potential_requests=0,
                  omitted_requests_history_cap=0, kept_requests=0,
                  partial_candidate_requests=0, history_boundary_tie_advances=0,
                  candidate_positions_skipped_at_ties=0)
    source_lengths, history_lengths, candidate_lengths = [], [], []
    user_items, observed_users = {}, set()
    used_keys = np.zeros(args.video_capacity, dtype=np.bool_)
    unique_request_ids, query_occurrences = 0, 0
    # Random writes and sorting stay on local disk, not the output NFS mount.
    with tempfile.TemporaryDirectory(prefix="training-replay-", dir=args.temp_dir) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "requests.sqlite"))
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE requests (ts INTEGER, source_row INTEGER, uid INTEGER, "
                   "history INTEGER, candidates TEXT, unique_ids INTEGER)")
        with open(args.sequence_file, newline="") as source, (output / "sequences.jsonl").open("w") as seq_out:
            for source_row, row in enumerate(csv.DictReader(source)):
                if args.max_users and source_row >= args.max_users:
                    break
                uid = int(row["user_id"])
                if not 0 <= uid < 2**32 or uid in observed_users:
                    raise ValueError(f"Invalid or duplicate user ID: {uid}")
                observed_users.add(uid)
                items = np.asarray(json.loads(row["video_id"]), dtype=np.int64)
                actions = np.asarray(json.loads(row["action_weights"]), dtype=np.int64)
                timestamps = np.asarray(json.loads(row["time_ms"]), dtype=np.int64)
                if not len(items) == len(actions) == len(timestamps):
                    raise ValueError(f"Unaligned item/action/time arrays for user {uid}")
                if len(items) and (items.min() < 0 or items.max() >= args.video_capacity):
                    raise ValueError(f"video_id outside configured capacity for user {uid}")
                if len(actions) and (actions.min() < 0 or actions.max() >= 246):
                    raise ValueError(f"action_weights outside [0, 246) for user {uid}")
                if np.any(timestamps[1:] < timestamps[:-1]):
                    order = np.argsort(timestamps, kind="stable")
                    items, actions, timestamps = items[order], actions[order], timestamps[order]
                context = {name: int(row[name]) for name in CONTEXTS}
                for name, size in CONTEXT_SIZES.items():
                    if not 0 <= context[name] < size:
                        raise ValueError(f"Invalid {name} for user {uid}")
                counts["source_users"] += 1
                counts["source_interactions"] += len(items)
                source_lengths.append(len(items))
                history, last_history, last_access = args.min_history, 0, 0
                unique_prefix, pending_rows, omitted = set(), [], 0
                while history < len(items):
                    # Move the boundary past an entire timestamp group. Thus all
                    # history events are strictly earlier than the arrival.
                    aligned = int(np.searchsorted(timestamps, timestamps[history - 1], side="right"))
                    counts["history_boundary_tie_advances"] += aligned - history
                    history = aligned
                    if history >= len(items):
                        break
                    counts["potential_requests"] += 1
                    end = min(history + args.num_candidates, len(items))
                    next_boundary = history + args.interactions_per_request
                    if next_boundary < len(items):
                        next_aligned = int(np.searchsorted(timestamps, timestamps[next_boundary - 1], side="right"))
                        counts["candidate_positions_skipped_at_ties"] += max(
                            0, next_aligned - max(end, next_boundary))
                    if history > args.max_history_len:
                        counts["omitted_requests_history_cap"] += 1
                        omitted += 1
                    else:
                        unique_prefix.update(map(int, items[last_access:end]))
                        candidates = items[history:end].tolist()
                        pending_rows.append((int(timestamps[history]), source_row, uid,
                                             history, json.dumps(candidates, separators=(",", ":")),
                                             len(unique_prefix)))
                        last_history, last_access = history, end
                        counts["kept_requests"] += 1
                        counts["partial_candidate_requests"] += len(candidates) < args.num_candidates
                        history_lengths.append(history)
                        candidate_lengths.append(len(candidates))
                        unique_request_ids += len(unique_prefix)
                        query_occurrences += end
                    history = next_boundary
                counts["users_with_history_cap_omissions"] += omitted > 0
                if not pending_rows:
                    counts["users_without_request"] += 1
                    continue
                db.executemany("INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)", pending_rows)
                compact_items = items[:last_access].astype(np.int32)
                user_items[uid] = compact_items
                used_keys[compact_items] = True
                sequence = {"cache_uid": uid, "context": context,
                            "video_id": items[:last_history].tolist(),
                            "action_weights": actions[:last_history].tolist()}
                seq_out.write(json.dumps(sequence, separators=(",", ":")) + "\n")
                if counts["source_users"] % 1000 == 0:
                    db.commit()
                    print(json.dumps({"source_users": counts["source_users"],
                                      "requests": counts["kept_requests"],
                                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
        if not counts["kept_requests"]:
            raise ValueError("No requests satisfy the history and candidate rules")
        db.commit()
        first_ts, last_ts = db.execute("SELECT min(ts), max(ts) FROM requests").fetchone()
        scale = args.time_scale
        if args.arrival_rps is not None:
            if last_ts == first_ts or counts["kept_requests"] < 2:
                raise ValueError("Arrival-rate scaling needs distinct request timestamps")
            scale = (counts["kept_requests"] - 1) / args.arrival_rps / ((last_ts - first_ts) / 1000)
        windows, window_rows, window_users = [], [], {}
        last_seen, reuse_gaps, reuse_ms = {}, [], []
        current_pages, total_pages, peak_pages, reused_tokens, history_tokens = {}, 0, 0, 0, 0
        window_keys = np.zeros_like(used_keys)

        def finish_window():
            window_keys.fill(False)
            for window_uid, (access, _) in window_users.items():
                window_keys[user_items[window_uid][:access]] = True
            video_keys = int(np.count_nonzero(window_keys))
            kv_pages = sum((2 * history + len(CONTEXTS) + args.page_size - 1) // args.page_size
                           for _, history in window_users.values())
            gaps = [r[4] for r in window_rows if r[4] is not None]
            windows.append({
                "window_id": len(windows), "first_request_id": window_rows[0][0],
                "last_request_id": window_rows[-1][0], "request_count": len(window_rows),
                "first_timestamp_ms": window_rows[0][1], "last_timestamp_ms": window_rows[-1][1],
                "active_users": len(window_users), "first_ever_user_requests": len(window_rows) - len(gaps),
                "mean_history_items": float(np.mean([r[2] for r in window_rows])),
                "mean_video_query_occurrences": float(np.mean([r[3] for r in window_rows])),
                "mean_video_unique_ids_per_request": float(np.mean([r[5] for r in window_rows])),
                "median_reuse_request_gap": float(np.median(gaps)) if gaps else None,
                "unique_video_ids": video_keys,
                "embedding_vector_bytes": (video_keys + len(window_users)) * args.embedding_dim * 4,
                "active_user_latest_kv_bytes": kv_pages * args.page_size * args.kv_bytes_per_token,
            })
            window_rows.clear()
            window_users.clear()

        with (output / "trace.jsonl").open("w") as trace:
            for request_id, (ts, source_row, uid, history, candidates_json, unique_ids) in enumerate(
                    db.execute("SELECT * FROM requests ORDER BY ts, source_row, history")):
                candidates = json.loads(candidates_json)
                tokens = 2 * history + len(CONTEXTS)
                previous = last_seen.get(uid)
                gap = request_id - previous[0] if previous else None
                if previous:
                    reuse_gaps.append(gap)
                    reuse_ms.append(ts - previous[1])
                    reused_tokens += previous[2]
                last_seen[uid] = (request_id, ts, tokens)
                history_tokens += tokens
                pages = (tokens + args.page_size - 1) // args.page_size
                total_pages += pages - current_pages.get(uid, 0)
                current_pages[uid] = pages
                peak_pages = max(peak_pages, total_pages)
                row = {"request_id": request_id, "user_id": uid, "cache_uid": uid,
                       "date": 0, "source_row": source_row, "timestamp_ms": ts,
                       "arrival_s": (ts - first_ts) / 1000 * scale, "workload_phase": "training_prefix",
                       "history_len": history, "candidates": candidates, "num_candidates": len(candidates),
                       "video_query_occurrences": history + len(candidates),
                       "unique_video_query_ids": unique_ids, "reuse_request_gap": gap,
                       "reuse_time_ms": ts - previous[1] if previous else None}
                trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                window_rows.append((request_id, ts, history, history + len(candidates), gap, unique_ids))
                window_users[uid] = (history + len(candidates), history)
                if len(window_rows) == args.window_requests:
                    finish_window()
            if window_rows:
                finish_window()
        db.close()

    with (output / "windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    np.save(output / "video_keys.npy", np.flatnonzero(used_keys).astype(np.int64))
    num_video_keys = int(np.count_nonzero(used_keys))
    stats = {
        "counts": counts, "source_sequence_length": distribution(source_lengths),
        "history_length": distribution(history_lengths), "candidate_count": distribution(candidate_lengths),
        "reuse_request_gap": distribution(reuse_gaps), "reuse_time_ms": distribution(reuse_ms),
        "retained_user_count": len(user_items), "video_query_occurrences": query_occurrences,
        "sum_per_request_unique_video_ids": unique_request_ids, "global_unique_video_ids": num_video_keys,
        "embedding_vector_bytes": (num_video_keys + len(user_items)) * args.embedding_dim * 4,
        "no_eviction_latest_user_kv_peak_bytes": peak_pages * args.page_size * args.kv_bytes_per_token,
        "no_eviction_kv_token_hit_rate_cold_start": reused_tokens / history_tokens,
        "history_request_coverage": counts["kept_requests"] / counts["potential_requests"],
        "model": {"layers": 3, "hidden_size": args.embedding_dim, "heads": 4, "kv_channels": 128,
                  "kv_bytes_per_token": args.kv_bytes_per_token, "page_size": args.page_size,
                  "embedding_dtype": "float32", "kv_dtype": "bfloat16"},
        "notes": ["Query counts are uncropped history plus candidates, before KV hits skip lookups.",
                  "Window key counts are unions; per-request unique counts are not added to estimate working sets.",
                  "KV peak retains the latest prefix for every seen user; it is sufficient capacity, not a minimum.",
                  "No-eviction token hit rate is a cold-start theoretical bound, not a measured GPU hit rate.",
                  "Request gap is the difference in request IDs, not distinct-user LRU stack distance.",
                  "Embedding bytes exclude set metadata, static context/action tables and model parameters."],
    }
    (output / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "dataset": "kuairand-27k", "seed": args.seed,
        "request_count": counts["kept_requests"], "sequence_count": len(user_items),
        "max_history_len": max(history_lengths), "num_candidates": args.num_candidates,
        "variable_num_candidates": True, "context_features": CONTEXTS,
        "time_scale": scale, "target_arrival_rps": args.arrival_rps, "timestamp_origin_ms": first_ts,
        "request_source": "training_sequences_chronological_prefixes",
        "candidate_source": "subsequent_logged_exposure_ids_not_online_candidate_sets",
        "workload": {"kind": "training_prefix", "window_requests": args.window_requests},
        "interactions_per_request": args.interactions_per_request, "max_source_users": args.max_users,
        "dynamic_table_rows": {"video_id": num_video_keys, "user_id": len(user_items)},
        "checkpoint_ps_key_files": {}, "filters": counts,
        "prepare_total_s": time.perf_counter() - started,
        "sources": {"sequence_file": {"path": str(Path(args.sequence_file).resolve()),
                    "bytes": Path(args.sequence_file).stat().st_size,
                    "mtime_ns": Path(args.sequence_file).stat().st_mtime_ns}},
        "artifacts": {name: sha256(output / name) for name in
                      ("trace.jsonl", "sequences.jsonl", "statistics.json", "windows.csv", "video_keys.npy")},
        "notes": ["cache_uid=user_id, date=0 is a sentinel; prefixes continue across real dates.",
                  "Candidate action labels are never supplied as historical input.",
                  "Oversized histories are excluded and counted; no sliding-window KV reuse is implied.",
                  "Candidates are real subsequent exposures, not logged ranking-request candidates.",
                  "Use each row's actual num_candidates; the manifest value is only the upper bound."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "request_count": manifest["request_count"],
                      "user_count": manifest["sequence_count"], "unique_video_ids": num_video_keys,
                      "history_request_coverage": stats["history_request_coverage"],
                      "prepare_total_s": manifest["prepare_total_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--max-history-len", type=int, default=4096)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--interactions-per-request", type=int, default=100)
    parser.add_argument("--window-requests", type=int, default=5000)
    parser.add_argument("--max-users", type=int, default=0, help="0 means all users; nonzero is diagnostic only")
    parser.add_argument("--video-capacity", type=int, default=32038725)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--kv-bytes-per-token", type=int, default=6144)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--arrival-rps", type=float)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--temp-dir", help="Local temporary directory; defaults to the system temp directory")
    args = parser.parse_args()
    for name in ("min_history", "max_history_len", "num_candidates", "interactions_per_request",
                 "window_requests", "video_capacity", "embedding_dim", "kv_bytes_per_token", "page_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_users < 0 or args.min_history > args.max_history_len:
        parser.error("Invalid user limit or history limits")
    if args.video_capacity > 2**31:
        parser.error("Video IDs must fit the compact signed int32 representation")
    if args.time_scale <= 0 or (args.arrival_rps is not None and args.arrival_rps <= 0):
        parser.error("Arrival scaling must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
