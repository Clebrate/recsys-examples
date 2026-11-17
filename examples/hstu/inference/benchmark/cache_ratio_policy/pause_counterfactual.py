#!/usr/bin/env python3
"""CPU replay removing explicit resize pauses from fixed observed batches.

Preserves original arrivals, batch membership, measured service (including
refill/cold misses) and all other dispatch gaps, including observed idle.
This is not an actual zero-cost-policy run, a causal estimate, or a bound.
No original logs are modified.
"""
import argparse
from array import array
import copy
import json
import math
from pathlib import Path
import sys

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
import cache_ratio_policy_benchmark as bench

ASSUMPTIONS = [
    "All logged requests and latency tails are retained; original arrivals are unchanged.",
    "Observed batch membership and service durations are fixed, including all observed refill and cold-miss work.",
    "Only explicit runtime_event.resize_pause intervals are removed; observer/action duration summaries are not subtracted again.",
    "Every other observed ready-to-dispatch gap is retained, including observed idle; arrivals can absorb removed pause time.",
    "The first batch is unchanged; pauses after the final ready do not shorten completed-request horizon.",
    "This is a fixed-observation replay, not an actual zero-cost-policy world, a causal gain estimate, or an upper bound.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"{label} must be finite and nonnegative")
    return value


def index_value(value, label):
    require(type(value) is int and 0 <= value < 2**64, f"{label} must be a nonnegative integer")
    return value


def records(path):
    require(path.is_file(), f"Missing input log: {path.name}")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path.name}")
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            require(isinstance(row, dict), f"{path.name}:{line_number} is not an object")
            yield row


def metrics(arrivals, dispatches, ready, batch_count):
    queue = [(d-a)*1000 for a, d in zip(arrivals, dispatches)]
    service = [(r-d)*1000 for d, r in zip(dispatches, ready)]
    e2e = [(r-a)*1000 for a, r in zip(arrivals, ready)]
    horizon = ready[-1] - arrivals[0]
    require(horizon > 0, "Completed-request horizon must be positive")
    qpop = bench.population_stats(arrivals, dispatches)
    spop = bench.population_stats(arrivals, ready)
    return {
        "requests": len(arrivals), "batches": batch_count,
        "queue_ms": bench.fixed.distribution(queue),
        "service_ms": bench.fixed.distribution(service),
        "e2e_ms": bench.fixed.distribution(e2e),
        "queue_population": qpop, "system_population": spop,
        "mean_queued_requests_over_horizon": qpop["area_request_ms"] / (horizon*1000),
        "mean_outstanding_requests_over_horizon": spop["area_request_ms"] / (horizon*1000),
        "arrival_to_drain_s": horizon,
        "first_arrival_s": arrivals[0], "last_arrival_s": arrivals[-1], "last_ready_s": ready[-1],
        "drain_after_last_arrival_s": ready[-1] - arrivals[-1],
        "completion_throughput_requests_s": len(arrivals) / horizon,
        "mean_batch_size": len(arrivals) / batch_count,
    }


def _analyze_run(directory):
    directory = Path(directory)
    ids, batch_ids = array("Q"), array("Q")
    arrivals, dispatches, ready = array("d"), array("d"), array("d")
    for row in records(directory / "requests.jsonl"):
        rid = index_value(row["request_id"], "request_id")
        bid = index_value(row["batch_index"], "request batch_index")
        a, d, r = (number(row[name], name) for name in ("arrival_s", "dispatch_s", "ready_s"))
        require(a <= d <= r, f"Invalid request clock ordering for request {rid}")
        require(not arrivals or arrivals[-1] <= a, "Requests must retain ordered original arrivals")
        require(not dispatches or dispatches[-1] <= d, "Unordered request dispatches")
        require(not ready or ready[-1] <= r, "Unordered request completion times")
        for name, value in (("queue_ms", (d-a)*1000), ("service_ms", (r-d)*1000), ("e2e_ms", (r-a)*1000)):
            if name in row:
                measured = number(row[name], name)
                require(math.isclose(measured, value, rel_tol=1e-8, abs_tol=1e-5),
                        f"Recorded {name} disagrees with clock")
        ids.append(rid); batch_ids.append(bid)
        arrivals.append(a); dispatches.append(d); ready.append(r)
    require(bool(ids), "Empty request log is unsupported")
    require(len(set(ids)) == len(ids), "Repeated request_id")

    bd, br, latest_arrival = array("d"), array("d"), array("d")
    boundaries = array("Q", [0])
    for index, row in enumerate(records(directory / "batches.jsonl")):
        require(index_value(row["batch_index"], "batch_index") == index,
                "Missing, repeated or reordered batch index")
        members = row["request_ids"]
        require(isinstance(members, list) and bool(members), "Empty or invalid batch membership")
        for rid in members:
            index_value(rid, "batch request_id")
        require(index_value(row["batch_size"], "batch_size") == len(members), "Batch size disagrees with membership")
        start, end = boundaries[-1], boundaries[-1] + len(members)
        require(end <= len(ids) and list(ids[start:end]) == members, "Batch/request membership coverage mismatch")
        require(all(b == index for b in batch_ids[start:end]), "Request/batch index mismatch")
        d, r = (number(row[name], name) for name in ("dispatch_s", "ready_s"))
        require(d <= r and (not br or br[-1] <= d), "Overlapping or negative batch service")
        require(all(v == d for v in dispatches[start:end]) and all(v == r for v in ready[start:end]),
                "Request/batch timestamps disagree")
        bd.append(d); br.append(r); latest_arrival.append(max(arrivals[start:end]))
        boundaries.append(end)
    require(boundaries[-1] == len(ids) and bool(br), "Batch log does not cover every request")
    batch_count = len(br)
    actual = metrics(arrivals, dispatches, ready, batch_count)

    intervals, missing = [], []
    action_count = attempt_count = event_count = missing_action_count = 0
    for row in records(directory / "policy_actions.jsonl"):
        event_count += 1
        index = index_value(row["batch_index"], "event batch_index")
        require(index < batch_count, "Event refers to an absent batch")
        if "completed_requests" in row:
            require(index_value(row["completed_requests"], "completed_requests") == boundaries[index+1],
                    "Event completion coverage disagrees with batch")
        recorded = number(row["event_recorded_s"], "event_recorded_s") if "event_recorded_s" in row else None
        if recorded is not None:
            require(recorded >= br[index], "Event recorded before its batch was ready")
        event = row["runtime_event"]
        require(isinstance(event, dict), "runtime_event must be an object")
        action, pause = event.get("action"), event.get("resize_pause")
        rejection = event.get("rejection")
        require(rejection is None or isinstance(rejection, dict), "Non-null rejection must be an object")
        if action is not None:
            require(isinstance(action, dict), "Non-null action must be an object")
            action_count += 1
        attempted = (action is not None or event.get("resize_interval") is not None
                     or (rejection or {}).get("reason") == "native_preflight_declined")
        if attempted:
            attempt_count += 1
            if pause is None:
                missing_action_count += int(action is not None)
                missing.append({"batch_index": index,
                                "action_status": (action or {}).get("status", "unknown"),
                                "rejection_reason": (rejection or {}).get("reason")})
        if pause is None:
            continue
        require(isinstance(pause, dict), "resize_pause must be an object")
        start, end = (number(pause[name], "resize_pause."+name) for name in ("start_s", "end_s"))
        require(end >= start, "Negative resize pause duration")
        scope = pause["scope"]
        require(isinstance(scope, str) and bool(scope.strip()), "resize_pause.scope must be nonempty")
        require(start >= br[index], "Resize pause starts before its batch ready / outside gap")
        if index+1 < batch_count:
            require(end <= bd[index+1], "Resize pause extends past next dispatch / outside gap")
        if recorded is not None:
            require(end <= recorded, "Resize pause ends after event was recorded")
        intervals.append((start, end, index, scope))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        require(current[0] >= previous[1], "Overlapping resize pause intervals")
    durations_by_gap = {}
    for start, end, index, _ in intervals:
        durations_by_gap.setdefault(index, []).append(end-start)
    gap_pauses = [math.fsum(durations_by_gap.get(i, [])) for i in range(batch_count)]
    for index, duration in enumerate(gap_pauses[:-1]):
        require(duration <= bd[index+1]-br[index], "Resize pause total exceeds its dispatch gap")
    within = math.fsum(gap_pauses[:-1])
    total = math.fsum(end-start for start, end, _, _ in intervals)
    result = {
        "schema_version": 1, "status": "unsupported" if missing else "supported",
        "analysis": "fixed_observed_batches_direct_resize_pause_removal",
        "actual": actual, "counterfactual": None, "relative_throughput": None,
        "resize_pauses": {"count": len(intervals), "total_s": total,
                          "within_horizon_s": within, "after_final_ready_s": gap_pauses[-1],
                          "scopes": sorted({entry[3] for entry in intervals})},
        "coverage": {"logged_requests": len(ids), "matched_requests": boundaries[-1],
                     "logged_batches": batch_count, "policy_events": event_count,
                     "actions": action_count, "actions_missing_pause": missing_action_count,
                     "resize_attempts": attempt_count, "attempts_missing_pause": len(missing),
                     "all_logged_requests_retained": True, "all_latency_tails_retained": True,
                     "scope": "Complete logged request/batch coverage; full-source completeness is audited separately"},
        "assumptions": list(ASSUMPTIONS),
    }
    if missing:
        result["reason"] = "Resize attempts lack direct resize_pause timestamps; duration summaries cannot substitute"
        result["unsupported_actions"] = missing
        return result

    if within == 0:
        cf = copy.deepcopy(actual)  # Exact identity, including floating-point summaries.
    else:
        cf_dispatches, cf_ready = array("d"), array("d")
        previous_ready = None
        for index in range(batch_count):
            d, r = bd[index], br[index]
            if index == 0:
                cf_d, cf_r = d, r
            else:
                # Equivalent to prev_cf_ready + (actual_d-prev_actual_ready) - pause.
                # Carrying the saved shift preserves identity before the first pause.
                candidate = d - (br[index-1]-previous_ready) - gap_pauses[index-1]
                cf_d = max(latest_arrival[index], candidate)
                cf_r = r if cf_d == d else cf_d + (r-d)
                require(cf_d <= d+1e-10 and cf_r <= r+1e-10, "Replay unexpectedly delayed an observed batch")
            count = boundaries[index+1]-boundaries[index]
            cf_dispatches.extend([cf_d]*count); cf_ready.extend([cf_r]*count)
            previous_ready = cf_r
        cf = metrics(arrivals, cf_dispatches, cf_ready, batch_count)
        cf["service_ms"] = copy.deepcopy(actual["service_ms"])
    result["counterfactual"] = cf
    result["relative_throughput"] = cf["completion_throughput_requests_s"] / actual["completion_throughput_requests_s"]
    result["horizon_reduction_s"] = actual["arrival_to_drain_s"] - cf["arrival_to_drain_s"]
    return result


def analyze_run(directory):
    """Return actual/replayed metrics; missing action pauses yield unsupported.

    Invalid clocks, overlap/gap violations and incomplete request/batch coverage
    raise ValueError. No input log or clock is changed.
    """
    try:
        return _analyze_run(directory)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid counterfactual input: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    result = analyze_run(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
