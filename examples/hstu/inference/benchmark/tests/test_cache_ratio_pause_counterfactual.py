"""Hand-calculated CPU checks of the fixed-observation pause replay."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1]
ANALYZER = DIRECTORY / "cache_ratio_policy" / "pause_counterfactual.py"
spec = importlib.util.spec_from_file_location("pause_counterfactual", ANALYZER)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


class PauseReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, batches, events=()):
        requests, batch_rows = [], []
        for index, (arrivals, dispatch, ready) in enumerate(batches):
            members = list(range(len(requests), len(requests) + len(arrivals)))
            for rid, arrival in zip(members, arrivals):
                requests.append({"request_id": rid, "batch_index": index,
                                 "arrival_s": arrival, "dispatch_s": dispatch, "ready_s": ready,
                                 "queue_ms": (dispatch-arrival)*1000,
                                 "service_ms": (ready-dispatch)*1000,
                                 "e2e_ms": (ready-arrival)*1000})
            batch_rows.append({"batch_index": index, "request_ids": members,
                               "batch_size": len(members), "dispatch_s": dispatch, "ready_s": ready})
        for name, rows in (("requests", requests), ("batches", batch_rows), ("policy_actions", events)):
            (self.directory / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return cf.analyze_run(self.directory)

    @staticmethod
    def pause(batch, start, end, action=True):
        return {"batch_index": batch, "event_recorded_s": end,
                "runtime_event": {"action": {"status": "applied"} if action else None,
                                  "resize_pause": {"start_s": start, "end_s": end,
                                                   "scope": "resize plus post-resize drain"}}}

    def test_backlog_chain_retains_service_and_all_requests(self):
        result = self.write([([0], 0, 2), ([1], 5, 7), ([2], 7, 9)], [self.pause(0, 2, 5)])
        actual, replay = result["actual"], result["counterfactual"]
        self.assertEqual((actual["queue_ms"]["mean"], replay["queue_ms"]["mean"]), (3000, 1000))
        self.assertEqual((actual["e2e_ms"]["mean"], replay["e2e_ms"]["mean"]), (5000, 3000))
        self.assertEqual((actual["arrival_to_drain_s"], replay["arrival_to_drain_s"]), (9, 6))
        self.assertEqual(replay["queue_population"], {"area_request_ms": 3000, "peak_requests": 1, "final_pending": 0})
        self.assertEqual(actual["service_ms"], replay["service_ms"])
        self.assertEqual(replay["requests"], 3)
        self.assertEqual(result["relative_throughput"], 1.5)
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        cf.analyze_run(self.directory)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_arrival_absorbs_pause_so_naive_horizon_subtraction_is_wrong(self):
        result = self.write([([0], 0, 1), ([10], 10, 11)], [self.pause(0, 1, 4)])
        self.assertEqual(result["actual"], result["counterfactual"])
        self.assertEqual(result["resize_pauses"]["within_horizon_s"], 3)
        self.assertEqual(result["horizon_reduction_s"], 0)
        self.assertEqual(result["relative_throughput"], 1)
        self.assertNotEqual(result["counterfactual"]["completion_throughput_requests_s"], 2/(11-3))

    def test_batch_latest_arrival_limits_shift(self):
        result = self.write([([0, 1], 1, 3), ([2, 4], 6, 8)], [self.pause(0, 3, 6)])
        self.assertEqual(result["actual"]["queue_ms"]["mean"], 1750)
        self.assertEqual(result["counterfactual"]["queue_ms"]["mean"], 750)
        self.assertEqual(result["counterfactual"]["e2e_ms"]["mean"], 2750)
        self.assertEqual(result["counterfactual"]["arrival_to_drain_s"], 6)
        self.assertEqual(result["counterfactual"]["requests"], 4)

    def test_other_gaps_and_observed_refill_service_remain(self):
        # Original gap=5, explicit pauses=2, so 3 seconds of other gap remain.
        result = self.write([([0], 0, 2), ([1], 7, 17)],
                            [self.pause(0, 2, 3), self.pause(0, 4, 5)])
        self.assertEqual(result["counterfactual"]["last_ready_s"], 15)
        self.assertEqual(result["counterfactual"]["service_ms"], result["actual"]["service_ms"])
        self.assertEqual(result["counterfactual"]["service_ms"]["max"], 10000)
        self.assertEqual(result["resize_pauses"]["within_horizon_s"], 2)

    def test_no_pause_and_final_pause_are_exact_identity(self):
        batches = [([0.1], 0.2, 0.3), ([0.25], 0.4, 0.65)]
        for events in ([], [self.pause(1, 0.65, 10.65)],
                       [{"batch_index": 0, "runtime_event": {"action": None}}],
                       [{"batch_index": 0, "runtime_event": {"action": None,
                           "rejection": {"reason": "observation_only"}}}]):
            with self.subTest(events=events):
                result = self.write(batches, events)
                self.assertEqual(result["status"], "supported")
                self.assertEqual(result["actual"], result["counterfactual"])
                self.assertEqual(result["relative_throughput"], 1)
                self.assertEqual(result["resize_pauses"]["within_horizon_s"], 0)
                if events and events[0]["runtime_event"].get("resize_pause"):
                    self.assertEqual(result["resize_pauses"]["after_final_ready_s"], 10)
                    self.assertEqual(result["resize_pauses"]["total_s"], 10)

    def test_missing_pause_for_action_or_real_rejection_is_unsupported(self):
        attempted = [{"action": {"status": status, "action_ms": 999999}} for status in ("applied", "unchanged", "rejected")]
        attempted += [{"action": None, "rejection": {"reason": "native_preflight_declined"}},
                      {"action": None, "resize_interval": {"start_s": 1, "end_s": 2}}]
        for event in attempted:
            with self.subTest(event=event):
                event["observer_ms"] = 999999
                result = self.write([([0], 0, 1), ([1], 2, 3)], [{"batch_index": 0, "runtime_event": event}])
                self.assertEqual(result["status"], "unsupported")
                self.assertIsNone(result["counterfactual"])
                self.assertIsNone(result["relative_throughput"])
                self.assertEqual(result["actual"]["requests"], 2)
                self.assertEqual(result["coverage"]["attempts_missing_pause"], 1)
        supported = self.pause(0, 1, 2, action=False)
        supported["runtime_event"]["rejection"] = {"reason": "native_preflight_declined"}
        self.assertEqual(self.write([([0], 0, 1), ([1], 2, 3)], [supported])["status"], "supported")

    def test_invalid_pause_clock_scope_and_overlap_fail(self):
        batches = [([0], 0, 2), ([1], 6, 7)]
        invalid = [(1, 3), (2, 7), (4, 3), (-1, 3), (float("nan"), 3), (2, float("inf"))]
        for start, end in invalid:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.write(batches, [self.pause(0, start, end)])
        for field in ("start_s", "end_s", "scope"):
            event = self.pause(0, 2, 3)
            del event["runtime_event"]["resize_pause"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.write(batches, [event])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.write(batches, [self.pause(0, 2, 4), self.pause(0, 3, 5)])
        # Touching intervals do not overlap.
        self.assertEqual(self.write(batches, [self.pause(0, 2, 3), self.pause(0, 3, 4)])["resize_pauses"]["total_s"], 2)

    def test_bad_request_clock_or_incomplete_coverage_fail(self):
        with self.assertRaises(ValueError):
            self.write([([2], 1, 3)])
        self.write([([0], 0, 1), ([1], 2, 3)])
        batches = self.directory / "batches.jsonl"
        batches.write_text(batches.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "cover every request"):
            cf.analyze_run(self.directory)

    def test_cli_and_import_are_cpu_only(self):
        self.write([([0], 0, 1)])
        proc = subprocess.run([sys.executable, str(ANALYZER), str(self.directory)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(proc.stdout)["relative_throughput"], 1)
        code = "import runpy,sys; runpy.run_path(sys.argv[1], run_name='test_import'); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code, str(ANALYZER)], capture_output=True, text=True, check=True)


if __name__ == "__main__":
    unittest.main()
