"""CPU checks for populated-cache calibration and exact payload accounting."""
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "cache_ratio_policy/calibrate_resize.py"
SPEC = importlib.util.spec_from_file_location("resize_calibration", SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


class CalibrationTests(unittest.TestCase):
    def observation(self, user=1, video=1):
        return {"embedding": {"tables": {
            name: {"enabled": True, "complete": True, "overflow": False,
                   "resident_rows": count, "capacity_rows": 100}
            for name, count in (("user_id", user), ("video_id", video))
        }}}

    def test_requests_are_not_proof_of_population(self):
        self.assertFalse(calibration.populated_embedding(self.observation(0, 0)))
        self.assertFalse(calibration.populated_embedding(self.observation(0, 1)))
        self.assertFalse(calibration.populated_embedding(self.observation(1, 0)))
        self.assertTrue(calibration.populated_embedding(self.observation()))

    def test_missing_or_incomplete_native_observation_rejected(self):
        bad = self.observation()
        bad["embedding"]["tables"].pop("video_id")
        with self.assertRaises(ValueError):
            calibration.populated_embedding(bad)
        for field, value in (("complete", False), ("overflow", True),
                             ("resident_rows", None), ("resident_rows", 101)):
            bad = self.observation()
            bad["embedding"]["tables"]["video_id"][field] = value
            with self.assertRaises(ValueError):
                calibration.populated_embedding(bad)

    def state(self, kv):
        return {"kv": {"payload_bytes": kv},
                "embedding": {"payload_bytes": calibration.bench.BUDGET - kv}}

    def test_exact_roundtrip_and_equal_opposite_transfers(self):
        initial = calibration.bench.BUDGET // 2
        kv = initial
        for delta in calibration.TRANSFERS:
            calibration.verify_transfer(self.state(kv), self.state(kv + delta), delta)
            kv += delta
        self.assertEqual(kv, initial)

    def test_noop_and_donor_only_changes_are_not_successes(self):
        before = self.state(calibration.bench.BUDGET // 2)
        with self.assertRaises(ValueError):
            calibration.verify_transfer(before, before, calibration.TRANSFER_BYTES)
        after = self.state(before["kv"]["payload_bytes"] + calibration.TRANSFER_BYTES)
        after["embedding"]["payload_bytes"] += calibration.TRANSFER_BYTES
        with self.assertRaises(ValueError):
            calibration.verify_transfer(before, after, calibration.TRANSFER_BYTES)


if __name__ == "__main__":
    unittest.main()
