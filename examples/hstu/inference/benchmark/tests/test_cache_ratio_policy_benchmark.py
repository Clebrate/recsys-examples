"""CPU tests of identity, arrival and accounting helpers; no mock model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
import cache_ratio_policy_benchmark as benchmark

def session_module(name):
    path = DIRECTORY / "cache_ratio_policy" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PolicyModeTests(unittest.TestCase):
    def test_diagnostic_cli_keeps_unknown_cost_and_legacy_default(self):
        required = ['--trace-dir', '/trace', '--workload-lock', '/lock',
                    '--runtime-root', '/runtime', '--output-dir', '/out', '--mode', 'adaptive']
        parser = benchmark.parser()
        self.assertEqual(parser.parse_args(required).resize_cost_mode, 'amortized')
        args = parser.parse_args(required + ['--resize-cost-mode', 'diagnostic'])
        self.assertEqual(args.resize_cost_mode, 'diagnostic')
        self.assertIsNone(args.resize_calibration)


class ArrivalTests(unittest.TestCase):
    def rows(self):
        return [{"request_id": i, "arrival_s": t, "tokens": 10 + i}
                for i, t in enumerate((100.0, 100.0, 102.0))]

    def test_full_scale_exact_hash_and_single_application(self):
        rows = self.rows()
        expected = hashlib.sha256(b"[0,0.0]\n[1,0.0]\n[2,0.01]\n").hexdigest()
        result = benchmark.scale_arrivals(rows, 0.005, 3, expected)
        self.assertEqual(result["transforms_applied"], 1)
        self.assertEqual([r["arrival_s"] for r in rows], [0.0, 0.0, 0.01])
        with self.assertRaisesRegex(ValueError, "already transformed"):
            benchmark.scale_arrivals(rows, 0.005, 3, expected)

    def test_truncation_cannot_redefine_full_arrivals(self):
        with self.assertRaisesRegex(ValueError, "count"):
            benchmark.scale_arrivals(self.rows()[:2], 0.005, 3, "unused")

    def test_wrong_scale_and_hash_fail(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            benchmark.scale_arrivals(self.rows(), 0.006, 3, "unused")
        with self.assertRaisesRegex(ValueError, "hash"):
            benchmark.scale_arrivals(self.rows(), 0.005, 3, "wrong")

    def test_snapshot_groups_ties_and_counts_offered_tokens(self):
        rows = self.rows()
        _, snapshot = benchmark.arrival_index(rows)
        self.assertEqual(snapshot(99), (0, 0))
        self.assertEqual(snapshot(100), (2, 21))
        self.assertEqual(snapshot(102), (3, 33))


class AccountingTests(unittest.TestCase):
    def test_queue_area_peak_and_equal_time_departure(self):
        result = benchmark.population_stats([0, 1, 1, 4], [1, 2, 3, 4])
        self.assertEqual(result, {"peak_requests": 2, "area_request_ms": 4000.0, "final_pending": 0})

    def test_immediate_service_has_no_waiting_population(self):
        self.assertEqual(benchmark.population_stats([0, 0, 1], [0, 0, 1])["peak_requests"], 0)

    def test_departure_before_arrival_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.population_stats([1], [0])

    def test_unordered_departures_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.population_stats([0, 0], [2, 1])

    def test_windows_keep_boundary_completions_and_tail(self):
        rows = list(benchmark.fixed_windows([0, 9, 10], [0, 10, 10], [1, 10, 20]))
        self.assertEqual([r["completed"] for r in rows], [1, 1, 1])
        self.assertEqual(sum(r["arrivals"] for r in rows), 3)
        self.assertEqual(rows[-1]["outstanding_at_end_left"], 0)

    def test_quantiles_keep_single_large_tail(self):
        value = benchmark.fixed.distribution([1.0] * 999 + [5000.0])
        self.assertEqual(value["max"], 5000)
        self.assertGreater(value["mean"], 5)
        self.assertEqual(value["p99_9"], 1)


class IdentityTests(unittest.TestCase):
    def test_hash_receipt_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"same\n")
            result = benchmark.sha_file(path)
            self.assertEqual(result["sha256"], hashlib.sha256(b"same\n").hexdigest())
            self.assertEqual(path.read_bytes(), b"same\n")

    def test_missing_calibration_stays_unknown(self):
        cost, evidence = benchmark.calibration(None, {})
        self.assertIsNone(cost)
        self.assertEqual(evidence["status"], "missing")

    def test_calibration_rejects_stale_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.json"
            path.write_text(json.dumps({"schema_version": 1, "backend": "nvemb_vmm_general_rehash",
                "chunk_bytes": benchmark.CHUNK, "payload_budget_bytes": benchmark.BUDGET,
                "resize_cost_ms": 123, "method": "conservative measured max", "source_receipts": ["x"],
                "library_sha256": {"libnve-common.so": "old"}}))
            with self.assertRaisesRegex(ValueError, "library differs"):
                benchmark.calibration(path, {"/runtime/libnve-common.so": {"sha256": "new"}})

    def test_serial_lease_checks_whole_round_and_guard(self):
        runner = session_module("run_serial")
        self.assertTrue(runner.fits(10000, 4000, 3, 1800, 600))
        self.assertFalse(runner.fits(9999, 4000, 3, 1800, 600))
        self.assertFalse(runner.fits(5000, 4000, 1, 1800, 600))

    def test_serial_rejects_guard_below_child_cleanup_budget(self):
        runner = session_module("run_serial")
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-B", runner.__file__, "--trace-dir", temporary,
                       "--workload-lock", temporary, "--runtime-root", temporary,
                       "--results-root", str(Path(temporary) / "unused"),
                       "--lease-end-utc", "2030-01-01T00:00:00Z", "--scope", "smoke",
                       "--lease-guard-s", "119", "--plan-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("guard>=120s", result.stderr)
            self.assertFalse((Path(temporary) / "unused").exists())

    def test_summary_does_not_invent_completed_runs(self):
        summary = session_module("summarize_results")
        with tempfile.TemporaryDirectory() as directory:
            result = summary.summarize(Path(directory))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["runs"], [])
            self.assertEqual(result["comparisons"], [])

    def test_gc_accepts_image_frozen_objects_and_requires_automatic_gc(self):
        # Use a fresh CPU process so this test never changes the test runner's
        # pre-existing frozen set or GC eligibility.
        script = f"""
import gc, sys
sys.path.insert(0, {str(DIRECTORY)!r})
import cache_ratio_policy_benchmark as benchmark
gc.freeze()
before = gc.get_freeze_count()
assert before > 0
try:
    record = benchmark.freeze_loaded_inputs()
    assert record['preexisting_frozen_objects'] == before
    assert record['frozen_objects'] >= before
    assert gc.isenabled() and record['automatic_gc_enabled_after_freeze']
    gc.disable()
    try:
        benchmark.freeze_loaded_inputs()
    except ValueError as error:
        assert 'Automatic GC' in str(error)
    else:
        raise AssertionError('Disabled automatic GC was accepted')
finally:
    gc.enable()
    gc.unfreeze()
    gc.collect()
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-B", "-c", script], check=True, capture_output=True, text=True)

    def test_summary_signature_matches_image_frozen_count(self):
        summary = session_module("summarize_results")
        report = {"scope": "smoke", "expected_requests": 32,
                  "input": {"manifest": {"sha256": "manifest"}, "artifacts": {}, "arrival": {}},
                  "backend": {"seed": 303, "initialized_embedding_rows": {}, "probe_request_ids": []},
                  "model_parameters": {}, "startup_protocol": "same", "gpu_identity": {}, "code_identity": {},
                  "gc": {"protocol": "same", "thresholds": [700, 10, 10], "preexisting_frozen_objects": 375},
                  "loaded_libraries": {}, "configuration": {"round_index": 1},
                  "initial_capacity": {"payload_budget_bytes": benchmark.BUDGET,
                      "kv": {"payload_bytes": 126 * benchmark.CHUNK, "chunk_bytes": benchmark.CHUNK},
                      "embedding": {"payload_bytes": 130 * benchmark.CHUNK, "tables": {}},
                      "peak_limit_bytes": 32 * benchmark.GIB}}
        first = summary.signature(report)
        self.assertEqual(first["gc_preexisting_frozen_objects"], 375)
        report["gc"]["preexisting_frozen_objects"] = 376
        second = summary.signature(report)
        self.assertNotEqual(first, second)
        self.assertEqual([key for key in first if first[key] != second[key]], ["gc_preexisting_frozen_objects"])
        cache = {"seed_identity": {"manifest_sha256": "seed-a"}}
        seeded = summary.signature(report, cache)
        self.assertEqual(seeded["compiler_cache_seed"], cache["seed_identity"])
        self.assertNotEqual(seeded, summary.signature(report, {"seed_identity": {"manifest_sha256": "seed-b"}}))
        report["resize_calibration"] = {"file": {"sha256": "changed-cost"}, "record": {"resize_cost_ms": 123}}
        self.assertNotEqual(seeded, summary.signature(report, cache))
        calibrated = summary.signature(report, cache)
        report["configuration"]["host_staging_limit_gib"] = 40
        self.assertNotEqual(calibrated, summary.signature(report, cache))


class CompilerCacheTests(unittest.TestCase):
    def setUp(self):
        self.runner = session_module("run_serial")

    def seed(self, root, portable=False):
        (root / "triton/key").mkdir(parents=True)
        (root / "cuda").mkdir()
        (root / "triton/key/kernel.bin").write_bytes(b"original compiled bytes")
        (root / "cuda/driver.bin").write_bytes(b"driver cache")
        children = ["kernel.bin"] if portable else {"kernel.bin": "/original/triton/key/kernel.bin"}
        self.runner.write(root / "triton/key/__grp__kernel.json", {"child_paths": children})
        manifest = {"schema_version": 1, "source_roots": {"triton": "/original/triton", "cuda": "/original/cuda"},
                    "files": self.runner.compiler_cache_files(root)}
        self.runner.write(root / "manifest.json", manifest)
        return self.runner.verify_compiler_seed(root)[1]

    def test_private_copies_relocate_legacy_groups_without_mutating_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed, first, second = root / "seed", root / "first", root / "second"
            identity = self.seed(seed)
            receipt = self.runner.prepare_compiler_cache(seed, first, identity)
            source_file, copied_file = seed / "triton/key/kernel.bin", first / "triton/key/kernel.bin"
            self.assertNotEqual((source_file.stat().st_dev, source_file.stat().st_ino),
                                (copied_file.stat().st_dev, copied_file.stat().st_ino))
            self.assertEqual(receipt["relocated_triton_groups"], 1)
            group = json.loads((first / "triton/key/__grp__kernel.json").read_text())
            self.assertEqual(group["child_paths"]["kernel.bin"], str(copied_file))
            copied_file.write_bytes(b"new state from first arm")
            other = self.runner.prepare_compiler_cache(seed, second, identity)
            self.assertEqual(other["seed_identity"], receipt["seed_identity"])
            self.assertEqual((second / "triton/key/kernel.bin").read_bytes(), b"original compiled bytes")
            self.assertEqual(json.loads((seed / "triton/key/__grp__kernel.json").read_text())["child_paths"]["kernel.bin"],
                             "/original/triton/key/kernel.bin")

    def test_portable_group_keeps_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); seed = root / "seed"
            receipt = self.runner.prepare_compiler_cache(seed, root / "arm", self.seed(seed, portable=True))
            self.assertEqual(receipt["relocated_triton_groups"], 0)
            self.assertEqual((seed / "triton/key/__grp__kernel.json").read_bytes(),
                             (root / "arm/triton/key/__grp__kernel.json").read_bytes())

    def test_file_tampering_and_new_manifest_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary) / "seed"; identity = self.seed(seed)
            (seed / "cuda/driver.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "file/content"):
                self.runner.verify_compiler_seed(seed)
            manifest = json.loads((seed / "manifest.json").read_text())
            manifest["files"] = self.runner.compiler_cache_files(seed)
            self.runner.write(seed / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                self.runner.verify_compiler_seed(seed, identity)

    def test_symlinks_and_external_group_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); seed = root / "seed"; self.seed(seed)
            link = seed / "triton/link"
            link.symlink_to(seed / "cuda/driver.bin")
            with self.assertRaisesRegex(ValueError, "link or special"):
                self.runner.verify_compiler_seed(seed)
            link.unlink()
            self.runner.write(seed / "triton/key/__grp__kernel.json", {"child_paths": {"kernel.bin": "/elsewhere/kernel.bin"}})
            manifest = json.loads((seed / "manifest.json").read_text())
            manifest["files"] = self.runner.compiler_cache_files(seed)
            self.runner.write(seed / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "outside its declared"):
                self.runner.prepare_compiler_cache(seed, root / "arm", self.runner.verify_compiler_seed(seed)[1])


if __name__ == "__main__":
    unittest.main()
