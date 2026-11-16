# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile/run the exact CPU planner included by the native SDK patch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativeRehashTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is unavailable")
    def test_native_host_planner(self):
        source = Path(__file__).with_name("nvemb_rehash_test.cpp")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "test_rehash"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-O2",
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
            self.assertIn("13 native host planner cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
