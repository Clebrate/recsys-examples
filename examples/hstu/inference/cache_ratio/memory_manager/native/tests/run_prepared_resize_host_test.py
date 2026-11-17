#!/usr/bin/env python3
"""Compile/run the production prepared-plan owner and LFU planner without CUDA.

Requires git and a C++17 compiler. The three production headers are extracted by
applying their added-file sections from the shipping native patch in a temporary
directory; no SDK installation, CUDA toolkit, or GPU is used.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    patch = (here.parent / "nvemb_preserving_resize.patch").read_text()
    required = {
        "include/nvemb_prepared_resize.hpp",
        "include/nvemb_general_rehash.hpp",
        "include/ecache/ec_vmm_rehash_common.hpp",
    }
    sections = []
    found = set()
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        if not section.startswith("diff --git "):
            continue
        name = section.splitlines()[0].split(" b/", 1)[1]
        if name in required:
            if "new file mode 100644\n--- /dev/null\n" not in section:
                raise RuntimeError(f"Expected a complete added production header: {name}")
            sections.append(section)
            found.add(name)
    if found != required:
        raise RuntimeError(f"Missing production headers: {sorted(required - found)}")
    with tempfile.TemporaryDirectory(prefix="nvemb-prepared-host-") as temp:
        root = Path(temp)
        selected = root / "headers.patch"
        selected.write_text("".join(sections))
        subprocess.run(["git", "apply", "--check", str(selected)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(selected)], cwd=root, check=True)
        binary = root / "prepared_resize_test"
        subprocess.run([
            args.cxx, "-std=c++17", "-O2", "-pthread", "-Wall", "-Wextra", "-Werror",
            "-I", str(root / "include"), str(here / "prepared_resize_test.cpp"),
            "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
