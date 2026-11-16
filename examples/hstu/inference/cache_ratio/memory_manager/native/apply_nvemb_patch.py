# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the experimental preserving-resize patch to a separate NVEmb source checkout.

Base: NVIDIA/nv-embedding-cache 803e278db2419261369a3d68c69a27858f7ca6d0.
Build both nve-common and the Python binding from the patched checkout using
that repository's installation workflow. A binding-only overlay linked to an
old nve-common does not provide this ABI. This script does not build or load it.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

BASE_FILES = {
    "include/ecache/ec_set_associative.h": "4f0971cda5d166465accf08c6692feff1c340c49ea09e3824871e771716dc94e",
    "include/ecache/ec_set_associative_device_modify.cuh": "d932df44f1e5d025bade9666cf6ebb397862b8bfe21d978e0b52cb396c3730d0",
    "include/gpu_table.hpp": "4cbf0c3ae6be7ea2745d093efc7f285c8d45dd5e9ee42a6940ce32f45d1ebe34",
    "include/hierarchical_embedding_layer.hpp": "511793fafca4413f76d2b6c45767a94d9db06ffc188625f233d16387ff3b0936",
    "include/layer_utils.hpp": "6e5e847e12e593ca0f8c6fae9a92a6334e6aeeab79a5b6db183c3e951ead866e",
    "python/pynve/bindings/binding_layers.hpp": "2133309145a2078eb22c449bb7ee4e26479a7bad4901df9dd5e1c1a3b819448e",
    "python/pynve/bindings/python_binding.cu": "2cf00f5e332d0b1d3da391688fd563445f416541f85fe7c26c4870d336758766",
    "src/gpu_table.cu": "76d07b8d50c66f234391d0a99e1ff718ac4bad4b52754e675cd902ceecbaad59",
    "src/hierarchical_embedding_layer.cu": "cd36d71ddb683c515a713d7e5a3bc36ded2ec8ac0d902cf12f13adbc6f531f03"
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="validate without changing sources"
    )
    args = parser.parse_args()
    source = args.source_dir.resolve()
    for name, expected in BASE_FILES.items():
        path = source / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise SystemExit(f"Source differs from the reviewed NVEmb base: {path}")
    patch = Path(__file__).with_name("nvemb_preserving_resize.patch").resolve()
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    if not args.check:
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    print(
        "Native patch checked"
        if args.check
        else "Native patch applied; build nve-common and pynve together"
    )


if __name__ == "__main__":
    main()
