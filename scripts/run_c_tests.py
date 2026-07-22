#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile and run every modules/*/c/test/*.c against the full portable engine.

Build inputs come from packaging/discovery.py so tests cannot drift from the
packaged build. Honours CC and VFHE_SANITIZE (e.g. "address,undefined").

Usage:
    python scripts/run_c_tests.py [arith misc ...]   # default: all modules
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import EXE_SUFFIX, LINK_MATH, ROOT, compiler, discovery, error

MODULES_DIR = ROOT / "modules"
UNITY_SRC = ROOT / "external" / "unity" / "src" / "unity.c"
CC = compiler("cc")
CTX = discovery.tool_build_context(ROOT)

SANITIZE_FLAGS: list[str] = []
if checks := os.environ.get("VFHE_SANITIZE", "").strip():
    SANITIZE_FLAGS = discovery.sanitizer_flags(checks)


def run_module(module_dir: Path, workdir: Path) -> bool:
    """Compile and run one module's C test suites; True when all pass."""
    all_passed = True
    for test in sorted((module_dir / "c" / "test").rglob("*.c")):
        binary = workdir / f"{module_dir.name}_{test.stem}{EXE_SUFFIX}"
        # Link Unity only where included; plain main() tests would fail to
        # resolve its setUp/tearDown.
        uses_unity = "unity.h" in test.read_text()
        compiled = subprocess.run(
            [
                CC,
                "-Wall",
                "-Wextra",
                "-std=gnu11",
                *SANITIZE_FLAGS,
                *CTX.define_flags,
                *CTX.include_flags,
                str(test),
                *([str(UNITY_SRC)] if uses_unity else []),
                *map(str, CTX.kernels),
                *LINK_MATH,
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            error(f"compile failed: {test.relative_to(ROOT)}\n{compiled.stderr}")
            all_passed = False
            continue
        test_run = subprocess.run([str(binary)], capture_output=True, text=True)
        sys.stdout.write(test_run.stdout)  # Unity's report is the product
        if test_run.returncode != 0:
            error(
                f"test failed: {test.relative_to(ROOT)} "
                f"(exit {test_run.returncode})\n{test_run.stderr}"
            )
            all_passed = False
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help="module names to run (default: all)")
    args = parser.parse_args()

    if not UNITY_SRC.exists():
        error(
            f"Unity not found at {UNITY_SRC.relative_to(ROOT)}.\n"
            f"Run: git submodule update --init external/unity"
        )
        return 2

    selected = set(args.modules)
    module_dirs = sorted(p for p in MODULES_DIR.iterdir() if p.is_dir())
    if selected:
        module_dirs = [m for m in module_dirs if m.name in selected]
        unknown = selected - {m.name for m in module_dirs}
        if unknown:
            error(f"unknown module(s): {', '.join(sorted(unknown))}")
            return 2

    all_passed = True
    with tempfile.TemporaryDirectory() as workdir:
        for module_dir in module_dirs:
            all_passed &= run_module(module_dir, Path(workdir))

    print("\nC tests:", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
