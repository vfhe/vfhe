#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile-check the x86 AVX-512 engine paths.

The default (portable) build never exercises the arith AVX-512 IFMA kernels or
the AES-NI rng backend, so they can bit-rot unnoticed. This compiles every
module C/asm source with the SIMD feature flags and PORTABLE_BUILD dropped --
object-only, no link -- and fails if any translation unit does not build.

On an x86_64 host it compiles natively; on an Apple-Silicon dev machine it
cross-compiles with ``-target x86_64-apple-darwin`` (matching the manual check
used during development). On a non-x86 Linux host there is no cross toolchain,
so it skips with a clear message.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import run_c_tests as rt
from run_c_tests import discovery

SIMD_FLAGS = [
    "-mavx512f",
    "-mavx512ifma",
    "-mavx512dq",
    "-mavx512vl",
    "-mavx2",
    "-maes",
]

# Only the module sources we own -- vendored BLAKE3 has its own SIMD story.
MODULE_SOURCES = [k for k in rt.ALL_KERNELS if str(rt.MODULES_DIR) in str(k)]


def target_flags() -> "list[str] | None":
    if discovery.is_x86_host():  # already x86_64
        return []
    if sys.platform == "darwin":  # Apple Silicon: cross-compile to x86_64
        return ["-target", "x86_64-apple-darwin"]
    return None  # non-x86 Linux: no cross toolchain available


def main() -> int:
    tflags = target_flags()
    if tflags is None:
        print("check_simd_build: no x86 toolchain on this host; skipping.")
        return 0

    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "out.o"
        for src in MODULE_SOURCES:
            cmd = [
                rt.CC,
                *tflags,
                *SIMD_FLAGS,
                "-std=c11",
                "-c",
                *rt.ALL_INCLUDES,
                str(src),
                "-o",
                str(obj),
            ]
            cc = subprocess.run(cmd, capture_output=True, text=True)
            if cc.returncode != 0:
                print(f"[FAIL] {src.relative_to(rt.ROOT)}\n{cc.stderr}")
                fails += 1

    print(
        f"\nSIMD compile-check: {len(MODULE_SOURCES) - fails}/{len(MODULE_SOURCES)} ok"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
