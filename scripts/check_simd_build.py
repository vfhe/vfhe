#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile-check the x86 AVX-512 engine paths the portable build never builds.

Without this the SIMD kernels and the AES-NI rng backend bit-rot unnoticed.
Object-only compile, no link. Non-x86 hosts cross-compile (macOS) or skip;
--require fails instead of skipping, so a skip cannot masquerade as a pass.

Usage:
    python scripts/check_simd_build.py [--require]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import ROOT, compiler, discovery, error, log

CC = compiler("cc")
INCLUDE_FLAGS = discovery.tool_build_context(ROOT).include_flags

SIMD_FLAGS = [
    "-mavx512f",
    "-mavx512ifma",
    "-mavx512dq",
    "-mavx512vl",
    "-mavx2",
    "-maes",
    "-mrdrnd",
]

# Only the module sources we own (vendored BLAKE3 has its own SIMD story),
# selected for x86: that is the architecture being compiled, not the host's.
MODULE_SOURCES = discovery.module_sources(ROOT / "modules", discovery.X86_ALIASES)


def target_flags() -> list[str] | None:
    """Flags to target x86-64 from this host, or None if it cannot."""
    if discovery.is_x86_host():  # already x86_64
        return []
    if sys.platform == "darwin":  # Apple Silicon: cross-compile to x86_64
        return ["-target", "x86_64-apple-darwin"]
    return None  # non-x86, non-mac: no cross toolchain available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="store_true",
        help="fail instead of skipping when the host cannot target x86",
    )
    args = parser.parse_args()

    cross_flags = target_flags()
    if cross_flags is None:
        if args.require:
            error(
                "check_simd_build: no x86 toolchain on this host, so the "
                "SIMD kernels were not checked at all. Run this on an x86_64 or "
                "macOS host, or drop --require to allow the skip."
            )
            return 1
        log("check_simd_build: no x86 toolchain on this host; skipping.")
        return 0

    if not MODULE_SOURCES:
        error("check_simd_build: discovery found no module sources")
        return 1

    failures = 0
    with tempfile.TemporaryDirectory() as workdir:
        obj = Path(workdir) / "out.o"
        for source in MODULE_SOURCES:
            compiled = subprocess.run(
                [
                    CC,
                    *cross_flags,
                    *SIMD_FLAGS,
                    "-std=gnu11",
                    "-c",
                    *INCLUDE_FLAGS,
                    str(source),
                    "-o",
                    str(obj),
                ],
                capture_output=True,
                text=True,
            )
            if compiled.returncode != 0:
                error(f"compile failed: {source.relative_to(ROOT)}\n{compiled.stderr}")
                failures += 1

    passed = len(MODULE_SOURCES) - failures
    print(f"\nSIMD compile-check: {passed}/{len(MODULE_SOURCES)} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
