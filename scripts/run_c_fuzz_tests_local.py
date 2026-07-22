#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the modules/*/c/fuzz/ libFuzzer harnesses and fuzz each briefly.

Needs a clang with the libFuzzer runtime: Apple's lacks it, so on macOS point
CC at a brew-installed LLVM clang. Corpora accumulate in .cache/fuzz/.

Usage:
    python scripts/run_c_fuzz_tests_local.py [fuzz_ntt ...] [--time 300]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _common import EXE_SUFFIX, LINK_MATH, ROOT, compiler, discovery, error, log

CC = compiler("clang")
CTX = discovery.tool_build_context(ROOT)


def build(target: Path, out_dir: Path) -> Path | None:
    """Compile one harness with clang + libFuzzer; None on a failed build."""
    binary = out_dir / f"{target.stem}{EXE_SUFFIX}"
    compiled = subprocess.run(
        [
            CC,
            "-fsanitize=fuzzer,address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
            "-g",
            "-O1",
            "-std=gnu11",
            *CTX.define_flags,
            *CTX.include_flags,
            str(target),
            *map(str, CTX.kernels),
            *LINK_MATH,
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        error(f"build failed: {target.relative_to(ROOT)}\n{compiled.stderr}")
        if "libclang_rt.fuzzer" in compiled.stderr:
            log(
                "hint: this clang has no libFuzzer runtime (Apple's does not); "
                "brew install llvm and set CC to its clang."
            )
        return None
    log(f"[built] {binary.name}")
    return binary


def fuzz(binary: Path, corpus: Path, seconds: int) -> bool:
    """Run one built harness against its corpus; True unless it crashes."""
    corpus.mkdir(exist_ok=True)
    log(f"[fuzz] {binary.name} for {seconds}s")
    fuzz_run = subprocess.run(
        [
            str(binary),
            str(corpus),
            f"-max_total_time={seconds}",
            "-timeout=30",
            "-print_final_stats=1",
        ]
    )
    if fuzz_run.returncode != 0:
        error(f"{binary.name} crashed (exit {fuzz_run.returncode})")
    return fuzz_run.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="fuzz target stems (default: all)")
    parser.add_argument(
        "--time", type=int, default=60, help="seconds per target (default: 60)"
    )
    args = parser.parse_args()

    targets = sorted((ROOT / "modules").glob("*/c/fuzz/*.c"))
    if args.targets:
        targets = [t for t in targets if t.stem in args.targets]
    if not targets:
        error("no matching fuzz targets under modules/*/c/fuzz/")
        return 2

    out_dir = ROOT / ".cache" / "fuzz"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for target in targets:
        binary = build(target, out_dir)
        if binary is None:
            all_ok = False
        else:
            all_ok &= fuzz(binary, out_dir / f"{target.stem}_corpus", args.time)

    print("\nfuzzing:", "OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
