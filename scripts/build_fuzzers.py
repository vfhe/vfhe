#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build (and optionally run) the libFuzzer harnesses under modules/*/c/fuzz/.

Reuses the kernel/include discovery from run_c_tests.py so a fuzz target links
exactly the same portable engine the unit tests do, plus:

  * clang with ``-fsanitize=fuzzer,address,undefined`` (libFuzzer + ASan/UBSan),
  * ``-DVFHE_RNG_TESTING`` so any sampling inside a harness is reproducible.

Usage:
    python scripts/build_fuzzers.py                 # build only
    python scripts/build_fuzzers.py --run --time 60 # build, then fuzz each 60s
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run_c_tests as rt

FUZZ_CC = "clang"
FUZZ_SANITIZE = "fuzzer,address,undefined"


def find_targets(selected: "set[str]") -> "list[Path]":
    targets = sorted(rt.MODULES_DIR.glob("*/c/fuzz/*.c"))
    if selected:
        targets = [t for t in targets if t.stem in selected]
    return targets


def build(target: Path, out_dir: Path) -> "Path | None":
    binary = out_dir / target.stem
    cmd = [
        FUZZ_CC,
        "-g",
        "-O1",
        f"-fsanitize={FUZZ_SANITIZE}",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
        "-std=gnu11",
        "-DVFHE_RNG_TESTING",
        *rt.PORTABLE_DEFS,
        *rt.ALL_INCLUDES,
        str(target),
        *map(str, rt.ALL_KERNELS),
        "-lm",
        "-o",
        str(binary),
    ]
    cc = subprocess.run(cmd, capture_output=True, text=True)
    if cc.returncode != 0:
        print(f"[FAIL build] {target.relative_to(rt.ROOT)}\n{cc.stderr}")
        return None
    print(f"[built] {binary.name}")
    return binary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="fuzz target stems (default: all)")
    ap.add_argument("--run", action="store_true", help="run each target after building")
    ap.add_argument(
        "--time", type=int, default=60, help="seconds per target with --run"
    )
    ap.add_argument("--out", default=None, help="output dir (default: .cache/fuzz)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else rt.ROOT / ".cache" / "fuzz"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = find_targets(set(args.targets))
    if not targets:
        print("no fuzz targets found under modules/*/c/fuzz/")
        return 2

    ok = True
    for target in targets:
        binary = build(target, out_dir)
        if binary is None:
            ok = False
            continue
        if args.run:
            corpus = out_dir / f"{target.stem}_corpus"
            corpus.mkdir(exist_ok=True)
            print(f"[fuzz] {binary.name} for {args.time}s")
            run = subprocess.run(
                [
                    str(binary),
                    str(corpus),
                    f"-max_total_time={args.time}",
                    "-print_final_stats=1",
                ]
            )
            if run.returncode != 0:
                print(f"[CRASH] {target.stem} exited {run.returncode}")
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
