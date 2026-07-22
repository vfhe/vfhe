#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the modules/*/c/fuzz/ harnesses with ClusterFuzzLite's toolchain.

Compiles with the $CC/$CXX/$CFLAGS/$LIB_FUZZING_ENGINE/$OUT contract into
$OUT; ClusterFuzzLite runs the results. Build inputs come from
packaging/discovery.py, so new harnesses and modules need no change here.
Container-only: outside it the required environment is absent.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def _log(message: str) -> None:
    """Progress goes to stderr; results own stdout."""
    print(message, file=sys.stderr)


def _error(message: str) -> None:
    """Errors go to stderr, prefixed ::error:: so CI annotates them."""
    print(f"::error::{message}", file=sys.stderr)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
import discovery  # noqa: E402  (needs ROOT/packaging on sys.path first)

CTX = discovery.tool_build_context(ROOT)


def build(targets: list[Path]) -> bool:
    """Build every harness into $OUT; True if all succeed.

    $LIB_FUZZING_ENGINE supplies main(); kernels compile once, shared across
    targets.
    """
    try:
        cc = os.environ["CC"]
        cxx = os.environ["CXX"]
        engine = os.environ["LIB_FUZZING_ENGINE"]
        out = Path(os.environ["OUT"])
    except KeyError as missing:
        _error(f"{missing} is not set; run this under ClusterFuzzLite")
        return False
    cflags = shlex.split(os.environ.get("CFLAGS", ""))
    cxxflags = shlex.split(os.environ.get("CXXFLAGS", ""))
    obj_dir = Path(os.environ.get("WORK", out)) / "vfhe-obj"
    obj_dir.mkdir(parents=True, exist_ok=True)

    def compile_to_object(source: Path, obj: Path) -> bool:
        cmd = [
            cc,
            *cflags,
            "-std=gnu11",
            *CTX.define_flags,
            *CTX.include_flags,
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        compiled = subprocess.run(cmd, capture_output=True, text=True)
        if compiled.returncode != 0:
            _error(f"compile failed: {source}\n{compiled.stderr}")
        return compiled.returncode == 0

    # Kernels are identical for every target: compile them once.
    kernel_objects = []
    for index, source in enumerate(CTX.kernels):
        obj = obj_dir / f"k{index}_{source.stem}.o"
        if not compile_to_object(source, obj):
            return False
        kernel_objects.append(obj)

    all_built = True
    for target in targets:
        harness_obj = obj_dir / f"harness_{target.stem}.o"
        if not compile_to_object(target, harness_obj):
            all_built = False
            continue
        binary = out / target.stem  # $OUT/<name>, no extension: ClusterFuzzLite layout
        linked = subprocess.run(
            [
                cxx,
                *cxxflags,
                str(harness_obj),
                *map(str, kernel_objects),
                engine,
                "-lm",
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        if linked.returncode != 0:
            _error(f"link failed: {target.stem}\n{linked.stderr}")
            all_built = False
            continue
        _log(f"[built] {binary}")
    return all_built


def main() -> int:
    targets = sorted((ROOT / "modules").glob("*/c/fuzz/*.c"))
    if not targets:
        _error("no fuzz targets under modules/*/c/fuzz/")
        return 2
    return 0 if build(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
