#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Link every modules/*/c/fuzz/ harness against the kernels meson built.

build.sh builds the instrumented archive (meson inherits the container's
CFLAGS); this compiles each harness like a kernel and links it with
$LIB_FUZZING_ENGINE, which supplies main() and is C++ — hence $CXX for the
link. Binaries land in $OUT under the bare name ClusterFuzzLite expects.
Container-only: outside it the required environment is absent.

Usage (build.sh's, inside the container):
    build_c_fuzz_tests.py --engine            the engine this machine can fuzz
    build_c_fuzz_tests.py <path to libvfhe_<engine>.a>
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import _engines  # noqa: E402  (needs ROOT/tools on sys.path first)


def error(message: str) -> None:
    """Not _common's: that one annotates only when $GITHUB_ACTIONS is set,
    which it is not inside this container, though the log still lands in a
    workflow that renders the annotation."""
    print(f"::error::{message}", file=sys.stderr)


def choose_engine() -> str:
    """The best engine this machine executes, because a fuzzer runs what it
    builds: emulating one would cost 10-50x on top of the sanitizer's, and the
    corpus would advance that much slower. Hosted runners therefore fuzz the
    portable engine and the project's own machine fuzzes its SIMD one."""
    cpuinfo = Path("/proc/cpuinfo")
    flags = cpuinfo.read_text() if cpuinfo.exists() else ""
    for engine in _engines.buildable(ROOT):  # best first
        if engine.requires is None or re.search(
            rf"\b{re.escape(engine.requires)}\b", flags
        ):
            return engine.name
    error("no engine is buildable here (tools/_engines.py)")
    raise SystemExit(1)


def engine_of(archive: Path) -> _engines.Engine:
    """The engine an archive holds, so the harnesses compile with its flags."""
    engine = _engines.by_name(ROOT, archive.stem.removeprefix("libvfhe_"))
    if engine is None:
        error(f"{archive.name} names no engine in tools/_engines.py")
        raise SystemExit(1)
    return engine


def run(step: list[str], what: str) -> bool:
    done = subprocess.run(step, capture_output=True, text=True)
    if done.returncode:
        error(f"{what} failed:\n{done.stderr}")
    return not done.returncode


def build(archive: Path, targets: list[Path]) -> bool:
    """Compile and link every harness into $OUT; True if all succeed."""
    try:
        cc, cxx = os.environ["CC"], os.environ["CXX"]
        engine, out = os.environ["LIB_FUZZING_ENGINE"], Path(os.environ["OUT"])
    except KeyError as missing:
        error(f"{missing} is not set; run this under ClusterFuzzLite")
        return False
    work = Path(os.environ.get("WORK", out))
    cxxflags = shlex.split(os.environ.get("CXXFLAGS", ""))
    kernels = engine_of(archive)
    as_kernel = [
        *shlex.split(os.environ.get("CFLAGS", "")),
        "-std=gnu11",
        *kernels.cflags,
        *[f"-I{d}" for d in _engines.find_include_dirs(ROOT)],
    ]

    built = True
    for target in targets:
        obj = work / f"{target.stem}.o"
        binary = out / target.stem
        compiled = run([cc, *as_kernel, "-c", str(target), "-o", str(obj)], target.name)
        if not compiled:
            built = False
            continue
        link = [
            cxx,
            *cxxflags,
            str(obj),
            str(archive),
            engine,
            "-lm",
            "-o",
            str(binary),
        ]
        linked = run(link, f"linking {target.stem}")
        if not linked:
            built = False
            continue
        print(f"[built] {binary}", file=sys.stderr)
    return built


def main() -> int:
    if len(sys.argv) != 2:
        error("usage: build_c_fuzz_tests.py --engine | <libvfhe_<engine>.a>")
        return 2
    # build.sh asks which engine to compile before there is an archive to link.
    if sys.argv[1] == "--engine":
        print(choose_engine())
        return 0
    archive = Path(sys.argv[1])
    if not archive.is_file():
        error(f"no kernel archive at {archive}; did meson compile run?")
        return 2
    targets = sorted((ROOT / "modules").glob("*/c/fuzz/*.c"))
    if not targets:
        error("no fuzz targets under modules/*/c/fuzz/")
        return 2
    succeeded = build(archive, targets)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
