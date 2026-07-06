#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile and run every module's Unity C test suite.

Discovers each ``modules/*/c/test/*.c`` and builds it against the full portable
engine (all module kernels + vendored BLAKE3), using the shared
``native/discovery.py`` so the test build never drifts from the packaged CFFI
build. Honours ``VFHE_SANITIZE`` (ASan/UBSan flags) and ``CC``. Pass module
names as arguments to restrict the run (e.g. ``run_c_tests.py arith base``).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules"
UNITY_DIR = ROOT / "external" / "unity" / "src"
CC = os.environ.get("CC", "cc")

sys.path.insert(0, str(ROOT / "native"))
import discovery  # noqa: E402  (needs ROOT on sys.path first)

# Optional sanitizers: VFHE_SANITIZE="address,undefined" (or "thread", ...) turns
# on the matching -fsanitize instrumentation with no-recover semantics so the
# first diagnostic aborts the run with a non-zero exit. Used by the nightly CI.
_SANITIZE = os.environ.get("VFHE_SANITIZE", "").strip()
SANITIZE_FLAGS: "list[str]" = (
    [
        f"-fsanitize={_SANITIZE}",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
        "-g",
        "-O1",
    ]
    if _SANITIZE
    else []
)

_ALIASES = discovery.host_arch_aliases()
BLAKE3_DIR = discovery.blake3_dir(ROOT)

# Same engine the packaged build compiles, linked into every test binary.
ALL_KERNELS = discovery.module_sources(
    MODULES_DIR, _ALIASES
) + discovery.blake3_sources(ROOT)

# Portable baseline (scalar engine + BLAKE3 SIMD off) so the tests build everywhere.
PORTABLE_DEFS = discovery.macros_as_cli(
    discovery.portable_macros(discovery.is_x86_host())
)

ALL_INCLUDES = [f"-I{d}" for d in discovery.module_include_dirs(MODULES_DIR)] + [
    f"-I{UNITY_DIR}",
    f"-I{BLAKE3_DIR}",
]
UNITY_SRC = UNITY_DIR / "unity.c"


def run_module(mod_path: Path, workdir: Path) -> bool:
    tests = sorted((mod_path / "c" / "test").rglob("*.c"))
    if not tests:
        return True

    ok = True
    for test in tests:
        binary = workdir / f"{mod_path.name}_{test.stem}"
        compile_cmd = [
            CC,
            "-Wall",
            "-Wextra",
            "-std=c11",
            *SANITIZE_FLAGS,
            *PORTABLE_DEFS,
            *ALL_INCLUDES,
            str(test),
            str(UNITY_SRC),
            *map(str, ALL_KERNELS),
            "-lm",
            "-o",
            str(binary),
        ]
        cc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if cc.returncode != 0:
            print(f"[FAIL compile] {test.relative_to(ROOT)}\n{cc.stderr}")
            ok = False
            continue
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode != 0:
            print(
                f"[FAIL run] {test.relative_to(ROOT)} (exit {run.returncode})\n{run.stderr}"
            )
            ok = False
    return ok


def main() -> int:
    if not UNITY_SRC.exists():
        print(
            f"Unity not found at {UNITY_SRC.relative_to(ROOT)}.\n"
            f"Run: git submodule update --init external/unity"
        )
        return 2

    selected = set(sys.argv[1:])
    mods = sorted(p for p in MODULES_DIR.iterdir() if p.is_dir())
    if selected:
        mods = [m for m in mods if m.name in selected]
        missing = selected - {m.name for m in mods}
        if missing:
            print(f"unknown module(s): {', '.join(sorted(missing))}")
            return 2

    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for mod in mods:
            all_ok &= run_module(mod, Path(tmp))

    print("\nC tests:", "PASSED" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
