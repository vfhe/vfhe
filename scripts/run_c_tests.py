#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules"
UNITY_DIR = ROOT / "external" / "unity" / "src"
CC = os.environ.get("CC", "cc")

_ARCH_ALIASES = ({"x86_64", "amd64", "x86-64", "x64"}, {"aarch64", "arm64"})


def _arch_aliases() -> "set[str]":
    machine = platform.machine().lower()
    for group in _ARCH_ALIASES:
        if machine in group:
            return group
    return {machine}


def _arch_ok(path: Path, aliases: "set[str]") -> bool:
    """False only if ``path`` lives under an ``arch/<other-arch>/`` directory."""
    parts = [p.lower() for p in path.parts]
    for i, seg in enumerate(parts[:-1]):
        if seg == "arch":
            return parts[i + 1] in aliases
    return True


_ALIASES = _arch_aliases()
ALL_KERNELS = [
    p
    for pattern in ("*.c", "*.S")
    for p in sorted(MODULES_DIR.glob(f"*/c/src/**/{pattern}"))
    if _arch_ok(p, _ALIASES)
]
_INCLUDE_DIRS = {d for d in MODULES_DIR.glob("*/c/include")}
_INCLUDE_DIRS |= {h.parent for h in MODULES_DIR.glob("*/c/src/**/*.h")}
ALL_INCLUDES = [f"-I{d}" for d in sorted(_INCLUDE_DIRS)] + [f"-I{UNITY_DIR}"]
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
            *ALL_INCLUDES,
            str(test),
            str(UNITY_SRC),
            *map(str, ALL_KERNELS),
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
