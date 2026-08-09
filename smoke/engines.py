#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the install carries every engine it claims, not just the loaded one.

Executing another engine needs its ISA (or an emulator), which is the test
suites' business. What only an install can answer is whether the artifact
*shipped* each engine the picker may choose: its extension, its static archive,
and the JSON a runtime-compiled module needs to match its ABI. A wheel that
dropped one would still import and pass every other smoke test.
"""

import importlib.util
from pathlib import Path

from _vfhe_engines import ENGINES
from vfhe.misc.libvfhe import active_engine, runnable_engines


def find_source_dir() -> Path:
    """The installed build inputs (vfhe/_source), where the archives live. Asked
    by spec: `vfhe.misc.libvfhe` as an attribute is the singleton, not the module."""
    spec = importlib.util.find_spec("vfhe.misc.libvfhe")
    if spec is None or spec.origin is None:
        raise SystemExit("vfhe.misc.libvfhe is not installed")
    return Path(spec.origin).parent.parent / "_source"


def check(label: str, ok: bool) -> bool:
    print(f"  {label:<52} [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    lib = find_source_dir() / "lib"
    print(f"vfhe engines: {', '.join(name for name, _ in ENGINES)}\n")

    ok = True
    for name, _ in ENGINES:
        ok &= check(
            f"{name}: extension installed",
            importlib.util.find_spec(f"_vfhe_native_{name}") is not None,
        )
        ok &= check(
            f"{name}: archive for runtime modules",
            (lib / f"libvfhe_{name}.a").is_file(),
        )
        ok &= check(
            f"{name}: engine-{name}.json", (lib / f"engine-{name}.json").is_file()
        )

    ok &= check(
        f"the loaded engine ({active_engine()}) is one of them",
        active_engine() in [n for n, _ in ENGINES],
    )
    ok &= check("this CPU can run at least one", bool(runnable_engines()))

    print("\n" + ("OK: every declared engine is installed." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
