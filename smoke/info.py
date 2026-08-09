#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: `python -m vfhe.info` answers from an installed package.

Standalone and self-verifying: runs the module the way a bug reporter does — a
subprocess of this interpreter — and checks it reports a real version and the
loaded engine. In a source tree the version is unknown by design, so only an
install proves this path.
"""

import subprocess
import sys

from vfhe.misc.libvfhe import active_engine

EXPECTED = ("vfhe", "engine", "python", "platform")


def run_module() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "vfhe.info"], capture_output=True, text=True, check=True
    )
    return result.stdout


def check(label: str, ok: bool) -> bool:
    print(f"  {label:<34} [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    output = run_module()
    print("python -m vfhe.info reports:\n")
    for line in output.splitlines():
        print(f"    {line}")
    print()

    labels = [line.split()[0] for line in output.splitlines() if line.strip()]
    ok = check("every fact present", list(EXPECTED) == labels)
    ok &= check("the loaded engine is named", active_engine() in output)
    ok &= check("an installed version, not a tree", "source tree" not in output)

    print("\n" + ("OK: the install describes itself." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
