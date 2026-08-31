# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: `python -m vfhe.info` answers from an installed package.

In a source tree the version is unknown by design, so only an install proves it.
"""

import subprocess
import sys

from _report import check, exit_status
from vfhe.engine import active_engine

EXPECTED = ["vfhe", "engine", "python", "platform"]


def main() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "vfhe.info"], capture_output=True, text=True, check=True
    ).stdout
    lines = out.splitlines()

    print("python -m vfhe.info reports:\n")
    for line in lines:
        print(f"    {line}")
    print()

    labels = [line.split()[0] for line in lines if line.strip()]
    ok = check("every fact present", labels == EXPECTED)
    ok &= check("the loaded engine is named", active_engine() in out)
    ok &= check("an installed version, not a tree", "source tree" not in out)

    return exit_status(ok, "the install describes itself.")


if __name__ == "__main__":
    raise SystemExit(main())
