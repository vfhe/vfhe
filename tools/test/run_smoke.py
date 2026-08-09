#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Run every smoke test in smoke/ against the interpreter running this script.

So the install under test is whichever environment invokes it — the scratch
venv `tools/install.py` prepared, or any venv you already have. Each test runs
from a temporary directory, never the repo root: a smoke test that could import
the source tree would not be testing the install.

Usage: <venv>/bin/python tools/test/run_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))  # the shared parts live one level up

from _common import ROOT, error, log  # noqa: E402  (a part; TOOLS above)


def find_tests() -> list[Path]:
    return sorted((ROOT / "smoke").glob("*.py"))


def run_test(test: Path, workdir: Path) -> int:
    log(f"[smoke] {test.name}")

    result = subprocess.run([sys.executable, str(test)], cwd=workdir)
    return result.returncode


def report(total: int, failed: list[str]) -> None:
    if failed:
        error(f"smoke tests failed: {', '.join(failed)}")
    else:
        print(f"\nsmoke tests: all {total} passed")


def main() -> int:
    if sys.argv[1:]:
        log(__doc__ or "")
        return 2
    tests = find_tests()

    with tempfile.TemporaryDirectory() as workdir:
        statuses = [(t.name, run_test(t, Path(workdir))) for t in tests]
    failed = [name for name, status in statuses if status != 0]

    report(len(tests), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
