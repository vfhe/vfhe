#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run every smoke/*.py with the given interpreter.

Each smoke test is a standalone, self-verifying program that exits non-zero
on failure; vfhe must be importable by the chosen interpreter. Tests run from
.cache/smoke so the source tree is never on sys.path.

Usage:
    python scripts/run_smoke_tests.py --python .cache/install/venv/bin/python
    python scripts/run_smoke_tests.py                # current interpreter
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import ROOT, error, log

SMOKE_DIR = ROOT / "smoke"
WORK_DIR = ROOT / ".cache" / "smoke"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter to run the tests with (default: current)",
    )
    args = parser.parse_args()

    python = args.python.resolve()  # tests run from WORK_DIR; keep it absolute
    if not python.is_file():
        error(f"no such interpreter: {args.python}")
        return 2

    smoke_tests = sorted(SMOKE_DIR.glob("*.py"))
    if not smoke_tests:
        error("no smoke tests under smoke/")
        return 2

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for smoke_test in smoke_tests:
        log(f"[smoke] {smoke_test.name}")
        result = subprocess.run([str(python), str(smoke_test)], cwd=WORK_DIR)
        if result.returncode != 0:
            failed.append(smoke_test.name)

    if failed:
        error(f"smoke tests failed: {', '.join(failed)}")
        return 1
    print(f"\nsmoke tests: all {len(smoke_tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
