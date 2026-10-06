# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: `python -m vfhe.info` answers from an installed package.

In a source tree the version is unknown by design, so only an install proves it.
"""

import importlib
import json
import subprocess
import sys

from _report import check, exit_status
from vfhe.util._engine import Engine

EXPECTED = ["version", "engine", "engine_file", "runs", "python", "platform"]


def main() -> int:
    importlib.import_module("vfhe.util.bindings")
    out = subprocess.run(
        [sys.executable, "-m", "vfhe.info"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(f"python -m vfhe.info reports:\n{out}")
    facts = json.loads(out)

    ok = check("every fact present", list(facts) == EXPECTED)
    active = Engine.active()
    ok &= check(
        "the loaded engine is named",
        active is not None and facts["engine"] == active.value,
    )
    ok &= check(
        "the loaded engine's file is named",
        active is not None
        and facts["engine_file"] == sys.modules[active.module].__file__,
    )
    ok &= check("the version is the install's", facts["version"] is not None)

    return exit_status(ok, "the install describes itself.")


if __name__ == "__main__":
    raise SystemExit(main())
