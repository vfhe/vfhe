#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install an sdist into a scratch venv and prove it installs.

The venv is recreated at --venv each run, so nothing touches the local
environment; pip compiles the native extension. Prints the venv's Python path
to stdout.

Usage:
    python scripts/check_install.py dist/vfhe-*.tar.gz [--venv DIR]
"""

from __future__ import annotations

import argparse
import subprocess
import venv
from pathlib import Path

from _common import ROOT, error, log, venv_python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path, help="path to the sdist tarball")
    parser.add_argument(
        "--venv",
        type=Path,
        default=ROOT / ".cache" / "install" / "venv",
        help="where to (re)create the venv (default: .cache/install/venv)",
    )
    args = parser.parse_args()

    if not args.sdist.is_file():
        error(f"no such sdist: {args.sdist}")
        return 2
    sdist = args.sdist.resolve()
    venv_dir = args.venv.resolve()

    log(f"[venv] recreating {venv_dir}")
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    python = venv_python(venv_dir)

    log(f"[install] {sdist.name} (compiles the native extension)")
    installed = subprocess.run([str(python), "-m", "pip", "install", str(sdist)])
    if installed.returncode != 0:
        error("pip install of the sdist failed")
        return 1

    print(python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
