#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Put one distribution in a scratch venv — a user's environment, reproduced.

What to install is either a file this build produced or a requirement an index
serves, so the same command prepares a release candidate and a published
release. Whoever needs that environment reads it afterwards: the smoke tests
run against it, the SBOM scans it.

Usage:
    python tools/install.py <venv dir> dist/vfhe-1.2.3.tar.gz
    python tools/install.py <venv dir> vfhe==1.2.3 [index url]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import venv
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))  # the shared parts live beside this file

from _common import error, log  # noqa: E402  (a part; TOOLS above)

USAGE = __doc__ or ""
PYPI = "https://pypi.org/simple/"
INDEX_ATTEMPTS = 5
INDEX_WAIT_SECONDS = 30


def install_command(python: Path, package: str, index: str) -> list[str]:
    command = [str(python), "-m", "pip", "install", package]
    if index:
        # An alternate index carries vfhe alone; its dependencies stay on PyPI.
        command += ["--index-url", index, "--extra-index-url", PYPI]
    return command


def install_package(python: Path, package: str, index: str) -> bool:
    """A file installs once; a requirement gets retried, because an index needs
    a moment to serve a release that was just published."""
    attempts = 1 if Path(package).is_file() else INDEX_ATTEMPTS
    command = install_command(python, package, index)

    for attempt in range(1, attempts + 1):
        log(f"[install] {package} (attempt {attempt}/{attempts})")
        if subprocess.run(command).returncode == 0:
            return True
        if attempt < attempts:
            time.sleep(INDEX_WAIT_SECONDS)
    return False


def resolve_package(package: str) -> str:
    """A path becomes absolute (the venv is built elsewhere); a requirement
    passes through untouched."""
    if Path(package).is_file():
        return str(Path(package).resolve())
    return package


def main() -> int:
    match sys.argv[1:]:
        case [directory, package]:
            index = ""
        case [directory, package, index]:
            pass
        case _:
            log(USAGE)
            return 2

    package = resolve_package(package)
    # abspath because callers run from another cwd; never resolve(), which would
    # follow the venv's symlink out to the base interpreter.
    directory = Path(os.path.abspath(directory))

    log(f"[venv] recreating {directory}")
    venv.EnvBuilder(with_pip=True, clear=True).create(directory)

    python = directory / "bin" / "python"
    if not python.is_file():
        error(f"no interpreter at {python}; this expects a POSIX venv layout")
        return 1

    if not install_package(python, package, index):
        return 1
    log(f"[venv] {package} installed in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
