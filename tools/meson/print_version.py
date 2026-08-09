#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Print the package version for meson's ``project()``.

In a git checkout, setuptools-scm derives it from tags; an sdist has no git,
so ``prepare_sdist.py`` freezes the version into ``.version`` and this
prefers that file. Run by meson, not by hand.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def version_from_git_tags() -> str:
    scm = subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return scm.stdout.strip()


def main() -> int:
    frozen = ROOT / ".version"
    version = frozen.read_text().strip() if frozen.exists() else version_from_git_tags()

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
