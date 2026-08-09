#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Make the sdist self-contained (meson dist script).

``meson dist`` archives only the main git tree, so BLAKE3's vendored sources
are copied in, and the version (argv[1], from meson) is frozen into
``.version`` — an sdist has no git for setuptools-scm to read. Run by meson.
"""

import os
import shutil
import sys
from pathlib import Path

BLAKE3 = "external/blake3"


def vendor_blake3(source_root: Path, dist_root: Path) -> None:
    """Only the sources the build compiles and the licences they ship under —
    not the whole submodule."""
    vendored = source_root / BLAKE3
    # meson dist may already have archived the submodule path; copy in either way.
    (dist_root / BLAKE3 / "c").mkdir(parents=True, exist_ok=True)

    for pattern in ("c/*.c", "c/*.h", "c/*.S", "LICENSE_*"):
        for path in sorted(vendored.glob(pattern)):
            shutil.copy2(path, dist_root / BLAKE3 / path.relative_to(vendored))


def freeze_version(dist_root: Path, version: str) -> None:
    (dist_root / ".version").write_text(version + "\n")


def main() -> int:
    source_root = Path(os.environ["MESON_SOURCE_ROOT"])
    dist_root = Path(os.environ["MESON_DIST_ROOT"])

    vendor_blake3(source_root, dist_root)
    freeze_version(dist_root, sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
