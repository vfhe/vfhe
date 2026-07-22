# SPDX-License-Identifier: Apache-2.0
"""Shared script plumbing: repo root, discovery import, logging, OS differences.

Importable because `python scripts/<name>.py` puts this folder on sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "packaging"))
import discovery  # noqa: E402, F401  (re-exported to the scripts)

IS_WINDOWS = os.name == "nt"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
LINK_MATH = [] if IS_WINDOWS else ["-lm"]  # libm is part of the CRT on Windows


def log(message: str) -> None:
    """Progress goes to stderr; results own stdout."""
    print(message, file=sys.stderr)


def error(message: str) -> None:
    """Errors go to stderr, prefixed ::error:: so CI annotates them."""
    print(f"::error::{message}", file=sys.stderr)


def compiler(posix_default: str) -> str:
    """$CC if set, else the given default (clang on Windows: no `cc` shim)."""
    return os.environ.get("CC") or ("clang" if IS_WINDOWS else posix_default)


def venv_python(venv_dir: Path) -> Path:
    """The venv's interpreter; the directory layout differs on Windows."""
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"
