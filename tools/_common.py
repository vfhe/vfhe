# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Repo paths, logging, and what this CPU can run.

Importable because `python tools/<name>.py` puts this folder on sys.path.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"  # meson's tree: everything compiled, gitignored


def log(message: str) -> None:
    print(message, file=sys.stderr)


def error(message: str) -> None:
    prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") else "error: "
    print(f"{prefix}{message}", file=sys.stderr)


def find_tool(name: str) -> str:
    """An executable's full path: subprocess calls take no partial ones."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} not found on PATH")
    return path


def host_supports(capability: str | None) -> bool:
    """Whether this CPU satisfies an engine's `requires`, per the build's own
    probe extension — so the tools and the engine picker act on one truth.
    Needs `make build`."""
    if not capability:
        return True

    sys.path.insert(0, str(BUILD_DIR))
    try:
        from _vfhe_cpu import lib
    except ImportError as exc:  # a caller that forgot to build
        raise RuntimeError("run `make build` before probing the CPU") from exc
    finally:
        sys.path.pop(0)

    supported = bool(lib.vfhe_cpu_supports(capability.encode()))
    return supported
