# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The compiler a build uses: its command, its version, and the flags meson.build sets per engine."""

from __future__ import annotations

import os
import shlex
import subprocess
import sysconfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def command() -> str:
    """Returns the compiler command: ``$CC`` when set, else the one Python was built with, else ``cc``."""
    return os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"


def version(cc: str) -> str:
    """Returns `cc` followed by the first line ``cc --version`` prints.

    ``<cc> version-unknown`` when `cc` is missing, exits nonzero or takes over
    30 s.
    """
    try:
        r = subprocess.run(  # noqa: S603 - the configured compiler
            [*shlex.split(cc), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return f"{cc} version-unknown"
    first_line = r.stdout.split("\n", 1)[0].strip()
    return f"{cc} {first_line}"


def flags(engine: str) -> Sequence[str]:
    """Returns the compile flags meson.build sets for `engine`.

    They come from ``_vfhe_flags``, which meson generates and installs beside the
    extensions. Optimisation, LTO and the C standard are meson options, which the
    generated ``meson.build`` sets as the root one does. Raises `RuntimeError` for
    an engine this build lacks.
    """
    from _vfhe_flags import ENGINES  # pyright: ignore[reportMissingImports]

    if engine not in ENGINES:
        raise RuntimeError(f"this install has no {engine} engine")
    return ENGINES[engine]
