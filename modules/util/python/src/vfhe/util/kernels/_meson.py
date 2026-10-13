# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Hands an engine build to meson: fills ``meson.build.in``, configures once, then builds and installs."""

from __future__ import annotations

import importlib.resources
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable, Sequence


def _strings(items: Iterable[object]) -> str:
    """Formats `items` as a meson list body of string literals."""
    return ", ".join(f"'{item}'" for item in items)


def write(
    directory: pathlib.Path,
    *,
    module: str,
    c_args: Sequence[str],
    include_dirs: Sequence[pathlib.Path],
    lib_sources: Sequence[pathlib.Path],
    user_sources: Sequence[pathlib.Path],
) -> None:
    """Writes ``directory/meson.build`` from ``meson.build.in``: vfhe's C compiled with `c_args`, `user_sources` without ``-DVFHE_BUILDING_LIBRARY``, all linked into the extension `module` and installed into `directory`.

    The file is rewritten only when its text changes, since a rewrite makes
    meson reconfigure.
    """
    values = {
        "DIRECTORY": str(directory),
        "MODULE": module,
        "PYTHON": sys.executable,
        "C_ARGS": _strings(c_args),
        "USER_C_ARGS": _strings(a for a in c_args if a != "-DVFHE_BUILDING_LIBRARY"),
        "INCLUDE_DIRS": _strings(include_dirs),
        "LIB_SOURCES": _strings(lib_sources),
        "USER_SOURCES": _strings(user_sources),
    }
    template = importlib.resources.files("vfhe.util.kernels") / "meson.build.in"
    text = re.sub(r"@(\w+)@", lambda m: values[m.group(1)], template.read_text())
    path = directory / "meson.build"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def build(directory: pathlib.Path, *, cc: str) -> None:
    """Configures ``directory/build`` once with `cc`, then builds and installs the module into `directory`.

    Raises `RuntimeError` when meson or ninja is missing, and when the build
    fails, with meson's output in the message.
    """
    build_dir = directory / "build"
    commands = []
    if not (build_dir / "build.ninja").exists():
        commands.append(["meson", "setup", str(build_dir), str(directory)])

    commands.append(["meson", "install", "-C", str(build_dir)])

    for command in commands:
        try:
            ran = subprocess.run(  # noqa: S603 - whichever meson PATH provides
                command, capture_output=True, text=True, env={**os.environ, "CC": cc}
            )
        except FileNotFoundError:
            raise RuntimeError(
                "meson is not on PATH; meson and ninja come with "
                "`pip install vfhe[kernels]`"
            ) from None
        if ran.returncode:
            raise RuntimeError(
                "building the engine failed; meson and ninja come with "
                "`pip install vfhe[kernels]`:\n" + ran.stdout + ran.stderr
            )
