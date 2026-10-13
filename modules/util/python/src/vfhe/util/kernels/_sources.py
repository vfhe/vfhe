# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""What a build takes from the install: vfhe's version, the build cache, and vfhe's C unpacked from the shipped tarball."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.resources
import os
import pathlib
import shutil
import tarfile
import tempfile


def version() -> str:
    """Returns the installed version of vfhe; ``unknown`` in a source tree."""
    try:
        return importlib.metadata.version("vfhe")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def cache_root() -> pathlib.Path:
    """Returns the build cache: ``$VFHE_BUILD_DIR``, otherwise ``$XDG_CACHE_HOME/vfhe`` or ``~/.cache/vfhe``.

    Projects share it: every build has a directory of its own under ``extensions/``,
    and vfhe's C is unpacked once per version under ``src/``.
    """
    env = os.environ.get("VFHE_BUILD_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return pathlib.Path(base).resolve() / "vfhe"


@dataclasses.dataclass(frozen=True)
class SourceTree:
    """vfhe's C by role, as the shipped tarball lays it out."""

    cdefs: list[pathlib.Path]
    """What Python may call, in cffi's cdef dialect: ``modules/*/python/cdef/*.h``."""
    headers: list[pathlib.Path]
    """The public headers: ``modules/*/c/include/*.h``."""
    include_dirs: list[pathlib.Path]
    """Where the compiler finds headers: the public ones, arith's internals, blake3's."""
    sources: list[pathlib.Path]
    """Every ``.c`` and ``.S``: meson put exactly what it compiles into the tarball."""


def source_tree() -> SourceTree:
    """Returns vfhe's C by role, from the install's ``vfhe/_source/kernels-src.tar.gz``, unpacked once per version.

    Safe to call from several processes at once; the first to finish unpacking
    wins. A reinstall under the same version reuses the unpacked tree; delete
    ``src/<version>`` to refresh it. Raises `RuntimeError` when the install has
    no tarball.
    """
    tarball = importlib.resources.files("vfhe") / "_source" / "kernels-src.tar.gz"
    if not tarball.is_file():
        raise RuntimeError(
            f"vfhe's C sources are not available: {tarball} is missing, so this "
            "install cannot compile custom kernels."
        )

    out = cache_root() / "src" / version()

    if not out.is_dir():
        # Unpack into a sibling and rename, so an interrupted extraction never
        # leaves a partial tree at `out`.
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(tempfile.mkdtemp(dir=out.parent))
        try:
            with tarball.open("rb") as f, tarfile.open(fileobj=f) as t:
                # `filter` exists from Python 3.12 and the 3.10.12 / 3.11.4
                # security releases on; this is the documented check for it.
                if hasattr(tarfile, "data_filter"):
                    t.extractall(staging, filter="data")
                else:
                    t.extractall(staging)  # noqa: S202 - our own tarball
            staging.replace(out)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            # Another process won the race when `out` exists; its tree is the same.
            if not out.is_dir():
                raise

    return SourceTree(
        cdefs=sorted(out.glob("modules/*/python/cdef/*.h")),
        headers=sorted(out.glob("modules/*/c/include/*.h")),
        include_dirs=[
            *sorted(out.glob("modules/*/c/include")),
            out / "modules/arith/c/src",
            out / "external/blake3/blake3/c",
        ],
        sources=sorted([*out.rglob("*.c"), *out.rglob("*.S")]),
    )
