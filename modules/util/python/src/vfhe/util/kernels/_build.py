# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Holds `ExtensionBuilder`, which compiles the caller's kernels and vfhe's C into one engine."""

from __future__ import annotations

import functools
import hashlib
import pathlib
import platform
import sys
from typing import TYPE_CHECKING

from vfhe.util._engine import Engine

from . import _compiler, _ffi, _meson, _sources

if TYPE_CHECKING:
    from collections.abc import Iterable


class ExtensionBuilder:
    """Compiles your C kernels and vfhe's C into one engine.

    `sources` are your ``.c`` and ``.S`` files. `declarations` are ``.h`` files in
    cffi's cdef dialect: prototypes and types, with no preprocessor line beyond a
    literal ``#define``. Each is both declared to cffi, so `vfhe.util.bindings.lib`
    exposes its functions, and included after vfhe's headers in the wrapper.
    `engine` defaults to the one this process would load, `cc` to ``$CC``, else
    the compiler Python was built with. Raises `ValueError` for a name that is
    no engine and `RuntimeError` for an engine this install lacks.

    The build lands in a cache directory named after what identifies it: vfhe's
    version, the engine, the compiler, Python, and the paths of your files. Their
    contents are ninja's business, so an edit recompiles what it reached, in
    place.

    >>> ExtensionBuilder(["mine.c"], ["mine.h"]).build()
    PosixPath('/home/me/.cache/vfhe/extensions/avx512ifma-3f9c2a1b7e6d5c4f')
    """

    def __init__(
        self,
        sources: Iterable[str | pathlib.Path] = (),
        declarations: Iterable[str | pathlib.Path] = (),
        *,
        engine: str | None = None,
        cc: str | None = None,
    ) -> None:
        self._engine = Engine(engine) if engine else Engine.select()
        self._c_args = _compiler.flags(self._engine.value)
        self._sources = [pathlib.Path(s).resolve() for s in sources]
        self._declarations = [pathlib.Path(d).resolve() for d in declarations]
        self._cc = cc or _compiler.command()

    @functools.cached_property
    def directory(self) -> pathlib.Path:
        """Returns the cache directory of this build, ``extensions/<engine>-<hash>``."""
        inputs = [
            _sources.version(),
            self._engine.value,
            _compiler.version(self._cc),
            platform.python_version(),
            *map(str, sorted([*self._sources, *self._declarations])),
        ]
        key = hashlib.sha256("\0".join(inputs).encode()).hexdigest()[:16]
        return _sources.cache_root() / "extensions" / f"{self._engine.value}-{key}"

    def compile(self) -> pathlib.Path:
        """Compiles the engine into `directory` and returns it.

        Recompiles only what changed; delete `directory` to start over. Leaves
        this process untouched. Needs meson and ninja on ``PATH``
        (``pip install vfhe[kernels]``). Raises `RuntimeError` when they or vfhe's
        C sources are missing, and when the build fails.
        """
        out = self.directory
        out.mkdir(parents=True, exist_ok=True)

        tree = _sources.source_tree()
        module = self._engine.module
        wrapper = out / f"{module}.c"

        _ffi.emit(
            module,
            wrapper,
            [*tree.cdefs, *self._declarations],
            [*tree.headers, *self._declarations],
        )

        _meson.write(
            out,
            module=module,
            c_args=self._c_args,
            include_dirs=[
                *tree.include_dirs,
                *sorted({p.parent for p in [*self._sources, *self._declarations]}),
            ],
            lib_sources=tree.sources,
            user_sources=[*self._sources, wrapper],
        )
        _meson.build(out, cc=self._cc)
        return out

    def build(self) -> pathlib.Path:
        """Compiles the engine if needed, puts `directory` first on ``sys.path`` and returns it.

        Returns at once when this build is already loaded. Raises `RuntimeError`
        when another engine is loaded.
        """
        active = Engine.active()
        if active is None:
            sys.path.insert(0, str(self.compile()))
            return self.directory

        loaded = (
            pathlib.Path(sys.modules[active.module].__file__ or "").resolve().parent
        )

        if loaded == self.directory:
            return loaded

        raise RuntimeError(
            f"{active.module} is already loaded from {loaded} and cannot be replaced. "
            "Call build() before importing vfhe's C, or build ahead of time with "
            "`python -m vfhe.util.kernels` and put its directory on PYTHONPATH."
        )
