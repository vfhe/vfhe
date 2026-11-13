# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Compiles the caller's kernels and vfhe's C into one engine: `directory`, `compile`, `build`.

Each takes the same inputs. `sources` are your ``.c`` and ``.S`` files.
`declarations` are ``.h`` files in cffi's cdef dialect: prototypes and types,
with no preprocessor line beyond a literal ``#define``. Each is both declared
to cffi, so `vfhe.util.bindings.lib` exposes its functions, and included after
vfhe's headers in the wrapper. `engine` defaults to the one this process would
load, `cc` to ``$CC``, else the compiler Python was built with. Each raises
`ValueError` for a name that is no engine and `RuntimeError` for an engine this
install lacks.
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import sys
from typing import TYPE_CHECKING

from vfhe.util._engine import Engine

from . import _compiler, _ffi, _meson, _sources

if TYPE_CHECKING:
    from collections.abc import Iterable

    Paths = Iterable[str | pathlib.Path]


def _inputs(
    sources: Paths, declarations: Paths, engine: str | None, cc: str | None
) -> tuple[Engine, list[pathlib.Path], list[pathlib.Path], str]:
    """The arguments resolved: the engine, the files as absolute paths, the compiler."""
    chosen = Engine(engine) if engine else Engine.select()
    _compiler.flags(chosen.value)  # raises for an engine this install lacks
    return (
        chosen,
        [pathlib.Path(s).resolve() for s in sources],
        [pathlib.Path(d).resolve() for d in declarations],
        cc or _compiler.command(),
    )


def directory(
    sources: Paths = (),
    declarations: Paths = (),
    *,
    engine: str | None = None,
    cc: str | None = None,
) -> pathlib.Path:
    """Returns the cache directory of this build, ``extensions/<engine>-<hash>``.

    The hash names what identifies the build: vfhe's version, the engine, the
    compiler, Python, and the paths of your files. Their contents are ninja's
    business, so an edit recompiles what it reached, in place.
    """
    chosen, files_c, files_h, compiler = _inputs(sources, declarations, engine, cc)
    names = [
        _sources.version(),
        chosen.value,
        _compiler.version(compiler),
        platform.python_version(),
        *map(str, sorted([*files_c, *files_h])),
    ]
    key = hashlib.sha256("\0".join(names).encode()).hexdigest()[:16]
    return _sources.cache_root() / "extensions" / f"{chosen.value}-{key}"


def compile(
    sources: Paths = (),
    declarations: Paths = (),
    *,
    engine: str | None = None,
    cc: str | None = None,
) -> pathlib.Path:
    """Compiles the engine into its `directory` and returns it.

    Recompiles only what changed; delete the directory to start over. Leaves
    this process untouched. Needs meson and ninja on ``PATH``
    (``pip install vfhe[kernels]``). Raises `RuntimeError` when they or vfhe's
    C sources are missing, and when the build fails.
    """
    chosen, files_c, files_h, compiler = _inputs(sources, declarations, engine, cc)
    out = directory(sources, declarations, engine=engine, cc=cc)
    out.mkdir(parents=True, exist_ok=True)

    tree = _sources.source_tree()
    module = chosen.module
    wrapper = out / f"{module}.c"

    _ffi.emit(
        module,
        wrapper,
        [*tree.cdefs, *files_h],
        [*tree.headers, *files_h],
    )
    _meson.write(
        out,
        module=module,
        c_args=_compiler.flags(chosen.value),
        include_dirs=[
            *tree.include_dirs,
            *sorted({p.parent for p in [*files_c, *files_h]}),
        ],
        lib_sources=tree.sources,
        user_sources=[*files_c, wrapper],
    )
    _meson.build(out, cc=compiler)
    return out


def build(
    sources: Paths = (),
    declarations: Paths = (),
    *,
    engine: str | None = None,
    cc: str | None = None,
) -> pathlib.Path:
    """Compiles the engine if needed, puts its directory first on ``sys.path`` and returns it.

    Returns at once when this build is already loaded. Raises `RuntimeError`
    when another engine is loaded.
    """
    active = Engine.active()
    if active is None:
        out = compile(sources, declarations, engine=engine, cc=cc)
        sys.path.insert(0, str(out))
        return out

    wanted = directory(sources, declarations, engine=engine, cc=cc)
    loaded = pathlib.Path(sys.modules[active.module].__file__ or "").resolve().parent
    if loaded == wanted:
        return loaded
    raise RuntimeError(
        f"{active.module} is already loaded from {loaded} and cannot be replaced. "
        "Call build() before importing vfhe's C, or build ahead of time with "
        "`python -m vfhe.util.kernels` and put its directory on PYTHONPATH."
    )
