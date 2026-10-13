# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Emits the C of a cffi extension module: its ``PyInit`` and a stub for every function the cdefs declare.

meson runs this file for the shipped engines::

    python _ffi.py MODULE OUT.c CDEF... -- HEADER...

`ExtensionBuilder` calls `emit` with your declarations added to both lists.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
from typing import TYPE_CHECKING

import cffi

if TYPE_CHECKING:
    from collections.abc import Iterable


def emit(
    module: str,
    out: pathlib.Path,
    cdefs: Iterable[pathlib.Path],
    headers: Iterable[pathlib.Path],
) -> None:
    """Writes `out`, the C of extension module `module`: a stub for every function in `cdefs`, compiled against `headers`.

    `cdefs` are in cffi's cdef dialect. `headers` are included by name, in order,
    so their directories belong on the include path of whatever compiles `out`;
    a cdef file may be among them when its prototypes are valid C.
    """
    ffi = cffi.FFI()
    for cdef in cdefs:
        ffi.cdef(cdef.read_text())
    ffi.set_source(module, "\n".join(f'#include "{h.name}"' for h in headers))
    # cffi prints "generating …" to stdout, which the kernels CLI reserves for the directory.
    with contextlib.redirect_stdout(io.StringIO()):
        ffi.emit_c_code(str(out))


if __name__ == "__main__":
    module, out, *paths = sys.argv[1:]
    split = paths.index("--")
    emit(
        module,
        pathlib.Path(out),
        map(pathlib.Path, paths[:split]),
        map(pathlib.Path, paths[split + 1 :]),
    )
