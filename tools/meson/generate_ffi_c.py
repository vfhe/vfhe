#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Emit a cffi extension's C source from cdef declarations and headers.

Compiling it is the build system's job (meson.build), which also names the
inputs — so this stays a generator with no opinion about what the build
contains. Used once per engine extension, and once for the CPU probe.

Positional, with ``--`` between the two lists, since meson.build is the
only caller:

    generate_ffi_c.py <module name> <output.c> <cdef>... -- <header>...
"""

from __future__ import annotations

import sys
from pathlib import Path

from cffi import FFI


def split_inputs(argv: list[str]) -> tuple[str, str, list[Path], list[Path]]:
    """`<name> <out> <cdefs>... -- <headers>...`, the `--` separating the two
    lists so both can be variadic."""
    if "--" not in argv:
        raise SystemExit(f"{__file__}: {__doc__}")

    cut = argv.index("--")
    name, out = argv[0], argv[1]
    cdefs = [Path(path) for path in argv[2:cut]]
    headers = [Path(path) for path in argv[cut + 1 :]]
    return name, out, cdefs, headers


def emit_extension_source(
    name: str, out: str, cdefs: list[Path], headers: list[Path]
) -> None:
    ffi = FFI()
    for cdef in cdefs:
        ffi.cdef(cdef.read_text())

    # Real headers are #included so the compiler sees the true definitions.
    includes = "\n".join(f'#include "{header.name}"' for header in headers)
    ffi.set_source(name, includes)
    ffi.emit_c_code(out)


def main() -> int:
    name, out, cdefs, headers = split_inputs(sys.argv[1:])
    emit_extension_source(name, out, cdefs, headers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
