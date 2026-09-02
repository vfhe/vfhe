#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Emits a cffi extension's C source from cdef declarations and the real headers.

ffi_c.py MODULE OUT.c CDEF... -- HEADER...
"""

import sys
from pathlib import Path

from cffi import FFI

name, out, *paths = sys.argv[1:]
split = paths.index("--")
cdefs, headers = paths[:split], paths[split + 1 :]

includes = ['#include "' + Path(header).name + '"' for header in headers]

ffi = FFI()
for cdef in cdefs:
    ffi.cdef(Path(cdef).read_text())
ffi.set_source(name, "\n".join(includes))
ffi.emit_c_code(out)
