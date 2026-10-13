# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Exports the loaded engine as cffi exposes it: `lib` and `ffi`.

- `lib` holds the C functions: ``lib.vfhe_set_num_threads(4)``.
- `ffi` builds and reads the C data they take: ``ffi.new``, ``ffi.cast``,
  ``ffi.buffer``, ``ffi.NULL``.

Importing this package loads an extension, which stays loaded for the life of
the process. A custom engine needs `kernels.ExtensionBuilder.build()` to run first.

>>> from vfhe.util.bindings import ffi, lib
>>> out = ffi.new("uint64_t[]", 8)
>>> lib.vfhe_num_threads()
1
"""

from ._core import ffi, lib

__all__ = ["ffi", "lib"]
