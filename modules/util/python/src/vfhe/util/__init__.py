# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Holds the runtime beneath every vfhe module: which engine runs and how Python reaches C.

- `bindings` loads the engine and exports `ffi` and `lib`; every other module
  reaches C through them.
- `kernels` compiles your C into an engine of your own.
- `io` serializes vfhe objects.

Importing `vfhe.util` imports none of them; import the one you use:

>>> from vfhe.util import bindings
"""
