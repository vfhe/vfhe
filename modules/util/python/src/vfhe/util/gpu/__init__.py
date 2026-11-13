# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Loads the CUDA extension, when the install has it and the machine a device.

The extension `_vfhe_cuda` exists in an install built where `nvcc` was found,
and loads on its own: the engine `bindings` loads is the same with or without
it. `available()` is the call every program may make. A module with kernels
of its own reaches them through `extension()`, as `vfhe.arith`'s ``cuda``
backend does.

>>> from vfhe.util import gpu
>>> gpu.available()
False
"""

from ._core import available, check, extension

__all__ = ["available", "check", "extension"]
