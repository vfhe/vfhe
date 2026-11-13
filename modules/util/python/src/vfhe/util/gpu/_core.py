# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The loading of the CUDA extension."""

from __future__ import annotations

import functools
import importlib
import importlib.util
from typing import Any

MODULE = "_vfhe_cuda"


@functools.cache
def _extension() -> tuple[Any, Any] | None:
    """Returns the extension's ``ffi`` and ``lib``; None when the install has no extension."""
    if importlib.util.find_spec(MODULE) is None:
        return None
    module = importlib.import_module(MODULE)
    return module.ffi, module.lib


def _device_count() -> int:
    """Returns how many CUDA devices this machine can run kernels on.

    0 when the install has no extension, the machine no driver, or the driver
    no device.
    """
    extension = _extension()
    return 0 if extension is None else extension[1].vfhe_gpu_device_count()


def available() -> bool:
    """Reports whether a kernel can run here."""
    return _device_count() > 0


def extension() -> tuple[Any, Any]:
    """Returns the extension's ``ffi`` and ``lib``, for a module's own kernels.

    Raises `RuntimeError` when no device is available.
    """
    loaded = _extension()
    if loaded is None or not available():
        raise RuntimeError("no CUDA device is available")
    return loaded


def check(error: int) -> None:
    """Raises `RuntimeError` with the runtime's description when a kernel returned `error`."""
    if error:
        ffi, lib = extension()
        raise RuntimeError(ffi.string(lib.vfhe_gpu_error_string(error)).decode())
