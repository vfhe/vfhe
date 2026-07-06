# SPDX-License-Identifier: Apache-2.0
"""Error mapping for the native arith engine.

The C engine reports failures as negative int status codes (``arith/error.h``).
This module mirrors those codes as an exception hierarchy and provides
:func:`check`, the single place where codes become exceptions.
"""

from __future__ import annotations


class VfheError(Exception):
    """Base class for all native-engine errors."""


class DomainError(VfheError):
    """An operand was in the wrong representation domain (COEFF vs EVAL)."""


class NotInvertibleError(VfheError):
    """An element has no inverse (a zero evaluation slot was encountered)."""


class UnsupportedError(VfheError):
    """The operation is not supported for this ring shape (e.g. split_degree > 1)."""


class ArgumentError(VfheError):
    """Invalid argument (bad mask, size, or aliasing)."""


#: Status-code -> exception class, mirroring ``vfhe_status`` in arith/error.h.
_CODE_TO_ERROR: dict[int, type[VfheError]] = {
    -1: DomainError,
    -2: NotInvertibleError,
    -3: UnsupportedError,
    -4: ArgumentError,
}


def check(rc: int) -> None:
    """Raise the exception matching a native status code (no-op on success).

    Args:
        rc: status returned by a native call; 0 means success.

    Raises:
        VfheError: the subclass matching ``rc``.
    """
    if rc == 0:
        return
    raise _CODE_TO_ERROR.get(rc, VfheError)(f"native call failed with status {rc}")
