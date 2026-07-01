# SPDX-License-Identifier: Apache-2.0
from _vfhe_native import lib


def mul64(a: int, b: int) -> int:
    """Low 64 bits of ``a * b`` (hand-written assembly kernel)."""
    return lib.asm_mul64(a, b)
