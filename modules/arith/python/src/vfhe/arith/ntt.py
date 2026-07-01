# SPDX-License-Identifier: Apache-2.0
from array import array
from collections.abc import Iterable

from _vfhe_native import ffi, lib  # cffi-built native library (out-of-line API mode)


def ntt_forward_32(coeffs: Iterable[int], modulus: int) -> array:
    """Forward 32-bit NTT of ``coeffs`` (a sequence of uint64) modulo ``modulus``.

    Crosses into C once for the whole array (no per-element marshalling).
    """
    buf = array("Q", coeffs)  # contiguous uint64
    n = len(buf)
    out = array("Q", bytes(8 * n))  # zero-initialised output of the same length
    if n == 0:
        return out
    proc = ffi.new("NTT_proc *")
    proc.n = n
    proc.modulus = modulus
    in_ptr = ffi.cast("uint64_t *", ffi.from_buffer(buf))
    out_ptr = ffi.cast("uint64_t *", ffi.from_buffer(out, require_writable=True))
    lib.ntt_forward_32(out_ptr, in_ptr, proc[0])  # struct passed by value
    return out
