# SPDX-License-Identifier: Apache-2.0
"""Multi-precision (big-integer) bridge for RNS polynomials.

:class:`Multiprecision` builds the CRT constants for a prime set and lifts an
RNS polynomial into base-2^52-digit big integers (native ``mp_polynomial_t``
and ``mp_scalar_t`` handles), used where exact integer arithmetic beyond a
single prime is needed. The module functions reconstruct Python ints from
those handles.

The native digit type differs between portable and AVX-512 builds, so scalar
digits are always read through the engine's accessors (``mp_scalar_digit``);
only ``mp_polynomial`` rows (always plain ``uint64_t``) are read directly.
"""

from __future__ import annotations

from math import ceil, log2, prod

from _vfhe_native import ffi, lib

from .errors import check


def mp_polynomial_to_list(mp_poly) -> list[int]:
    """Reconstruct integer coefficients from an ``mp_polynomial_t`` handle."""
    d, N = mp_poly.d, mp_poly.N
    coeffs = mp_poly.coeffs
    return [
        sum(int(coeffs[i][j]) * (1 << (52 * i)) for i in range(d)) for j in range(N)
    ]


def mp_scalar_to_int(mp_scalar) -> int:
    """Reconstruct the integer held by an ``mp_scalar_t`` handle."""
    d = lib.mp_scalar_digit_count(mp_scalar)
    return sum(
        int(lib.mp_scalar_digit(mp_scalar, i)) * (1 << (52 * i)) for i in range(d)
    )


class Multiprecision:
    """Builds CRT constants and lifts RNS polynomials to big-integer form."""

    def __init__(self) -> None:
        self.lib = lib
        self.vector_size = lib.mp_vector_size()

    def compute_crt_consts(self, primes) -> dict:
        """CRT reconstruction constants for a prime list.

        Returns a dict with the interpolation scalars ``pw`` (one per prime),
        the composite modulus ``q``, and the Barrett pair ``(m, k)`` for it --
        exactly the arguments ``mp_polynomial_from_poly`` expects.
        """
        ell = len(primes)
        ql = prod(primes)
        q = self.load(ql)
        prime_size = max(map(log2, primes))
        k = ceil(prime_size * (ell + 1))

        QhatQ = []
        for i in range(ell):
            Qi = prod([primes[j] for j in range(ell) if i != j])
            hatQi = pow(Qi, -1, primes[i])
            QhatQ.append((Qi * hatQi) % ql)
        pw = ffi.new("mp_scalar_t[]", [self.load(int(x)) for x in QhatQ])

        m_val = 2**k // ql
        assert m_val < 2**52
        m = self.load_small(m_val)
        return {"pw": pw, "q": q, "m": m, "k": k}

    def from_polynomial(self, poly, crt_consts):
        """Lift a :class:`~vfhe.arith.polynomial.Polynomial` to big integers."""
        poly.to_coeff()
        res = lib.mp_polynomial_new(poly.ring.N, poly.ring.ell + 1)
        check(
            lib.mp_polynomial_from_poly(
                res,
                poly.obj,
                crt_consts["pw"],
                crt_consts["q"],
                crt_consts["m"],
                crt_consts["k"],
            )
        )
        return res

    def load(self, x: list | int):
        """Load an int (or its base-2^52 digit list) as an ``mp_scalar_t``."""
        if isinstance(x, int):
            x = [(x >> (52 * i)) & ((1 << 52) - 1) for i in range(ceil(log2(x) / 52))]
        return lib.mp_scalar_load(ffi.new("uint64_t[]", x), len(x))

    def load_small(self, x: int):
        """Load a single machine word as a broadcast digit column."""
        return lib.mp_vector_load(x)
