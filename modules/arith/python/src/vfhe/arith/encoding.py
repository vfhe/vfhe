# SPDX-License-Identifier: Apache-2.0
"""Complex-FFT encoding layer (CKKS-style slot packing).

A :class:`ComplexRing` precomputes the forward/inverse FFT twiddle factors
for a given slot count (including the non-standard "special" CKKS roots,
generated in high precision Python-side); a :class:`ComplexPolynomial` holds
the slots as an interleaved real/imaginary ``double`` array and transforms
between slot and coefficient domains, ready to be rounded into an RNS
:class:`~vfhe.arith.ring.Ring` element.
"""

from __future__ import annotations

from math import log2

from _vfhe_native import ffi, lib

from .errors import check
from .polynomial import Polynomial
from .ring import Ring


class ComplexRing:
    """Precomputed FFT twiddle factors for length-``N`` CKKS encoding.

    Args:
        N: number of complex slots.
        special_rous: generate the CKKS "special" roots of unity (high
            precision, bit-reversed 5^k ordering) and load both twiddle
            directions into the native engine.
    """

    def __init__(self, N, special_rous=True):
        self.lib = lib
        self.N = N
        self.logN = int(log2(N))
        self.N2 = 2 * N

        if special_rous:
            rous = self.gen_special_rous_hp(self.N2)
            self.twiddles_fwd = lib.cfft_load_twiddles_fwd(
                ffi.new("double[]", [i.real for i in rous]),
                ffi.new("double[]", [i.imag for i in rous]),
                self.N2,
            )
            inv_rous = [i**-1 for i in rous]
            self.twiddles_inv = lib.cfft_load_twiddles_inv(
                ffi.new("double[]", [i.real for i in inv_rous]),
                ffi.new("double[]", [i.imag for i in inv_rous]),
                self.N2,
            )

    def alloc_polynomial(self):
        """Allocate a zeroed slot buffer (2N doubles, split real/imag)."""
        return ffi.new("double[]", self.N2)

    def exp_complex_polys_ifft_scale_round_to_RNS_batch(
        self,
        e_polys: list[ComplexPolynomial],
        ring: Ring,
        temp_delta: float,
        n_threads: int = 0,
    ) -> list[Polynomial]:
        """Batch-encode: per row, IFFT, scale by ``temp_delta``, round to RNS.

        Runs the whole pipeline in native threads (``n_threads=0`` selects
        the engine default). The input rows are consumed in place.
        """
        n = len(e_polys)
        if n == 0:
            return []
        rows = ffi.new("void*[]", n)
        outs = ffi.new("void*[]", n)
        res = [Polynomial(ring) for _ in range(n)]
        for i in range(n):
            rows[i] = ffi.cast("void*", e_polys[i].obj)
            outs[i] = ffi.cast("void*", res[i].obj)
        lib.cfft_ifft_scale_round_batch(
            rows, outs, n, self.N, self.logN, self.twiddles_inv, temp_delta, n_threads
        )
        return res

    @staticmethod
    def gen_special_rous(rou, N):
        """CKKS special root schedule from a given primitive root (float)."""
        brev = lambda x, size: int(bin(x)[2:].rjust(size, "0")[::-1], 2)
        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                result[2 ** (k - 1) + i] = rou ** (
                    (5 ** brev(i, k - 1)) * N // (2 ** (k + 1))
                )
        return result

    @staticmethod
    def gen_special_rous_hp(N):
        """CKKS special root schedule at 100-bit precision (via mpmath)."""
        N = int(N)
        import mpmath as mp

        mp.mp.prec = 100  # type: ignore[assignment]  # settable at runtime; stub marks it read-only
        rou = mp.exp(2 * mp.pi * 1j / (2 * mp.mpf(N)))
        brev = lambda x, size: int(bin(x)[2:].rjust(size, "0")[::-1], 2)
        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                expo = ((5 ** brev(i, k - 1)) * N // (2 ** (k + 1))) % (2 * N)
                result[2 ** (k - 1) + i] = mp.power(rou, expo)
        return [complex(i) for i in result]


class ComplexPolynomial:
    """``N`` complex slots stored as an interleaved real/imag ``double`` array.

    :meth:`FFT` maps the coefficient domain to slots and :meth:`IFFT` back;
    :meth:`round_to_RNS` quantizes the current coefficients into a ring
    element.
    """

    def __init__(self, ring: ComplexRing):
        self.ring = ring
        self.obj = ring.alloc_polynomial()

    def __iter__(self):
        N = self.ring.N
        return iter([self.obj[i] + self.obj[i + N] * 1j for i in range(N)])

    def IFFT(self) -> None:
        """Inverse FFT in place (bit-reverse, transform, 1/N scale)."""
        lib.cfft_bit_reverse(self.obj, self.ring.N, self.ring.logN)
        lib.cfft_inverse(self.obj, self.ring.twiddles_inv, self.ring.N)
        lib.cfft_scale(self.obj, 1.0 / self.ring.N, self.ring.N)

    def FFT(self) -> None:
        """Forward FFT in place (transform, then bit-reverse to natural order)."""
        lib.cfft_forward(self.obj, self.ring.twiddles_fwd, self.ring.N)
        lib.cfft_bit_reverse(self.obj, self.ring.N, self.ring.logN)

    def __imul__(self, other):
        if type(other) is int or type(other) is float:
            lib.cfft_scale(self.obj, float(other), self.ring.N)
            return self
        raise TypeError(f"cannot scale ComplexPolynomial by {type(other)}")

    def __setitem__(self, idx, val: float | complex):
        if type(val) is float or type(val) is int:
            self.obj[idx] = val
            self.obj[idx + self.ring.N] = 0.0
        elif type(val) is complex:
            self.obj[idx] = float(val.real)
            self.obj[idx + self.ring.N] = float(val.imag)
        else:
            raise TypeError(f"cannot assign {type(val)} into ComplexPolynomial")

    def from_array(self, v: list[float | complex]) -> ComplexPolynomial:
        """Load slot values (real or complex) starting at slot 0."""
        for i in range(len(v)):
            self[i] = v[i]
        return self

    def round_to_RNS(self, ring: Ring) -> Polynomial:
        """Round each coordinate in Python and load into a ring element."""
        array = [round(self.obj[i]) for i in range(self.ring.N2)]
        return Polynomial(ring).from_array(array)

    def round_to_RNS_native(self, ring: Ring) -> Polynomial:
        """Same as :meth:`round_to_RNS` with the rounding + load in one C call."""
        res = Polynomial(ring)
        check(lib.cfft_round_to_poly(res.obj, self.obj))
        return res
