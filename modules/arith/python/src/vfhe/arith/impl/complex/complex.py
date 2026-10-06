# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numbers
from math import log2
from typing import TYPE_CHECKING, ClassVar

# The AVX-512 complex FFT casts these buffers to __m512d and uses aligned loads,
# so they must be 64-byte aligned; more than cffi's default. Shared with the other
# over-aligned wrappers rather than kept private here.
from vfhe.arith._alloc import aligned64 as _aligned64
from vfhe.arith.impl.rns.polynomial import (
    Polynomial,
    RNSPolynomial,
    RNSRing,
    repr,
)
from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Sequence


class ComplexRing:
    #: The (implementation, backend) this parent was built for.
    spec: ClassVar[Spec]

    def __init__(self, N: int, special_rous: bool = True) -> None:
        self.lib = lib
        self.N = N
        self.logN = int(log2(N))
        self.N2 = 2 * N

        if special_rous:
            rous = self.gen_special_rous_hp(self.N2)
            rous_real = [i.real for i in rous]
            rous_imag = [i.imag for i in rous]
            self.CT_rous = self.lib.load_rous_CT(
                ffi.new("double[]", rous_real), ffi.new("double[]", rous_imag), self.N2
            )
            # inv rous
            inv_rous = [i**-1 for i in rous]
            rous_inv_real = [i.real for i in inv_rous]
            rous_inv_imag = [i.imag for i in inv_rous]
            self.GS_rous = self.lib.load_rous_GS(
                ffi.new("double[]", rous_inv_real),
                ffi.new("double[]", rous_inv_imag),
                self.N2,
            )

    def alloc_polynomial(self):
        return _aligned64("double[]", self.N2)

    def exp_complex_polys_ifft_scale_round_to_RNS_batch(
        self,
        e_polys: list[ComplexPolynomial],
        ring: RNSRing,
        temp_delta: float,
    ) -> list[RNSPolynomial]:
        """Batch: for each exp-domain ComplexPolynomial, IFFT, scale by temp_delta, round to RNS/NTT."""
        n = len(e_polys)
        if n == 0:
            return []
        res = [Polynomial(ring) for _ in range(n)]
        rows = ffi.new(
            "void*[]", [ffi.cast("void *", e_polys[i].obj) for i in range(n)]
        )
        outs = ffi.new("void*[]", [ffi.cast("void *", res[i].obj) for i in range(n)])
        self.lib.complex_polys_ifft_scale_round_to_RNS_batch(
            rows,
            outs,
            n,
            self.N,
            self.logN,
            self.GS_rous,
            temp_delta,
        )
        for p in res:
            p.repr = repr.ntt
        return res

    @staticmethod
    # special RoUs for CKKS
    def gen_special_rous(rou, N):
        def brev(x, size):
            return int(bin(x)[2:].rjust(size, "0")[::-1], 2)

        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                result[2 ** (k - 1) + i] = rou ** (
                    (5 ** brev(i, k - 1)) * N // (2 ** (k + 1))
                )
        return result

    @staticmethod
    def gen_special_rous_hp(N):
        N = int(N)
        import mpmath as mp

        mp.mp.prec = 100  # type: ignore
        rou = mp.exp(2 * mp.pi * 1j / (2 * mp.mpf(N)))

        def brev(x, size):
            return int(bin(x)[2:].rjust(size, "0")[::-1], 2)

        result = [1] * N
        for k in range(int(log2(N)) - 1, 0, -1):
            for i in range(2 ** (k - 1)):
                expo = ((5 ** brev(i, k - 1)) * N // (2 ** (k + 1))) % (2 * N)
                result[2 ** (k - 1) + i] = mp.power(rou, expo)
        return [complex(i) for i in result]


_BUILTIN_NUMBERS = frozenset((complex, float, int))


class ComplexPolynomial:
    def __init__(self, ring: ComplexRing):
        self.ring = ring
        self.obj = ring.alloc_polynomial()

    def __iter__(self):
        array = [
            self.obj[i] + self.obj[i + self.ring.N] * 1j for i in range(self.ring.N)
        ]
        return iter(array)

    def IFFT(self):
        self.ring.lib.bit_reverse_array(self.obj, self.ring.N, self.ring.logN)
        self.ring.lib.GS_RN(self.obj, self.ring.GS_rous, self.ring.N)
        self.ring.lib.complex_poly_scale_double(
            self.obj, 1.0 / self.ring.N, self.ring.N
        )

    def FFT(self):
        self.ring.lib.CT_NR(self.obj, self.ring.CT_rous, self.ring.N)
        self.ring.lib.bit_reverse_array(self.obj, self.ring.N, self.ring.logN)

    def __imul__(self, other):
        if isinstance(other, numbers.Real):
            self.ring.lib.complex_poly_scale_double(self.obj, float(other), self.ring.N)
            return self
        else:
            raise NotImplementedError(f"cannot scale by {type(other).__name__}")

    def __setitem__(self, idx, val: numbers.Complex | complex):
        if not isinstance(val, numbers.Complex):
            raise NotImplementedError(f"cannot assign a {type(val).__name__}")
        val = complex(val)
        self.obj[idx] = val.real
        self.obj[idx + self.ring.N] = val.imag

    def from_array(self, v: Sequence[numbers.Complex | complex]) -> ComplexPolynomial:
        """Sets the first ``len(v)`` values; any ``numbers.Complex`` is accepted."""
        if len(v) > self.ring.N:
            raise ValueError(f"Expected at most {self.ring.N} values, got {len(v)}")
        # Builtin types first: the ABC check is much slower.
        for val in v:
            if type(val) not in _BUILTIN_NUMBERS and not isinstance(
                val, numbers.Complex
            ):
                raise NotImplementedError(f"cannot assign a {type(val).__name__}")
        values = [complex(val) for val in v]
        n = len(values)
        self.obj[0:n] = [val.real for val in values]
        self.obj[self.ring.N : self.ring.N + n] = [val.imag for val in values]
        return self

    def round_to_RNS(self, ring: RNSRing) -> RNSPolynomial:
        array = [round(self.obj[i]) for i in range(self.ring.N2)]
        return Polynomial(ring).from_array(array)

    # Duplicate of round_to_RNS with no Python per-coefficient overhead (C rounding + int_array_to_RNS).
    def round_to_RNS_cpp(self, ring: RNSRing) -> RNSPolynomial:
        res = Polynomial(ring)
        self.ring.lib.complex_poly_round_to_RNS(res.obj, self.obj, ring.N)
        res.repr = repr.ntt
        return res


#: C[X]/(X^N + 1) held as double-precision coefficient pairs, with the FFT as
#: its multiplication transform. Floating point is approximate, so it carries
#: no EXACT flag, and a ring over C has no modulus tower.
COMPLEX_FFT = register(
    Spec(
        implementation="complex",
        backend="fft",
        parent_cls=ComplexRing,
        element_cls=ComplexPolynomial,
        capabilities=Capability.CORE | Capability.SAMPLING,
        constraints=Constraints(),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
ComplexRing.spec = COMPLEX_FFT
