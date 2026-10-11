# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
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
from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Sequence

_NOT_FINITE = "a value to round is NaN or infinite (a non-finite input or scale)"


class ComplexRing:
    #: The (implementation, backend) this parent was built for.
    spec: ClassVar[Spec]

    def __init__(self, N: int) -> None:
        self.lib = lib
        self.N = N
        self.logN = int(log2(N))
        self.N2 = 2 * N
        #: The negacyclic FFT of length 2N, and where each slot lands in its
        #: output (see ``complex_slot_positions`` in arith.h).
        self.plan = ffi.gc(self.lib.fft_new_plan(self.N2), self.lib.fft_free_plan)
        self.slot_positions = ffi.new("uint32_t[]", N)
        self.lib.complex_slot_positions(self.slot_positions, N)

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
        finite = self.lib.complex_polys_ifft_scale_round_to_RNS_batch(
            rows,
            outs,
            n,
            self.plan,
            self.slot_positions,
            temp_delta,
        )
        if not finite:
            raise ValueError(_NOT_FINITE)
        for p in res:
            p.repr = repr.ntt
        return res

    def values_ifft_scale_round_to_RNS_batch(
        self,
        values: Sequence[object],
        ring: RNSRing,
        scale: float,
        shifts: Sequence[int] | None = None,
        n_threads: int = 0,
    ) -> list[RNSPolynomial]:
        """For each vector in ``values``: rotated left by ``shifts[k]`` (default
        0), repeated to ``N`` values, inverse transformed, scaled by ``scale``
        and rounded into ``ring``, in the NTT domain.

        Each vector's length must be a power of two dividing ``N``. A contiguous
        one-dimensional buffer of complex doubles (format ``Zd``, e.g. a numpy
        ``complex128`` array) is read in place; anything else is converted, and
        may hold any ``numbers.Complex``.
        """
        count = len(values)
        if shifts is not None and len(shifts) != count:
            raise ValueError(f"Expected {count} shifts, got {len(shifts)}")
        if count == 0:
            return []
        rows, lengths = zip(*(_interleaved_values(v) for v in values), strict=True)
        for n in lengths:
            if n < 1 or n & (n - 1) or self.N % n:
                raise ValueError(
                    f"Expected a number of values dividing {self.N}, got {n}"
                )
        res = [Polynomial(ring) for _ in range(count)]
        finite = self.lib.complex_values_ifft_scale_round_to_RNS_batch(
            ffi.new(
                "RNS_Polynomial[]", [ffi.cast("RNS_Polynomial", p.obj) for p in res]
            ),
            ffi.new("double *[]", [ffi.cast("double *", row) for row in rows]),
            ffi.new("uint64_t[]", lengths),
            ffi.new(
                "uint64_t[]",
                [0] * count
                if shifts is None
                else [k % n for k, n in zip(shifts, lengths, strict=True)],
            ),
            count,
            self.plan,
            self.slot_positions,
            scale,
            n_threads,
        )
        if not finite:
            raise ValueError(_NOT_FINITE)
        for p in res:
            p.repr = repr.ntt
        return res


_BUILTIN_NUMBERS = frozenset((complex, float, int))


def _interleaved_values(v: object) -> tuple[ffi.CData, int]:
    """``v``'s values as an array of complex doubles, and how many there are.
    A contiguous buffer of complex doubles is shared, not copied."""
    try:
        view = memoryview(v)  # type: ignore[arg-type]
    except TypeError:
        view = None
    if view is not None:
        if view.format == "Zd" and view.ndim == 1 and view.c_contiguous:
            return ffi.from_buffer("double _Complex[]", view), len(view)
        try:
            v = view.tolist()
        except NotImplementedError:  # a format memoryview cannot unpack
            v = list(v)  # type: ignore[call-overload]
    values = v if isinstance(v, list | tuple) else list(v)  # type: ignore[call-overload]
    try:
        return ffi.new("double _Complex[]", values), len(values)
    except TypeError:
        pass
    for val in values:
        if type(val) not in _BUILTIN_NUMBERS and not isinstance(val, numbers.Complex):
            raise NotImplementedError(f"cannot assign a {type(val).__name__}")
    return ffi.new("double _Complex[]", [complex(x) for x in values]), len(values)


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
        """Slot form to coefficient form, in place."""
        self.ring.lib.complex_poly_ifft(
            self.obj, self.ring.plan, self.ring.slot_positions
        )

    def FFT(self):
        """Coefficient form to slot form, in place."""
        self.ring.lib.complex_poly_fft(
            self.obj, self.ring.plan, self.ring.slot_positions
        )

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
        if not self.ring.lib.complex_poly_round_to_RNS(res.obj, self.obj, ring.N):
            raise ValueError(_NOT_FINITE)
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
