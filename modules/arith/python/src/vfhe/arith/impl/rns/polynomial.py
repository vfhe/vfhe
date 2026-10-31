# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import array
import contextlib
import dataclasses
import itertools
import math
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING, Any

from vfhe.arith.base import FieldVector, Polynomial, Ring, check_spec_keywords
from vfhe.arith.impl.field.field import ExtensionField
from vfhe.arith.impl.field.vector import ExtensionFieldVector
from vfhe.arith.impl.mp.multiprecision import Multiprecision
from vfhe.arith.number_theory import crt, is_prime
from vfhe.arith.registry import register
from vfhe.arith.spec import Capability, Constraints, Spec
from vfhe.util.bindings import ffi, lib

from .rns_base import registry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self


#: The fields `RNSRing.residue_field` returns, one per (N, split_degree, prime).
_RESIDUE_FIELDS: dict[tuple[int, int, int], ExtensionField] = {}
#: The native tables of the residue-field isomorphisms (`rns_crt_maps_new`),
#: one per (base address, base row, direction). Bases live as long as the
#: process, so an address keeps naming the same base.
_CRT_MAPS: dict[tuple[int, int, bool], Any] = {}
#: Per (base address, prime mask): the inverse `_CRT_MAPS` of every row of the
#: mask, in an array indexed by base row, as `sample_exceptional` hands them
#: to the kernel.
_INVERSE_CRT_MAPS_BY_ROW: dict[tuple[int, int], Any] = {}


def next_power_of_2(x):
    return 1 << math.ceil(math.log2(x))


#: `array.array` type code per row width, resolved by size because the named C
#: widths are platform-dependent. Packing a row through one of these and
#: copying it in beats writing a cdata pointer element by element.
_ROW_CODE = {
    width: next(c for c in "ILQ" if array.array(c).itemsize == width)
    for width in (4, 8)
}


def _row(p, idx: int):
    """Row ``idx`` of a native RNS polynomial, at the width its prime uses.

    A prime under the native narrow bound stores its coefficients as 32-bit
    words, and ``RNS_Base.narrow_mask`` is the only authority on which. Both
    arrays are typed, so picking the wrong one is a type error on the C side
    and a wrong value here; go through this rather than indexing directly.
    """
    if (p.base.narrow_mask >> idx) & 1:
        return p.rows32[idx]
    return p.rows64[idx]


def _to_natural_order(row: list, d: int) -> list:
    """A row of a ring with split degree `d`, from the native layout to
    natural order: entry ``u + d t`` of the result is entry ``u * (N/d) + t``
    of `row`, for ``N = len(row)``. A transpose, done by slices along the
    shorter side."""
    n = len(row)
    k = n // d
    out = [0] * n
    if d <= k:
        for u in range(d):
            out[u::d] = row[u * k : (u + 1) * k]
    else:
        for t in range(k):
            out[t * d : (t + 1) * d] = row[t::k]
    return out


def _from_natural_order(values: list, d: int) -> list:
    """The inverse of `_to_natural_order`."""
    n = len(values)
    k = n // d
    out = [0] * n
    if d <= k:
        for u in range(d):
            out[u * k : (u + 1) * k] = values[u::d]
    else:
        for t in range(k):
            out[t::k] = values[t * d : (t + 1) * d]
    return out


@dataclasses.dataclass(frozen=True)
class RingParameters:
    """What makes two `RNSRing` objects the same ring: degree, split degree and
    the primes, in the ring's order. Stable across processes, where `RNSRing.mask`
    is not."""

    N: int
    split_degree: int
    primes: tuple[int, ...]


class RNSRing(Ring):
    """Z_q[X]/(X^N + 1) with q an RNS product, over the incomplete NTT.

    The modulus is a set of word-sized primes and an element is one row of
    residues per prime, so arithmetic is elementwise per row and a level of a
    modulus tower is a subset of the primes -- which is what `mask` names and
    what makes `quotient_ring` and `round_division` cheap.

    Primes are drawn from a process-wide base shared by every ring of the same
    ``(N, split_degree)``, and prime *indices* are global to it: two rings
    agree on the index of a prime they share, which is what lets a per-prime
    native array built for one ring be indexed by another. The base grows when
    a ring introduces a prime the process has not seen, so size such an array
    with `rns_rows`, never with the base's current length.

    `Ring` documents the parameters, and ``Ring(...)`` builds this class.
    """

    #: Number of primes in this ring's modulus. Not a level index -- see the
    #: `lvl` versus `ell` split in the scheme classes.
    ell: int
    #: The primes, in the shared base's order.
    primes: list[int]
    #: Each prime's index in the shared base, in the same order as `primes`.
    prime_indices: list[int]
    #: Bitmask over the shared base selecting this ring's primes: how one ring
    #: is matched to another across a scheme, within one process.
    mask: int
    #: Per-prime widths in bits, in the order of `primes`.
    prime_size: list[int]
    #: The smallest of `prime_size`: what bounds an integer scalar operand.
    smallest_prime: int
    #: Split degree of the incomplete transform: coefficients per residue slot.
    split_degree: int
    #: The product of `primes`, as a Python int.
    q_l: int
    #: Per-`k` constants for `RNSPolynomial.rescale_to_power_of_two`, built on
    #: first use and held for the ring's lifetime.
    _pow2_consts: dict[int, tuple]
    #: The handle `arith_ring` returns, looked up on first use: the native
    #: rings are shared and never freed, and N, mask and base do not change.
    _arith_ring: Any = None

    def __init__(
        self,
        N: int,
        mod_size: int | None = None,
        split_degree: int | None = None,
        primes: list[int] | None = None,
        mask: int | None = None,
        prime_size: int | list[int] = 49,
        exceptional_set_size: int = 128,
        *,
        implementation: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Build the ring; see `Ring` for what each parameter means.

        ``implementation`` and ``backend`` are the keywords `Ring` uses to
        pick a class. Reaching here through `Ring` they are already consumed;
        given directly they are checked, so that naming a spec this class does
        not serve is an error rather than an ignored argument.
        """
        check_spec_keywords(RNSRing, implementation, backend)
        if not (
            (mod_size is not None)
            or (type(prime_size) is list)
            or (primes is not None)
            or (mask is not None)
        ):
            raise ValueError("must provide mod_size, prime_size, primes, or mask")

        if mask is not None:
            if primes is None:
                raise ValueError("must provide primes when mask is given")
            temp_split_degree = split_degree
            if not temp_split_degree:
                smallest_in_pool = min(math.ceil(math.log2(p)) for p in primes)
                temp_split_degree = next_power_of_2(
                    exceptional_set_size / smallest_in_pool
                )

            key = (N, temp_split_degree)
            prime_map = registry().prime_to_index[key]
            pool_mask = sum(1 << prime_map[p] for p in primes)
            # ``mask`` indexes the shared base, so it can name primes outside
            # ``primes``; those would be dropped without a trace, leaving a
            # smaller ring than was asked for.
            if mask & ~pool_mask:
                missing = mask & ~pool_mask
                raise ValueError(
                    "mask selects base indices "
                    f"{[i for i in range(missing.bit_length()) if (missing >> i) & 1]}"
                    " that are not among the given primes; pass a pool that "
                    "covers them (RNSRing.union builds the ring over two "
                    "rings' primes)"
                )
            active_primes = [p for p in primes if ((mask >> prime_map[p]) & 1)]
            prime_size = [math.ceil(math.log2(p)) for p in active_primes]
            primes = active_primes

        # A pool given on its own fixes the sizes; with mod_size it is drawn
        # from by size, as without it.
        if primes is not None and (
            prime_size is None
            or (mod_size is None and not isinstance(prime_size, list))
        ):
            prime_size = [math.ceil(math.log2(i)) for i in primes]

        if isinstance(prime_size, list):
            self.ell = len(prime_size)
            self.prime_size = prime_size
            self.smallest_prime = min(prime_size)
        else:
            self.ell = math.ceil(mod_size / prime_size)  # type: ignore
            self.prime_size = [prime_size] * self.ell
            self.smallest_prime = prime_size

        if not split_degree:
            split_degree = next_power_of_2(exceptional_set_size / self.smallest_prime)  # type: ignore
        self.split_degree = split_degree
        self.N = N
        self.byte_size = self.N * self.ell * 8
        self.lib = lib

        if primes:
            if len(primes) < self.ell:
                raise ValueError(
                    f"not enough primes for quotient ring for size 2^{mod_size}"
                )
            self.primes = primes[: self.ell]
        else:
            self.primes = self.gen_primes()

        if self.split_degree > 1:
            # See gen_prime: with an even multiplier the residues are not
            # fields of size p^split_degree.
            order = 2 * self.N // self.split_degree
            even = [
                p
                for p in self.primes
                if (p - 1) % order == 0 and (p - 1) // order % 2 == 0
            ]
            if even:
                raise ValueError(
                    f"primes {even} are 1 mod {2 * order}: with split_degree "
                    f"{self.split_degree} a prime must be p = a * {order} + 1 "
                    "with a odd, or X^N + 1 does not split into irreducible "
                    f"factors of degree {self.split_degree}"
                )

        self.q_l = math.prod(self.primes)

        self.bit_size = math.ceil(math.log2(self.q_l))

        self.prime_indices = registry().register_ring_primes(
            self.primes, self.N, self.split_degree
        )
        self.base = registry().bases[(self.N, self.split_degree)]
        self.mask = sum(1 << idx for idx in self.prime_indices)
        self._pow2_consts: dict[int, tuple] = {}

    @classmethod
    def from_native(cls, N: int, mask: int, base: Any) -> RNSRing:
        """The ring over the shared base ``base`` whose primes ``mask``
        selects, for a value that came back from C with those parameters.
        """
        address = int(ffi.cast("uintptr_t", base))
        for (n, split_degree), known in registry().bases.items():
            if n == N and int(ffi.cast("uintptr_t", known)) == address:
                primes = registry().primes[(n, split_degree)]
                return Ring(N, mask=mask, split_degree=split_degree, primes=primes)
        raise ValueError("the base is not one this process built")

    @property
    def arith_ring(self):
        """This ring's handle in the generic C interface.

        Shared and never freed, so it is safe to hold: a structure built over
        a ring may point at it, and nothing can prove the last one is gone.
        """
        handle = self._arith_ring
        if handle is None:
            handle = self._arith_ring = self.lib.arith_rns_ring_get(
                self.N, self.mask, self.base
            )
        return handle

    def _pow2_rescale_consts(self, k: int) -> tuple:
        """Everything `rescale_to_power_of_two` needs for this ``k``.

        The reconstruction constants with 2^k folded into their per-prime
        factor, so the scaling costs nothing beyond the multiply the CRT
        already does; (q+1)/2 as native digits; and -q^-1 mod 2^k.
        """
        if k not in self._pow2_consts:
            mp = Multiprecision()
            consts = dict(mp.compute_crt_consts(self.primes))
            consts["hat_q"] = [
                h * pow(2, k, p) % p
                for h, p in zip(consts["hat_q"], self.primes, strict=True)
            ]
            half = ffi.new("uint64_t[]", mp.limbs((self.q_l + 1) // 2, consts["d"]))
            q_inv_neg = -pow(self.q_l, -1, 1 << k) % (1 << k)
            self._pow2_consts[k] = (mp, consts, half, q_inv_neg)
        return self._pow2_consts[k]

    @property
    def exceptional_set_size(self) -> int:
        """|A| for this ring: the residue fields have size p_i^split_degree,
        and a difference must be invertible in every one of them, so the
        smallest bounds the set."""
        return min(self.primes) ** self.split_degree

    @property
    def rns_rows(self) -> int:
        """Rows a native per-prime array must have to be indexed by this ring.

        One past this ring's highest prime index, derived from ``mask``. Prime
        indices are handed out append-only, so this is fixed for the ring's
        lifetime and safe to store alongside an allocation sized with it. The
        kernels only touch rows the mask selects, so an array this long is
        enough even though the shared base may hold more primes.
        """
        return self.mask.bit_length()

    def _base_l(self) -> int:
        """The shared RNS base's *current* prime count.

        A live read: it grows whenever any ring of the same
        ``(N, split_degree)`` is built with a prime this process has not seen.
        Size native arrays with `rns_rows`, which is stable, rather than with
        this.
        """
        return ffi.cast("RNS_Base", self.base).l

    def get_rou_matrix(self):
        w = self.lib.rns_base_get_rou_matrix(self.base)
        row_len = self.N // self.split_degree
        return [[w[idx][k] for k in range(row_len)] for idx in self.prime_indices]

    def residue_field(self, index: int) -> ExtensionField:
        """A residue field of ``R_q`` at the prime ``p = primes[index]``: the
        field `crt_components` maps into.

        Let ``d = split_degree`` and ``k = N/d``. Modulo ``p``, ``X^N + 1`` is
        the product of ``k`` irreducible factors ``X^d - w_s``, ``s < k``, one
        per slot of the incomplete transform (``w_s`` is
        ``get_rou_matrix()[index][s]``). So the maximal ideals of ``R_q``
        containing ``p`` are ``P_s = (p, X^d - w_s)``, and each residue field
        ``R_q/P_s`` is ``F_p[x]/(x^d - w_s)``, with ``x`` the class of ``X``: a
        field with ``p^d`` elements. This method returns
        ``R_q/P_0 = F_p[x]/(x^d - w_0)``.

        There is one field object per ``(N, split_degree, p)`` in the
        process: fields compare by identity, and the components computed
        through any ring with this prime have to be in the same one.
        """
        prime = self.primes[index]
        key = (self.N, self.split_degree, prime)
        field = _RESIDUE_FIELDS.get(key)
        if field is None:
            rou = self.lib.rns_base_get_rou_matrix(self.base)
            w0 = int(rou[self.prime_indices[index]][0])
            field = ExtensionField(prime, self.split_degree, w0)
            _RESIDUE_FIELDS[key] = field
        return field

    def _crt_maps(self, index: int, to_slot_zero: bool):
        """The native tables (`rns_crt_maps_new`) of the isomorphisms between
        the residue fields at ``primes[index]``: towards `residue_field` if
        `to_slot_zero`, as `crt_components` applies them, and back otherwise.
        Built once per base, row and direction."""
        row = self.prime_indices[index]
        key = (int(ffi.cast("uintptr_t", self.base)), row, to_slot_zero)
        maps = _CRT_MAPS.get(key)
        if maps is None:
            maps = ffi.gc(
                self.lib.rns_crt_maps_new(self.base, row, to_slot_zero),
                self.lib.rns_crt_maps_free,
            )
            _CRT_MAPS[key] = maps
        return maps

    def _inverse_crt_maps_by_row(self):
        """The maps `_crt_maps(index, False)` of every prime, in a native
        array indexed by base row (the other entries unset). Built once per
        base and prime set; the maps it points to live in `_CRT_MAPS`."""
        key = (int(ffi.cast("uintptr_t", self.base)), self.mask)
        handles = _INVERSE_CRT_MAPS_BY_ROW.get(key)
        if handles is None:
            handles = ffi.new("RNS_CrtMaps[]", self.rns_rows)
            for index, row in enumerate(self.prime_indices):
                handles[row] = self._crt_maps(index, False)
            _INVERSE_CRT_MAPS_BY_ROW[key] = handles
        return handles

    def crt_components(
        self,
        elements: Sequence[RNSPolynomial],
        index: int,
        element_major: bool = False,
    ) -> ExtensionFieldVector:
        """The CRT components of `elements` at the prime ``p = primes[index]``:
        each element's residues modulo the maximal ideals of ``R_q`` that
        contain ``p``, all carried into the field `residue_field(index)`.

        Let ``k = N/split_degree``. The maximal ideals of ``R_q`` containing
        ``p`` are ``P_s = (p, X^d - w_s)``, ``s < k``, with ``d``, ``w_s`` and
        the residue fields ``R_q/P_s`` as in `residue_field`, which returns
        ``R_q/P_0``. Every ``w_s`` is an odd power ``w_0^(x_s)`` of ``w_0``,
        and the automorphism ``X -> X^(x_s)`` of ``R_q`` maps ``P_s`` onto
        ``P_0``; component ``s`` of an element is its residue modulo ``P_s``
        carried into ``R_q/P_0`` by the isomorphism this induces. Each
        component is a surjective ring homomorphism ``R_q -> R_q/P_0``, and
        the ``k`` components together induce an isomorphism
        ``R_q/pR_q -> (R_q/P_0)^k``. So the components at every prime
        determine the element; `from_crt_components` is the inverse.

        Entry ``s * len(elements) + b`` of the result is component ``s`` of
        ``elements[b]``: component ``s`` of all the elements is a contiguous
        window. With `element_major`, entry ``b * k + s`` is: the components
        of one element are a contiguous window.

        The elements may belong to any ring with this prime and this ring's
        ``N`` and ``split_degree``. They are read in the mul domain, through a
        converted copy when they are not in it.
        """
        views = self._crt_operands(elements, index)
        out = ExtensionFieldVector(
            self.residue_field(index), len(views) * (self.N // self.split_degree)
        )
        handles = ffi.new(
            "RNS_Polynomial[]", [ffi.cast("RNS_Polynomial", v.obj) for v in views]
        )
        self.lib.polynomial_RNS_crt_components(
            out._struct, handles, len(views), self._crt_maps(index, True), element_major
        )
        return out

    def lift_mod_2_128(
        self, elements: Sequence[RNSPolynomial]
    ) -> tuple[array.array, array.array]:
        """Every coefficient of `elements` as the integer in ``[0, q)`` its
        residues define, modulo ``2^128``, as two ``array('Q')``s: entry
        ``e * N + k`` of ``(low, high)`` is coefficient ``k`` of
        ``elements[e]`` as ``low + 2^64 high``.

        The elements must be of this ring. They are read in the coefficient
        domain, through a converted copy when they are not in it.
        """
        if any(element.ring != self for element in elements):
            raise ValueError("the elements must be of this ring")
        views = [element._viewed_as(repr.coeff) for element in elements]
        size = len(views) * self.N
        low = array.array("Q", bytes(8 * size))
        high = array.array("Q", bytes(8 * size))
        if views:
            handles = ffi.new(
                "RNS_Polynomial[]", [ffi.cast("RNS_Polynomial", v.obj) for v in views]
            )
            self.lib.polynomial_RNS_lift_mod_2_128(
                ffi.from_buffer("uint64_t[]", low),
                ffi.from_buffer("uint64_t[]", high),
                handles,
                len(views),
            )
        return low, high

    def crt_linear_combination(
        self,
        parts: Sequence[Sequence[RNSPolynomial]],
        index: int,
        coefficients: FieldVector,
    ) -> ExtensionFieldVector:
        """A linear combination of CRT components at ``p = primes[index]``,
        in the field `residue_field(index)`: entry b of the result is the sum,
        over c and s, of ``coefficients[c * k + s]`` times component s
        (`crt_components`) of ``parts[c][b]``, with ``k = N/split_degree``.
        Every ``parts[c]`` holds the same number of elements.

        It equals multiplying the components laid out element-major by the
        coefficients and summing each element's k entries over c and s, but
        the components are never formed: each coefficient of an entry is one
        inner product of the elements' mul-domain words with a table built
        from the coefficients (`polynomial_RNS_crt_linear_combination`). This
        is the precomputation of [ACGS24, ePrint 2023/1949, App. C, Eqs. 3-4]
        for a linear combination of slots: each slot is an inner product with
        the element's words, so the combination is one inner product with
        the coefficients' combination of those vectors.
        """
        field = self.residue_field(index)
        k = self.N // self.split_degree
        if coefficients.field is not field or len(coefficients) != len(parts) * k:
            raise ValueError(
                f"the coefficients are {len(parts)} * {k} elements of residue_field({index})"
            )
        count = len(parts[0]) if parts else 0
        if any(len(part) != count for part in parts):
            raise ValueError("the parts hold different numbers of elements")
        # The views own the rows the kernel reads, so they live until it returns.
        views = [self._crt_operands(part, index) for part in parts]
        arrays = [
            ffi.new(
                "RNS_Polynomial[]", [ffi.cast("RNS_Polynomial", v.obj) for v in part]
            )
            for part in views
        ]
        handles = ffi.new("RNS_Polynomial *[]", arrays)
        out = ExtensionFieldVector(field, count)
        self.lib.polynomial_RNS_crt_linear_combination(
            out._struct,
            handles,
            len(parts),
            count,
            coefficients._struct,
            self._crt_maps(index, True),
        )
        return out

    def _crt_operands(
        self, elements: Sequence[RNSPolynomial], index: int
    ) -> list[RNSPolynomial]:
        """`elements` in the mul domain (converted copies where they are
        not), after checking they are over a ring with ``primes[index]`` and
        this ring's ``N`` and ``split_degree``."""
        row = self.prime_indices[index]
        for element in elements:
            ring = element.ring
            if (
                ring.N != self.N
                or ring.split_degree != self.split_degree
                or not (element.rns_mask >> row) & 1
            ):
                raise ValueError(
                    f"an element is not over a ring with prime {self.primes[index]}, "
                    f"N = {self.N} and split_degree = {self.split_degree}"
                )
        return [element._viewed_as(repr.ntt) for element in elements]

    def from_crt_components(
        self, vectors: Sequence[FieldVector]
    ) -> list[RNSPolynomial]:
        """The inverse of `crt_components`: the elements of this ring whose
        components at ``primes[i]`` are ``vectors[i]``, for every ``i``, which
        is the CRT reconstruction. Returned in the mul domain.
        """
        if len(vectors) != self.ell:
            raise ValueError(
                f"expected one vector per prime ({self.ell}), got {len(vectors)}"
            )
        k = self.N // self.split_degree
        count, rest = divmod(len(vectors[0]), k)
        if rest:
            raise ValueError(f"the components of an element are {k} entries")
        out = [Polynomial(self) for _ in range(count)]
        handles = ffi.new(
            "RNS_Polynomial[]", [ffi.cast("RNS_Polynomial", p.obj) for p in out]
        )
        for index, vector in enumerate(vectors):
            if vector.field is not self.residue_field(index):
                raise ValueError(f"vector {index} is not over residue_field({index})")
            if len(vector) != count * k:
                raise ValueError("the vectors hold components of different counts")
            self.lib.polynomial_RNS_from_crt_components(
                handles, vector._struct, count, self._crt_maps(index, False)
            )
        for p in out:
            p.repr = repr.ntt
        return out

    def quotient_ring(self, mod_size=None, ell=None, mask=None):
        if not ((mod_size is not None) ^ (ell is not None) ^ (mask is not None)):
            raise ValueError("must provide mod_size, ell, or mask")
        if mod_size is not None:
            res = Ring(self.N, mod_size, self.split_degree, primes=self.primes)
        elif ell is not None:
            res = Ring(
                self.N,
                prime_size=self.prime_size[:ell],
                split_degree=self.split_degree,
                primes=self.primes,
            )
        else:
            res = Ring(
                self.N, mask=mask, split_degree=self.split_degree, primes=self.primes
            )
        return res

    def is_quotient_ring(self, parent: RNSRing) -> bool:
        return parent.mask & self.mask == self.mask

    def modulus_ratio(self, other_ring: RNSRing, return_pointer: bool = False):
        if not (other_ring.is_quotient_ring(self)):
            raise ValueError("other_ring must be a quotient ring of this ring")

        scaling_mask = self.mask & ~other_ring.mask
        delta_big_int = math.prod(
            [
                self.primes[i]
                for i, idx in enumerate(self.prime_indices)
                if (scaling_mask >> idx) & 1
            ]
        )
        if return_pointer:
            delta_arr = [0] * self.rns_rows
            for k, idx in enumerate(self.prime_indices):
                delta_arr[idx] = delta_big_int % self.primes[k]
            return ffi.new("uint64_t[]", delta_arr)
        return delta_big_int

    @property
    def parameters(self) -> RingParameters:
        """What this ring is, for equality and for matching a stored ring to a live one."""
        return RingParameters(self.N, self.split_degree, tuple(self.primes))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RNSRing) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(self.parameters)

    def intersec(self, other: RNSRing) -> RNSRing:
        if self == other:
            return self
        if self.ell > other.ell:
            if not (other.is_quotient_ring(self)):
                raise ValueError("failed: not (other.is_quotient_ring(self))")
            return other
        if not (self.is_quotient_ring(other)):
            raise ValueError("failed: not (self.is_quotient_ring(other))")
        return self

    def union(self, other: RNSRing) -> RNSRing:
        """The ring over both rings' primes together.

        Neither ring need contain the other, so this is not `intersec`
        reversed: it is the smallest ring both are quotients of. The primes
        come back in the base's own order, which is the order every per-prime
        array is indexed by, so a caller never has to know how primes are
        numbered.
        """
        if self.N != other.N or self.split_degree != other.split_degree:
            raise ValueError("rings must share N and split_degree to be combined")
        order = registry().prime_to_index[(self.N, self.split_degree)]
        primes = sorted(set(self.primes) | set(other.primes), key=lambda p: order[p])
        return type(self)(
            self.N,
            primes=primes,
            prime_size=[math.ceil(math.log2(p)) for p in primes],
            split_degree=self.split_degree,
        )

    # A prime p = a * rou_order + 1 below 2^prime_size, not in exclude_list,
    # rou_order = 2N / split_degree, scanning a downwards.
    #
    # With split_degree d > 1, a must be odd. The point of d > 1 is soundness,
    # not the transform: X^N + 1 must split mod p into the N/d irreducible
    # factors X^d - zeta^(2j+1), zeta a primitive 2N/d-th root of unity, so
    # that R_p is a product of fields of size p^d [CCCFGS25, Sec. 3.1]. That
    # needs p - 1 of 2-adicity exactly log2(2N/d) -- a odd -- since otherwise
    # zeta is a square and every factor splits further, leaving fields of size
    # p only. An even a still
    # gives a correct incomplete transform; the residue fields are what is lost.
    #
    # With d = 1 the ring splits into F_p anyway and any a will do. Full-size
    # primes (at least 2^(prime_size - 1)) come first, odd a before even, so a
    # ring with a full-size odd-a prime keeps it; a smaller prime is taken
    # only when the size has none.
    @staticmethod
    def gen_prime(rou_order, prime_size, exclude_list=None, *, split_degree):
        exclude_list = exclude_list if exclude_list is not None else []
        top = (2**prime_size - 1) // rou_order
        # The largest a whose prime is below 2^(prime_size - 1).
        low = ((1 << (prime_size - 1)) - 1) // rou_order
        full_size = [((top | 1) - 2, low)]
        smaller = [(low if low & 1 else low - 1, 0)]
        if split_degree == 1:
            full_size.append((top & ~1, low))
            smaller.append((low & ~1, 0))
        for first, stop in full_size + smaller:
            for a in range(first, stop, -2):
                candidate = a * rou_order + 1
                if candidate not in exclude_list and is_prime(candidate):
                    return candidate
        raise ValueError(
            f"no prime p = 1 (mod {rou_order}) below 2^{prime_size} is left; "
            "use larger primes or a smaller ring"
        )

    @staticmethod
    def gen_primes_near(
        rou_order: int,
        prime_size: int,
        count: int,
        exclude_list: list[int] | None = None,
        *,
        split_degree: int,
    ) -> list[int]:
        """``count`` primes ``p = a * rou_order + 1`` nearest ``2^prime_size``,
        alternately above and below it: the nearest above, the nearest below,
        the next above, and so on, none in ``exclude_list``.

        The ones above have ``prime_size + 1`` bits. As in `gen_prime`, ``a``
        is odd when ``split_degree > 1``; with ``split_degree`` 1 any ``a``
        will do, and the nearest is taken. Alternating keeps the product of
        consecutive primes close to the matching power of ``2^prime_size``.
        """
        excluded = set(exclude_list) if exclude_list is not None else set()
        step = 2 if split_degree > 1 else 1
        target = 1 << prime_size
        # The smallest a with a * rou_order + 1 above the target, and the
        # largest below it, moved onto the allowed parity.
        up = (target - 1) // rou_order + 1
        down = (target - 2) // rou_order
        if step == 2:
            up |= 1
            down -= 1 - (down & 1)
        primes: list[int] = []
        while len(primes) < count:
            from_above = len(primes) % 2 == 0
            while True:
                a = up if from_above else down
                if a < 1:
                    raise ValueError(
                        f"no prime p = 1 (mod {rou_order}) below 2^{prime_size} "
                        "is left; use larger primes or a smaller ring"
                    )
                candidate = a * rou_order + 1
                if from_above:
                    up += step
                else:
                    down -= step
                if candidate not in excluded and is_prime(candidate):
                    primes.append(candidate)
                    break
        return primes

    def gen_primes(self):
        primes = []
        for p_size in self.prime_size:
            primes.append(
                RNSRing.gen_prime(
                    2 * self.N // self.split_degree,
                    p_size,
                    exclude_list=primes,
                    split_degree=self.split_degree,
                )
            )
        return primes

    # in this file used to test against the definition of Ring, used in other files when efficiency is not required

    def alloc_polynomial(self):
        return self.lib.polynomial_new_RNS_polynomial(self.N, self.mask, self.base)

    def scalar_array(self, value):
        scale_arr = [0] * self.rns_rows
        if isinstance(value, int):
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = value % self.primes[k]
        else:
            vals = list(value)
            if len(vals) != self.ell:
                raise ValueError(f"expected {self.ell} values, got {len(vals)}")
            for k, idx in enumerate(self.prime_indices):
                scale_arr[idx] = vals[k]
        return ffi.new("uint64_t[]", scale_arr)

    def scalar(self, value) -> RNSScalar:
        """``value`` as a multiplier of this ring's elements: an integer, or
        one integer per prime in `primes` order, each reduced by its prime.

        It is built once and applied without allocating, by ``p * scalar`` or
        ``p *= scalar`` for any element ``p`` of this ring or of a quotient of
        it.
        """
        return RNSScalar(self, value)

    def random_element(self, ntt=True):
        return Polynomial(self).sample_uniform(ntt)

    def random_gaussian_element(self, sigma, ntt=True):
        return Polynomial(self).sample_gaussian(sigma, ntt)

    def random_exceptional(self, seed=None, ntt=True):
        return Polynomial(self).sample_exceptional(seed, ntt)


class Representation(Enum):
    """Which representation an `RNSPolynomial` currently holds.

    ``empty``
        Allocated, never written. Not a data representation: converting to it
        is an error.
    ``coeff``
        Residues of the polynomial's coefficients, one row per prime. The
        canonical form: equality, digests and integer readback are defined
        here.
    ``ntt``
        The same value under the number-theoretic transform, where
        multiplication is elementwise. The mul domain.

    `Domain` is the same distinction in the vocabulary every implementation
    shares; `domain_of` maps one to the other.
    """

    empty = 1
    ntt = 2
    coeff = 3


#: The name this enum has always had here, kept because it is what the
#: keyword arguments taking one are called. `Representation` is the same
#: object under a name that does not shadow the builtin.
repr = Representation

#: This implementation's representation flag as a generic arithmetic domain:
#: the two say the same thing about an element, in the two vocabularies.
_DOMAIN = {repr.empty: 0, repr.coeff: 1, repr.ntt: 2}
_MUL_DOMAIN = _DOMAIN[repr.ntt]


def domain_of(representation) -> int:
    """The `ArithDomain` value matching a `Representation`."""
    return _DOMAIN[representation]


def _check_status(status: int, operation: str) -> None:
    """Raises when a call of the generic interface reports a failure."""
    if status != 0:
        raise ValueError(f"arith_{operation} failed with {status}")


class RNSScalar:
    """A multiplier of the elements of `ring` and of its quotients: one
    residue per prime, held natively as an ``ArithScalar``. Built by
    `RNSRing.scalar`."""

    def __init__(self, ring: RNSRing, value) -> None:
        if isinstance(value, Integral):
            residues = [int(value) % p for p in ring.primes]
        else:
            residues = [int(v) for v in value]
            if len(residues) != ring.ell:
                raise ValueError(f"expected {ring.ell} values, got {len(residues)}")
            residues = [v % p for v, p in zip(residues, ring.primes, strict=True)]
        self.ring = ring
        self.residues = tuple(residues)
        native = ffi.new("ArithScalar *")
        _check_status(
            ring.lib.arith_scalar_new(
                ring.arith_ring, ring.scalar_array(residues), native
            ),
            "scalar_new",
        )
        self._native = native

    def serves(self, ring: RNSRing) -> bool:
        """Whether this scalar multiplies the elements of `ring`: a quotient
        of its own ring, over the same base."""
        return ring.base == self.ring.base and ring.is_quotient_ring(self.ring)

    def __del__(self) -> None:
        native = getattr(self, "_native", None)
        if native is not None and native.handle != ffi.NULL:
            # interpreter shutdown may already have torn the lib down
            with contextlib.suppress(Exception):
                self.ring.lib.arith_scalar_free(self.ring.arith_ring, native)

    def __repr__(self) -> str:
        return f"RNSScalar({list(self.residues)})"


class RNSPolynomial(Polynomial):
    """An element of an `RNSRing`: one row of residues per prime.

    The value is held natively and reached through `obj`; `repr` records which
    of the two representations (`Representation`) those rows are currently in,
    and the transforms are what move between them. Readers convert a copy, so
    reading never changes the representation of the object read.
    """

    #: The parent, whose primes index this element's rows.
    ring: RNSRing
    #: The native ``RNS_Polynomial`` handle, freed with this object.
    obj: Any
    #: The ``ArithElement`` `as_element` returns, built on first use.
    _element: Any = None

    if TYPE_CHECKING:
        # What a direct construction yields, with the native handle this
        # class alone can adopt; the front declares the common keywords.
        def __new__(
            cls, ring: RNSRing, repr: Any = ..., obj: Any = None
        ) -> RNSPolynomial: ...

    def __init__(
        self,
        ring: RNSRing,
        repr: Representation = Representation.empty,
        obj: Any = None,
    ) -> None:
        """Allocate an element of ``ring``, holding no value yet.

        ``obj`` adopts a native ``RNS_Polynomial`` over ``ring``'s primes
        instead, in representation ``repr``; it is freed with this object.
        """
        self.ring = ring
        self.obj = ring.alloc_polynomial() if obj is None else obj
        #: Which representation `obj`'s rows hold right now.
        self.repr: Representation = repr

    @property
    def rns_mask(self):
        return ffi.cast("RNS_Polynomial", self.obj).rns_mask

    def from_array(self, array: list):
        array = list(array) + ([0] * (self.ring.N - len(array)))
        # int_array_to_RNS reads the buffer as int64_t; wrap signed values into
        # their two's-complement uint64 form.
        array = [x & 0xFFFFFFFFFFFFFFFF for x in array]
        self.ring.lib.int_array_to_RNS(self.obj, ffi.new("uint64_t[]", array))
        self.repr = repr.ntt
        return self

    def from_bigint_array(self, array: list):
        rows = [ffi.NULL] * self.ring.rns_rows
        keep = []  # keep row buffers alive across the (copying) C call
        for k, idx in enumerate(self.ring.prime_indices):
            p = self.ring.primes[k]
            # Zero-initialized at the full degree, so a shorter array is
            # padded with zero coefficients as in from_array.
            row = ffi.new("uint64_t[]", self.ring.N)
            row[0 : len(array)] = [v % p for v in array]
            keep.append(row)
            rows[idx] = row
        matrix = ffi.new("uint64_t*[]", rows)
        self.ring.lib.array_to_RNS(self.obj, matrix)
        self.repr = repr.ntt
        return self

    def fast_inverse(self):
        if self.repr != repr.ntt:
            raise ValueError("fast_inverse requires NTT representation")
        out = Polynomial(self.ring)
        rc = self.ring.lib.polynomial_RNS_inverse(out.obj, self.obj)
        if rc == -1:
            raise ValueError("polynomial_RNS_inverse: invalid ring parameters")
        if rc == -2:
            raise ValueError("polynomial_RNS_inverse: zero slot is not invertible")
        if rc != 0:
            raise RuntimeError(f"polynomial_RNS_inverse failed with code {rc}")
        out.repr = repr.ntt
        return out

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            self.ring.lib.free_RNS_polynomial(self.obj)

    # def from_int(self, i:int) -> Polynomial:
    def multiply(self, in1, in2):
        if not (in1.repr == in2.repr == repr.ntt):
            raise ValueError("failed: not (in1.repr == in2.repr == repr.ntt)")
        _check_status(
            self.ring.lib.arith_mul_handles(
                self.ring.arith_ring, self.obj, in1.obj, in2.obj, _MUL_DOMAIN
            ),
            "mul",
        )
        self.repr = in1.repr

    def negate(self, in1=None):
        if not in1:
            in1 = self
        self.ring.lib.polynomial_RNSc_negate(self.obj, in1.obj)
        self.repr = in1.repr

    def sub(self, in1, in2):
        if in1.repr != in2.repr:
            raise ValueError("failed: in1.repr != in2.repr")
        _check_status(
            self.ring.lib.arith_sub_handles(
                self.ring.arith_ring, self.obj, in1.obj, in2.obj, _DOMAIN[in1.repr]
            ),
            "sub",
        )
        self.repr = in1.repr

    def add(self, in1, in2):
        if in1.repr != in2.repr:
            raise ValueError("failed: in1.repr != in2.repr")
        _check_status(
            self.ring.lib.arith_add_handles(
                self.ring.arith_ring, self.obj, in1.obj, in2.obj, _DOMAIN[in1.repr]
            ),
            "add",
        )
        self.repr = in1.repr

    def automorphism(self, gen):
        """``X -> X^gen``.

        Stays in the NTT domain when ``self`` is there and the ring is fully split
        (a permutation of the evaluation points); otherwise returns coefficients.
        """
        if gen >= self.ring.N * 2:
            raise ValueError("failed: gen >= self.ring.N * 2")
        res = Polynomial(self.ring)
        if self.repr == repr.ntt and self.ring.split_degree == 1 and gen % 2 == 1:
            idx = ffi.new("uint32_t[]", self.ring.N)
            self.ring.lib.polynomial_RNS_automorphism_index(idx, self.ring.N, gen)
            self.ring.lib.polynomial_RNS_permute(res.obj, self.obj, idx)
            res.repr = repr.ntt
            return res
        self.to_coeff()
        _check_status(
            self.ring.lib.arith_permute(
                self.ring.arith_ring, res.as_element(), self.as_element(), gen
            ),
            "permute",
        )
        res.repr = repr.coeff
        return res

    def to_NTT(self):
        if self.repr == repr.ntt:
            return
        _check_status(
            self.ring.lib.arith_to_mul(self.ring.arith_ring, self.as_element()),
            "to_mul",
        )
        self.repr = repr.ntt

    def to_coeff(self):
        if self.repr == repr.coeff:
            return
        _check_status(
            self.ring.lib.arith_to_canonical(self.ring.arith_ring, self.as_element()),
            "to_canonical",
        )
        self.repr = repr.coeff

    def to_repr(self, target) -> None:
        if target == repr.ntt:
            self.to_NTT()
        elif target == repr.coeff:
            self.to_coeff()
        else:
            raise ValueError(f"cannot convert to {target}: not a data representation")

    def _viewed_as(self, target) -> RNSPolynomial:
        """``self`` in `target` representation, converting a *copy* if needed.

        Every reader goes through this, so that reading a polynomial's value
        never changes the representation of the object read.
        """
        if self.repr == target:
            return self
        view = self.copy()
        view.to_repr(target)
        return view

    def sample_uniform(self, ntt=True):
        self.ring.lib.polynomial_gen_random_RNSc_polynomial(self.obj)
        self.repr = repr.ntt if ntt else repr.coeff
        return self

    def sample_gaussian(self, sigma, ntt=True):
        _check_status(
            self.ring.lib.arith_sample_gaussian(
                self.ring.arith_ring, self.as_element(), sigma
            ),
            "sample_gaussian",
        )
        self.repr = repr.coeff
        if ntt:
            self.to_NTT()
        return self

    def sample_exceptional(self, seed=None, ntt=True):
        ring = self.ring
        maps = ring._inverse_crt_maps_by_row()
        if seed is None:
            ring.lib.polynomial_RNS_sample_exceptional(self.obj, ffi.NULL, 0, maps)
        else:
            ring.lib.polynomial_RNS_sample_exceptional(self.obj, seed, len(seed), maps)
        self.repr = repr.ntt
        if not ntt:
            self.to_coeff()
        return self

    def base_extend(
        self, ring: RNSRing | None = None, out: RNSPolynomial | None = None
    ) -> RNSPolynomial:
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif isinstance(out, Polynomial):
            out_: RNSPolynomial = out
        if not (self.ring.is_quotient_ring(out_.ring)):
            raise ValueError("Not a quotient ring")
        return self.lift_to(out=out_)

    def convert_base(
        self,
        ring: RNSRing | None = None,
        out: RNSPolynomial | None = None,
        exact: bool = False,
    ) -> RNSPolynomial:
        return self.lift_to(ring=ring, out=out, exact=exact)

    def lift_to(
        self,
        ring: RNSRing | None = None,
        out: RNSPolynomial | None = None,
        params=None,
        exact: bool = False,
    ) -> RNSPolynomial:
        self.to_coeff()
        if out is None and ring is not None:
            out_ = Polynomial(ring)
        elif isinstance(out, Polynomial):
            out_: RNSPolynomial = out
        if params is None:
            params = registry().get_conversion_params(
                self.ring.N,
                self.ring.split_degree,
                self.rns_mask,
                out_.rns_mask,
                exact=exact,
            )
        self.ring.lib.polynomial_base_conversion_RNSc(out_.obj, self.obj, params)
        out_.repr = repr.coeff
        return out_

    def mod_reduce(
        self, ring: RNSRing | None = None, out: RNSPolynomial | None = None
    ) -> RNSPolynomial:
        # Primes are transformed independently, so dropping some is a row copy
        # in either domain.
        if not (out is not None or ring is not None):
            raise ValueError("Must provide ring or out")
        if out is None:
            out_ = Polynomial(ring)  # type: ignore
        else:
            out_: RNSPolynomial = out
        if not (out_.ring.is_quotient_ring(self.ring)):
            raise ValueError("Not a quotient ring")
        self.ring.lib.polynomial_RNSc_mod_reduce(out_.obj, self.obj)
        out_.repr = self.repr
        return out_

    # floor division, in-place
    def floor_division(self, ring: RNSRing) -> RNSPolynomial:
        if ring.ell >= self.ring.ell:
            raise ValueError("new ring is not smaller than the current one")
        if not (ring.is_quotient_ring(self.ring)):
            raise ValueError("Not a quotient ring or contiguous RNS-component subset")
        self.to_coeff()
        divide_mask = self.rns_mask & ~ring.mask
        self.ring.lib.polynomial_floor_division_RNSc_wo_free(self.obj, divide_mask)
        self.ring = ring
        return self

    # round division, in-place
    def round_division(self, ring: RNSRing) -> RNSPolynomial:
        if ring.ell >= self.ring.ell:
            raise ValueError("new ring is not smaller than the current one")
        if not (ring.is_quotient_ring(self.ring)):
            raise ValueError("Not a quotient ring or contiguous RNS-component subset")
        self.to_coeff()
        _check_status(
            self.ring.lib.arith_round_division(
                self.ring.arith_ring, self.as_element(), ring.arith_ring
            ),
            "round_division",
        )
        self.ring = ring
        return self

    def round_division_approximate(self, ring: RNSRing) -> RNSPolynomial:
        if not ring.is_quotient_ring(self.ring):
            raise ValueError("Not a quotient ring")
        self.to_coeff()
        status = self.ring.lib.arith_round_division_approximate(
            self.ring.arith_ring, self.as_element(), ring.arith_ring
        )
        if status != 0:
            raise ValueError(f"arith_round_division_approximate failed with {status}")
        self.ring = ring
        return self

    def decompose_digit(
        self,
        ring: RNSRing,
        primes: list[int],
        log_base: int = 0,
        level: int = 0,
        dropped: int = 0,
        centered: bool = False,
    ) -> RNSPolynomial:
        mask = 0
        for p in primes:
            if p not in self.ring.primes:
                raise ValueError(f"{p} is not a prime of this ring")
            mask |= 1 << self.ring.prime_indices[self.ring.primes.index(p)]
        self.to_coeff()
        out = Polynomial(ring)
        status = self.ring.lib.arith_decompose_digit(
            self.ring.arith_ring,
            out.as_element(),
            self.as_element(),
            mask,
            log_base,
            level,
            dropped,
            int(centered),
        )
        if status != 0:
            raise ValueError(
                "a radix digit takes one prime; the primes must be this ring's"
            )
        out.repr = repr.coeff
        return out

    def rescale_to_power_of_two(self, k: int):
        """``round(2^k * c / q)`` for each coefficient, as ``N`` machine words.

        The rescale for a target modulus that is not a product of base primes,
        which is what `round_division` and `scaled_lift` between them cannot
        express. Coefficients are taken as centered representatives and the
        results modulo 2^k, so the words are the two's-complement pattern of a
        signed value for any ``k`` up to 64 -- at 64 exactly what a `uint64_t`
        holds. Exact: the rounding is not carried out in floating point.

        Leaves this polynomial in the coefficient domain and does not otherwise
        change it.

        :param k: Width of the target modulus, 1 to 64.
        :returns: A native ``uint64_t[N]``; ``ffi.buffer`` of it is the bytes,
            ``list`` of it the coefficients.
        """
        if not 1 <= k <= 64:
            raise ValueError(f"k must be between 1 and 64, got {k}")
        mp, consts, half, q_inv_neg = self.ring._pow2_rescale_consts(k)
        out = ffi.new("uint64_t[]", self.ring.N)
        self.ring.lib.mp_polynomial_scale_to_2k(
            out, mp.from_polynomial(self, consts), half, q_inv_neg, k
        )
        return out

    def scaled_lift(self, ring: RNSRing, delta=None) -> RNSPolynomial:
        """Lifts the polynomial to a larger ring and scales it by the delta factor."""
        self.to_coeff()
        out = Polynomial(ring)
        self.ring.lib.polynomial_RNSc_scaled_lift(
            out.obj, self.obj, delta if delta is not None else ffi.NULL
        )
        out.repr = repr.coeff
        return out

    def __eq__(self, value) -> bool:
        if type(value) is int:
            return all(all([value == i[0]] + [j == 0 for j in i[1:]]) for i in self)
        elif type(value) is list:
            s_list = list(self)
            return all(
                all([value[i] == s_list[i][0]] + [j == 0 for j in s_list[i][1:]])
                for i in range(self.ring.ell)
            )
        else:
            view = self._viewed_as(value.repr)
            return bool(self.ring.lib.polynomial_eq(view.obj, value.obj))

    def __repr__(self) -> str:
        return str(list(self))

    def get_hash_pointer(self):
        # keep the view alive: it owns the buffer the call reads
        view = self._viewed_as(repr.ntt)
        return self.ring.lib.polynomial_RNS_get_hash_p(view.obj)

    def get_hash(self):
        view = self._viewed_as(repr.ntt)
        result = ffi.new("uint64_t[4]")
        self.ring.lib.polynomial_RNS_get_hash(result, view.obj)
        return [result[i] for i in range(4)]

    def digest(self) -> bytes:
        """The 32-byte BLAKE3 digest of the value, taken in the mul domain:
        the bytes `get_hash` returns as four words."""
        view = self._viewed_as(repr.ntt)
        result = ffi.new("uint8_t[32]")
        self.ring.lib.polynomial_RNS_get_hash(ffi.cast("uint64_t *", result), view.obj)
        return bytes(result)

    @staticmethod
    def digest_sequence(polys: list[RNSPolynomial]) -> bytes:
        """The BLAKE3 digest of one message: for each of `polys` in order,
        the rows `get_hash` hashes, taken in the mul domain like it."""
        state = ffi.new("uint8_t[]", lib.hash_stream_size())
        lib.hash_stream_init(state)
        for p in polys:
            view = p._viewed_as(repr.ntt)  # a local: the view owns the rows read
            p.ring.lib.polynomial_RNS_hash_update(state, view.obj)
        out = ffi.new("uint8_t[32]")
        lib.hash_stream_digest(state, out)
        return bytes(out)

    def __hash__(self):
        raise TypeError(
            "not a Python hashable object. Call polynomial.get_hash() for cryptographic hash"
        )

    # for compatibility with list, not an efficient iterator
    def __iter__(self):
        return iter(self.get_coeff_matrix(repr=repr.coeff))

    def get_coeff_matrix(self, repr=repr.coeff):
        """One list of ``N`` coefficients per prime, in `RNSRing.primes` order.

        The rows are in the order `to_uint64` gives, which says what that
        order is in each representation.
        """
        # `view` must stay referenced: `p` points into the buffer it owns, and a
        # temporary would be freed before the rows below are read.
        view = self._viewed_as(repr)
        p = ffi.cast("RNS_Polynomial", view.obj)
        N = self.ring.N
        values = [ffi.unpack(_row(p, idx), N) for idx in self.ring.prime_indices]
        if self.ring.split_degree == 1:
            return values
        return [_to_natural_order(row, self.ring.split_degree) for row in values]

    def from_coeff_matrix(self, matrix, repr=repr.coeff):
        """Overwrite with one list of ``N`` coefficients per prime; returns ``self``."""
        p = ffi.cast("RNS_Polynomial", self.obj)
        N = self.ring.N
        d = self.ring.split_degree
        for k, idx in enumerate(self.ring.prime_indices):
            row = _row(p, idx)
            values = matrix[k]
            if d > 1:
                values = list(values)
                if len(values) < N:
                    raise IndexError(f"row {k} holds {len(values)} values, not {N}")
                values = _from_natural_order(values[:N], d)
            width = 4 if (p.base.narrow_mask >> idx) & 1 else 8
            code = _ROW_CODE[width]
            try:
                packed = array.array(code, values)
            except TypeError:
                packed = array.array(code, [int(v) for v in values])
            if len(packed) < N:
                raise IndexError(f"row {k} holds {len(packed)} values, not {N}")
            ffi.memmove(row, packed, N * width)
        self.repr = repr
        return self

    def to_uint64(self, repr=repr.coeff) -> array.array:
        """The rows in representation `repr`, as one ``array('Q')`` of
        ``ell * N`` words: words ``i * N`` to ``(i + 1) * N - 1`` are the row
        of ``ring.primes[i]``. ``numpy.frombuffer(words, numpy.uint64)
        .reshape(ell, N)`` reads it as a matrix without copying.

        In `Representation.coeff`, word ``i * N + j`` is coefficient ``j``
        modulo ``primes[i]``. In `Representation.ntt` a row is the ``N / d``
        slots of the incomplete transform one after another, ``d =
        split_degree``: words ``i * N + s * d`` to ``i * N + s * d + d - 1``
        are the coefficients, low degree first, of the residue modulo
        ``X^d - w_s`` (for ``d = 1`` the value at ``w_s``), with ``w_s =
        get_rou_matrix()[i][s] = w_0^(2 brv(s) + 1)`` and ``brv`` the
        reversal of ``log2(N / d)`` bits. `get_coeff_matrix` returns the same
        rows as lists of Python ints.

        Reads a converted copy when ``self`` is in the other representation.
        """
        view = self._viewed_as(repr)  # a local: the call reads its rows
        words = array.array("Q", bytes(8 * self.ring.ell * self.ring.N))
        self.ring.lib.polynomial_RNS_get_natural(
            ffi.from_buffer("uint64_t[]", words), view.obj
        )
        return words

    def from_uint64(self, values, repr=repr.coeff) -> RNSPolynomial:
        """Overwrite with rows laid out as `to_uint64` returns them, holding
        representation `repr`; returns ``self``.

        `values` is an ``array('Q')`` or any C-contiguous buffer of
        ``ell * N`` unsigned 64-bit integers, such as a ``numpy.uint64`` array
        of shape ``(ell, N)``. Every word must be below its row's prime;
        ValueError otherwise, with ``self`` unchanged.
        """
        view = memoryview(values)
        if view.itemsize != 8 or view.format not in ("Q", "L"):
            raise TypeError("values must be a buffer of 64-bit unsigned integers")
        if not view.c_contiguous:
            raise TypeError("values must be C-contiguous")
        size = self.ring.ell * self.ring.N
        if view.nbytes != 8 * size:
            raise ValueError(f"{view.nbytes // 8} values for {size} residues")
        bad = self.ring.lib.polynomial_RNS_set_natural(
            self.obj, ffi.from_buffer("uint64_t[]", view.cast("B"))
        )
        if bad >= 0:
            raise ValueError(
                f"row {bad} holds a value not below its prime {self.ring.primes[bad]}"
            )
        self.repr = repr
        return self

    # returns the list of coefficients of a Polynomial element
    def get_polynomial(self, signed: bool = False) -> list[int]:
        rns = list(self)
        if self.ring.ell == 1:
            unsigned_res = list(itertools.chain.from_iterable(rns))
        else:
            unsigned_res = [
                crt([rns[i][j] for i in range(self.ring.ell)], self.ring.primes)
                for j in range(self.ring.N)
            ]
        if not signed:
            return unsigned_res
        else:
            return [
                min(i, -(self.ring.q_l - i), key=lambda x: abs(x)) for i in unsigned_res
            ]

    def as_element(self):
        """This polynomial as an `ArithElement` for the generic C interface.

        A view, not a copy: the returned struct points at the same storage, so
        it must not outlive this polynomial. The struct is this polynomial's
        own, built on first use and returned again by later calls with the
        current representation as its domain, so one call's struct is valid
        until the next.
        """
        element = self._element
        if element is None:
            element = self._element = ffi.new("ArithElement *", {"handle": self.obj})
        element.domain = _DOMAIN[self.repr]
        return element

    def copy(self) -> RNSPolynomial:
        res = Polynomial(self.ring)
        _check_status(
            self.ring.lib.arith_copy_handles(
                self.ring.arith_ring, res.obj, self.obj, _DOMAIN[self.repr]
            ),
            "copy",
        )
        res.repr = self.repr
        return res

    def decompose(self, base):
        res = []
        lst = self.get_polynomial()
        for _ in range(self.ring.bit_size // base):
            dec = []
            for i in range(len(lst)):
                dec.append(lst[i] & ((1 << base) - 1))
                lst[i] >>= base
            res.append(Polynomial(self.ring).from_array(dec))
        return res

    def __mod__(self, value: int):
        return self.decompose_digit(self.ring, [value])

    def __copy__(self) -> RNSPolynomial:
        return self.copy()

    def __itruediv__(self, value) -> Self:
        self.to_coeff()
        value_idx = self.ring.primes.index(value)
        self.ring.lib.polynomial_round_division_RNSc_wo_free(self.obj, 1 << value_idx)
        return self

    def __mul__(self, other) -> RNSPolynomial:
        if isinstance(other, RNSPolynomial):
            self.to_NTT()
            other.to_NTT()
            res = Polynomial(self.ring.intersec(other.ring))
            res.multiply(self, other)
        elif isinstance(other, Integral):
            if other == 1:
                return self.copy()
            res = Polynomial(self.ring)
            self._scale_integer(res, int(other))
        elif isinstance(other, RNSScalar):
            res = Polynomial(self.ring)
            self._scale_by(res, other)
        elif type(other) is list:
            res = Polynomial(self.ring)
            self._scale_by(res, RNSScalar(self.ring, other))
        else:
            raise NotImplementedError(f"cannot multiply by {type(other).__name__}")
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self) + other

    def __imul__(self, other):
        if isinstance(other, RNSPolynomial):
            self.to_NTT()
            other.to_NTT()
            self.ring.lib.polynomial_multo_RNS_polynomial(self.obj, other.obj)
        elif isinstance(other, Integral):
            if other == 1:
                return self
            self._scale_integer(self, int(other))
        elif isinstance(other, RNSScalar):
            self._scale_by(self, other)
        elif type(other) is list:
            self._scale_by(self, RNSScalar(self.ring, other))
        else:
            raise NotImplementedError(f"cannot scale by {type(other).__name__}")
        return self

    def _scale_integer(self, out: RNSPolynomial, k: int) -> RNSPolynomial:
        """``out = k * self`` for any integer ``k``, in whichever representation
        ``self`` is in; ``out`` may be ``self``.

        A ``k`` in ``[0, 2^64)`` goes to the native kernel as it is, which
        reduces it by each prime; any other ``k`` becomes an `RNSScalar`.
        """
        if not 0 <= k < 2**64:
            return self._scale_by(out, RNSScalar(self.ring, k))
        _check_status(
            self.ring.lib.arith_scale_int(
                self.ring.arith_ring, out.as_element(), self.as_element(), k
            ),
            "scale_int",
        )
        out.repr = self.repr
        return out

    def _scale_by(self, out: RNSPolynomial, scalar: RNSScalar) -> RNSPolynomial:
        """``out = scalar * self``, in whichever representation ``self`` is
        in; ``out`` may be ``self``."""
        if not scalar.serves(self.ring):
            raise ValueError(
                "the scalar is not over this ring or a ring it is a quotient of"
            )
        _check_status(
            self.ring.lib.arith_scale_by(
                self.ring.arith_ring,
                out.as_element(),
                self.as_element(),
                scalar._native[0],
            ),
            "scale_by",
        )
        out.repr = self.repr
        return out

    def _add_integer(self, out: RNSPolynomial, other: int) -> RNSPolynomial:
        """``out = self + other``, in whichever representation ``self`` is in.

        ``out`` may be ``self``; the C kernels accept out == in. They read the
        addend as a two's-complement int64, which is why it is range-checked
        and wrapped into the unsigned value cffi's ``uint64_t`` parameter takes,
        and why subtraction is just addition of the negation.
        """
        if not (-(2**63) <= other < 2**63):
            raise ValueError(f"integer operand {other} exceeds int64")
        wrapped = other & 0xFFFFFFFFFFFFFFFF
        if self.repr == repr.coeff:
            self.ring.lib.polynomial_RNSc_add_integer(out.obj, self.obj, wrapped)
        else:
            self.ring.lib.polynomial_RNS_add_integer(out.obj, self.obj, wrapped)
        out.repr = self.repr
        return out

    def __add__(self, other):
        if type(other) is int:
            if other == 0:
                return self.copy()
            return self._add_integer(Polynomial(self.ring), other)
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = Polynomial(self.ring.intersec(other.ring))
        res.add(self, other)
        return res

    def __iadd__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            return self._add_integer(self, other)
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if isinstance(other, RNSPolynomial):
            self.add(self, other)
            return self
        else:  # assume it's rlwe
            return other + self

    def __isub__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            return self._add_integer(self, -other)
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        self.sub(self, other)
        return self

    def __neg__(self):
        res = Polynomial(self.ring)
        res.negate(self)
        return res

    def __sub__(self, other):
        if type(other) is int:
            if other == 0:
                return self.copy()
            return self._add_integer(Polynomial(self.ring), -other)
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = Polynomial(self.ring.intersec(other.ring))
        res.sub(self, other)
        return res


#: RNS residues under the number-theoretic transform: the default parent for
#: polynomials in Z_q[X]/(X^N + 1), exact, with a full modulus tower.
RNS_NTT = register(
    Spec(
        implementation="rns",
        backend="ntt",
        parent_cls=RNSRing,
        element_cls=RNSPolynomial,
        capabilities=(
            Capability.CORE
            | Capability.QUOTIENT_POLY_RING
            | Capability.TOWER
            | Capability.SAMPLING
            | Capability.EXACT
        ),
        # 62 bits, not 64: values are lazy in [0, 4q) between butterfly
        # stages, so the widest reduction radix, 2^64, needs 4q <= 2^64.
        constraints=Constraints(max_prime_bits=62),
    )
)

# `spec` binds to the class because each class here serves exactly one; a
# class serving several must set it per instance instead.
RNSRing.spec = RNS_NTT
