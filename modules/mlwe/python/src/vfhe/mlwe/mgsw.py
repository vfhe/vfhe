# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses

from vfhe.arith import Polynomial, RNSPolynomial, domain_of, repr
from vfhe.util.bindings import ffi, lib

from .mlwe import (
    MLWE,
    GadgetParams,
    MLWE_Key,
    MLWE_Scheme,
    MLWE_Set,
    SchemeParameters,
    _monomial,
)


@dataclasses.dataclass(frozen=True)
class MGSWParameters:
    """What makes two MGSW schemes the same: the MLWE scheme, how many primes
    the gadget covers, and the gadget. Keys do not depend on ``balanced``."""

    mlwe: SchemeParameters
    ell: int
    gadget: GadgetParams


# RNS MGSW scheme (similar to RGSW)
class MGSW_Scheme:
    def __init__(
        self,
        MLWE_scheme: MLWE_Scheme,
        ell: int | None = None,
        gadget_params: GadgetParams | None = None,
        balanced: bool | None = None,
    ):
        """MGSW over ``MLWE_scheme``, encrypting against one of the two gadgets.

        ``ell`` is how many of the ring's primes the gadget covers (all of them
        by default). ``gadget_params`` choose the gadget (the RNS one by
        default; see :class:`GadgetParams`): the radix gadget, each prime
        contributing base-``2^log_base`` digits instead of a single residue,
        bounds the products an external product accumulates by the radix
        rather than by the primes, and keeping fewer digits is the approximate
        decomposition, for keys of a level over one prime. ``balanced`` picks
        the digit as in :class:`MLWE_Scheme`, and defaults to
        ``MLWE_scheme``'s.
        """
        self.mlwe_scheme = MLWE_scheme
        self.ell = ell if ell else MLWE_scheme.rings[0].ell
        self.ring = MLWE_scheme.special_rings[0]
        self.gadget_params = (
            gadget_params if gadget_params is not None else GadgetParams()
        )
        self.balanced = MLWE_scheme.balanced if balanced is None else balanced
        self.native_gadget_params = self.gadget_params.native(self.balanced)

    @property
    def parameters(self) -> MGSWParameters:
        """What this scheme is, for equality and for matching a stored scheme to a live one."""
        return MGSWParameters(self.mlwe_scheme.parameters, self.ell, self.gadget_params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MGSW_Scheme) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(self.parameters)

    def gadget_scalars(self, lvl: int = 0) -> list[list[int]]:
        """The gadget elements, as the per-prime scaling vectors to encrypt.

        The key lives in ``self.ring`` whatever the level; only how far the
        gadget reaches and what the external product's rescale divides by
        follow ``lvl``, so ``ell - lvl`` primes carry it, level ``lvl``'s
        first, wherever they sit in ``self.ring``.
        """
        return self.mlwe_scheme.gadget_scalars(
            lvl, self.gadget_params, ring=self.ring, primes=self.ell - lvl
        )

    def gadget_size(self, lvl: int = 0) -> int:
        """How many gadget elements a key encrypted for ``lvl`` holds per
        component: the length of :meth:`gadget_scalars`, without building them.
        """
        groups = self.mlwe_scheme._gadget_groups(  # noqa: SLF001 - same package
            lvl, self.ring, self.ell - lvl, self.gadget_params.primes_per_digit
        )
        log_base, digits = self.gadget_params.log_base, self.gadget_params.digits
        if not log_base:
            return len(groups)
        every = [
            lib.gadget_radix_digits(self.ring.primes[g[0]], log_base) for g in groups
        ]
        return sum(e if digits is None else min(digits, e) for e in every)

    def encrypt(
        self, msg: RNSPolynomial, key: MLWE_Key, lvl: int = 0, n_threads: int = 0
    ):
        """An MGSW encryption of ``msg``: for each component ``s_j`` of the key
        one encryption of ``-s_j * msg`` per gadget element, then the same for
        ``msg`` itself, drawn as one :meth:`MLWE_Scheme.sample_scaled` batch.
        """
        # Base extend msg to self.ring if needed
        if msg.ring != self.ring:
            msg = msg.base_extend(self.ring)

        key_special = MLWE_Key(key.key, key.sigma_err, self.mlwe_scheme, ring=self.ring)
        msgs = []
        for j in range(self.mlwe_scheme.r):
            sm = -key.poly[j] * msg
            if sm.ring != self.ring:
                sm = sm.base_extend(self.ring)
            msgs.append(sm)
        msgs.append(msg)
        samples = self.mlwe_scheme.sample_scaled(
            msgs, self.gadget_scalars(lvl), key_special, n_threads=n_threads
        )
        return MGSW(self, obj=samples)

    def encrypt_constants(
        self, values: list[int], key: MLWE_Key, lvl: int = 0, n_threads: int = 0
    ) -> list[MGSW]:
        """MGSW encryptions of the constant polynomials ``values``, as one batch.

        A constant folds into the gadget
        scalars -- ``s_j * c`` times ``g`` is ``s_j`` times ``c * g`` -- so the
        messages are the same for every value and only the scalars differ, and
        the whole key is one draw.
        """
        key_special = MLWE_Key(key.key, key.sigma_err, self.mlwe_scheme, ring=self.ring)
        one = Polynomial(self.ring).from_array([1])
        msgs = [-s_j for s_j in key_special.poly] + [one]
        gadget = self.gadget_scalars(lvl)
        scalars = [[c * g_p for g_p in g] for c in values for g in gadget]
        samples = self.mlwe_scheme.sample_scaled(
            msgs, scalars, key_special, n_threads=n_threads
        )
        width = len(gadget)
        return [
            MGSW(
                self,
                obj=[
                    samples[j * len(scalars) + v * width + k]
                    for j in range(len(msgs))
                    for k in range(width)
                ],
            )
            for v in range(len(values))
        ]

    def trivial(self, msg: RNSPolynomial, lvl: int = 0) -> MGSW:
        """The noiseless MGSW of ``msg``: the gadget matrix times ``msg``,
        which decrypts as an encryption of ``msg`` under any key (row ``j`` of
        the ``-s_j`` block carries ``msg`` times the gadget on its mask
        component ``j``, the last block on the body). ``lvl`` as in
        :meth:`encrypt`.
        """
        if msg.ring != self.ring:
            msg = msg.base_extend(self.ring)
        if msg.repr != repr.ntt:
            msg = msg.copy()
            msg.to_NTT()
        element = ffi.new("ArithElement[]", 1)
        element[0].handle = msg.obj
        element[0].domain = domain_of(msg.repr)
        ring = self.ring
        rows = [
            ring.scalar_array([v % p for v, p in zip(g, ring.primes, strict=True)])
            for g in self.gadget_scalars(lvl)
        ]
        r = self.mlwe_scheme.r
        handles = ffi.new("void*[]", (r + 1) * len(rows))
        lib.mgsw_trivial(
            handles,
            ring.arith_ring,
            r,
            element,
            ffi.new("uint64_t*[]", rows),
            len(rows),
        )
        return MGSW(self, obj=self._rows(handles))

    def automorphism(
        self,
        c: MGSW,
        g: int,
        aut_key: MLWE_Set,
        relin_key: MLWE_Set,
        n_threads: int = 0,
    ) -> MGSW:
        """The MGSW of ``m(X^g)`` from ``c``, an MGSW of ``m``.

        The rows encrypting ``m`` times the gadget go through the automorphism
        and ``aut_key`` (``s(X^g) -> s``); the rows encrypting ``-s_j m`` times
        it are rebuilt from those through ``relin_key``, a relinearization key
        (:meth:`MLWE_Scheme.gen_rlk`). Both keys must consume samples over the
        rows' ring, :attr:`ring`: without special primes, keys of level 0;
        with them, keys of a scheme where that ring is a level (one more
        special prime above it).
        """
        two_n = 2 * self.ring.N
        if g % 2 == 0 or not 0 < g < two_n:
            raise ValueError(f"g must be odd and in (0, 2N) (2N = {two_n})")
        r = self.mlwe_scheme.r
        if len(relin_key.mlwe) != r * (r + 3) // 2:
            raise ValueError("relin_key is not a relinearization key of this rank")
        for key in (aut_key, relin_key):
            key_ring = next(x for x in key.mlwe if x is not None)[0].ring
            if self.ring.mask & ~key_ring.mask or self.ring.base != key_ring.base:
                raise ValueError("the keys must consume samples over the MGSW's ring")
        c.to_NTT()
        width = c.gadget_size
        handles = ffi.new("void*[]", len(c.obj))
        lib.mgsw_automorphism(
            handles,
            ffi.new("void*[]", [x.obj for x in c.obj]),
            width,
            g,
            aut_key.obj,
            relin_key.obj,
            n_threads,
        )
        return MGSW(self, obj=self._rows(handles))

    def _rows(self, handles) -> list[MLWE]:
        lvl = self.mlwe_scheme.level_of_ring(self.ring, strict=False)
        out = []
        for h in handles:
            row = MLWE(self.mlwe_scheme, lvl=lvl, ring=self.ring, obj=h)
            row.repr = repr.ntt
            out.append(row)
        return out


class MGSW:
    def __init__(self, scheme: MGSW_Scheme, obj: list[MLWE] | None = None):
        self.scheme = scheme
        self.obj = (
            obj
            if obj
            else [
                MLWE(self.scheme.mlwe_scheme)
                for _ in range(
                    self.scheme.mlwe_scheme.r * self.scheme.ell + self.scheme.ell
                )
            ]
        )

    def to_NTT(self):
        for c in self.obj:
            c.to_NTT()

    def to_coeff(self):
        for c in self.obj:
            c.to_coeff()

    def multiply_by_monomial(self, exponent: int, minus_one: bool = False) -> MGSW:
        """An MGSW of ``m * X^exponent`` (or ``m * (X^exponent - 1)`` with
        ``minus_one``) from this MGSW of ``m``: every row multiplied as
        :meth:`MLWE.multiply_by_monomial` does, the transformed monomial built
        once for all of them. Rows keep their representation."""
        rows = self.obj
        if all(row.repr == repr.ntt for row in rows):
            monomial = _monomial(rows[0].ring, exponent, minus_one)
            out = []
            for row in rows:
                product = row.new_like()
                product.multiply_poly(row, monomial)
                out.append(product)
            return MGSW(self.scheme, obj=out)
        return MGSW(
            self.scheme,
            obj=[row.multiply_by_monomial(exponent, minus_one) for row in rows],
        )

    @property
    def gadget_size(self) -> int:
        """Gadget keys per component of the key: what C strides the array by.

        Read off the key itself rather than the scheme, since a key encrypted
        at a level carries the gadget for that level's primes only.
        """
        return len(self.obj) // (self.scheme.mlwe_scheme.r + 1)

    def check_level(self, sample: MLWE) -> None:
        """Raises unless this key was encrypted for ``sample``'s level.

        A product reads one gadget element per digit of the sample and divides
        by the ratio the key's gadget was scaled with, both of which follow the
        level the key was encrypted for.
        """
        lvl = sample.lvl
        if lvl < 0 or self.gadget_size != self.scheme.gadget_size(lvl):
            raise ValueError(
                f"an MGSW with {self.gadget_size} gadget elements per component "
                f"cannot multiply a sample at level {lvl}; encrypt it with "
                f"lvl={lvl}"
            )

    def external_product(self, other: MLWE) -> MLWE:
        self.check_level(other)
        res = other.new_like(lvl=other.lvl)
        self.to_NTT()
        other.to_coeff()

        # Pass the array of RNS_MLWE handles (self.obj) to C
        mgsw_ptr_array = ffi.new("void*[]", [c.obj for c in self.obj])

        lib.mgsw_external_product(
            res.obj,
            mgsw_ptr_array,
            other.obj,
            self.gadget_size,
            self.scheme.mlwe_scheme.special_primes,
            self.scheme.native_gadget_params,
        )
        res.repr = repr.ntt

        return res

    def internal_product(self, other: MGSW, n_threads: int = 0) -> MGSW:
        """The MGSW of the product of the two messages: the external product
        of ``self`` with every row of ``other``, laid out as ``other`` and over
        its ring.

        ``self`` must consume samples over ``other``'s rows' ring. Without
        special primes that is the ring both live in, and ``self`` is
        encrypted for level 0. With special primes ``other``'s rows live over
        its key ring, which is no level of its own scheme: ``self`` then
        belongs to a scheme where that ring is a level (one more special
        prime above it), encrypted for that level -- which keeps the rows
        over the key ring, the noise of a product with a special prime.
        """
        row_ring = other.obj[0].ring
        mine = self.scheme.mlwe_scheme
        lvl = mine.level_of_ring(row_ring, strict=False)
        if (
            lvl < 0
            or row_ring.base != mine.rings[lvl].base
            or self.gadget_size != self.scheme.gadget_size(lvl)
            or mine.r != other.scheme.mlwe_scheme.r
        ):
            raise ValueError(
                "this MGSW cannot multiply the other's rows: it must be "
                "encrypted for the level of its scheme whose ring is theirs"
            )
        self.to_NTT()
        other.to_NTT()
        handles = ffi.new("void*[]", len(other.obj))
        lib.mgsw_internal_product(
            handles,
            ffi.new("void*[]", [c.obj for c in self.obj]),
            self.gadget_size,
            ffi.new("void*[]", [c.obj for c in other.obj]),
            len(other.obj),
            self.scheme.native_gadget_params,
            n_threads,
        )
        return MGSW(other.scheme, obj=other.scheme._rows(handles))  # noqa: SLF001

    def __mul__(self, other: MLWE | MGSW) -> MLWE | MGSW:
        if isinstance(other, MGSW):
            return self.internal_product(other)
        if isinstance(other, MLWE):
            return self.external_product(other)
        return NotImplemented


def CMUX(in1: MLWE, in2: MLWE, selector: MGSW) -> MLWE:
    selector.check_level(in1)
    res = in1.new_like(lvl=in1.lvl)
    in1.to_coeff()
    in2.to_coeff()
    selector.to_NTT()

    mgsw_ptr_array = ffi.new("void*[]", [c.obj for c in selector.obj])

    lib.mgsw_CMUX(
        res.obj,
        in1.obj,
        in2.obj,
        mgsw_ptr_array,
        selector.gadget_size,
        in1.scheme.special_primes,
        selector.scheme.native_gadget_params,
    )
    res.repr = repr.ntt
    return res


def CMUX_batch(
    in1: list[MLWE], in2: list[MLWE], selector: MGSW, n_threads: int = 0
) -> list[MLWE]:
    """``CMUX(in1[i], in2[i], selector)`` for every ``i``: one selector over
    many pairs, the multiplexers run in parallel. Each pair is at one level,
    which ``selector`` was encrypted for. The results are left canonical.
    """
    if len(in1) != len(in2):
        raise ValueError("in1 and in2 must have the same length")
    for a, b in zip(in1, in2, strict=True):
        selector.check_level(a)
        if b.lvl != a.lvl or b.r != a.r:
            raise ValueError("the two inputs of a multiplexer must be at one level")
    if not in1:
        return []
    for c in (*in1, *in2):
        c.to_coeff()
    selector.to_NTT()
    out = [a.new_like(lvl=a.lvl) for a in in1]
    lib.mgsw_CMUX_batch(
        ffi.new("void*[]", [c.obj for c in out]),
        ffi.new("void*[]", [c.obj for c in in1]),
        ffi.new("void*[]", [c.obj for c in in2]),
        len(in1),
        ffi.new("void*[]", [c.obj for c in selector.obj]),
        selector.gadget_size,
        selector.scheme.native_gadget_params,
        n_threads,
    )
    for c in out:
        c.repr = repr.coeff
    return out


def NCMUX(
    in1: MLWE, in2: MLWE, selector: MGSW, aut_minus1: MLWE_Set | list[MLWE_Set]
) -> MLWE:
    aut_minus1 = aut_minus1 if isinstance(aut_minus1, MLWE_Set) else aut_minus1[in1.lvl]
    selector.check_level(in1)
    res = in1.new_like(lvl=in1.lvl)
    in1.to_coeff()
    in2.to_coeff()
    selector.to_NTT()

    mgsw_ptr_array = ffi.new("void*[]", [c.obj for c in selector.obj])

    lib.mgsw_NCMUX(
        res.obj,
        in1.obj,
        in2.obj,
        mgsw_ptr_array,
        aut_minus1.obj,
        selector.gadget_size,
        in1.scheme.special_primes,
        selector.scheme.native_gadget_params,
    )
    res.repr = repr.ntt
    return res
