# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import math
import operator
import secrets
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from vfhe.arith.polynomial import Polynomial, Ring, repr
from vfhe.misc.libvfhe import ffi, lib

if TYPE_CHECKING:
    from .lwe import LWE_Key

# Ciphertext operations preserve the concrete ciphertext class of their input
# (e.g. a CKKS_Ciphertext stays a CKKS_Ciphertext); see MLWE.new_like.
CtT = TypeVar("CtT", bound="MLWE")


class LibMLWE:
    def __init__(self) -> None:
        self.lib = lib


lib_rlwe = LibMLWE()


class MLWE_Scheme:
    # Concrete ciphertext class this scheme allocates when it has no input
    # ciphertext to derive the type from (see :meth:`sample`). Subclasses set it
    # to their own ciphertext class; operations that transform an existing
    # ciphertext use ``MLWE.new_like`` instead. Bound after MLWE is defined.
    ciphertext_type: type[MLWE]

    def __init__(
        self,
        rings: list[Ring] | Ring,
        special_primes: int = 0,
        special_rings: list[Ring] | None = None,
        max_lvl: int | None = None,
        module_rank: int = 1,
    ) -> None:
        """Create a leveled scheme in one of two initialization modes.

        - Single ``Ring``: the level chain is derived automatically by dropping the top prime one at a time. ``special_primes`` reserves that many top
          primes as the special (key-switching) primes, and ``max_lvl`` caps the
          number of levels.
        - List of ``Ring``: the levels are given explicitly. ``rings[i]`` is
          level ``i`` and ``special_rings[i]`` its special-prime-extended ring
          (required when ``special_primes > 0``; defaults to ``rings``
          otherwise). This allows non-nested level rings, e.g. for rational
          rescaling.
        """
        if isinstance(rings, Ring):
            max_lvl = max_lvl if max_lvl is not None else len(rings.primes) - 1
            if special_primes == 0:
                self.rings = [
                    rings.quotient_ring(ell=rings.ell - i) for i in range(max_lvl)
                ]
                self.special_rings = self.rings
            else:
                self.rings = [
                    rings.quotient_ring(ell=rings.ell - i)
                    for i in range(special_primes, special_primes + max_lvl)
                ]
                special_mask = ((1 << special_primes) - 1) << self.rings[0].ell
                self.special_rings = [
                    rings.quotient_ring(mask=self.rings[i].mask | special_mask)
                    for i in range(len(self.rings))
                ]
        else:
            # Explicit per-level rings: use the provided chain directly, with
            # special_rings[i] aligned to level i (defaults to rings when there
            # are no special primes).
            max_lvl = max_lvl if max_lvl is not None else len(rings)
            if special_rings is None:
                special_rings = rings
            self.rings = [rings[i] for i in range(max_lvl)]
            self.special_rings = [special_rings[i] for i in range(max_lvl)]

        self.r = module_rank
        self.N = self.rings[0].N
        self.ell = len(self.rings)
        self.max_lvl = max_lvl
        self.tmp = Polynomial(self.rings[0])
        self.rlk: MLWE_Set | list | None = None

    @property
    def ring(self) -> Ring:
        return self.rings[0]

    @property
    def extended_rank(self) -> int:
        """Rank of the extended (not-yet-relinearized) product of two samples.

        The product lives in the symmetric square of the ciphertext module:
        ``r*(r+1)/2`` quadratic components plus r linear ones (2 at rank 1). See
        :meth:`tensor_product`.
        """
        return self.r * (self.r + 3) // 2

    @property
    def special_primes(self) -> int:
        if self.special_rings and self.rings:
            return self.special_rings[0].ell - self.rings[0].ell
        return 0

    def level_of_ring(self, ring: Ring, strict: bool = True) -> int:
        """Return the level index of ``ring`` in ``self.rings``.

        Rings are identified by their RNS ``mask`` (their prime set), so an
        on-the-fly quotient with the same primes as a scheme level still
        resolves to that level. A ring outside ``self.rings`` (e.g. a
        special/extended ring) raises when ``strict`` is set, or resolves to the
        sentinel level ``-1`` when ``strict`` is ``False``.
        """
        for lvl, r in enumerate(self.rings):
            if r.mask == ring.mask:
                return lvl
        if strict:
            raise ValueError(
                "ring is not one of the scheme's levels (e.g. a special/extended "
                "ring); pass lvl explicitly"
            )
        return -1

    def key_gen_sparse(self, h, sigma_err, ternary=True):
        assert self.N * self.r >= h
        key = [[0] * self.N for _ in range(self.r)]
        crt_h = 0
        sign = 1
        while crt_h < h:
            rnd = secrets.randbelow(self.N * self.r)
            if key[rnd // self.N][rnd % self.N]:
                continue
            key[rnd // self.N][rnd % self.N] = sign
            if ternary:
                sign *= -1
            crt_h += 1
        return MLWE_Key(key, sigma_err, self)

    def _gen_ksk_components(
        self, key_out: MLWE_Key, key_poly: list[Polynomial], lvl: int
    ) -> list[list[MLWE]]:
        """Sample the gadget ciphertexts for one key-switch key per key poly.

        Returns the raw component lists (before wrapping in an ``MLWE_Set``) so
        callers such as relinearization can interleave NULL pass-through slots.
        """
        assert self == key_out.scheme, "Scheme mismatch"
        assert lvl is not None, "Level must be specified"
        result = []
        special_q = self.special_rings[lvl].modulus_ratio(self.rings[lvl])
        key_out_special = MLWE_Key(
            key_out.key, key_out.sigma_err, self, ring=self.special_rings[lvl]
        )
        for j in range(len(key_poly)):
            poly_j = key_poly[j]
            if poly_j.ring != self.special_rings[lvl]:
                poly_j = Polynomial(self.special_rings[lvl]).from_bigint_array(
                    poly_j.get_polynomial(signed=True)
                )
            result_i = []
            for i in range(self.special_rings[lvl].ell):
                scaling_factor = (
                    [0] * i
                    + [special_q % self.special_rings[lvl].primes[i]]
                    + [0] * (self.special_rings[lvl].ell - 1 - i)
                )
                out = MLWE(self, lvl=lvl, ring=self.special_rings[lvl])
                self.sample(poly_j * scaling_factor, key_out_special, out=out)
                result_i.append(out)
            result.append(result_i)
        return result

    def gen_ksk_for_level(
        self, key_out: MLWE_Key, key_in: MLWE_Key | list[Polynomial], lvl: int
    ):
        key_poly = key_in if isinstance(key_in, list) else key_in.poly
        return MLWE_Set(self._gen_ksk_components(key_out, key_poly, lvl))

    def gen_rlk_for_level(
        self, key_out: MLWE_Key, quad_polys: list[Polynomial], lvl: int
    ):
        # One real key-switch key per quadratic component (r*(r+1)/2 of them),
        # followed by r NULL slots for the linear components, which keep the
        # target key and are copied through by the key-switch.
        components = self._gen_ksk_components(key_out, quad_polys, lvl)
        return MLWE_Set(components + [None] * self.r)

    def quadratic_key_polys(self, key: MLWE_Key) -> list[Polynomial]:
        """The quadratic key terms of the tensored product, in slot order.

        These are ``-(s_i * s_j)`` for every pair ``i <= j`` in lexicographic
        order -- the extended key that :meth:`tensor_product`'s quadratic slots
        decrypt under (see that method for the layout). For rank 1 this is just
        ``[-(s_0 * s_0)]``.
        """
        return [
            -(key.poly[i] * key.poly[j])
            for i in range(self.r)
            for j in range(i, self.r)
        ]

    def gen_rlk(
        self,
        key_out: MLWE_Key,
        quad_polys: MLWE_Key | list[Polynomial],
        lvl: int | None = None,
    ):
        """Relinearization key for the rank-r product.

        ``quad_polys`` holds the ``r*(r+1)/2`` quadratic key terms in the slot
        order of :meth:`tensor_product` (e.g. ``[-(s_0 * s_0)]`` for rank 1);
        pass the :class:`MLWE_Key` itself to have them derived by
        :meth:`quadratic_key_polys`. The resulting key-switch set has these real
        keys plus r NULL slots for the linear components, consumed by
        :meth:`relinearize`/:meth:`multiply`.
        """
        quad_polys = (
            quad_polys
            if isinstance(quad_polys, list)
            else self.quadratic_key_polys(quad_polys)
        )
        assert len(quad_polys) == self.r * (self.r + 1) // 2, (
            "expected one quadratic key per pair of key components"
        )
        if lvl is not None:
            return self.gen_rlk_for_level(key_out, quad_polys, lvl)
        return [
            self.gen_rlk_for_level(key_out, quad_polys, lvl)
            for lvl in range(len(self.rings))
        ]

    def gen_ksk(
        self,
        key_out: MLWE_Key,
        key_in: MLWE_Key | list[Polynomial],
        lvl: int | None = None,
    ):
        key_poly = key_in if isinstance(key_in, list) else key_in.poly
        assert self == key_out.scheme, "Scheme mismatch"
        if lvl is not None:
            return self.gen_ksk_for_level(key_out, key_poly, lvl)
        return [
            self.gen_ksk_for_level(key_out, key_poly, lvl)
            for lvl in range(len(self.rings))
        ]

    def gen_ksk_automorphism(
        self, key_out: MLWE_Key, key_in: MLWE_Key, g: int, lvl: int | None = None
    ):
        key_perm = [i.automorphism(g) for i in key_in.poly]
        return self.gen_ksk(key_out, key_perm, lvl)

    def gen_ksk_automorphism_set(
        self,
        key_out: MLWE_Key,
        key_in: MLWE_Key,
        generators: list[int],
        lvl: int | None = None,
    ):
        return [self.gen_ksk_automorphism(key_out, key_in, g, lvl) for g in generators]

    def keyswitch(self, c: CtT, ksk: MLWE_Set | list[MLWE_Set]) -> CtT:
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = c.new_like(lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_RNSc_GHS_hybrid_keyswitch(out.obj, c.obj, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def gen_ksk_trace(
        self,
        key_out: MLWE_Key,
        key_in: MLWE_Key,
        gens: list[int] | None = None,
        lvl: int | None = None,
    ):
        log_N = int(math.log2(self.N))
        gens = (
            gens
            if gens is not None
            else [(1 << (log_N - i + 1)) + 1 for i in range(1, log_N + 1)]
        )
        result = [self.gen_ksk_automorphism(key_out, key_in, g, lvl) for g in gens]
        # The lvl argument decides which shape gen_ksk_automorphism returned.
        if lvl is not None:
            return MLWE_Set().flatten_array(cast("list[MLWE_Set]", result))
        leveled = cast("list[list[MLWE_Set]]", result)
        result_leveled = []
        for lvl in range(len(leveled[0])):
            result_lvl_i = [leveled[i][lvl] for i in range(len(gens))]
            result_leveled.append(MLWE_Set().flatten_array(result_lvl_i))
        return result_leveled

    def automorphism(self, c: CtT, gen: int, ksk: MLWE_Set | list[MLWE_Set]) -> CtT:
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = c.new_like(lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_automorphism_RNSc_GHS(out.obj, c.obj, gen, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def trace(self, c: CtT, ksk: MLWE_Set | list[MLWE_Set]) -> CtT:
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c.lvl]
        out = c.new_like(lvl=c.lvl, ring=self.special_rings[c.lvl])
        c.to_coeff()
        lib_rlwe.lib.mlwe_trace(out.obj, c.obj, ksk.obj, c.lvl)
        out.repr = repr.coeff
        out.ring = self.rings[c.lvl]
        return out

    def full_packing_keyswitch_scaled(
        self, vec: list[MLWE], ksk: MLWE_Set | list[MLWE_Set]
    ):
        # log_size is the packing depth (log2 of the vector length), based on the number of elements to pack
        log_size = int(math.log2(len(vec)))
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[vec[0].lvl]
        assert (1 << log_size) == len(vec)
        for c in vec:
            c.to_coeff()

        # create C array of handles for vec
        vec_ptr_array = ffi.new("void*[]", [c.obj for c in vec])

        lib_rlwe.lib.mlwe_full_packing_keyswitch_scaled(
            vec_ptr_array, log_size, ksk.obj, vec[0].lvl
        )

        # The output is in vec[0]
        vec[0].repr = repr.coeff
        return vec[0]

    def sample(
        self,
        msg: Polynomial,
        key: MLWE_Key,
        out: MLWE | None = None,
        lvl: int | None = None,
    ) -> MLWE:
        """Samples an MLWE ciphertext of the given message polynomial under the given key.

        Args:
            msg: The message polynomial.
            key: The MLWE key.
            out: Optional MLWE object to store the result.

        Returns:
            The sampled MLWE ciphertext.
        """
        if not out:
            out = self.ciphertext_type(self, lvl=lvl)
        msg.to_coeff()
        lib_rlwe.lib.mlwe_RNSc_sample(out.obj, key.obj, msg.obj)
        out.repr = repr.coeff
        return out

    def phase(self, rlwe: MLWE, key: MLWE_Key, out: Polynomial | None = None):
        if not out:
            out = Polynomial(rlwe.ring)
        if key.ring != rlwe.ring:
            key_at_ring = MLWE_Key(key.key, key.sigma_err, self, ring=rlwe.ring)
        else:
            key_at_ring = key
        rlwe.to_NTT()
        lib_rlwe.lib.mlwe_RNS_phase(out.obj, rlwe.obj, key_at_ring.obj)
        out.repr = repr.ntt
        return out

    def tensor_product(self, in1: MLWE, in2: MLWE) -> list[Polynomial]:
        """Symmetric tensor product of the r+1 components of ``in1`` and ``in2``.

        Since ``phase(c) = b - sum_i a_i * s_i``, the product of the two phases
        is quadratic in the key::

            b1*b2 - sum_i (a1_i*b2 + b1*a2_i) * s_i + sum_{i<=j} q_ij * s_i*s_j

        with ``q_ij = a1_i*a2_j + a1_j*a2_i`` (``i < j``) and
        ``q_ii = a1_i*a2_i``. The returned polynomials follow the extended key's
        slot order: the ``r*(r+1)/2`` quadratic components ``q_ij`` (pairs
        ``i <= j``, lexicographic), then the r linear components, then the
        constant term ``b1*b2`` last. Their keys are ``-(s_i * s_j)`` for the
        quadratic slots (see :meth:`quadratic_key_polys`) and ``s_i`` for the
        linear ones, which is why relinearization key-switches the former and
        passes the latter through.

        At rank 1 the layout coincides with the convolution of the coefficient
        vectors ``(a, b)``; at higher ranks it does not (distinct key terms would
        collide in a single index-sum slot).

        Args:
            in1: The first MLWE ciphertext.
            in2: The second MLWE ciphertext.

        Returns:
            A list of ``extended_rank + 1`` Polynomial objects.
        """
        assert in1.ring == in2.ring, "Ciphertexts must be in the same ring"
        assert in1.scheme.r == in2.scheme.r, "Ciphertexts must have the same rank"
        in1.to_NTT()
        in2.to_NTT()
        out_polys = [Polynomial(in1.ring) for _ in range(self.extended_rank + 1)]
        for p in out_polys:
            p.repr = repr.ntt
        out_pointers = ffi.new("void*[]", [p.obj for p in out_polys])
        lib_rlwe.lib.mlwe_tensor_product(out_pointers, in1.obj, in2.obj)
        return out_polys

    def multiply(
        self,
        in1: CtT,
        in2: MLWE,
        ksk: MLWE_Set | list[MLWE_Set] | None = None,
    ) -> CtT:
        """Tensors the two ciphertexts and (optionally) relinearizes.

        Args:
            in1: The first MLWE ciphertext.
            in2: The second MLWE ciphertext.
            ksk: The relinearization key set. If ``None``, relinearization is
                skipped and the extended product (rank
                :attr:`extended_rank`) is returned; feed it to
                :meth:`relinearize` to reduce it back to rank r.

        Returns:
            A rank-r product when ``ksk`` is given, otherwise the extended
            product of rank :attr:`extended_rank` (with ``is_extended`` set).
        """
        assert in1.ring == in2.ring, "Ciphertexts must be in the same ring"
        assert in1.scheme == in2.scheme, "Ciphertexts must be from the same scheme"
        assert in1.lvl == in2.lvl, "Ciphertexts must have the same level"
        in1.to_NTT()
        in2.to_NTT()

        if ksk is None:
            out = in1.new_like(lvl=in1.lvl, rank=in1.scheme.extended_rank)
            out.is_extended = True
            lib_rlwe.lib.mlwe_multiply(out.obj, in1.obj, in2.obj, ffi.NULL)
            out.repr = repr.ntt
            return out

        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[in1.lvl]
        # Relinearization key-switches into the extended ring, so the output
        # needs the special-prime capacity even though it lands back in base.
        out = in1.new_like(lvl=in1.lvl, ring=in1.scheme.special_rings[in1.lvl])
        lib_rlwe.lib.mlwe_multiply(out.obj, in1.obj, in2.obj, ksk.obj)
        out.repr = repr.ntt
        out.ring = in1.scheme.rings[in1.lvl]
        return out

    def relinearize(self, c_ext: CtT, ksk: MLWE_Set | list[MLWE_Set]) -> CtT:
        """Relinearize an extended product back to rank r.

        Reuses the GHS hybrid key-switch: the rlk key-switches the quadratic
        components and copies the linear ones through (their NULL slots).
        """
        ksk = ksk if isinstance(ksk, MLWE_Set) else ksk[c_ext.lvl]
        out = c_ext.new_like(lvl=c_ext.lvl, ring=self.special_rings[c_ext.lvl])
        c_ext.to_coeff()
        lib_rlwe.lib.mlwe_RNSc_GHS_hybrid_keyswitch(
            out.obj, c_ext.obj, ksk.obj, c_ext.lvl
        )
        out.repr = repr.coeff
        out.ring = self.rings[c_ext.lvl]
        return out


class MLWE_Key:
    def __init__(
        self,
        key: list[list[int]],
        sigma_err: float,
        scheme: MLWE_Scheme,
        ring: Ring | None = None,
    ):
        assert len(key) == scheme.r
        self.key = key
        self.scheme = scheme
        self.sigma_err = sigma_err
        if ring is None:
            ring = scheme.rings[0]
        self.ring = ring
        # the key is copied verbatim into signed int64 coeffs; wrap negatives into
        # two's-complement uint64.
        concat_key = [
            x & 0xFFFFFFFFFFFFFFFF for x in functools.reduce(operator.iadd, key, [])
        ]
        self.obj = lib_rlwe.lib.mlwe_new_RNS_key_from_array(
            ffi.new("uint64_t[]", concat_key),
            ring.N,
            scheme.r,
            ring.ell,
            ring.base,
            sigma_err,
        )
        self.poly = [Polynomial(ring).from_array(s_i) for s_i in key]
        for p in self.poly:
            p.to_NTT()
        struct = ffi.cast("RNS_MLWE_Key", self.obj)
        for i in range(scheme.r):
            ring.lib.polynomial_copy_RNS_polynomial(struct.s_RNS[i], self.poly[i].obj)

    def __del__(self) -> None:
        if hasattr(self, "obj") and self.obj:
            lib_rlwe.lib.free_mlwe_RNS_key(self.obj)

    def extract_lwe_key(self) -> LWE_Key:
        from .lwe import LWE_Key

        lwe_coeffs = functools.reduce(operator.iadd, self.key, [])
        n = len(lwe_coeffs)
        return LWE_Key(ring=self.scheme.rings[0], key=lwe_coeffs, n=n)


class MLWE_Set:
    def __init__(self, mlwe: Sequence[list[MLWE] | None] | None = None):
        if mlwe is None:
            return
        self.mlwe = mlwe
        self.dim = 2
        result_obj = ffi.new("void*[]", len(mlwe))
        for j in range(len(mlwe)):
            # A ``None`` component is a NULL key-switch key: the matching
            # ciphertext component keeps its key and is copied through (used by
            # relinearization for the linear part of the product).
            component = mlwe[j]
            if component is None:
                result_obj[j] = ffi.NULL
                continue
            ell = len(component)
            for x in component:
                x.to_NTT()
            tmp = ffi.new("void*[]", [i.obj for i in component])
            result_obj[j] = lib_rlwe.lib.mlwe_create_copy_array(tmp, ell)
        self.obj = result_obj

    # Turn an array of n-D MLWE_Set into a (n+1)-D MLWE_set
    @staticmethod
    def flatten_array(array: list[MLWE_Set]):
        out = MLWE_Set()
        out.mlwe = []
        out.dim = array[0].dim + 1
        result_obj = ffi.new("void*[]", len(array))
        out._children = array  # type: ignore  # keep child MLWE_Set buffers alive
        for j in range(len(array)):
            out.mlwe.append(array[j].mlwe)
            result_obj[j] = ffi.cast("void *", array[j].obj)
        out.obj = result_obj
        return out


class MLWE:
    def __init__(
        self,
        scheme: MLWE_Scheme,
        lvl: int | None = None,
        ring: Ring | None = None,
        rank: int | None = None,
    ) -> None:
        if lvl is None:
            lvl = scheme.level_of_ring(ring, strict=False) if ring is not None else 0
        if ring is None:
            ring = scheme.rings[lvl]
        # Rank defaults to the scheme's module rank; an extended (non-relinearized)
        # product carries the larger MLWE_Scheme.extended_rank.
        self.r = rank if rank is not None else scheme.r
        self.obj = lib_rlwe.lib.mlwe_alloc_RNS_sample(
            ring.N, self.r, ring.mask, ring.base
        )
        self.ring = ring
        self.scheme = scheme
        self.repr = repr.empty
        self.lvl = lvl
        # Marks an extended-rank product that still needs relinearization.
        self.is_extended = False

    @property
    def ell(self) -> int:
        return self.ring.ell

    def new_like(
        self: CtT,
        lvl: int | None = None,
        ring: Ring | None = None,
        rank: int | None = None,
    ) -> CtT:
        """Allocate an empty ciphertext of the same concrete type as ``self``.

        Operations that derive a new ciphertext from an existing one allocate it
        through here, so a subclass (e.g. ``CKKS_Ciphertext``) keeps its type and
        its extra metadata instead of decaying into a plain ``MLWE``. ``lvl``
        defaults to ``self.lvl``; ``ring`` and ``rank`` default as in
        :meth:`__init__`.
        """
        out = type(self)(
            self.scheme, lvl=self.lvl if lvl is None else lvl, ring=ring, rank=rank
        )
        out._inherit(self)
        return out

    def _inherit(self, other: MLWE) -> None:
        """Copy subclass-specific metadata from ``other`` into a fresh ``self``.

        No-op for plain MLWE; subclasses that add fields (e.g. the CKKS scaling
        factor) override this so :meth:`new_like` carries them over.
        """

    def __del__(self) -> None:
        if hasattr(self, "obj") and self.obj:
            lib_rlwe.lib.free_mlwe_RNS_sample(self.obj)

    def to_NTT(self):
        if self.repr == repr.ntt:
            return
        lib_rlwe.lib.mlwe_RNSc_to_RNS(self.obj, self.obj)
        self.repr = repr.ntt

    def to_coeff(self):
        if self.repr == repr.coeff:
            return
        lib_rlwe.lib.mlwe_RNS_to_RNSc(self.obj, self.obj)
        self.repr = repr.coeff

    def multiply_poly(self, in_rlwe, in_poly):
        assert in_rlwe.ring == in_poly.ring, "trying to mul things in different rings"
        assert in_rlwe.repr == in_poly.repr == repr.ntt
        lib_rlwe.lib.mlwe_RNS_mul_by_poly(self.obj, in_rlwe.obj, in_poly.obj)
        self.repr = repr.ntt

    def multiply_scalar(self, in_rlwe, pointer_to_int_list):
        if self is not in_rlwe:
            lib_rlwe.lib.mlwe_copy_RNS_sample(self.obj, in_rlwe.obj)
        lib_rlwe.lib.mlwe_scale_RNS_mlwe_RNS(self.obj, pointer_to_int_list)
        self.repr = in_rlwe.repr

    def add_MLWE(self, in1, in2):
        assert in1.repr == in2.repr
        lib_rlwe.lib.mlwe_add_RNSc_sample(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def sub_MLWE(self, in1, in2):
        assert in1.repr == in2.repr
        lib_rlwe.lib.mlwe_sub_RNSc_sample(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def add_poly(self, in1: MLWE, in2: Polynomial):
        assert in1.ring == in2.ring, "trying to add things in different rings"
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            lib_rlwe.lib.mlwe_RNS_add_polynomial(self.obj, in1.obj, in2.obj)
        else:
            lib_rlwe.lib.mlwe_add_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def sub_poly(self, in1, in2):
        assert in1.repr == in2.repr
        if in1.repr == repr.ntt:
            lib_rlwe.lib.mlwe_RNS_sub_polynomial(self.obj, in1.obj, in2.obj)
        else:
            lib_rlwe.lib.mlwe_sub_RNSc_polynomial(self.obj, in1.obj, in2.obj)
        self.repr = in1.repr

    def get_a_poly(self, j: int) -> Polynomial:
        res = Polynomial(self.ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj, self.obj_a_i(j))
        res.repr = self.repr
        return res

    def get_b_poly(self) -> Polynomial:
        res = Polynomial(self.ring)
        self.ring.lib.polynomial_copy_RNS_polynomial(res.obj, self.obj_b())
        res.repr = self.repr
        return res

    def get_a_digit(self, j: int, i: int) -> Polynomial:
        res = Polynomial(self.ring)
        # polynomial_RNSc_mod_reduce(out, in); reduces to the base RNS limb.
        lib_rlwe.lib.polynomial_RNSc_mod_reduce(res.obj, self.obj_a_i(j))
        res.repr = repr.coeff
        return res

    def get_b_digit(self, i: int) -> Polynomial:
        res = Polynomial(self.ring)
        lib_rlwe.lib.polynomial_RNSc_mod_reduce(res.obj, self.obj_b())
        res.repr = repr.coeff
        return res

    def obj_a_i(self, j):
        # self.obj points to RNS_MLWE { a, b, r }; a is an array of RNS_Polynomial
        return ffi.cast("RNS_MLWE", self.obj).a[j]

    def obj_b(self):
        # b is the second member of RNS_MLWE
        return ffi.cast("RNS_MLWE", self.obj).b

    def copy(self):
        res = self.new_like()
        lib_rlwe.lib.mlwe_copy_RNS_sample(res.obj, self.obj)
        res.repr = self.repr
        return res

    def copy_from(self, other: MLWE):
        lib_rlwe.lib.mlwe_copy_RNS_sample(self.obj, other.obj)
        self.repr = other.repr

    def round_division(
        self: CtT, ring: Ring | None = None, lvl: int | None = None
    ) -> CtT:
        """Round-divide the ciphertext down into a smaller (quotient) ring.

        The destination is given either as ``ring`` or as a level index ``lvl``
        into ``scheme.rings`` (exactly one must be supplied). The dropped primes
        are those in the current ring but not the destination. Afterwards
        ``self.lvl`` is set to the destination's index in ``scheme.rings`` (its
        level), not its prime count.
        """
        assert (ring is None) != (lvl is None), "provide exactly one of ring or lvl"
        if lvl is None:
            assert ring is not None
            lvl = self.scheme.level_of_ring(ring)
        else:
            ring = self.scheme.rings[lvl]
        assert ring.is_quotient_ring(self.ring), (
            "destination must be a quotient of the current ring"
        )
        self.to_coeff()
        divide_mask = self.ring.mask ^ ring.mask
        lib_rlwe.lib.mlwe_round_division_RNSc(self.obj, divide_mask)
        self.lvl = lvl
        self.ring = ring
        return self

    def __copy__(self):
        return self.copy()

    def _base_extend_cond(self, other: Polynomial):
        if other.ring != self.ring:
            res = other.base_extend(self.ring)
            res.to_NTT()
            return res
        return other

    def __add__(self, other):
        if type(other) is int:
            if other == 0:
                return self.copy()
            else:
                raise NotImplementedError
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = self.new_like()
        if isinstance(other, MLWE):
            res.add_MLWE(self, other)
        if type(other) is Polynomial:
            res.add_poly(self, other)
        return res

    def __iadd__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            else:
                raise NotImplementedError
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if isinstance(other, MLWE):
            self.add_MLWE(self, other)
        if type(other) is Polynomial:
            self.add_poly(self, other)
        return self

    def __sub__(self, other) -> MLWE:
        if type(other) is int:
            if other == 0:
                return self.copy()
            else:
                raise NotImplementedError
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        res = self.new_like()
        if isinstance(other, MLWE):
            res.sub_MLWE(self, other)
        if type(other) is Polynomial:
            res.sub_poly(self, other)
        return res

    def __isub__(self, other):
        if type(other) is int:
            if other == 0:
                return self
            else:
                raise NotImplementedError
        if self.repr != other.repr:
            self.to_NTT()
            other.to_NTT()
        if isinstance(other, MLWE):
            self.sub_MLWE(self, other)
        if type(other) is Polynomial:
            self.sub_poly(self, other)
        return self

    def __mul__(self, other) -> MLWE:
        if type(other) is Polynomial:
            res = self.new_like()
            if other.ring != self.ring:
                assert other.ring.is_quotient_ring(self.ring)
                other_sr = other.base_extend(self.ring)
            else:
                other_sr = other
            self.to_NTT()
            other_sr.to_NTT()
            res.multiply_poly(self, other_sr)
        elif type(other).__name__ == "MGSW":
            return other * self
        elif isinstance(other, MLWE):
            assert self.scheme.rlk is not None, (
                "Relinearization key (rlk) must be set in the scheme"
            )
            return self.scheme.multiply(self, other, self.scheme.rlk)
        else:  # assuming other is a pointer to int
            res = self.new_like()
            res.multiply_scalar(self, other)
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


MLWE_Scheme.ciphertext_type = MLWE
