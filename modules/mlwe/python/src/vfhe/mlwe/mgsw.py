# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vfhe.arith import RNSPolynomial, repr
from vfhe.engine import ffi, lib

from .mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set


# RNS MGSW scheme (similar to RGSW)
class MGSW_Scheme:
    def __init__(
        self,
        MLWE_scheme: MLWE_Scheme,
        ell: int | None = None,
        radix_log_base: int | None = None,
    ):
        """MGSW over ``MLWE_scheme``, encrypting against one of the two gadgets.

        ``ell`` is how many of the ring's primes the gadget covers (all of them
        by default). ``radix_log_base`` selects the radix gadget over the RNS
        one -- each prime contributing base-``2^radix_log_base`` digits instead
        of a single residue -- which bounds the products an external product
        accumulates by the radix rather than by the primes. See
        :meth:`MLWE_Scheme.gadget_scalars`.
        """
        self.mlwe_scheme = MLWE_scheme
        self.ell = ell if ell else MLWE_scheme.rings[0].ell
        self.ring = MLWE_scheme.special_rings[0]
        self.radix_log_base = radix_log_base

    def gadget_scalars(self, lvl: int = 0) -> list[list[int]]:
        """The gadget elements, as the per-prime scaling vectors to encrypt.

        The key lives in ``self.ring`` whatever the level; only how far the
        gadget reaches and what the external product's rescale divides by
        follow ``lvl``, so the first ``ell - lvl`` primes carry it.
        """
        return self.mlwe_scheme.gadget_scalars(
            lvl, self.radix_log_base, ring=self.ring, primes=self.ell - lvl
        )

    def encrypt(self, msg: RNSPolynomial, key: MLWE_Key, lvl: int = 0):
        result = []
        # Base extend msg to self.ring if needed
        if msg.ring != self.ring:
            msg = msg.base_extend(self.ring)

        key_special = MLWE_Key(key.key, key.sigma_err, self.mlwe_scheme, ring=self.ring)
        scalars = self.gadget_scalars(lvl)

        # MGSW ciphertext is a matrix of MLWE ciphertexts: for each component
        # of the secret key s_j (j=0..r-1), one encryption of s_j * msg per
        # gadget element, then the same for msg itself.
        for j in range(self.mlwe_scheme.r):
            sm = -key.poly[j] * msg
            if sm.ring != self.ring:
                sm = sm.base_extend(self.ring)
            for scaling_factor in scalars:
                out = MLWE(self.mlwe_scheme, ring=self.ring)
                self.mlwe_scheme.sample(sm * scaling_factor, key_special, out=out)
                result.append(out)

        for scaling_factor in scalars:
            out = MLWE(self.mlwe_scheme, ring=self.ring)
            self.mlwe_scheme.sample(msg * scaling_factor, key_special, out=out)
            result.append(out)

        return MGSW(self, obj=result)


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

    @property
    def gadget_size(self) -> int:
        """Gadget keys per component of the key: what C strides the array by.

        Read off the key itself rather than the scheme, since a key encrypted
        at a level carries the gadget for that level's primes only.
        """
        return len(self.obj) // (self.scheme.mlwe_scheme.r + 1)

    def external_product(self, other: MLWE) -> MLWE:
        res = MLWE(self.scheme.mlwe_scheme)
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
            self.scheme.radix_log_base or 0,
        )
        res.repr = repr.ntt

        return res

    def __mul__(self, other: MLWE) -> MLWE:
        if isinstance(other, MLWE):
            return self.external_product(other)
        return NotImplemented


def CMUX(in1: MLWE, in2: MLWE, selector: MGSW) -> MLWE:
    # Old Python implementation:
    # return selector*(in2 - in1) + in1

    res = MLWE(in1.scheme)
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
        selector.scheme.radix_log_base or 0,
    )
    res.repr = repr.ntt
    return res


def NCMUX(
    in1: MLWE, in2: MLWE, selector: MGSW, aut_minus1: MLWE_Set | list[MLWE_Set]
) -> MLWE:
    # Old Python implementation:
    # tmp = in2.scheme.automorphism(in2, 2 * in2.ring.N - 1, aut_minus1)
    # return CMUX(in1, tmp, selector)

    aut_minus1 = aut_minus1 if isinstance(aut_minus1, MLWE_Set) else aut_minus1[in1.lvl]
    res = MLWE(in1.scheme, lvl=in1.lvl)
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
        selector.scheme.radix_log_base or 0,
    )
    res.repr = repr.ntt
    return res
