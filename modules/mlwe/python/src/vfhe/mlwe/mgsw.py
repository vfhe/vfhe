# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vfhe.arith import Polynomial, RNSPolynomial, repr
from vfhe.engine import ffi, lib

from .mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set


# RNS MGSW scheme (similar to RGSW)
class MGSW_Scheme:
    def __init__(
        self,
        MLWE_scheme: MLWE_Scheme,
        ell: int | None = None,
        radix_log_base: int | None = None,
        balanced: bool | None = None,
    ):
        """MGSW over ``MLWE_scheme``, encrypting against one of the two gadgets.

        ``ell`` is how many of the ring's primes the gadget covers (all of them
        by default). ``radix_log_base`` selects the radix gadget over the RNS
        one -- each prime contributing base-``2^radix_log_base`` digits instead
        of a single residue -- which bounds the products an external product
        accumulates by the radix rather than by the primes. See
        :meth:`MLWE_Scheme.gadget_scalars`. ``balanced`` picks the RNS gadget's
        digit as in :class:`MLWE_Scheme`, and defaults to ``MLWE_scheme``'s.
        """
        self.mlwe_scheme = MLWE_scheme
        self.ell = ell if ell else MLWE_scheme.rings[0].ell
        self.ring = MLWE_scheme.special_rings[0]
        self.radix_log_base = radix_log_base
        self.balanced = MLWE_scheme.balanced if balanced is None else balanced

    def gadget_scalars(self, lvl: int = 0) -> list[list[int]]:
        """The gadget elements, as the per-prime scaling vectors to encrypt.

        The key lives in ``self.ring`` whatever the level; only how far the
        gadget reaches and what the external product's rescale divides by
        follow ``lvl``, so the first ``ell - lvl`` primes carry it.
        """
        return self.mlwe_scheme.gadget_scalars(
            lvl, self.radix_log_base, ring=self.ring, primes=self.ell - lvl
        )

    def encrypt(
        self, msg: RNSPolynomial, key: MLWE_Key, lvl: int = 0, n_threads: int = 0
    ):
        """An MGSW encryption of ``msg``: for each component ``s_j`` of the key
        one encryption of ``-s_j * msg`` per gadget element, then the same for
        ``msg`` itself. Drawn on up to ``n_threads`` threads (0: the library
        limit); see :meth:`MLWE_Scheme.sample_scaled`.
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

        What a bootstrapping key is made of. A constant folds into the gadget
        scalars -- ``s_j * c`` times ``g`` is ``s_j`` times ``c * g`` -- so the
        messages are the same for every value and only the scalars differ, and
        the whole key is one draw on up to ``n_threads`` threads (0: the
        library limit).
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
            self.scheme.balanced,
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
        selector.scheme.balanced,
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
        selector.scheme.balanced,
    )
    res.repr = repr.ntt
    return res
