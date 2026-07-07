from __future__ import annotations

from vfhe.arith.polynomial import Polynomial, repr
from vfhe.misc.libvfhe import ffi, lib

from .mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set


# RNS MGSW scheme (similar to RGSW)
class MGSW_Scheme:
    def __init__(self, MLWE_scheme: MLWE_Scheme, ell: "int|None" = None):
        self.mlwe_scheme = MLWE_scheme
        self.ell = ell if ell else MLWE_scheme.rings[0].ell
        self.ring = MLWE_scheme.special_rings[0]

    def encrypt(self, msg: Polynomial, key: MLWE_Key, lvl: int = 0):
        result = []
        special_q = self.ring.modulus_ratio(self.mlwe_scheme.rings[lvl])
        # Base extend msg to self.ring if needed
        if msg.ring != self.ring:
            msg = msg.base_extend(self.ring)

        key_special = MLWE_Key(key.key, key.sigma_err, self.mlwe_scheme, ring=self.ring)

        # MGSW ciphertext is a matrix of MLWE ciphertexts
        # For each component of the secret key s_j (j=0..r-1) and for each digit i=0..ell-1
        # We encrypt s_j * msg * Q/q_i
        for j in range(self.mlwe_scheme.r):
            sm = -key.poly[j] * msg
            if sm.ring != self.ring:
                sm = sm.base_extend(self.ring)
            for i in range(self.ell - lvl):
                scaling_factor = (
                    [0] * i
                    + [special_q % self.ring.primes[i]]
                    + [0] * (self.ring.ell - 1 - i)
                )
                out = MLWE(self.mlwe_scheme, ring=self.ring)
                self.mlwe_scheme.sample(sm * scaling_factor, key_special, out=out)
                result.append(out)

        # Finally, for each digit i=0..ell-1, we encrypt msg * Q/q_i
        for i in range(self.ell - lvl):
            scaling_factor = (
                [0] * i
                + [special_q % self.ring.primes[i]]
                + [0] * (self.ring.ell - 1 - i)
            )
            out = MLWE(self.mlwe_scheme, ring=self.ring)
            self.mlwe_scheme.sample(msg * scaling_factor, key_special, out=out)
            result.append(out)

        return MGSW(self, obj=result)


class MGSW:
    def __init__(self, scheme: MGSW_Scheme, obj: "list[MLWE]|None" = None):
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

    def external_product(self, other: MLWE) -> MLWE:
        res = MLWE(self.scheme.mlwe_scheme)
        self.to_NTT()
        other.to_coeff()
        ell = self.scheme.ell

        # Pass the array of RNS_MLWE handles (self.obj) to C
        mgsw_ptr_array = ffi.new("void*[]", [c.obj for c in self.obj])

        lib.mgsw_external_product(
            res.obj,
            mgsw_ptr_array,
            other.obj,
            ell,
            self.scheme.mlwe_scheme.special_primes,
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
        selector.scheme.ell,
        in1.scheme.special_primes,
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
        selector.scheme.ell,
        in1.scheme.special_primes,
    )
    res.repr = repr.ntt
    return res
