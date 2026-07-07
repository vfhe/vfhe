from __future__ import annotations

import math

from vfhe.arith.number_theory import crt
from vfhe.arith.polynomial import Polynomial, repr
from vfhe.mlwe.lwe import LWE, LWE_Key, lib_lwe
from vfhe.mlwe.mgsw import CMUX, MGSW_Scheme
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, lib_rlwe


class CGGI16_Key:
    def __init__(self):
        self.bk = []  # list[MGSW]
        self.b_prec = 0


class CGGI16:
    def __init__(self, scheme: MLWE_Scheme, gsw_ell: "int|None" = None):
        self.scheme = scheme
        self.mgsw_scheme = MGSW_Scheme(scheme, ell=gsw_ell)
        self.ring = scheme.ring

    def generate_bootstrap_key(
        self, input_key: MLWE_Key | LWE_Key, output_key: MLWE_Key
    ) -> CGGI16_Key:
        bk = CGGI16_Key()
        if isinstance(input_key, MLWE_Key):
            lwe_key = input_key.extract_lwe_key()
        else:
            lwe_key = input_key

        s = lwe_key.get_s()
        n = lwe_key.n

        for i in range(n):
            poly = Polynomial(self.ring).from_array([s[i]] + [0] * (self.ring.N - 1))
            mgsw = self.mgsw_scheme.encrypt(poly, output_key)
            mgsw.to_NTT()
            bk.bk.append(mgsw)

        return bk

    def functional_bootstrap_wo_extract(
        self, out: MLWE, tv: MLWE, rlwe_in: LWE, bk: CGGI16_Key, torus_base: int
    ):
        # 1. Setup dimensions
        N = self.ring.N
        n = rlwe_in.n
        q_lwe = math.prod(rlwe_in.ring.primes[: rlwe_in.l])

        # Get b and a values of input LWE
        in_b_limbs = rlwe_in.get_b()
        b_val = crt(in_b_limbs, rlwe_in.ring.primes)

        in_a_limbs = rlwe_in.get_a()
        a_vals = [
            crt([in_a_limbs[j][i] for j in range(rlwe_in.l)], rlwe_in.ring.primes)
            for i in range(n)
        ]

        # 2. Phase offset / initial rotation amount
        prec_offset = q_lwe // (4 * torus_base)
        rot_b = round(((b_val + prec_offset) * (2 * N)) / q_lwe) % (2 * N)

        # Initial rotation: rotated_tv = tv * X^{-rot_b} = tv * X^{2N - rot_b}
        rot_tv = (2 * N - rot_b) % (2 * N)

        # TRLWE rotated_tv: starts with tv rotated
        out.to_coeff()
        tv.to_coeff()
        self.ring.lib.mlwe_RNSc_mul_by_xai(out.obj, tv.obj, rot_tv)
        out.repr = repr.coeff

        # 3. Blind rotate
        for i in range(n):
            a_i_val = round((a_vals[i] * (2 * N)) / q_lwe) % (2 * N)
            if a_i_val == 0:
                continue
            # print(f"DEBUG: Blind rotate i={i}, a_i_val={a_i_val}")

            out.to_coeff()
            out_rotated = MLWE(self.scheme)
            self.ring.lib.mlwe_RNSc_mul_by_xai(out_rotated.obj, out.obj, a_i_val)
            out_rotated.repr = repr.coeff

            res = CMUX(out, out_rotated, bk.bk[i])
            out.copy_from(res)

    def LUT_packing(self, lut: list[int], size: int, LUT_prec: int):
        rlwe_tv = MLWE(self.scheme)
        N = self.ring.N
        ell = self.ring.ell
        quotient_ring = self.ring.quotient_ring(ell=ell)
        q = quotient_ring.q_l
        normal_ring = self.scheme.rings[0]

        scaled_lut = [(val * q) // (1 << LUT_prec) for val in lut]

        coeffs = [0] * N
        for i in range(N):
            idx = i // (N // size)
            if idx < len(scaled_lut):
                coeffs[i] = scaled_lut[idx]

        poly = Polynomial(normal_ring).from_bigint_array(coeffs)
        lib_rlwe.lib.mlwe_RNS_trivial_sample_of_zero(rlwe_tv.obj)
        rlwe_tv.repr = repr.ntt

        rlwe_tv += poly
        return rlwe_tv

    def functional_bootstrap(
        self, out: LWE, tv: MLWE, rlwe_in: LWE, bk: CGGI16_Key, torus_base: int
    ):
        # void functional_bootstrap(TLWE out, TRLWE tv, TLWE in, Bootstrap_Key key, int torus_base)
        rotated_tv = MLWE(self.scheme)
        self.functional_bootstrap_wo_extract(rotated_tv, tv, rlwe_in, bk, torus_base)

        rotated_tv.to_coeff()
        lwe_obj = self.ring.lib.mlwe_extract_LWE(rotated_tv.obj, 0)

        if out.obj is not None:
            lib_lwe.lib.free_lwe_sample(out.obj)
        out.obj = lwe_obj
        out.n = self.scheme.r * self.ring.N
