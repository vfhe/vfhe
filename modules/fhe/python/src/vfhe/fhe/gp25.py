from __future__ import annotations

import math
from math import log2

from vfhe.arith.polynomial import Polynomial, repr
from vfhe.misc.libvfhe import ffi, lib
from vfhe.mlwe.lwe import LWE, LWE_Key
from vfhe.mlwe.mgsw import CMUX, MGSW, NCMUX, MGSW_Scheme
from vfhe.mlwe.mlwe import MLWE, MLWE_Key, MLWE_Scheme, MLWE_Set, lib_rlwe


def mod_switch(v, q, p):
    return round((v * p) / q) % p


class SAB_Key:
    def __init__(self):
        self.s = []  # list[list[list[MGSW]]] -> a_idx -> h -> r_prec -> MGSW
        self.s_sign = []  # list[list[MGSW]] -> a_idx -> h -> MGSW
        self.packing_key: "MLWE_Set | None" = None
        self.hw_reducing_key: "MLWE_Set | None" = None
        self.trace_repack_key: "MLWE_Set | list | None" = None
        self.h = 0
        self.b_prec = 0
        self.r_prec = 0


class GP25:
    def __init__(
        self,
        scheme: MLWE_Scheme,
        gsw_ell: "int|None" = None,
        debug_key: "MLWE_Key|None" = None,
        threads: int = 1,
        trace_repack: bool = False,
    ):
        self.scheme = scheme
        self.mgsw_scheme = MGSW_Scheme(scheme, ell=gsw_ell)
        self.ring = scheme.rings[0]
        self.aut_minus1 = None  # generate when genering sab
        self.debug_key = debug_key
        self.threads = threads
        self.trace_repack = trace_repack

    def generate_sparse_ternary_key(
        self, input_key: MLWE_Key, output_key: MLWE_Key, h: int, r_prec: int
    ) -> SAB_Key:
        sab = SAB_Key()
        sab.h = h
        sab.r_prec = r_prec

        in_N = input_key.scheme.ring.N
        in_r = input_key.scheme.r
        r_max = 1 << r_prec
        cnt_h = 0

        for i in range(in_r):
            sab.s.append([])
            sab.s_sign.append([])

            key_poly = input_key.key[i]
            previous = in_N

            for j in range(in_N):
                idx = in_N - j - 1
                coeff = key_poly[idx]

                if coeff != 0:
                    is_negative = 1 if coeff == -1 else 0
                    current = idx
                    r_diff = previous - current

                    assert r_diff < r_max
                    sab.s[i].append(self.encrypt_bits(r_diff, r_prec, output_key))
                    sign_poly = Polynomial(self.ring).from_array(
                        [is_negative] + [0] * (self.ring.N - 1)
                    )
                    sab.s_sign[i].append(
                        self.mgsw_scheme.encrypt(sign_poly, output_key)
                    )
                    sab.s_sign[i][-1].to_NTT()

                    previous = current
                    cnt_h += 1

            assert cnt_h == h
            assert previous < r_max
            sab.s[i].append(self.encrypt_bits(previous, r_prec, output_key))

        ## generate additional keys
        extracted_lwe_key = self.extract_lwe_key(output_key)
        self.aut_minus1 = self.scheme.gen_ksk_automorphism(
            output_key, output_key, 2 * self.ring.N - 1
        )

        if self.trace_repack:
            log_N = int(math.log2(self.ring.N))
            gens = [(1 << j) + 1 for j in range(1, log_N + 1)]
            sab.trace_repack_key = self.scheme.gen_ksk_trace(
                output_key, output_key, gens=gens
            )
        else:
            lvl = 0
            sab.packing_key = self.gen_packing_ksk(output_key, extracted_lwe_key, lvl)
            sab.trace_repack_key = None

        return sab

    def encrypt_bits(self, val: int, bits: int, output_key: MLWE_Key) -> list[MGSW]:
        res = []
        for i in range(bits):
            bit = (val >> i) & 1
            poly = Polynomial(self.ring).from_array([bit] + [0] * (self.ring.N - 1))
            res.append(self.mgsw_scheme.encrypt(poly, output_key))
            res[-1].to_NTT()
        return res

    def sab_rlwe_bootstrap(
        self,
        out: MLWE,
        rlwe_in: MLWE,
        tv: MLWE,
        sab_key: SAB_Key,
        in_modulus: "int|None" = None,
    ):
        """
        Implements the Sparse Amortized Bootstrap (SAB) for RLWE.
        """
        rlwe_in.to_coeff()
        b_poly = rlwe_in.get_b_poly()

        # 1. Setup tv_xb
        acc = self.setup_tv_xb(b_poly, tv, sab_key.b_prec, q=in_modulus)

        # 2. Blind rotate
        self.sab_blind_rotate(acc, rlwe_in, sab_key, q=in_modulus)

        if self.trace_repack:
            # 4. Repacking keyswitch with full_packing_keyswitch_scaled
            assert sab_key.trace_repack_key is not None
            out_repacked = self.scheme.full_packing_keyswitch_scaled(
                acc, sab_key.trace_repack_key
            )
        else:
            # 3. Extract LWE samples from the RLWE samples in acc
            extracted = []
            for i in range(self.ring.N):
                lwe = self.rlwe_extract_lwe(acc[i], 0)
                extracted.append(lwe)

            # 4. Repacking keyswitch
            # out_repacked is an MLWE sample
            lvl = 0
            assert sab_key.packing_key is not None
            out_repacked = self.packing_keyswitch(extracted, sab_key.packing_key, lvl)

        # 5. HW reducing keyswitch
        if sab_key.hw_reducing_key is not None:
            out.copy_from(self.scheme.keyswitch(out_repacked, sab_key.hw_reducing_key))
        else:
            out.copy_from(out_repacked)

    def get_noise(self, c: MLWE, b_prec: int, key: MLWE_Key):
        phase = self.scheme.phase(c, key)
        phase.to_coeff()
        rns = phase.get_coeff_matrix()
        ell = self.ring.ell - self.scheme.special_primes
        primes = self.ring.primes[:ell]
        q = math.prod(primes)
        if ell == 1:
            coeffs = rns[0]
        else:
            from vfhe.arith.number_theory import crt

            coeffs = [
                crt([rns[i][j] for i in range(ell)], primes) for j in range(self.ring.N)
            ]
        msg = [mod_switch(i, q, 1 << b_prec) for i in coeffs]
        unsigned = [
            (coeffs[i] - mod_switch(msg[i], 1 << b_prec, q)) % q
            for i in range(self.ring.N)
        ]
        return [min(i, (-(q - i)), key=lambda x: abs(x)) for i in unsigned]

    def get_noise_stderr(self, c: MLWE, b_prec: int, key: MLWE_Key):
        noise = self.get_noise(c, b_prec, key)
        stderr = math.sqrt(sum([i**2 for i in noise]) / len(noise))
        return log2(stderr) if stderr != 0 else 0

    def extract_lwe_key(self, mlwe_key: MLWE_Key) -> LWE_Key:
        return mlwe_key.extract_lwe_key()

    def gen_packing_ksk(
        self, key_out: MLWE_Key, lwe_key: LWE_Key, lvl: int
    ) -> MLWE_Set:
        s = lwe_key.get_s()
        n = lwe_key.n

        result = []
        special_ring = self.scheme.special_rings[lvl]
        normal_ring = self.scheme.rings[lvl]
        special_q = special_ring.modulus_ratio(normal_ring)

        key_out_special = MLWE_Key(
            key_out.key, key_out.sigma_err, self.scheme, ring=special_ring
        )

        for i in range(n):
            res_i = []
            si_poly = Polynomial(special_ring).from_array(
                [s[i]] + [0] * (self.ring.N - 1)
            )
            for j in range(special_ring.ell):
                scaling_factor = (
                    [0] * j
                    + [special_q % special_ring.primes[j]]
                    + [0] * (special_ring.ell - 1 - j)
                )
                out = MLWE(self.scheme, lvl=lvl, ring=special_ring)
                self.scheme.sample(si_poly * scaling_factor, key_out_special, out=out)
                res_i.append(out)
            result.append(res_i)
        return MLWE_Set(result)

    def sab_LUT_packing(self, lut: list[int], size: int, LUT_prec: int):
        rlwe_tv = MLWE(self.scheme)
        N = self.ring.N
        ell = self.ring.ell
        quotient_ring = self.ring.quotient_ring(ell=ell)
        q = quotient_ring.q_l
        normal_ring = self.scheme.rings[0]

        if self.trace_repack:
            inv_N = pow(N, -1, q)
            scaled_lut = [(val * q) // (1 << LUT_prec) for val in lut]
            scaled_lut = [(i * inv_N) % q for i in scaled_lut]
        else:
            # Scale LUT entries
            scaled_lut = [(val * q) // (1 << LUT_prec) for val in lut]

        coeffs = [0] * N
        coeffs[0] = scaled_lut[0]
        for i in range(1, N):
            idx = i // (N // size)
            if idx < len(scaled_lut):
                val = scaled_lut[idx]
                coeffs[N - i] = (q - val) % q

        poly = Polynomial(normal_ring).from_bigint_array(coeffs)
        # Trivial zero sample
        lib_rlwe.lib.mlwe_RNS_trivial_sample_of_zero(rlwe_tv.obj)
        rlwe_tv.repr = repr.ntt

        rlwe_tv += poly
        return rlwe_tv

    def rlwe_extract_lwe(self, rlwe: MLWE, idx: int) -> LWE:
        # Needs coefficient representation
        rlwe.to_coeff()
        lwe_obj = lib_rlwe.lib.mlwe_extract_LWE(rlwe.obj, idx)
        return LWE(ring=rlwe.ring, obj=lwe_obj)

    def packing_keyswitch(
        self, extracted: list[LWE], packing_key: MLWE_Set, lvl: int
    ) -> MLWE:
        res = MLWE(self.scheme, lvl=lvl, ring=self.scheme.rings[lvl])

        # Pass the array of LWE handles to C. packing_key.obj is the matrix of
        # MLWE handles (ksk).
        lwe_ptr_array = ffi.new("void*[]", [c.obj for c in extracted])

        lib_rlwe.lib.mlwe_full_packing_keyswitch(
            res.obj, lwe_ptr_array, len(extracted), packing_key.obj, lvl
        )
        res.repr = repr.ntt
        res.ring = self.scheme.rings[lvl]
        return res

    def setup_tv_xb(
        self, b: Polynomial, tv: MLWE, b_prec: int, q: "int|None" = None
    ) -> list[MLWE]:
        acc = []
        log_N2 = int(math.log2(2 * self.ring.N))
        prec_offset = 1 << (log_N2 - b_prec - 1)

        # Assuming b has a method to get coefficient array
        b_coeffs = b.get_polynomial()
        if q is None:  # default: the input ring modulus
            q = b.ring.q_l

        for i in range(self.ring.N):
            b_val = b_coeffs[i]
            rot = (round(b_val * (2 * self.ring.N) / q) + prec_offset) % (
                2 * self.ring.N
            )

            rotated = self.mul_by_xai(tv, rot)
            acc.append(rotated)
        return acc

    def sab_blind_rotate(
        self, acc: list[MLWE], rlwe_in: MLWE, sab_key: SAB_Key, q: "int|None" = None
    ):
        N = self.ring.N
        r = rlwe_in.scheme.r
        if q is None:  # default: the input ring modulus
            q = rlwe_in.scheme.ring.q_l

        for j in range(r):
            a_poly = rlwe_in.get_a_poly(j)
            a_coeffs = a_poly.get_polynomial()

            # Mod switch a -> 2N (round(a * 2N / q))
            a_mod_switched = [
                round(a_val * (2 * N) / q) % (2 * N) for a_val in a_coeffs
            ]

            self.sparse_mul(acc, a_mod_switched, j, sab_key)

    def sparse_mul(self, p: list[MLWE], a: list[int], a_idx: int, sab_key: SAB_Key):
        for i in range(sab_key.h):
            self.RGSW_monomial_mul(p, sab_key.s[a_idx][i])
            self.sub_a(p, a, i, a_idx, sab_key)
        self.RGSW_monomial_mul(p, sab_key.s[a_idx][sab_key.h])

    def RGSW_monomial_mul(self, p0: list[MLWE], e: list[MGSW]):
        # r_prec = len(e)
        # in_N = len(p0)
        #
        # p = [p0, [MLWE(self.scheme) for _ in range(in_N)]]
        #
        # for i in range(r_prec):
        #     power = 1 << i
        #     out_idx = (i + 1) & 1
        #     in_idx = out_idx ^ 1
        #
        #     for j in range(power):
        #         p[out_idx][j] = self.NCMUX(p[in_idx][j], p[in_idx][in_N - power + j], e[i])
        #
        #     for j in range(in_N - power):
        #         p[out_idx][j + power] = self.CMUX(p[in_idx][j + power], p[in_idx][j], e[i])
        #
        # if (r_prec & 1) == 1:
        #     for i in range(in_N):
        #         p0[i].copy_from(p[1][i])

        for c in p0:
            c.to_coeff()

        in_N = len(p0)
        p0_ptr_array = ffi.new("void*[]", [c.obj for c in p0])

        # e is a matrix of MGSW handle arrays: build each row, then an array of
        # row pointers (kept alive in c_mgsw_arrays through the C call).
        c_mgsw_arrays = []
        for mgsw in e:
            mgsw.to_NTT()
            c_mgsw_arrays.append(ffi.new("void*[]", [c.obj for c in mgsw.obj]))
        e_ptr_array = ffi.new(
            "void*[]", [ffi.cast("void *", arr) for arr in c_mgsw_arrays]
        )

        assert self.aut_minus1 is not None
        aut_minus1 = (
            self.aut_minus1
            if isinstance(self.aut_minus1, MLWE_Set)
            else self.aut_minus1[e[0].obj[0].lvl]
        )
        if self.threads > 1:
            lib.gp25_RGSW_monomial_mul_mt(
                p0_ptr_array,
                in_N,
                e_ptr_array,
                len(e),
                aut_minus1.obj,
                e[0].scheme.ell,
                self.scheme.special_primes,
                self.threads,
            )
        else:
            lib.gp25_RGSW_monomial_mul(
                p0_ptr_array,
                in_N,
                e_ptr_array,
                len(e),
                aut_minus1.obj,
                e[0].scheme.ell,
                self.scheme.special_primes,
            )

        for c in p0:
            c.repr = repr.coeff

    def CMUX(self, in1: MLWE, in2: MLWE, selector: MGSW):
        # return selector*(in2 - in1) + in1
        return CMUX(in1, in2, selector)

    def NCMUX(self, in1: MLWE, in2: MLWE, selector: MGSW):
        # tmp = self.scheme.automorphism(in2, 2 * self.ring.N - 1, self.aut_minus1)
        # return self.CMUX(in1, tmp, selector)
        assert self.aut_minus1 is not None
        return NCMUX(in1, in2, selector, self.aut_minus1)

    def sub_a(
        self, p: list[MLWE], a: list[int], key_idx: int, a_idx: int, sab_key: SAB_Key
    ):
        # Per coefficient k (all independent): p[k] = p[k]*X^a[k] + s_sign (X) (p[k]*X^a[k]*(X^{-2a}-1)).
        # Done in one multithreaded C kernel (gp25_sub_a_mt) instead of a serial Python loop of
        # 2048 external products. (-2a) mod 2N is recomputed inside C from a[k].
        for c in p:
            c.to_coeff()
        in_N = len(p)
        s_sign = sab_key.s_sign[a_idx][key_idx]
        s_sign.to_NTT()
        lib.gp25_sub_a_mt(
            ffi.new("void*[]", [c.obj for c in p]),
            in_N,
            ffi.new("uint64_t[]", [int(x) for x in a]),
            ffi.new("void*[]", [c.obj for c in s_sign.obj]),
            s_sign.scheme.ell,
            self.scheme.special_primes,
            self.ring.N,
            self.threads,
        )
        for c in p:
            c.repr = repr.coeff

    def mul_by_xai(self, in1: MLWE, a: int) -> MLWE:
        out = MLWE(self.scheme)
        in1.to_coeff()
        self.ring.lib.mlwe_RNSc_mul_by_xai(out.obj, in1.obj, a)
        out.repr = repr.coeff
        return out

    def mul_by_xai_minus_1(self, in1: MLWE, a: int) -> MLWE:
        out = MLWE(self.scheme)
        in1.to_coeff()
        self.ring.lib.mlwe_RNSc_mul_by_xai_minus1(out.obj, in1.obj, a)
        out.repr = repr.coeff
        return out
