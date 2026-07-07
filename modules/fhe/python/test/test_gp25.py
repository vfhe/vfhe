# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.fhe GP25 sparse-amortized
bootstrap over the cffi boundary.

`test_lwe_extraction` and `test_packing_ksk` isolate the cheap C paths
(mlwe_extract_LWE, gen_packing_ksk + mlwe_full_packing_keyswitch). `test_sab`
runs a full ternary bootstrap, exercising gp25_RGSW_monomial_mul and
gp25_sub_a_mt end to end.
"""

import math
import random

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import GP25
from vfhe.mlwe import LWE, MLWE, LWE_Key, MLWE_Scheme


def mod_switch(v, q, p):
    return round((v * p) / q) % p


def get_min_prec(key):
    N = key.scheme.N
    r_max = 0
    for i in range(key.scheme.r):
        key_poly = key.key[i]
        previous = N
        for j in range(N):
            coeff = key_poly[N - j - 1]
            if coeff != 0:
                r_max = max(r_max, previous - (N - j - 1))
                previous = N - j - 1
        r_max = max(r_max, previous)
    return int(math.floor(math.log2(r_max))) + 1


def rs_sparse_ternary_key(scheme, h, sigma, target_r_prec):
    for _ in range(1 << 15):
        key = scheme.key_gen_sparse(h, sigma, ternary=True)
        r_prec = get_min_prec(key)
        if r_prec <= target_r_prec:
            return key, r_prec
    raise RuntimeError("rejection sampling limit")


def test_lwe_extraction():
    N = 256
    Rq = Ring(N, prime_size=[50, 50, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=1)
    key = scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(scheme)
    lwe_key = gp25.extract_lwe_key(key)

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = gp25.rlwe_extract_lwe(rlwe_sample, idx)
        phase = lwe_sample.phase(lwe_key, recompose=True)
        res = mod_switch(phase, Rq.q_l, Rp.primes[0])
        diff = (res - msg_coeffs[idx]) % Rp.primes[0]
        diff = min(diff, Rp.primes[0] - diff)
        assert diff < 1000


def test_packing_ksk():
    in_N = out_N = 256
    Rq = Ring(out_N, prime_size=[50, 50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=4)
    lwe_key = LWE_Key(ring=Rq, sec_sigma=3.2, err_sigma=3.2, n=in_N)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(out_scheme)
    packing_key = gp25.gen_packing_ksk(output_key, lwe_key, 0)

    extracted = []
    for i in range(in_N):
        m_i = mod_switch((i * 137), 2000, Rq.q_l)
        extracted.append(LWE(ring=Rq, m=[m_i % q for q in Rq.primes], key=lwe_key))

    out_repacked = gp25.packing_keyswitch(extracted, packing_key, 0)
    out_coeffs = out_scheme.phase(out_repacked, output_key).get_polynomial()
    for i in range(in_N):
        m_i = mod_switch(out_coeffs[i], Rq.q_l, 2000)
        diff = (m_i - (i * 137) % 2000) % 2000
        diff = min(diff, 2000 - diff)
        assert diff <= 5


@pytest.mark.complete
def test_sab():
    in_N = out_N = 256
    msg_prec = 5
    Rq = Ring(out_N, prime_size=[50, 50, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    in_ring = Rq.quotient_ring(ell=2)
    in_scheme = MLWE_Scheme(in_ring, special_primes=0, module_rank=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=4, max_lvl=1)

    input_key, r_prec = rs_sparse_ternary_key(in_scheme, 17, 3.2, 8)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(out_scheme, threads=1)
    sab_key = gp25.generate_sparse_ternary_key(input_key, output_key, 17, r_prec)
    sab_key.b_prec = msg_prec

    lut = [
        random.randint(0, (1 << (msg_prec - 1)) - 1) for _ in range(1 << (msg_prec - 1))
    ]
    rlwe_tv = gp25.sab_LUT_packing(lut, 1 << (msg_prec - 1), msg_prec)

    poly_in = [random.randint(0, (1 << (msg_prec - 1)) - 1) for _ in range(in_N)]
    poly_scaled = [mod_switch(i, 1 << msg_prec, Rp.q_l) for i in poly_in]
    delta = in_ring.modulus_ratio(Rp, return_pointer=True)
    rlwe_in = in_scheme.sample(
        Polynomial(Rp).from_array(poly_scaled).scaled_lift(in_ring, delta=delta),
        input_key,
    )

    out_bootstrap = MLWE(out_scheme)
    gp25.sab_rlwe_bootstrap(out_bootstrap, rlwe_in, rlwe_tv, sab_key)

    out_coeffs = (
        out_scheme.phase(out_bootstrap, output_key).round_division(Rp).get_polynomial()
    )
    for i in range(in_N):
        assert mod_switch(out_coeffs[i], Rp.q_l, 1 << msg_prec) == lut[poly_in[i]]
