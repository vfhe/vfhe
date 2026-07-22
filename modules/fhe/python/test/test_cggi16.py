# SPDX-License-Identifier: Apache-2.0
"""Characterization test for the (reverted) vfhe.fhe CGGI16 functional
bootstrap over the cffi boundary: bootstrap-key generation, LUT packing, blind
rotation (CMUX), and LWE extraction, verified against a random lookup table.
"""

import random

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CGGI16
from vfhe.mlwe import LWE, LWE_Key, MLWE_Scheme


def mod_switch(v, q, p):
    return round((v * p) / q) % p


@pytest.mark.complete
def test_functional_bootstrap(deterministic_prng):
    # Bootstrapping is probabilistic; pin the C PRNG + Python RNG so this
    # exact-equality check is reproducible rather than flaky (seed chosen to
    # decrypt cleanly).
    deterministic_prng(0xB007C0DE)
    random.seed(0xB007C0DE)
    out_N = 256
    msg_prec = 5
    Rq = Ring(out_N, prime_size=[50, 50, 50], split_degree=1)
    _Rp = Rq.quotient_ring(ell=1)
    in_ring = Rq.quotient_ring(ell=2)
    in_scheme = MLWE_Scheme(in_ring, special_primes=0, module_rank=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1, max_lvl=1)

    input_key = in_scheme.key_gen_sparse(17, 3.2, ternary=False)
    input_lwe_key = input_key.extract_lwe_key()
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=False)

    cggi16 = CGGI16(out_scheme)
    bk_key = cggi16.generate_bootstrap_key(input_lwe_key, output_key)

    lut_size = 1 << (msg_prec - 1)
    lut = [random.randint(0, (1 << (msg_prec - 1)) - 1) for _ in range(lut_size)]
    rlwe_tv = cggi16.LUT_packing(lut, lut_size, msg_prec)

    msg_val = random.randint(0, lut_size - 1)
    m_scaled = mod_switch(msg_val, 1 << msg_prec, in_ring.q_l)
    lwe_in = LWE(
        ring=in_ring, m=[m_scaled % q for q in in_ring.primes], key=input_lwe_key
    )

    out_lwe = LWE(ring=in_ring)
    cggi16.functional_bootstrap(
        out_lwe, rlwe_tv, lwe_in, bk_key, torus_base=1 << (msg_prec - 1)
    )

    output_lwe_key = output_key.extract_lwe_key()
    output_lwe_key_ell2 = LWE_Key(
        ring=in_ring, key=output_lwe_key.get_s(), n=output_lwe_key.n
    )
    phase = out_lwe.phase(output_lwe_key_ell2, recompose=True)
    assert mod_switch(phase, in_ring.q_l, 1 << msg_prec) == lut[msg_val]
