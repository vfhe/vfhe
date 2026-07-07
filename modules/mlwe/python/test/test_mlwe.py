# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.mlwe over the cffi boundary.

Encrypt/decrypt roundtrips, homomorphic add/sub/mul-by-poly, BV and GHS
key-switching, automorphism, ciphertext multiplication (relinearization), the
MGSW external product, and the LWE surface. Noise is stripped with
round_division back to the plaintext ring, so equality is exact.
"""

import math
import secrets

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.mlwe import LWE, LWE_Key, MGSW_Scheme, MLWE_Scheme

N = 256


@pytest.fixture
def bv():
    Rq = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=1)
    return Rq, Rp, scheme


@pytest.fixture
def ghs():
    Rq = Ring(N, prime_size=[45, 45, 45, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1)
    return Rq, Rp, scheme


def enc(scheme, Rp, m, key):
    delta = scheme.rings[0].modulus_ratio(Rp, return_pointer=True)
    return scheme.sample(m.scaled_lift(scheme.rings[0], delta=delta), key)


def test_encrypt_decrypt_add_sub_mul(bv):
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    m1 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    c1 = enc(scheme, Rp, m1, key)

    assert scheme.phase(c0, key).round_division(Rp) == m0
    assert scheme.phase(c1, key).round_division(Rp) == m1
    assert scheme.phase(c0 + c1, key).round_division(Rp) == m0 + m1
    assert scheme.phase(c0 - c1, key).round_division(Rp) == m0 - m1

    z = Rp.random_element()
    assert scheme.phase(c0 * z, key).round_division(Rp) == m0 * z


def test_bv_keyswitch(bv):
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.phase(c_out, key2).round_division(Rp) == m0


def test_ghs_keyswitch(ghs):
    Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.phase(c_out, key2).round_division(Rp) == m0


def test_ghs_automorphism(ghs):
    Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    auto5 = scheme.gen_ksk_automorphism(key, key, 5)
    c_out = scheme.automorphism(c0, 5, auto5)
    assert scheme.phase(c_out, key).round_division(Rp) == m0.automorphism(5)


def test_mlwe_multiplication(bv):
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_ksk(key, [-(s_0 * s_0)])

    m1 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    m2 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.phase(c1 * c2, key).round_division(Rp)

    ell_non_special = Rq.ell - scheme.special_primes
    P = math.prod(Rq.primes[Rp.ell : ell_non_special])
    P_poly = Polynomial(Rp).from_bigint_array([P] + [0] * (Rp.N - 1))
    diff = (m_out - m1 * m2 * P_poly).get_polynomial(signed=True)
    assert max(abs(c) for c in diff) < 1000


def test_mgsw_external_product_identity(bv):
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(scheme)

    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)

    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    res = ct_id.external_product(ct1)
    assert scheme.phase(res, key).round_division(Rp) == m1


def test_lwe_alloc_and_phase():
    ring = Ring(N, prime_size=[20], split_degree=1)
    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    sample = LWE(ring=ring, m=[12345], key=key)
    phase = sample.phase(key)
    assert isinstance(phase, list) and len(phase) == ring.ell
    # a-vector is length n over each RNS limb; b matches
    assert len(sample.get_a()[0]) == ring.N
    assert len(sample.get_b()) == ring.ell
