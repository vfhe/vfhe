# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
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


def _mul_error(Rq, Rp, scheme, m_out, m1, m2):
    """Max coefficient error of a decrypted product against m1*m2.

    The product of two delta-scaled plaintexts carries an extra factor of the
    (non-special) primes above the plaintext ring, so scale the expectation by
    that P before comparing.
    """
    ell_non_special = Rq.ell - scheme.special_primes
    P = math.prod(Rq.primes[Rp.ell : ell_non_special])
    P_poly = Polynomial(Rp).from_bigint_array([P] + [0] * (Rp.N - 1))
    diff = (m_out - m1 * m2 * P_poly).get_polynomial(signed=True)
    return max(abs(c) for c in diff)


def test_encrypt_decrypt_add_sub_mul(bv):
    _Rq, Rp, scheme = bv
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
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.phase(c_out, key2).round_division(Rp) == m0


def test_ghs_keyswitch(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.phase(c_out, key2).round_division(Rp) == m0


def test_ghs_automorphism(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    auto5 = scheme.gen_ksk_automorphism(key, key, 5)
    c_out = scheme.automorphism(c0, 5, auto5)
    assert scheme.phase(c_out, key).round_division(Rp) == m0.automorphism(5)


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_mlwe_multiplication(scheme_fixture, request):
    Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    m1 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    m2 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.phase(c1 * c2, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_mlwe_multiplication_deferred_relinearization(scheme_fixture, request):
    Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    m1 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    m2 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N)])
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    # Multiply without a key: the product is a larger, not-yet-relinearized ct.
    c_ext = scheme.multiply(c1, c2, None)
    assert c_ext.is_extended
    assert c_ext.r == scheme.extended_rank

    c_relin = scheme.relinearize(c_ext, rlk)
    assert c_relin.r == scheme.r

    m_out = scheme.phase(c_relin, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


def test_mgsw_external_product_identity(bv):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(scheme)

    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)

    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    res = ct_id.external_product(ct1)
    assert scheme.phase(res, key).round_division(Rp) == m1


# Module ranks above 1, paired with a ring dimension that keeps the lattice
# dimension N*r (and the runtime) in line with the rank-1 tests above.
RANK_DIMS = [(2, 128), (4, 64)]


def _rank_scheme(N_r, r, special_primes):
    """Rank-``r`` scheme over a ring of dimension ``N_r``, plus its plaintext ring.

    Mirrors the ``bv``/``ghs`` fixtures: one extra top prime is added as the
    special (key-switching) prime when ``special_primes`` is set.
    """
    prime_size = [45, 45, 45] + ([50] if special_primes else [])
    Rq = Ring(N_r, prime_size=prime_size, split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=special_primes, module_rank=r)
    return Rq, Rp, scheme


def _rank_key(scheme, N_r, r):
    # Keep the key density proportional to the lattice dimension N*r.
    return scheme.key_gen_sparse(N_r * r // 8, 3.2)


@pytest.mark.parametrize("r, N_r", [(1, 256), *RANK_DIMS])
def test_tensor_product_slot_layout(r, N_r):
    # The tensor product's documented contract: with the extended key made of
    # the quadratic terms -(s_i*s_j) (i <= j, lexicographic) followed by the
    # linear terms s_i, the extended phase equals the product of the two input
    # phases exactly (same ring, no rounding anywhere).
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=0)
    key = _rank_key(scheme, N_r, r)
    c1 = enc(scheme, Rp, Rp.random_element(), key)
    c2 = enc(scheme, Rp, Rp.random_element(), key)

    slots = scheme.tensor_product(c1, c2)
    assert len(slots) == scheme.extended_rank + 1

    ext_key = scheme.quadratic_key_polys(key) + key.poly
    assert len(ext_key) == scheme.extended_rank

    ext_phase = slots[-1]
    for slot, t in zip(slots[:-1], ext_key, strict=False):
        ext_phase = ext_phase - slot * t

    expected = scheme.phase(c1, key) * scheme.phase(c2, key)
    ext_phase.to_coeff()
    expected.to_coeff()
    assert ext_phase == expected


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_encrypt_decrypt_add_sub_mul_module_rank(r, N_r):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=0)
    key = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    m1 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    c1 = enc(scheme, Rp, m1, key)
    assert c0.r == r

    assert scheme.phase(c0, key).round_division(Rp) == m0
    assert scheme.phase(c0 + c1, key).round_division(Rp) == m0 + m1
    assert scheme.phase(c0 - c1, key).round_division(Rp) == m0 - m1

    z = Rp.random_element()
    assert scheme.phase(c0 * z, key).round_division(Rp) == m0 * z


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.parametrize("special_primes", [0, 1])
def test_keyswitch_module_rank(r, N_r, special_primes):
    # The hybrid key-switch consumes one gadget key per key component, so the
    # ksk grows with the rank.
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes)
    key = _rank_key(scheme, N_r, r)
    key2 = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert c_out.r == r
    assert scheme.phase(c_out, key2).round_division(Rp) == m0


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_ghs_automorphism_module_rank(r, N_r):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    auto5 = scheme.gen_ksk_automorphism(key, key, 5)
    c_out = scheme.automorphism(c0, 5, auto5)
    assert scheme.phase(c_out, key).round_division(Rp) == m0.automorphism(5)


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.parametrize("special_primes", [0, 1])
def test_multiplication_module_rank(r, N_r, special_primes):
    # The product is the symmetric tensor of the two ciphertexts, so the rlk
    # carries one key per quadratic pair -(s_i*s_j), i <= j.
    Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes)
    key = _rank_key(scheme, N_r, r)
    scheme.rlk = scheme.gen_rlk(key, key)

    m1 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N_r)])
    m2 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N_r)])
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.phase(c1 * c2, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_multiplication_deferred_relinearization_module_rank(r, N_r):
    # The extended product carries the r*(r+1)/2 quadratic components plus the r
    # linear ones, i.e. rank r*(r+3)/2 (2r only at rank 1).
    Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    rlk = scheme.gen_rlk(key, key)

    m1 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N_r)])
    m2 = Polynomial(Rp).from_array([secrets.choice([-1, 0, 1]) for _ in range(N_r)])
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    c_ext = scheme.multiply(c1, c2, None)
    assert c_ext.is_extended
    assert c_ext.r == r * (r + 3) // 2 == scheme.extended_rank

    c_relin = scheme.relinearize(c_ext, rlk)
    assert c_relin.r == r

    m_out = scheme.phase(c_relin, key).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


def test_lwe_alloc_and_phase():
    ring = Ring(N, prime_size=[20], split_degree=1)
    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    sample = LWE(ring=ring, m=[12345], key=key)
    phase = sample.phase(key)
    assert isinstance(phase, list) and len(phase) == ring.ell
    # a-vector is length n over each RNS limb; b matches
    assert len(sample.get_a()[0]) == ring.N
    assert len(sample.get_b()) == ring.ell
