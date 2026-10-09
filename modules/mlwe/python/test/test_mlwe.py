# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.mlwe over the cffi boundary.

Encrypt/decrypt roundtrips, homomorphic add/sub/mul-by-poly, BV and GHS
key-switching, automorphism, ciphertext multiplication (relinearization), the
MGSW external product, and the LWE surface. Noise is stripped with
round_division back to the plaintext ring, so equality is exact.
"""

import functools
import math
import operator
import random
from typing import cast

import pytest
import vfhe.engine as engine
from vfhe.arith import Polynomial, Ring, repr
from vfhe.arith.number_theory import crt
from vfhe.crypto import entropy
from vfhe.engine import ffi
from vfhe.mlwe import (
    CMUX,
    LWE,
    MGSW,
    MLWE,
    LWE_Key,
    MGSW_Scheme,
    MLWE_Key,
    MLWE_Scheme,
    MLWE_Set,
    PlaintextMatrix,
)
from vfhe.mlwe.io import seed_still_holds

N = 256


def _ternary(n):
    """`n` coefficients drawn uniformly from {-1, 0, 1}."""
    return [entropy.below(3) - 1 for _ in range(n)]


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

    assert scheme.linear_decrypt(c0, key).round_division(Rp) == m0
    assert scheme.linear_decrypt(c1, key).round_division(Rp) == m1
    assert scheme.linear_decrypt(c0 + c1, key).round_division(Rp) == m0 + m1
    assert scheme.linear_decrypt(c0 - c1, key).round_division(Rp) == m0 - m1

    z = Rp.random_element()
    assert scheme.linear_decrypt(c0 * z, key).round_division(Rp) == m0 * z


def test_bv_keyswitch(bv):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


def test_ghs_keyswitch(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key)
    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


def test_ghs_automorphism(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    auto5 = scheme.gen_ksk_automorphism(key, key, 5)
    c_out = scheme.automorphism(c0, 5, auto5)
    assert scheme.linear_decrypt(c_out, key).round_division(Rp) == m0.automorphism(5)


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_mlwe_multiplication(scheme_fixture, request):
    Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    m1 = Polynomial(Rp).from_array(_ternary(N))
    m2 = Polynomial(Rp).from_array(_ternary(N))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.linear_decrypt(c1 * c2, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_mlwe_multiplication_deferred_relinearization(scheme_fixture, request):
    Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])

    m1 = Polynomial(Rp).from_array(_ternary(N))
    m2 = Polynomial(Rp).from_array(_ternary(N))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    # Multiply without a key: the product is a larger, not-yet-relinearized ct.
    c_ext = scheme.multiply(c1, c2, None)
    assert c_ext.is_extended
    assert c_ext.r == scheme.extended_rank

    c_relin = scheme.relinearize(c_ext, rlk)
    assert c_relin.r == scheme.r

    m_out = scheme.linear_decrypt(c_relin, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


def test_mgsw_external_product_identity(bv):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(scheme)

    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)

    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    res = ct_id.external_product(ct1)
    assert scheme.linear_decrypt(res, key).round_division(Rp) == m1


@pytest.mark.parametrize("bit", [0, 1])
def test_mgsw_cmux(bv, bit):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    selector = MGSW_Scheme(scheme).encrypt(
        Polynomial(Rp).from_array([bit] + [0] * (N - 1)), key
    )
    m = [Rp.random_element() for _ in range(2)]
    out = CMUX(enc(scheme, Rp, m[0], key), enc(scheme, Rp, m[1], key), selector)
    assert scheme.linear_decrypt(out, key).round_division(Rp) == m[bit]


# --- the RNS gadget's digit -------------------------------------------------
#
# `balanced`, on by default, takes residue j centered, in (-p_j/2, p_j/2];
# off, it takes it as stored, in [0, p_j). Both are x mod p_j, so the keys are
# the same either way. Without special primes a key switch adds exactly
# -sum_j d_j * e_j to the linear decryption, and an external product by an
# encryption of 1 adds sum_j d_j * e_j over all r + 1 components, so the added
# noise has variance components * N * sigma^2 * sum_j E[d_j^2]: (p_j^2 - 1) / 12
# for a centered digit, about p_j^2 / 3 for the other, one bit more standard
# deviation.
SIGMA = 3.2
ONE = [1] + [0] * (N - 1)


def test_balanced_is_the_default(bv):
    Rq, _Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, SIGMA)
    assert scheme.balanced
    assert cast("MLWE_Set", scheme.gen_ksk(key, key, lvl=0)).balanced
    assert cast("MLWE_Set", scheme.gen_rlk(key, key, lvl=0)).balanced
    assert MGSW_Scheme(scheme).balanced

    plain = MLWE_Scheme(Rq, balanced=False)
    key = plain.key_gen_sparse(N // 8, SIGMA)
    assert not cast("MLWE_Set", plain.gen_ksk(key, key, lvl=0)).balanced
    assert not cast("MLWE_Set", plain.gen_rlk(key, key, lvl=0)).balanced
    assert not MGSW_Scheme(plain).balanced
    assert MGSW_Scheme(plain, balanced=True).balanced


def _digits(poly, ring, balanced):
    """The RNS gadget's digits of ``poly``, in the order it takes them, in ``ring``."""
    poly.to_coeff()
    residues = sorted(
        zip(
            poly.ring.prime_indices,
            poly.ring.primes,
            poly.get_coeff_matrix(),
            strict=True,
        )
    )
    return [
        Polynomial(ring).from_bigint_array(
            [r - p if balanced and r > p // 2 else r for r in row]
        )
        for _, p, row in residues
    ]


def _gadget_product(keys, polys, balanced):
    """``sum_i sum_j d_j(polys[i]) * keys[i][j]``, as its ``a`` and ``b``."""
    products = [
        (d * k.get_a_poly(0), d * k.get_b_poly())
        for row, poly in zip(keys, polys, strict=True)
        for d, k in zip(_digits(poly, row[0].ring, balanced), row, strict=True)
    ]
    return [
        functools.reduce(operator.add, terms) for terms in zip(*products, strict=True)
    ]


def _coeffs(poly):
    poly.to_coeff()
    return poly.get_polynomial()


@pytest.mark.parametrize("balanced", [False, True])
def test_gadget_products_take_the_chosen_digit(balanced):
    """A key switch and a CMUX, recomputed from their digits.

    Without special primes nothing is rounded, so both are exactly the gadget
    products of their digits -- with ``balanced=False``, of the ``[0, p_j)``
    residues the RNS gadget took before it had the option.
    """
    _Rq, Rp, scheme = _rank_scheme(N, 1, 0, balanced=balanced)
    key = scheme.key_gen_sparse(N // 8, SIGMA)
    key_out = scheme.key_gen_sparse(N // 8, SIGMA)

    ksk = cast("MLWE_Set", scheme.gen_ksk(key_out, key, lvl=0))
    c = enc(scheme, Rp, Rp.random_element(), key)
    a, b = _gadget_product([ksk.mlwe[0]], [c.get_a_poly(0)], balanced)
    out = scheme.keyswitch(c, ksk)
    assert _coeffs(out.get_a_poly(0)) == _coeffs(-a)
    assert _coeffs(out.get_b_poly()) == _coeffs(c.get_b_poly() - b)

    selector = MGSW_Scheme(scheme).encrypt(Polynomial(Rp).from_array(ONE), key)
    c0, c1 = (enc(scheme, Rp, Rp.random_element(), key) for _ in range(2))
    diff = c1 - c0
    ell = selector.gadget_size
    a, b = _gadget_product(
        [selector.obj[:ell], selector.obj[ell:]],
        [diff.get_a_poly(0), diff.get_b_poly()],
        balanced,
    )
    out = CMUX(c0, c1, selector)
    assert _coeffs(out.get_a_poly(0)) == _coeffs(c0.get_a_poly(0) + a)
    assert _coeffs(out.get_b_poly()) == _coeffs(c0.get_b_poly() + b)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("special_primes", [0, 1])
def test_gadget_products_decrypt_with_either_digit(balanced, special_primes):
    Rq, Rp, scheme = _rank_scheme(N, 1, special_primes, balanced=balanced)
    key = scheme.key_gen_sparse(N // 8, SIGMA)
    key2 = scheme.key_gen_sparse(N // 8, SIGMA)
    m = [Rp.random_element() for _ in range(2)]
    c = [enc(scheme, Rp, m_i, key) for m_i in m]

    out = scheme.keyswitch(c[0], scheme.gen_ksk(key2, key, lvl=0))
    assert scheme.linear_decrypt(out, key2).round_division(Rp) == m[0]

    out = scheme.automorphism(c[0], 5, scheme.gen_ksk_automorphism(key, key, 5, lvl=0))
    assert scheme.linear_decrypt(out, key).round_division(Rp) == m[0].automorphism(5)

    scheme.rlk = scheme.gen_rlk(key, key)
    t = [Polynomial(Rp).from_array(_ternary(N)) for _ in range(2)]
    m_out = scheme.linear_decrypt(
        enc(scheme, Rp, t[0], key) * enc(scheme, Rp, t[1], key), key
    ).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, t[0], t[1]) < 1000

    mgsw_scheme = MGSW_Scheme(scheme)
    one = mgsw_scheme.encrypt(Polynomial(Rp).from_array(ONE), key)
    assert (
        scheme.linear_decrypt(one.external_product(c[0]), key).round_division(Rp)
        == m[0]
    )
    for bit in (0, 1):
        selector = mgsw_scheme.encrypt(
            Polynomial(Rp).from_array([bit] + [0] * (N - 1)), key
        )
        out = CMUX(c[0], c[1], selector)
        assert scheme.linear_decrypt(out, key).round_division(Rp) == m[bit]


# The [0, p_j) digit's mean, p_j/2, adds a term the key's errors fix, which
# moves one key's measurement by about 0.4 bits; this many keys average it out.
KEY_SETS = 32


def _added_noise(scheme, Rp, operations):
    """log2 of the std of what each operation adds to the linear decryption.

    ``operations(key)`` gives the key the outputs decrypt under and the
    operations to measure, all on the same ciphertexts.
    """
    diffs = []
    for _ in range(KEY_SETS):
        key = scheme.key_gen_sparse(N // 8, SIGMA)
        key_out, ops = operations(key)
        c = enc(scheme, Rp, Rp.random_element(), key)
        before = scheme.linear_decrypt(c, key)
        before.to_coeff()
        diffs = diffs or [[] for _ in ops]
        for diff, op in zip(diffs, ops, strict=True):
            after = scheme.linear_decrypt(op(c), key_out)
            after.to_coeff()
            diff += (after - before).get_polynomial(signed=True)
    return [0.5 * math.log2(sum(d * d for d in diff) / len(diff)) for diff in diffs]


def _assert_balanced_saves_a_bit(measured, primes, components):
    balanced, plain = measured
    variance = components * N * SIGMA**2 * sum((p * p - 1) / 12 for p in primes)
    assert abs(balanced - 0.5 * math.log2(variance)) < 0.25
    assert abs(plain - balanced - 1) < 0.4


def test_keyswitch_noise_of_either_digit(bv):
    _Rq, Rp, scheme = bv

    def operations(key):
        key_out = scheme.key_gen_sparse(N // 8, SIGMA)
        ksk = scheme.gen_ksk(key_out, key, lvl=0)
        plain = MLWE_Set(ksk.mlwe, balanced=False)
        return key_out, [
            functools.partial(scheme.keyswitch, ksk=k) for k in (ksk, plain)
        ]

    measured = _added_noise(scheme, Rp, operations)
    _assert_balanced_saves_a_bit(measured, scheme.ring.primes, scheme.r)


def test_external_product_noise_of_either_digit(bv):
    _Rq, Rp, scheme = bv

    def operations(key):
        one = MGSW_Scheme(scheme).encrypt(Polynomial(Rp).from_array(ONE), key)
        plain = MGSW(MGSW_Scheme(scheme, balanced=False), obj=one.obj)
        return key, [one.external_product, plain.external_product]

    measured = _added_noise(scheme, Rp, operations)
    _assert_balanced_saves_a_bit(measured, scheme.ring.primes, scheme.r + 1)


# Module ranks above 1, paired with a ring dimension that keeps the lattice
# dimension N*r (and the runtime) in line with the rank-1 tests above.
RANK_DIMS = [(2, 128), (4, 64)]


def _rank_scheme(N_r, r, special_primes, balanced=True):
    """Rank-``r`` scheme over a ring of dimension ``N_r``, plus its plaintext ring.

    Mirrors the ``bv``/``ghs`` fixtures: one extra top prime is added as the
    special (key-switching) prime when ``special_primes`` is set.
    """
    prime_size = [45, 45, 45] + ([50] if special_primes else [])
    Rq = Ring(N_r, prime_size=prime_size, split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(
        Rq, special_primes=special_primes, module_rank=r, balanced=balanced
    )
    return Rq, Rp, scheme


def _rank_key(scheme, N_r, r):
    # Keep the key density proportional to the lattice dimension N*r.
    return scheme.key_gen_sparse(N_r * r // 8, 3.2)


@pytest.mark.parametrize("r, N_r", [(1, 256), *RANK_DIMS])
def test_tensor_product_slot_layout(r, N_r):
    # The tensor product's documented contract: with the extended key made of
    # the quadratic terms -(s_i*s_j) (i <= j, lexicographic) followed by the
    # linear terms s_i, the linear decryption under it equals the product of the
    # two inputs' linear decryptions exactly (same ring, no rounding anywhere).
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=0)
    key = _rank_key(scheme, N_r, r)
    c1 = enc(scheme, Rp, Rp.random_element(), key)
    c2 = enc(scheme, Rp, Rp.random_element(), key)

    slots = scheme.tensor_product(c1, c2)
    assert len(slots) == scheme.extended_rank + 1

    ext_key = scheme.quadratic_key_polys(key) + key.poly
    assert len(ext_key) == scheme.extended_rank

    ext_decryption = slots[-1]
    for slot, t in zip(slots[:-1], ext_key, strict=False):
        ext_decryption = ext_decryption - slot * t

    expected = scheme.linear_decrypt(c1, key) * scheme.linear_decrypt(c2, key)
    ext_decryption.to_coeff()
    expected.to_coeff()
    assert ext_decryption == expected


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_encrypt_decrypt_add_sub_mul_module_rank(r, N_r):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=0)
    key = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    m1 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    c1 = enc(scheme, Rp, m1, key)
    assert c0.r == r

    assert scheme.linear_decrypt(c0, key).round_division(Rp) == m0
    assert scheme.linear_decrypt(c0 + c1, key).round_division(Rp) == m0 + m1
    assert scheme.linear_decrypt(c0 - c1, key).round_division(Rp) == m0 - m1

    z = Rp.random_element()
    assert scheme.linear_decrypt(c0 * z, key).round_division(Rp) == m0 * z


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
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_ghs_automorphism_module_rank(r, N_r):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    auto5 = scheme.gen_ksk_automorphism(key, key, 5)
    c_out = scheme.automorphism(c0, 5, auto5)
    assert scheme.linear_decrypt(c_out, key).round_division(Rp) == m0.automorphism(5)


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.parametrize("special_primes", [0, 1])
def test_multiplication_module_rank(r, N_r, special_primes):
    # The product is the symmetric tensor of the two ciphertexts, so the rlk
    # carries one key per quadratic pair -(s_i*s_j), i <= j.
    Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes)
    key = _rank_key(scheme, N_r, r)
    scheme.rlk = scheme.gen_rlk(key, key)

    m1 = Polynomial(Rp).from_array(_ternary(N_r))
    m2 = Polynomial(Rp).from_array(_ternary(N_r))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.linear_decrypt(c1 * c2, key).round_division(Rp)

    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
def test_multiplication_deferred_relinearization_module_rank(r, N_r):
    # The extended product carries the r*(r+1)/2 quadratic components plus the r
    # linear ones, i.e. rank r*(r+3)/2 (2r only at rank 1).
    Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    rlk = scheme.gen_rlk(key, key)

    m1 = Polynomial(Rp).from_array(_ternary(N_r))
    m2 = Polynomial(Rp).from_array(_ternary(N_r))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    c_ext = scheme.multiply(c1, c2, None)
    assert c_ext.is_extended
    assert c_ext.r == r * (r + 3) // 2 == scheme.extended_rank

    c_relin = scheme.relinearize(c_ext, rlk)
    assert c_relin.r == r

    m_out = scheme.linear_decrypt(c_relin, key).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


def test_lwe_alloc_and_linear_decrypt():
    ring = Ring(N, prime_size=[20], split_degree=1)
    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    sample = LWE(ring=ring, m=[12345], key=key)
    decryption = sample.linear_decrypt(key)
    assert isinstance(decryption, list) and len(decryption) == ring.ell
    # a-vector is length n over each RNS limb; b matches
    assert len(sample.get_a()[0]) == ring.N
    assert len(sample.get_b()) == ring.ell


@pytest.mark.parametrize("n", [776, 778, 783])
def test_lwe_dimension_need_not_fill_a_vector(n):
    """The inner product <a, s> covers every coefficient, at any dimension.

    The element-wise kernels work on whole vector groups and leave the rest
    to a scalar body; a dimension that is not a multiple of the group is
    where a missing tail shows, as a linear decryption off by the last few
    terms.
    """
    ring = Ring(N, prime_size=[30, 45], split_degree=1)
    rng = random.Random(n)  # noqa: S311 - test data, not a key
    s = [rng.randint(0, 1) for _ in range(n)]
    # noiseless, so the linear decryption is m exactly
    key = LWE_Key(ring, key=s, n=n, err_sigma=0.0)
    q = ring.q_l
    m = q // 3
    sample = LWE(ring=ring, m=[m % p for p in ring.primes], key=key)
    assert sample.linear_decrypt(key, recompose=True) == m
    b = crt(sample.get_b(), ring.primes)
    a = sample.get_a()
    linear = b - sum(crt([limb[i] for limb in a], ring.primes) * s[i] for i in range(n))
    assert linear % q == m


def test_lwe_key_encrypts_with_its_noise_parameter_only():
    """A key given as coefficients takes err_sigma, or refuses to encrypt.

    One extracted from an MLWE key takes the MLWE key's.
    """
    ring = Ring(N, prime_size=[30, 45], split_degree=1)
    s = [i % 2 for i in range(N)]
    m = [p // 4 for p in ring.primes]

    bare = LWE_Key(ring, key=s)
    assert bare.err_sigma is None
    assert ffi.cast("LWE_Key", bare.obj).sigma < 0
    with pytest.raises(ValueError, match="noise parameter"):
        LWE(ring=ring, m=m, key=bare)

    noisy = LWE_Key(ring, key=s, err_sigma=3.2)
    assert noisy.err_sigma == 3.2
    assert ffi.cast("LWE_Key", noisy.obj).sigma == 3.2
    sample = LWE(ring=ring, m=m, key=noisy)
    # the bare key decrypts what the noisy one encrypted
    decryptions = sample.linear_decrypt(bare)
    assert isinstance(decryptions, list)
    for d, m_j, p in zip(decryptions, m, ring.primes, strict=True):
        err = (d - m_j) % p
        assert min(err, p - err) < 64

    scheme = MLWE_Scheme(ring, special_primes=0, module_rank=2)
    extracted = scheme.key_gen_sparse(16, 2.5, ternary=False).extract_lwe_key()
    assert extracted.err_sigma == 2.5
    assert ffi.cast("LWE_Key", extracted.obj).sigma == 2.5


def test_lwe_limbs_follow_the_ring_over_a_populated_base():
    # A ring built after another one holds primes the base registered first,
    # so its prime indices do not ascend and do not start at 0. Every limb has
    # to be reduced by its own prime, and a key rebuilt from its coefficients
    # has to decrypt what the generated one encrypted.
    Ring(N, prime_size=[45, 45], split_degree=1)
    ring = Ring(N, prime_size=[21, 45, 45], split_degree=1)
    assert ring.prime_indices != sorted(ring.prime_indices)

    key = LWE_Key(ring, sec_sigma=3.2, err_sigma=3.2)
    rebuilt = LWE_Key(ring, key=key.get_s())
    m = [p // 4 for p in ring.primes]
    sample = LWE(ring=ring, m=m, key=key)

    for limb, p in zip(sample.get_a(), ring.primes, strict=True):
        assert max(limb) < p
    for b, p in zip(sample.get_b(), ring.primes, strict=True):
        assert b < p
    decryptions = sample.linear_decrypt(rebuilt)
    assert isinstance(decryptions, list)
    for decryption, m_j, p in zip(decryptions, m, ring.primes, strict=True):
        err = (decryption - m_j) % p
        assert min(err, p - err) < 64


# --- radix gadget ----------------------------------------------------------
#
# The key-switch keys the radix gadget needs -- one per prime and per
# base-2^w digit of it -- and what that buys: every product the decomposition
# accumulates stays below 2^w instead of below the prime.
RADIX_LOG_BASE = 10


def _radix_keys(ring, log_base: int) -> int:
    """How many gadget keys per component the radix gadget takes over `ring`."""
    return sum(-(-p.bit_length() // log_base) for p in ring.primes[: ring.ell])


def _keys_per_component(ksk, lvl: int) -> int:
    """The gadget keys a leveled key-switch key holds for one component."""
    component = cast("list[MLWE_Set]", ksk)[lvl].mlwe[0]
    assert component is not None
    return len(component)


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_keyswitch_radix_gadget(scheme_fixture, request):
    # The gadget is chosen when the key is generated and travels with it, so
    # the key switch itself is the same call either way.
    _Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)

    ksk = scheme.gen_ksk(key2, key, radix_log_base=RADIX_LOG_BASE)
    assert _keys_per_component(ksk, c0.lvl) == _radix_keys(
        scheme.special_rings[c0.lvl], RADIX_LOG_BASE
    )

    c_out = scheme.keyswitch(c0, ksk)
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


def test_keyswitch_radix_beats_the_rns_gadget():
    # Two primes, no special ones, and a plaintext ring one prime below the
    # ciphertext's, so the noise budget is a single prime. Decomposing into
    # residues puts a whole prime into the noise, which that budget cannot
    # absorb; the radix gadget keeps every product below 2^w.
    Rq = Ring(N, prime_size=[45, 45], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()

    radix = scheme.keyswitch(
        enc(scheme, Rp, m0, key),
        scheme.gen_ksk(key2, key, radix_log_base=RADIX_LOG_BASE),
    )
    assert scheme.linear_decrypt(radix, key2).round_division(Rp) == m0

    rns = scheme.keyswitch(enc(scheme, Rp, m0, key), scheme.gen_ksk(key2, key))
    assert scheme.linear_decrypt(rns, key2).round_division(Rp) != m0


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.parametrize("special_primes", [0, 1])
def test_keyswitch_radix_module_rank(r, N_r, special_primes):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes)
    key = _rank_key(scheme, N_r, r)
    key2 = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)

    ksk = scheme.gen_ksk(key2, key, radix_log_base=RADIX_LOG_BASE)
    c_out = scheme.keyswitch(c0, ksk)
    assert c_out.r == r
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
def test_mlwe_multiplication_radix_rlk(scheme_fixture, request):
    # Relinearization reads the same gadget, around the key's NULL slots.
    Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)], radix_log_base=RADIX_LOG_BASE)

    m1 = Polynomial(Rp).from_array(_ternary(N))
    m2 = Polynomial(Rp).from_array(_ternary(N))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.linear_decrypt(c1 * c2, key).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


def test_mgsw_external_product_radix(bv):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(scheme, radix_log_base=RADIX_LOG_BASE)

    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)

    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    assert ct_id.gadget_size == _radix_keys(mgsw_scheme.ring, RADIX_LOG_BASE)

    res = ct_id.external_product(ct1)
    assert scheme.linear_decrypt(res, key).round_division(Rp) == m1


@pytest.mark.parametrize("radix", [None, RADIX_LOG_BASE])
def test_mgsw_products_at_a_level(ghs, radix):
    # A key encrypted for a level multiplies that level's samples, and what
    # it produces stays at that level.
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    one, zero = MGSW_Scheme(scheme, radix_log_base=radix).encrypt_constants(
        [1, 0], key, lvl=1
    )
    m = [Rp.random_element() for _ in range(2)]
    c = [enc(scheme, Rp, x, key).round_division(lvl=1) for x in m]

    for out, expected in (
        (one.external_product(c[0]), m[0]),
        (CMUX(c[0], c[1], one), m[1]),
        (CMUX(c[0], c[1], zero), m[0]),
    ):
        assert out.lvl == 1
        assert out.ring == scheme.rings[1]
        assert scheme.linear_decrypt(out, key).round_division(Rp) == expected


def test_mgsw_products_refuse_another_level(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    (one,) = MGSW_Scheme(scheme).encrypt_constants([1], key)
    c = enc(scheme, Rp, Rp.random_element(), key).round_division(lvl=1)
    with pytest.raises(ValueError, match="level 1"):
        one.external_product(c)
    with pytest.raises(ValueError, match="level 1"):
        CMUX(c, c, one)


@pytest.mark.parametrize("count", [N // 16, N])
def test_full_packing_keyswitch_scaled(ghs, count):
    # Sample k carries k + 1 in its constant coefficient and noise-sized
    # junk elsewhere; packing keeps the constants, at k * N / count, scaled by
    # count.
    _Rq, _Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    log_n = N.bit_length() - 1
    ksk = scheme.gen_ksk_trace(
        key, key, gens=[(1 << j) + 1 for j in range(1, log_n + 1)], lvl=0
    )
    delta = 1 << 60
    vec = [
        scheme.sample(
            Polynomial(scheme.rings[0]).from_bigint_array(
                [(k + 1) * delta] + [entropy.below(1 << 40) for _ in range(N - 1)]
            ),
            key,
        )
        for k in range(count)
    ]
    packed = scheme.full_packing_keyswitch_scaled(vec, ksk)
    d = scheme.linear_decrypt(packed, key).get_polynomial(signed=True)
    stride = N // count
    for k in range(count):
        assert round(d[k * stride] / (count * delta)) == k + 1


# --- LWE extraction and the packing key switch -------------------------------


def _mod_switch(v, q, p):
    return round((v * p) / q) % p


def _schemes_on_shared_primes(*shapes, prime_size=(50, 50, 50), special_primes=1):
    """One scheme per ``(dimension, rank)`` in ``shapes``, all on the same
    primes: drawn for the largest dimension (an NTT prime for twice it serves
    every smaller one) and given to the others, whose rings have bases of
    their own."""
    big = max(d for d, _ in shapes)
    rings = {big: Ring(big, prime_size=list(prime_size), split_degree=1)}
    schemes = []
    for d, r in shapes:
        if d not in rings:
            rings[d] = Ring(d, primes=list(rings[big].primes), split_degree=1)
        schemes.append(
            MLWE_Scheme(rings[d], special_primes=special_primes, module_rank=r)
        )
    return schemes


def test_lwe_extraction():
    N = 256
    Rq = Ring(N, prime_size=[50, 50, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=1)
    key = scheme.key_gen_sparse(64, 3.2, ternary=True)
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = scheme.extract_lwe(rlwe_sample, idx)
        decryption = lwe_sample.linear_decrypt(lwe_key, recompose=True)
        res = _mod_switch(decryption, Rq.q_l, Rp.primes[0])
        diff = (res - msg_coeffs[idx]) % Rp.primes[0]
        diff = min(diff, Rp.primes[0] - diff)
        assert diff < 1000


def test_lwe_extraction_over_a_populated_base():
    N = 256
    _, _, scheme = _scheme_over_a_populated_base()
    Rq = scheme.rings[0]
    Rp = Rq.quotient_ring(ell=1)
    key = scheme.key_gen_sparse(64, 3.2, ternary=True)
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = scheme.extract_lwe(rlwe_sample, idx)
        decryption = lwe_sample.linear_decrypt(lwe_key, recompose=True)
        res = _mod_switch(decryption, Rq.q_l, Rp.primes[0])
        diff = (res - msg_coeffs[idx]) % Rp.primes[0]
        assert min(diff, Rp.primes[0] - diff) < 1000


def _check_packing(out_scheme, output_key, count, radix_log_base=None):
    Rq = out_scheme.rings[0]
    lwe_key = LWE_Key(ring=Rq, sec_sigma=3.2, err_sigma=3.2, n=Rq.N)
    packing_key = out_scheme.gen_packing_ksk(
        output_key, lwe_key, radix_log_base=radix_log_base
    )
    extracted = []
    for i in range(count):
        m_i = _mod_switch((i * 137), 2000, Rq.q_l)
        extracted.append(LWE(ring=Rq, m=[m_i % q for q in Rq.primes], key=lwe_key))

    out_repacked = out_scheme.packing_keyswitch(extracted, packing_key)
    out_coeffs = out_scheme.linear_decrypt(out_repacked, output_key).get_polynomial()
    for j in range(Rq.N):
        expected = (j * 137) % 2000 if j < count else 0
        m_j = _mod_switch(out_coeffs[j], Rq.q_l, 2000)
        diff = (m_j - expected) % 2000
        assert min(diff, 2000 - diff) <= 5


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("keygen_threads", [1, 4])
def test_packing_ksk(balanced, keygen_threads):
    Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=4, balanced=balanced)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    engine.set_num_threads(keygen_threads)
    try:
        _check_packing(out_scheme, output_key, 256)
    finally:
        engine.set_num_threads()


def test_packing_ksk_radix():
    Rq = Ring(256, prime_size=[50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=0, max_lvl=1)
    output_key = out_scheme.key_gen_sparse(64, 3.2)
    _check_packing(out_scheme, output_key, 256, radix_log_base=10)


def test_packing_ksk_over_a_populated_base():
    _, _, out_scheme = _scheme_over_a_populated_base(module_rank=4)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    _check_packing(out_scheme, output_key, 256)


@pytest.mark.parametrize("n", [16, 128])
def test_packing_samples_of_another_dimension(n):
    # Samples extracted from R_N packed into R_n: the two rings have their own
    # bases, so the primes are matched by value.
    N = 64
    io, rotation = _schemes_on_shared_primes((n, 1), (N, 2))
    output_key = io.key_gen_sparse(16 * io.r, 3.2)
    rotation_key = rotation.key_gen_sparse(16 * rotation.r, 3.2)
    ring = rotation.rings[0]
    assert ring.base != io.rings[0].base
    delta = 1 << 70
    count = min(n, 40)
    extracted = [
        rotation.extract_lwe(
            rotation.sample(
                Polynomial(ring).from_bigint_array([(k + 1) * delta]), rotation_key
            ),
            0,
        )
        for k in range(count)
    ]
    packing_key = io.gen_packing_ksk(output_key, rotation_key.extract_lwe_key())
    out = io.packing_keyswitch(extracted, packing_key)
    d = io.linear_decrypt(out, output_key).get_polynomial(signed=True)
    assert [round(x / delta) for x in d] == [k + 1 for k in range(count)] + [0] * (
        n - count
    )


# --- Ring switching -----------------------------------------------------------
#
# Between R_N and R_n, N = k n, through the subring Z[X^k]: down keeps the
# coefficients k m of the message, up embeds it with Y = X^k. The subring maps
# only move coefficients, so the noise left is the key switch's.

RING_SWITCH_DELTA = 1 << 70


def _ring_switch_case(src, dst, lvl=0, **key_options):
    """Encrypts a message in every coefficient under a key of ``src`` at
    ``lvl``, switches it to ``dst`` and returns (message, decryption in
    units of the scale, noise bits)."""
    key_src = src.key_gen_sparse(16 * src.r, 3.2)
    key_dst = dst.key_gen_sparse(16 * dst.r, 3.2)
    out_lvl = dst.level_with_primes(src.rings[lvl])
    ksk = dst.gen_ring_switch_key(key_dst, key_src, out_lvl, **key_options)
    rng = random.Random(0xC0FFEE)  # noqa: S311 - test data, not a key
    msg = [rng.randrange(-1000, 1000) for _ in range(src.N)]
    poly = Polynomial(src.rings[lvl]).from_bigint_array(
        [m * RING_SWITCH_DELTA for m in msg]
    )
    out = dst.ring_switch(src.sample(poly, key_src, lvl=lvl), ksk)
    assert out.lvl == out_lvl
    assert out.ring.N == dst.N
    d = dst.linear_decrypt(out, key_dst).get_polynomial(signed=True)
    got = [round(x / RING_SWITCH_DELTA) for x in d]
    noise = max(abs(x - g * RING_SWITCH_DELTA) for x, g in zip(d, got, strict=True))
    return msg, got, noise.bit_length()


def _switched(msg, n):
    """What a switch of ``msg`` to dimension ``n`` decrypts to."""
    N = len(msg)
    if n <= N:
        return msg[:: N // n]
    k = n // N
    return [msg[m // k] if m % k == 0 else 0 for m in range(n)]


@pytest.mark.parametrize(
    ("N_from", "n_to"), [(256, 64), (256, 128), (64, 256), (128, 128)]
)
@pytest.mark.parametrize(("rank_from", "rank_to"), [(1, 1), (2, 1), (1, 3)])
def test_ring_switch(N_from, n_to, rank_from, rank_to):
    src, dst = _schemes_on_shared_primes((N_from, rank_from), (n_to, rank_to))
    msg, got, noise = _ring_switch_case(src, dst)
    assert got == _switched(msg, n_to)
    assert noise < 16


@pytest.mark.parametrize("direction", ["down", "up"])
def test_ring_switch_at_a_level(direction):
    dims = [(256, 1), (64, 2)] if direction == "down" else [(64, 2), (256, 1)]
    src, dst = _schemes_on_shared_primes(*dims, prime_size=(50, 50, 50, 50))
    msg, got, _ = _ring_switch_case(src, dst, lvl=1)
    assert got == _switched(msg, dst.N)


@pytest.mark.parametrize("hybrid", [True, False])
def test_ring_switch_radix(hybrid):
    src, dst = _schemes_on_shared_primes((256, 1), (64, 1))
    msg, got, noise = _ring_switch_case(
        src, dst, radix_log_base=RADIX_LOG_BASE, hybrid=hybrid
    )
    assert got == _switched(msg, 64)
    if not hybrid:
        # Without the special prime the digits' products stay: a BV switch.
        assert noise > 16


@pytest.mark.parametrize("count", [2, 3, 4])
def test_ring_switch_up_interleaves_several_samples(count):
    # Up by k = 4: sample i lands at coefficients i + 4 m, the rest are zero.
    src, dst = _schemes_on_shared_primes((64, 2), (256, 1))
    key_src = src.key_gen_sparse(32, 3.2)
    key_dst = dst.key_gen_sparse(16, 3.2)
    ksk = dst.gen_ring_switch_key(key_dst, key_src, 0)
    rng = random.Random(count)  # noqa: S311 - test data, not a key
    msgs = [[rng.randrange(-1000, 1000) for _ in range(64)] for _ in range(count)]
    samples = [
        src.sample(
            Polynomial(src.rings[0]).from_bigint_array(
                [m * RING_SWITCH_DELTA for m in msg]
            ),
            key_src,
        )
        for msg in msgs
    ]
    out = dst.ring_switch(samples, ksk)
    d = dst.linear_decrypt(out, key_dst).get_polynomial(signed=True)
    expected = [msgs[c % 4][c // 4] if c % 4 < count else 0 for c in range(256)]
    assert [round(x / RING_SWITCH_DELTA) for x in d] == expected


def test_ring_switch_takes_several_samples_only_going_up():
    src, dst = _schemes_on_shared_primes((256, 1), (64, 1))
    key_src = src.key_gen_sparse(16, 3.2)
    key_dst = dst.key_gen_sparse(16, 3.2)
    ksk = dst.gen_ring_switch_key(key_dst, key_src, 0)
    c = src.sample(Polynomial(src.rings[0]).from_array([1]), key_src)
    with pytest.raises(ValueError, match="at most n / N samples"):
        dst.ring_switch([c, c], ksk)
    up_src, up_dst = _schemes_on_shared_primes((64, 1), (128, 1))
    up_key_src = up_src.key_gen_sparse(16, 3.2)
    up_ksk = up_dst.gen_ring_switch_key(up_dst.key_gen_sparse(16, 3.2), up_key_src, 0)
    c = up_src.sample(Polynomial(up_src.rings[0]).from_array([1]), up_key_src)
    with pytest.raises(ValueError, match="at most n / N samples"):
        up_dst.ring_switch([c, c, c], up_ksk)


def test_ring_switch_checks_its_key():
    src, dst = _schemes_on_shared_primes((256, 1), (64, 1), prime_size=(50, 50, 50, 50))
    wide, _ = _schemes_on_shared_primes((256, 2), (64, 1))
    key_src = src.key_gen_sparse(16, 3.2)
    key_dst = dst.key_gen_sparse(16, 3.2)
    c = src.sample(Polynomial(src.rings[0]).from_array([1]), key_src)
    with pytest.raises(ValueError, match="ring-switch key"):
        # A key for another level.
        dst.ring_switch(c, dst.gen_ring_switch_key(key_dst, key_src, 1))
    with pytest.raises(ValueError, match="ring-switch key"):
        # A key for a source of another rank.
        key_wide = wide.key_gen_sparse(16, 3.2)
        dst.ring_switch(c, dst.gen_ring_switch_key(key_dst, key_wide, 0))


# --- MGSW x MGSW, automorphisms, trivial MGSW ---------------------------------
#
# Without special primes an MGSW's rows live in the ring it multiplies, so
# every operation takes keys of level 0 (with the radix gadget, which is what
# keeps BV noise down). With special primes the rows live over the key ring,
# which is a level only of a scheme with one more special prime: the second
# scheme of `_two_schemes`, holding the same secret.


def _monomial(ring, e):
    return Polynomial(ring).from_array([0] * e + [1] + [0] * (ring.N - e - 1))


def _bv_mgsw(module_rank=1):
    Rq = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=module_rank)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw = MGSW_Scheme(scheme, radix_log_base=RADIX_LOG_BASE)
    keys = {
        "aut": lambda g: scheme.gen_ksk_automorphism(
            key, key, g, lvl=0, radix_log_base=RADIX_LOG_BASE
        ),
        "rlk": lambda: scheme.gen_rlk(key, key, lvl=0, radix_log_base=RADIX_LOG_BASE),
    }
    return Rp, scheme, key, mgsw, mgsw, key, keys


def _two_schemes(module_rank=1):
    base = Ring(N, prime_size=[45, 45, 45, 50, 50], split_degree=1)
    scheme = MLWE_Scheme(
        base.quotient_ring(ell=4), special_primes=1, module_rank=module_rank
    )
    rows = MLWE_Scheme(base, special_primes=1, module_rank=module_rank)
    assert rows.rings[0].mask == scheme.special_rings[0].mask
    Rp = scheme.rings[0].quotient_ring(ell=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    rows_key = MLWE_Key(key.key, key.sigma_err, rows)
    keys = {
        "aut": lambda g: rows.gen_ksk_automorphism(rows_key, rows_key, g, lvl=0),
        "rlk": lambda: rows.gen_rlk(rows_key, rows_key, lvl=0),
    }
    return Rp, scheme, key, MGSW_Scheme(scheme), MGSW_Scheme(rows), rows_key, keys


CASES = {"bv_radix": _bv_mgsw, "special_primes": _two_schemes}


def _check_product(scheme, Rp, key, mgsw, factor):
    m = Rp.random_element()
    out = mgsw.external_product(enc(scheme, Rp, m, key))
    assert scheme.linear_decrypt(out, key).round_division(Rp) == m * factor


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("module_rank", [1, 2])
def test_mgsw_trivial(case, module_rank):
    Rp, scheme, key, mgsw, _, _, _ = CASES[case](module_rank)
    _check_product(scheme, Rp, key, mgsw.trivial(_monomial(Rp, 5)), _monomial(Rp, 5))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("module_rank", [1, 2])
def test_mgsw_internal_product(case, module_rank):
    Rp, scheme, key, mgsw, rows_mgsw, rows_key, _ = CASES[case](module_rank)
    a = rows_mgsw.encrypt(_monomial(Rp, 2), rows_key)
    b = mgsw.encrypt(_monomial(Rp, 3), key)
    product = a.internal_product(b)
    assert len(product.obj) == len(b.obj)
    assert all(x.ring == y.ring for x, y in zip(product.obj, b.obj, strict=True))
    _check_product(scheme, Rp, key, product, _monomial(Rp, 5))
    # From a trivial key, as a chain of monomial keys starts: X^2 * X^(N-1)
    # wraps to -X.
    chain = a.internal_product(mgsw.trivial(_monomial(Rp, N - 1)))
    _check_product(scheme, Rp, key, chain, -_monomial(Rp, 1))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("module_rank", [1, 2])
@pytest.mark.parametrize("g", [5, 2 * N - 1])
def test_mgsw_automorphism(case, module_rank, g):
    Rp, scheme, key, mgsw, _, _, keys = CASES[case](module_rank)
    m = Rp.random_element()
    c = mgsw.encrypt(m, key)
    out = mgsw.automorphism(c, g, keys["aut"](g), keys["rlk"]())
    _check_product(scheme, Rp, key, out, m.automorphism(g))


def test_mgsw_internal_product_refuses_rows_it_cannot_consume():
    _Rp, _scheme, key, mgsw, _, _, _ = _two_schemes()
    b = mgsw.encrypt(_monomial(_Rp, 1), key)
    with pytest.raises(ValueError, match="cannot multiply the other's rows"):
        b.internal_product(b)


def test_keyswitch_radix_single_component():
    # One prime left in the ciphertext ring: each residue is the value itself,
    # so the gadget degenerates to the plain powers of 2^w. The level has no
    # room left to round noise away, so the key switch is measured directly --
    # the linear decryption under the new key against the one it consumed.
    Rq = Ring(N, prime_size=[45, 45], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, max_lvl=2)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)

    c1 = enc(scheme, Rp, Rp.random_element(), key).round_division(lvl=1)
    assert c1.ell == 1

    ksk = scheme.gen_ksk(key2, key, radix_log_base=RADIX_LOG_BASE)
    assert _keys_per_component(ksk, c1.lvl) == _radix_keys(
        scheme.special_rings[c1.lvl], RADIX_LOG_BASE
    )

    before = scheme.linear_decrypt(c1, key)
    after = scheme.linear_decrypt(scheme.keyswitch(c1, ksk), key2)
    before.to_coeff()
    after.to_coeff()
    err = (after - before).get_polynomial(signed=True)
    assert max(abs(c) for c in err) < 1 << 25


def test_keyswitch_radix_at_a_level(ghs):
    # Above level 0 the key's ring keeps the special prime while the
    # ciphertext's has dropped a base one, so the gadget the key switch
    # consumes is a prefix of the one the key carries -- one whose digit
    # counts have to line up prime by prime.
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()

    c1 = enc(scheme, Rp, m0, key).round_division(lvl=1)
    assert c1.lvl == 1

    ksk = scheme.gen_ksk(key2, key, radix_log_base=RADIX_LOG_BASE)
    assert _keys_per_component(ksk, c1.lvl) == _radix_keys(
        scheme.special_rings[c1.lvl], RADIX_LOG_BASE
    )
    c_out = scheme.keyswitch(c1, ksk)
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


@pytest.mark.parametrize("lvl", [0, 1])
@pytest.mark.parametrize("radix", [4, RADIX_LOG_BASE])
def test_keyswitch_without_the_special_primes(ghs, lvl, radix):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c = enc(scheme, Rp, m0, key)
    if lvl:
        c.round_division(lvl=lvl)
    ksk = scheme.gen_ksk(key2, key, lvl=lvl, radix_log_base=radix, hybrid=False)
    # The key never reaches the special prime.
    assert all(x.ring == scheme.rings[lvl] for x in ksk.mlwe[0])
    c_out = scheme.keyswitch(c, ksk)
    assert c_out.ring == c.ring
    assert scheme.linear_decrypt(c_out, key2).round_division(Rp) == m0


def test_keyswitch_without_the_special_primes_at_one_prime(ghs):
    # At the last level the plaintext ring is the whole ring, so check the
    # noise the key switch adds instead of a decryption.
    _Rq, _Rp, scheme = ghs
    lvl = len(scheme.rings) - 1
    ring = scheme.rings[lvl]
    assert ring.ell == 1
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    c = scheme.sample(ring.random_element(), key.at_ring(ring), lvl=lvl)
    ksk = scheme.gen_ksk(key2, key, lvl=lvl, radix_log_base=4, hybrid=False)
    diff = scheme.linear_decrypt(scheme.keyswitch(c, ksk), key2) - (
        scheme.linear_decrypt(c, key)
    )
    assert max(abs(x) for x in diff.get_polynomial(signed=True)) < 2**20


def test_round_division_moves_the_native_ring(ghs):
    # Native code that allocates, rescales or key-switches from a sample takes
    # the ring from the sample itself, so it must name the ring the components
    # were divided into.
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c = enc(scheme, Rp, m0, key).round_division(lvl=1)
    assert ffi.cast("MLWE", c.obj).ring == scheme.rings[1].arith_ring
    assert scheme.linear_decrypt(c, key).round_division(Rp) == m0


def test_derived_samples_keep_the_shape(ghs):
    # copy / add / sub allocate their result like their input: a sample over a
    # special ring keeps its special prime, and an unrelinearized product its
    # extended rank, instead of falling back to the level's defaults.
    _Rq, _Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    c = scheme.sample(scheme.rings[0].random_element(), key)
    special = type(c)(scheme, lvl=-1, ring=scheme.special_rings[0])
    product = scheme.multiply(c, c, None)
    for x in (special, product):
        for out in (x.copy(), x + x, x - x):
            assert out.ring == x.ring
            assert out.r == x.r
            assert out.is_extended == x.is_extended
    copied = product.copy()
    for i in range(product.r):
        assert copied.get_a_poly(i) == product.get_a_poly(i)
    assert copied.get_b_poly() == product.get_b_poly()


def _log2_noise(scheme, out, c, gen, key):
    """Bits of what the automorphism's key switch added to the linear decryption."""
    expected = scheme.linear_decrypt(c, key).automorphism(gen)
    diff = scheme.linear_decrypt(out, key) - expected
    return max(abs(x) for x in diff.get_polynomial(signed=True)).bit_length()


def _check_hoisted(scheme, Rp, key, m0, c, radix):
    gens = [1, 5, 25, 2 * scheme.N - 1]
    ksks = [
        scheme.gen_ksk_automorphism(key, key, g, radix_log_base=radix) for g in gens
    ]
    hoisted = scheme.automorphisms(c, gens, ksks)
    for gen, ksk, out in zip(gens, ksks, hoisted, strict=True):
        assert out.lvl == c.lvl and out.ring == c.ring
        alone = scheme.automorphism(c, gen, ksk)
        assert _log2_noise(scheme, out, c, gen, key) <= (
            _log2_noise(scheme, alone, c, gen, key) + 2
        )
        # The RNS gadget without a special prime leaves about 2^52 of noise,
        # which a level's smaller modulus no longer absorbs, hoisted or not.
        expected = m0.automorphism(gen)
        if scheme.linear_decrypt(alone, key).round_division(Rp) == expected:
            assert scheme.linear_decrypt(out, key).round_division(Rp) == expected


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
@pytest.mark.parametrize("radix", [None, RADIX_LOG_BASE])
@pytest.mark.parametrize("lvl", [0, 1])
def test_hoisted_automorphisms_match_the_unhoisted_ones(
    scheme_fixture, radix, lvl, request
):
    _Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    c = enc(scheme, Rp, m0, key)
    if lvl:
        c.round_division(lvl=lvl)
    _check_hoisted(scheme, Rp, key, m0, c, radix)


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.parametrize("radix", [None, RADIX_LOG_BASE])
def test_hoisted_automorphisms_module_rank(r, N_r, radix):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    _check_hoisted(scheme, Rp, key, m0, enc(scheme, Rp, m0, key), radix)


@pytest.mark.parametrize("split_degree", [2, 4])
def test_hoisted_automorphisms_on_a_ring_that_is_not_fully_split(split_degree):
    # There the transform's points are not single evaluations, so the digits
    # are kept canonical and permuted before each product.
    Rq = Ring(N, prime_size=[45, 45, 45, 50], split_degree=split_degree)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m0 = Rp.random_element()
    _check_hoisted(scheme, Rp, key, m0, enc(scheme, Rp, m0, key), None)


def test_hoisted_automorphisms_refuse_a_key_of_another_gadget(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    c = enc(scheme, Rp, Rp.random_element(), key)
    ksks = [
        scheme.gen_ksk_automorphism(key, key, 5),
        scheme.gen_ksk_automorphism(key, key, 25, radix_log_base=RADIX_LOG_BASE),
    ]
    with pytest.raises(ValueError, match="gadget"):
        scheme.automorphisms(c, [5, 25], ksks)
    with pytest.raises(ValueError, match="generator"):
        scheme.automorphisms(c, [4], ksks[:1])


@pytest.fixture
def threads():
    """Lets the test's parallel calls use up to 8 threads (vfhe defaults to 1)."""
    engine.set_num_threads(8)
    yield
    engine.set_num_threads()


@pytest.mark.parametrize("n_threads", [1, 0])
@pytest.mark.usefixtures("threads")
def test_automorphism_batch(ghs, n_threads):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ms = [Rp.random_element() for _ in range(4)]
    cts = [enc(scheme, Rp, m, key) for m in ms]
    gens = [5, 1, 25, 2 * N - 1]
    ksks = [None if g == 1 else scheme.gen_ksk_automorphism(key, key, g) for g in gens]
    # NTT-domain inputs are accepted and not modified.
    cts[0].to_NTT()
    cts[1].to_NTT()
    outs = scheme.automorphism_batch(cts, gens, ksks, n_threads)
    assert [c.repr for c in cts] == [repr.ntt, repr.ntt, repr.coeff, repr.coeff]
    for m, g, out in zip(ms, gens, outs, strict=True):
        assert out.repr == repr.coeff
        assert scheme.linear_decrypt(out, key).round_division(Rp) == m.automorphism(g)
    with pytest.raises(ValueError, match="no key"):
        scheme.automorphism_batch(cts[:1], [5], [None])


def _same_sample(c1, c2):
    a, b = c1.copy(), c2.copy()
    a.to_coeff()
    b.to_coeff()
    return (
        a.r == b.r
        and all(a.get_a_poly(i) == b.get_a_poly(i) for i in range(a.r))
        and a.get_b_poly() == b.get_b_poly()
    )


def _check_automorphism_sum(scheme, Rp, key, gens, radix, n_threads):
    ms = [Rp.random_element() for _ in gens]
    cts = [enc(scheme, Rp, m, key) for m in ms]
    for c in cts[::2]:
        c.to_NTT()
    domains = [c.repr for c in cts]
    ksks = [
        None
        if g == 1
        else scheme.gen_ksk_automorphism(key, key, g, radix_log_base=radix)
        for g in gens
    ]
    out = scheme.automorphism_sum(cts, gens, ksks, n_threads)
    assert [c.repr for c in cts] == domains
    assert out.repr == repr.coeff and out.lvl == cts[0].lvl and out.ring == cts[0].ring
    expected = sum(
        (m.automorphism(g) for m, g in zip(ms, gens, strict=True)),
        start=Polynomial(Rp).from_array([0] * Rp.N),
    )
    assert scheme.linear_decrypt(out, key).round_division(Rp) == expected
    return cts, ksks, out


@pytest.mark.parametrize("radix", [None, RADIX_LOG_BASE])
@pytest.mark.usefixtures("threads")
def test_automorphism_sum(ghs, radix):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    gens = [5, 1, 25, 2 * N - 1, 125, 1, 3]
    cts, ksks, out = _check_automorphism_sum(scheme, Rp, key, gens, radix, 1)
    # The key products are summed exactly before the single division, so the
    # result does not depend on how the terms are split between threads.
    for n_threads in (2, 3, 0):
        assert _same_sample(scheme.automorphism_sum(cts, gens, ksks, n_threads), out)


def test_automorphism_sum_mixes_digit_choices(ghs):
    # Each term uses its own key's digit choice, so keys with and without
    # centered digits can share one sum.
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    gens = [5, 25, 125, 2 * N - 1]
    ms = [Rp.random_element() for _ in gens]
    cts = [enc(scheme, Rp, m, key) for m in ms]
    ksks = []
    for i, g in enumerate(gens):
        scheme.balanced = i % 2 == 0
        ksks.append(scheme.gen_ksk_automorphism(key, key, g))
    out = scheme.automorphism_sum(cts, gens, ksks)
    expected = sum(
        (m.automorphism(g) for m, g in zip(ms, gens, strict=True)),
        start=Polynomial(Rp).from_array([0] * Rp.N),
    )
    assert scheme.linear_decrypt(out, key).round_division(Rp) == expected


@pytest.mark.parametrize("r, N_r", RANK_DIMS)
@pytest.mark.usefixtures("threads")
def test_automorphism_sum_module_rank(r, N_r):
    _Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    _check_automorphism_sum(scheme, Rp, key, [5, 1, 25], None, 0)


def test_automorphism_sum_without_a_key_switch_is_the_plain_sum(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    cts = [enc(scheme, Rp, Rp.random_element(), key) for _ in range(3)]
    out = scheme.automorphism_sum(cts, [1, 1, 1], [None, None, None])
    assert _same_sample(out, cts[0] + cts[1] + cts[2])


def test_automorphism_sum_refusals(ghs):
    _Rq, Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    cts = [enc(scheme, Rp, Rp.random_element(), key) for _ in range(2)]
    ksk5 = scheme.gen_ksk_automorphism(key, key, 5)
    ksk25 = scheme.gen_ksk_automorphism(key, key, 25)
    with pytest.raises(ValueError, match="no key"):
        scheme.automorphism_sum(cts, [5, 25], [ksk5, None])
    # A key for another level is over another ring.
    with pytest.raises(ValueError, match="ring"):
        scheme.automorphism_sum(cts, [5, 25], [ksk5[0], ksk25[1]])
    lower = cts[1].copy().round_division(lvl=1)
    with pytest.raises(ValueError, match="share"):
        scheme.automorphism_sum([cts[0], lower], [5, 25], [ksk5, ksk25])


@pytest.mark.parametrize("n_threads", [1, 0])
@pytest.mark.usefixtures("threads")
def test_linear_combinations(bv, n_threads):
    # Plaintext coefficients scale the linear decryption exactly, so the
    # combination of the decryptions is the decryption of the combination.
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ring = scheme.rings[0]
    cts = [enc(scheme, Rp, Rp.random_element(), key) for _ in range(3)]
    small = [Polynomial(ring).from_array(_ternary(N)) for _ in range(5)]
    rows = [[small[0], None, small[1]], [small[2], small[3], small[4]]]
    outs = scheme.linear_combinations(cts, rows, n_threads)
    decryptions = [scheme.linear_decrypt(c, key) for c in cts]
    for row, out in zip(rows, outs, strict=True):
        expected = sum(
            (p * d for p, d in zip(row, decryptions, strict=True) if p is not None),
            start=Polynomial(ring).from_array([0] * N),
        )
        assert scheme.linear_decrypt(out, key) == expected


@pytest.mark.parametrize("n_threads", [1, 0])
@pytest.mark.usefixtures("threads")
def test_linear_combinations_with_a_prepared_matrix(bv, n_threads):
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ring = scheme.rings[0]
    cts = [enc(scheme, Rp, Rp.random_element(), key) for _ in range(3)]
    small = [Polynomial(ring).from_array(_ternary(N)) for _ in range(4)]
    rows = [[small[0], None, small[1]], [None, small[2], small[3]]]
    matrix = PlaintextMatrix(rows)
    assert all(p.repr == repr.ntt for p in small)
    expected = scheme.linear_combinations(cts, rows, n_threads)
    for _ in range(2):
        outs = scheme.linear_combinations(cts, matrix, n_threads)
        for out, exp in zip(outs, expected, strict=True):
            assert _same_sample(out, exp)
    with pytest.raises(ValueError, match="one coefficient per ciphertext"):
        scheme.linear_combinations(cts[:2], matrix)
    with pytest.raises(ValueError, match="one length"):
        PlaintextMatrix([[small[0]], [small[1], small[2]]])
    other = Polynomial(scheme.rings[1]).from_array(_ternary(N))
    with pytest.raises(ValueError, match="one ring"):
        PlaintextMatrix([[small[0], other]])


@pytest.mark.parametrize("scheme_fixture", ["bv", "ghs"])
@pytest.mark.parametrize("n_threads", [1, 0])
@pytest.mark.usefixtures("threads")
def test_multiply_batch_divided_to_a_level(scheme_fixture, n_threads, request):
    # The fused call matches the two batch calls bit for bit: it only skips a
    # forward and inverse transform that cancel.
    _Rq, Rp, scheme = request.getfixturevalue(scheme_fixture)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(s_0 * s_0)])
    cts = [
        enc(scheme, Rp, Polynomial(Rp).from_array(_ternary(N)), key) for _ in range(4)
    ]
    cts[1].to_NTT()
    # cts[0] appears twice, cts[2] and cts[3] once, in either domain.
    lhs, rhs = [cts[0], cts[0], cts[2]], [cts[1], cts[0], cts[3]]
    fused = scheme.multiply_batch(lhs, rhs, rlk, n_threads, lvl=1)
    assert all(c.repr == repr.ntt for c in cts)
    separate = scheme.round_division_batch(scheme.multiply_batch(lhs, rhs, rlk), 1)
    for f, s_ in zip(fused, separate, strict=True):
        assert f.lvl == 1 and f.ring == scheme.rings[1] and f.repr == repr.coeff
        assert ffi.cast("MLWE", f.obj).ring == scheme.rings[1].arith_ring
        assert _same_sample(f, s_)
    with pytest.raises(ValueError, match="key"):
        scheme.multiply_batch(lhs, rhs, None, lvl=1)


def test_gen_ksk_rejects_a_radix_larger_than_the_primes(bv):
    _Rq, _Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    with pytest.raises(ValueError, match="smallest prime"):
        scheme.gen_ksk(key, key, radix_log_base=64)


def test_special_primes_of_a_ring_that_is_not_the_first_of_its_base():
    # Prime indices are global to the (N, split_degree) base, so a ring built
    # after another one starts above index 0. The derived level chain has to
    # take its special primes from the full ring's mask; taking them from a
    # prime count gave the scheme a special ring equal to its level 0, which
    # silently downgrades every GHS key switch to BV.
    Ring(N, prime_size=[45, 45, 45, 50], split_degree=1)
    Rq = Ring(N, prime_size=[32, 32, 32, 32], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=2)

    assert scheme.special_primes == 1
    assert scheme.special_rings[0].mask == Rq.mask
    assert scheme.special_rings[0].ell == scheme.rings[0].ell + 1


def _scheme_over_a_populated_base(module_rank=1):
    """A GHS scheme whose ring is not the first built over its base.

    Its working primes' base indices do not ascend -- the first is new, the
    others were registered by an earlier ring -- and its special prime's index
    is below theirs. The gadget has to be laid out in the order the key switch
    walks the ciphertext's primes, which is neither ``ring.primes`` order nor
    base-index order over the whole key ring.
    """
    Ring(N, prime_size=[43, 43, 53], split_degree=1)
    Rq = Ring(N, prime_size=[37, 43, 43, 53], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=module_rank)
    work = scheme.rings[0].prime_indices
    special = set(scheme.special_rings[0].prime_indices) - set(work)
    assert work != sorted(work)
    assert min(special) < max(work)
    return Rq, Rq.quotient_ring(ell=1), scheme


@pytest.mark.parametrize("radix_log_base", [None, RADIX_LOG_BASE])
def test_keyswitch_over_a_populated_base(radix_log_base):
    _Rq, Rp, scheme = _scheme_over_a_populated_base()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    key2 = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key2, key, radix_log_base=radix_log_base)

    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)
    for c in (c0, c0.round_division(lvl=1)):
        out = scheme.keyswitch(c, ksk)
        assert scheme.linear_decrypt(out, key2).round_division(Rp) == m0


@pytest.mark.parametrize("radix_log_base", [None, RADIX_LOG_BASE])
def test_multiplication_over_a_populated_base(radix_log_base):
    Rq, Rp, scheme = _scheme_over_a_populated_base()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    s_0 = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s_0 * s_0)], radix_log_base=radix_log_base)

    m1 = Polynomial(Rp).from_array(_ternary(N))
    m2 = Polynomial(Rp).from_array(_ternary(N))
    c1 = enc(scheme, Rp, m1, key)
    c2 = enc(scheme, Rp, m2, key)

    m_out = scheme.linear_decrypt(c1 * c2, key).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.parametrize("radix_log_base", [None, RADIX_LOG_BASE])
def test_mgsw_external_product_over_a_populated_base(radix_log_base):
    _Rq, Rp, scheme = _scheme_over_a_populated_base()
    key = scheme.key_gen_sparse(N // 8, 3.2)
    mgsw_scheme = MGSW_Scheme(scheme, radix_log_base=radix_log_base)

    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)

    ct_id = mgsw_scheme.encrypt(Polynomial(Rp).from_array([1] + [0] * (N - 1)), key)
    res = ct_id.external_product(ct1)
    assert scheme.linear_decrypt(res, key).round_division(Rp) == m1


# --- Key generation on several threads ---------------------------------------
#
# Every key is sampled through MLWE_Scheme.sample_scaled, which draws each
# sample from seeds of its own: the thread count changes who draws a sample,
# never what it holds.


def _digests(samples):
    """Every component of every sample by digest, so that two keys compare
    equal exactly when they hold the same samples."""
    return [
        [c.get_a_poly(j).get_hash() for j in range(c.r)] + [c.get_b_poly().get_hash()]
        for c in samples
    ]


def _ksk_digests(ksk):
    return _digests([c for component in ksk.mlwe if component for c in component])


def _every_key(scheme, n_threads):
    """One key of every kind, pinned, drawn on up to ``n_threads`` threads."""
    mgsw_scheme = MGSW_Scheme(scheme)
    one = Polynomial(scheme.rings[0]).from_array([1])
    with entropy.deterministic(0x7EAD5):
        key = scheme.key_gen_sparse(scheme.N * scheme.r // 8, 3.2)
        key2 = scheme.key_gen_sparse(scheme.N * scheme.r // 8, 3.2)
        ksk = scheme.gen_ksk(key2, key, lvl=0, n_threads=n_threads)
        rlk = scheme.gen_rlk(key, key, lvl=0, n_threads=n_threads)
        auts = scheme.gen_ksk_automorphism_set(
            key, key, [3, 5, 9], lvl=0, n_threads=n_threads
        )
        mgsw = mgsw_scheme.encrypt(one, key, n_threads=n_threads)
        constants = mgsw_scheme.encrypt_constants([1, -1, 0], key, n_threads=n_threads)
    return [
        _ksk_digests(ksk),
        _ksk_digests(rlk),
        *[_ksk_digests(aut) for aut in auts],
        _digests(mgsw.obj),
        *[_digests(c.obj) for c in constants],
    ]


@pytest.mark.parametrize("r, N_r", [(1, 256), (4, 64)])
@pytest.mark.usefixtures("threads")
def test_keygen_does_not_depend_on_the_thread_count(r, N_r):
    """Pinned, every key generator gives the same key on 1, 3 and 8 threads."""
    _Rq, _Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    keys = [_every_key(scheme, n_threads) for n_threads in (1, 3, 0)]
    assert keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize("lvl", [0, 1])
@pytest.mark.usefixtures("threads")
def test_keygen_without_the_special_primes_does_not_depend_on_the_thread_count(
    ghs, lvl
):
    """Pinned, a ``hybrid=False`` key is the same on 1, 3 and 8 threads, and
    stays in the level's own ring on each."""
    _Rq, _Rp, scheme = ghs

    def bv_key(n_threads):
        with entropy.deterministic(0x7EAD5):
            key = scheme.key_gen_sparse(N // 8, 3.2)
            key2 = scheme.key_gen_sparse(N // 8, 3.2)
            ksk = scheme.gen_ksk(
                key2, key, lvl=lvl, radix_log_base=4, n_threads=n_threads, hybrid=False
            )
        assert all(c.ring == scheme.rings[lvl] for c in ksk.mlwe[0])
        return _ksk_digests(ksk)

    keys = [bv_key(n_threads) for n_threads in (1, 3, 0)]
    assert keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize("radix", [None, 9])
@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.usefixtures("threads")
def test_single_operations_do_not_depend_on_the_thread_count(radix, r):
    """A key switch, an automorphism, a product, a division and hoisted
    automorphisms split over the primes give the same samples on 1, 3 and 8
    threads. N = 4096 is large enough for them to run in parallel."""
    n = 4096
    Rq = Ring(n, prime_size=[45] * 5 + [50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=r)
    with entropy.deterministic(0x1A7E):
        key = scheme.key_gen_sparse(n * r // 8, 3.2)
        key2 = scheme.key_gen_sparse(n * r // 8, 3.2)
        ksk = scheme.gen_ksk(key2, key, radix_log_base=radix)
        auts = [
            scheme.gen_ksk_automorphism(key, key, g, radix_log_base=radix)
            for g in (5, 25)
        ]
        rlk = scheme.gen_rlk(key, key, radix_log_base=radix)
        c = enc(scheme, Rp, Rp.random_element(), key)

    def run(n_threads):
        out = [scheme.keyswitch(c.copy(), ksk, n_threads)]
        out.append(scheme.automorphism(c.copy(), 5, auts[0], n_threads))
        prod = scheme.multiply(c.copy(), c.copy(), rlk, n_threads)
        out.append(prod.copy())
        out.append(prod.round_division(lvl=1, n_threads=n_threads))
        with engine.local_num_threads(n_threads):
            out += scheme.automorphisms(c.copy(), [5, 25], auts)
        for x in out:
            x.to_coeff()
        return _digests(out)

    runs = [run(n_threads) for n_threads in (1, 3, 0)]
    assert runs[0] == runs[1] == runs[2]
    m_out = scheme.linear_decrypt(scheme.keyswitch(c.copy(), ksk, 3), key2)
    assert m_out.round_division(Rp) == scheme.linear_decrypt(
        c.copy(), key
    ).round_division(Rp)


@pytest.mark.parametrize("r, N_r", [(1, 256), *RANK_DIMS])
@pytest.mark.usefixtures("threads")
def test_keys_drawn_on_several_threads_switch_keys(r, N_r):
    """Keys drawn on 8 threads key-switch, automorph and relinearize."""
    Rq, Rp, scheme = _rank_scheme(N_r, r, special_primes=1)
    key = _rank_key(scheme, N_r, r)
    key2 = _rank_key(scheme, N_r, r)
    m0 = Rp.random_element()
    c0 = enc(scheme, Rp, m0, key)

    ksk = scheme.gen_ksk(key2, key)
    assert (
        scheme.linear_decrypt(scheme.keyswitch(c0, ksk), key2).round_division(Rp) == m0
    )

    gens = [3, 5]
    for g, aut in zip(
        gens, scheme.gen_ksk_automorphism_set(key, key, gens), strict=True
    ):
        c_out = scheme.automorphism(c0, g, aut)
        assert scheme.linear_decrypt(c_out, key).round_division(Rp) == m0.automorphism(
            g
        )

    scheme.rlk = scheme.gen_rlk(key, key)
    m1 = Polynomial(Rp).from_array(_ternary(N_r))
    m2 = Polynomial(Rp).from_array(_ternary(N_r))
    c1, c2 = enc(scheme, Rp, m1, key), enc(scheme, Rp, m2, key)
    m_out = scheme.linear_decrypt(c1 * c2, key).round_division(Rp)
    assert _mul_error(Rq, Rp, scheme, m_out, m1, m2) < 1000


@pytest.mark.usefixtures("threads")
def test_mgsw_constants_scale_by_their_values(bv):
    """`encrypt_constants([v])` encrypts the constant polynomial v: its external
    product multiplies a ciphertext's message by v."""
    _Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m1 = Rp.random_element()
    ct1 = enc(scheme, Rp, m1, key)
    values = [1, 0, -1, 2]
    constants = MGSW_Scheme(scheme).encrypt_constants(values, key)
    for v, mgsw in zip(values, constants, strict=True):
        res = mgsw.external_product(ct1)
        expected = m1 * Polynomial(Rp).from_array([v])
        assert scheme.linear_decrypt(res, key).round_division(Rp) == expected


@pytest.mark.usefixtures("threads")
def test_key_samples_keep_the_seeds_of_their_masks(ghs):
    """Key samples carry their mask seeds, as `sample`'s do, so a key is
    written as seeds and bodies."""
    _Rq, _Rp, scheme = ghs
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ksk = scheme.gen_ksk(key, key, lvl=0)
    samples = [c for component in ksk.mlwe for c in component]
    seeds = [seed_still_holds(c) for c in samples]
    assert all(seed is not None for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test_sample_scaled_rejects_what_it_cannot_encrypt(bv):
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    one = Polynomial(Rq).from_array([1])
    with pytest.raises(ValueError, match="key's ring"):
        scheme.sample_scaled([Polynomial(Rp).from_array([1])], [[1, 1, 1]], key)
    with pytest.raises(ValueError, match="values per scalar"):
        scheme.sample_scaled([one], [[1, 1]], key)
    rank_two = MLWE_Scheme(Rq, special_primes=0, module_rank=2)
    with pytest.raises(ValueError, match="rank"):
        scheme.sample_scaled([one], [[1, 1, 1]], rank_two.key_gen_sparse(N // 8, 3.2))
    assert scheme.sample_scaled([one], [], key) == []


def test_sample_scaled_leaves_the_messages_alone(bv):
    """A message in coefficient form is encrypted as it is, not converted in
    place under a caller that may share it."""
    Rq, Rp, scheme = bv
    key = scheme.key_gen_sparse(N // 8, 3.2)
    m = Polynomial(Rq).from_array([1])
    m.to_coeff()
    delta = Rq.modulus_ratio(Rp)
    (c,) = scheme.sample_scaled([m], [[delta % p for p in Rq.primes]], key)
    assert m.repr == repr.coeff
    assert scheme.linear_decrypt(c, key).round_division(Rp) == Polynomial(
        Rp
    ).from_array([1])


def test_an_adopted_sample_must_match_its_ring_and_rank(bv):
    Rq, Rp, scheme = bv
    sample = MLWE(scheme)
    with pytest.raises(ValueError, match="does not live over"):
        MLWE(scheme, ring=Rp, obj=sample.obj)
    with pytest.raises(ValueError, match="does not live over"):
        MLWE(scheme, ring=Rq, rank=2, obj=sample.obj)
