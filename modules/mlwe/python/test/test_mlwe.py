# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.mlwe over the cffi boundary.

Encrypt/decrypt roundtrips, homomorphic add/sub/mul-by-poly, BV and GHS
key-switching, automorphism, ciphertext multiplication (relinearization), the
MGSW external product, and the LWE surface. Noise is stripped with
round_division back to the plaintext ring, so equality is exact.
"""

import math
import random
from typing import cast

import pytest
import vfhe.engine as engine
from vfhe.arith import Polynomial, Ring
from vfhe.arith.number_theory import crt
from vfhe.crypto import entropy
from vfhe.engine import ffi
from vfhe.mlwe import LWE, LWE_Key, MGSW_Scheme, MLWE_Scheme, MLWE_Set

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
    outs = scheme.automorphism_batch(cts, gens, ksks, n_threads)
    for m, g, out in zip(ms, gens, outs, strict=True):
        assert scheme.linear_decrypt(out, key).round_division(Rp) == m.automorphism(g)
    with pytest.raises(ValueError, match="no key"):
        scheme.automorphism_batch(cts[:1], [5], [None])


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


def _scheme_over_a_populated_base():
    """A GHS scheme whose ring is not the first built over its base.

    Its working primes' base indices do not ascend -- the first is new, the
    others were registered by an earlier ring -- and its special prime's index
    is below theirs. The gadget has to be laid out in the order the key switch
    walks the ciphertext's primes, which is neither ``ring.primes`` order nor
    base-index order over the whole key ring.
    """
    Ring(N, prime_size=[43, 43, 53], split_degree=1)
    Rq = Ring(N, prime_size=[37, 43, 43, 53], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=1)
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
