# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The GP25 sparse-amortized bootstrap: input keys and their gaps, the
building blocks (rotation, signed monomials, extraction, packing) and the
whole bootstrap over the options it takes."""

import random

import pytest
from vfhe import engine
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import GP25, mod_switch
from vfhe.mlwe import LWE, MLWE, LWE_Key, MLWE_Key, MLWE_Scheme


def _scheme_over_a_populated_base(N, module_rank):
    """A GHS scheme whose ring is not the first built over its base.

    Its working primes' base indices do not ascend and its special prime's
    index is below theirs, so neither the gadget nor the LWE limbs line up
    with ``ring.primes`` order.
    """
    Ring(N, prime_size=[50, 50, 55], split_degree=1)
    Rq = Ring(N, prime_size=[47, 50, 50, 55], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=module_rank)
    work = scheme.rings[0].prime_indices
    special = set(scheme.special_rings[0].prime_indices) - set(work)
    assert work != sorted(work)
    assert min(special) < max(work)
    return scheme


def _key_with(scheme, positions):
    """A key whose components are non-zero (alternately 1 and -1) at
    ``positions``."""
    key = []
    for _ in range(scheme.r):
        poly = [0] * scheme.N
        for t, j in enumerate(positions):
            poly[j] = 1 - 2 * (t & 1)
        key.append(poly)
    return MLWE_Key(key, 3.2, scheme)


# --- input keys --------------------------------------------------------------


def test_gaps_end_at_the_lowest_coefficient():
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=0
    )
    assert GP25.gaps(_key_with(scheme, [40, 33])) == [[24, 7, 33]]
    assert GP25.gaps(_key_with(scheme, [0])) == [[64, 0]]


def test_every_gap_must_fit_including_the_last():
    # The last gap, from the lowest non-zero coefficient down to 0, is as
    # much a rotation as the others.
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=0
    )
    out_scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=1
    )
    output_key = out_scheme.key_gen_sparse(8, 3.2)
    gp25 = GP25(out_scheme)
    with pytest.raises(ValueError, match="a gap of the input key"):
        gp25.generate_bootstrap_key(_key_with(scheme, [40, 33]), output_key, 2, 5)
    with pytest.raises(ValueError, match="a gap of the input key"):
        gp25.generate_bootstrap_key(_key_with(scheme, [30, 3]), output_key, 2, 5)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("ternary", [False, True])
def test_sampled_input_keys_fit_their_gaps(rank, ternary):
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1),
        special_primes=0,
        module_rank=rank,
    )
    for _ in range(20):
        key = GP25.sample_input_key(scheme, 4, 5, 3.2, ternary=ternary)
        for poly, gaps in zip(key.key, GP25.gaps(key), strict=True):
            assert sum(c != 0 for c in poly) == 4
            assert set(poly) <= ({-1, 0, 1} if ternary else {0, 1})
            assert all(g < 1 << 5 for g in gaps)


def test_key_shape_is_checked():
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=0
    )
    out_scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=1
    )
    output_key = out_scheme.key_gen_sparse(8, 3.2)
    gp25 = GP25(out_scheme)
    key = _key_with(scheme, [40, 33, 20])
    with pytest.raises(ValueError, match="h = 2 non-zero"):
        gp25.generate_bootstrap_key(key, output_key, 2, 7)
    with pytest.raises(ValueError, match="binary"):
        gp25.generate_bootstrap_key(key, output_key, 3, 7, ternary=False)
    with pytest.raises(ValueError, match="gap_bits must be in"):
        gp25.generate_bootstrap_key(key, output_key, 3, 0)
    with pytest.raises(ValueError, match="log2"):
        gp25.generate_bootstrap_key(key, output_key, 3, 8)


# --- building blocks ---------------------------------------------------------


def _accumulators(scheme, key, n):
    """``n`` samples, sample k encrypting (k + 1) * Delta * X: distinct, and
    changed by X -> X^-1 (into -(k + 1) * Delta * X^(N - 1))."""
    delta = 1 << 60
    ring = scheme.rings[0]
    acc = [
        scheme.sample(Polynomial(ring).from_bigint_array([0, (k + 1) * delta]), key)
        for k in range(n)
    ]
    return acc, delta


def _constant_or_reflected(scheme, key, c, delta):
    """``k`` for a sample of ``k * Delta * X``, ``-k`` for ``-k * Delta * X^(N - 1)``."""
    d = scheme.linear_decrypt(c, key).get_polynomial(signed=True)
    N = len(d)
    if abs(d[1]) > delta // 2:
        return round(d[1] / delta)
    return round(d[N - 1] / delta)


@pytest.mark.parametrize("d", [0, 1, 5, 8, 15, 16])
def test_rotate(d):
    N, n = 64, 16
    scheme = MLWE_Scheme(
        Ring(N, prime_size=[50, 50, 50], split_degree=1), special_primes=1
    )
    key = scheme.key_gen_sparse(8, 3.2)
    gp25 = GP25(scheme)
    sab = gp25.generate_bootstrap_key(
        GP25.sample_input_key(
            MLWE_Scheme(Ring(n, prime_size=[50, 50], split_degree=1), special_primes=0),
            2,
            5,
            3.2,
        ),
        key,
        2,
        5,
    )
    acc, delta = _accumulators(scheme, key, n)
    bits = gp25.mgsw_scheme.encrypt_constants([(d >> b) & 1 for b in range(5)], key)
    gp25.rotate(acc, bits, sab)
    got = [_constant_or_reflected(scheme, key, c, delta) for c in acc]
    assert got == [k - d + 1 if k >= d else -(k - d + n + 1) for k in range(n)]


@pytest.mark.parametrize("sign", [None, 0, 1])
def test_multiply_by_signed_monomials(sign):
    N, n = 64, 8
    scheme = MLWE_Scheme(
        Ring(N, prime_size=[50, 50, 50], split_degree=1), special_primes=1
    )
    key = scheme.key_gen_sparse(8, 3.2)
    gp25 = GP25(scheme)
    acc, delta = _accumulators(scheme, key, n)
    a = [3 * k + 1 for k in range(n)]
    sign_key = (
        None if sign is None else gp25.mgsw_scheme.encrypt_constants([sign], key)[0]
    )
    gp25.multiply_by_signed_monomials(acc, a, sign_key)
    for k, c in enumerate(acc):
        e = (1 + (-a[k] if sign else a[k])) % (2 * N)
        expected = [0] * N
        expected[e % N] = (k + 1) * (1 if e < N else -1)
        d = scheme.linear_decrypt(c, key).get_polynomial(signed=True)
        assert [round(x / delta) for x in d] == expected


def test_lwe_extraction():
    N = 256
    Rq = Ring(N, prime_size=[50, 50, 50], split_degree=1)
    Rp = Rq.quotient_ring(ell=1)
    scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=1)
    key = scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(scheme)
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = gp25.rlwe_extract_lwe(rlwe_sample, idx)
        decryption = lwe_sample.linear_decrypt(lwe_key, recompose=True)
        res = mod_switch(decryption, Rq.q_l, Rp.primes[0])
        diff = (res - msg_coeffs[idx]) % Rp.primes[0]
        diff = min(diff, Rp.primes[0] - diff)
        assert diff < 1000


def test_lwe_extraction_over_a_populated_base():
    N = 256
    scheme = _scheme_over_a_populated_base(N, module_rank=1)
    Rq = scheme.rings[0]
    Rp = Rq.quotient_ring(ell=1)
    key = scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(scheme)
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = gp25.rlwe_extract_lwe(rlwe_sample, idx)
        decryption = lwe_sample.linear_decrypt(lwe_key, recompose=True)
        res = mod_switch(decryption, Rq.q_l, Rp.primes[0])
        diff = (res - msg_coeffs[idx]) % Rp.primes[0]
        assert min(diff, Rp.primes[0] - diff) < 1000


def _check_packing(out_scheme, gp25, output_key, count, stride):
    Rq = out_scheme.rings[0]
    lwe_key = LWE_Key(ring=Rq, sec_sigma=3.2, err_sigma=3.2, n=Rq.N)
    packing_key = gp25.gen_packing_ksk(output_key, lwe_key)
    extracted = []
    for i in range(count):
        m_i = mod_switch((i * 137), 2000, Rq.q_l)
        extracted.append(LWE(ring=Rq, m=[m_i % q for q in Rq.primes], key=lwe_key))

    out_repacked = gp25.packing_keyswitch(extracted, packing_key, stride)
    out_coeffs = out_scheme.linear_decrypt(out_repacked, output_key).get_polynomial()
    for j in range(Rq.N):
        expected = (j // stride * 137) % 2000 if j % stride == 0 else 0
        m_j = mod_switch(out_coeffs[j], Rq.q_l, 2000)
        diff = (m_j - expected) % 2000
        assert min(diff, 2000 - diff) <= 5


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("keygen_threads", [1, 4])
def test_packing_ksk(balanced, keygen_threads):
    Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=0, module_rank=4, balanced=balanced)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    gp25 = GP25(out_scheme)
    engine.set_num_threads(keygen_threads)
    try:
        _check_packing(out_scheme, gp25, output_key, 256, 1)
    finally:
        engine.set_num_threads()


@pytest.mark.parametrize("stride", [2, 16])
def test_packing_ksk_at_a_stride(stride):
    Rq = Ring(256, prime_size=[50, 50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=1)
    output_key = out_scheme.key_gen_sparse(64, 3.2)
    _check_packing(out_scheme, GP25(out_scheme), output_key, 256 // stride, stride)


def test_packing_ksk_radix():
    Rq = Ring(256, prime_size=[50, 50], split_degree=1)
    out_scheme = MLWE_Scheme(Rq, special_primes=0, max_lvl=1)
    output_key = out_scheme.key_gen_sparse(64, 3.2)
    gp25 = GP25(out_scheme, radix_log_base=10)
    _check_packing(out_scheme, gp25, output_key, 256, 1)


def test_packing_ksk_over_a_populated_base():
    out_scheme = _scheme_over_a_populated_base(256, module_rank=4)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    _check_packing(out_scheme, GP25(out_scheme), output_key, 256, 1)


def test_test_vector_needs_a_dividing_table():
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=1
    )
    with pytest.raises(ValueError, match="divide N"):
        GP25(scheme).test_vector([0, 1, 2])


# --- the bootstrap -----------------------------------------------------------

PRECISION = 4

CASES = {
    "default": {},
    "trace_repack": {"trace_repack": True},
    "binary": {"ternary": False},
    "input_rank_2": {"in_rank": 2},
    "output_rank_2": {"out_rank": 2},
    "unbalanced": {"balanced": False},
    "smaller_input": {"n": 16},
    "smaller_input_trace": {"n": 16, "trace_repack": True},
    "radix_without_special_primes": {"radix": 10, "primes": [50, 50], "special": 0},
    "several_levels": {"primes": [50, 50, 50, 50], "max_lvl": None},
}


def _bootstrap_case(
    N=64,
    n=64,
    h=4,
    in_rank=1,
    out_rank=1,
    ternary=True,
    trace_repack=False,
    radix=None,
    balanced=True,
    primes=(50, 50, 50),
    special=1,
    max_lvl=1,
):
    """Keys, a test vector and an input for one bootstrap, and what it must
    decrypt to."""
    in_ring = Ring(n, prime_size=[50, 50], split_degree=1)
    in_scheme = MLWE_Scheme(in_ring, special_primes=0, module_rank=in_rank)
    out_scheme = MLWE_Scheme(
        Ring(N, prime_size=list(primes), split_degree=1),
        special_primes=special,
        module_rank=out_rank,
        max_lvl=max_lvl,
        balanced=balanced,
    )
    bits = n.bit_length()
    input_key = GP25.sample_input_key(in_scheme, h, bits, 3.2, ternary=ternary)
    output_key = out_scheme.key_gen_sparse(16, 3.2)
    gp25 = GP25(out_scheme, radix_log_base=radix, trace_repack=trace_repack)
    key = gp25.generate_bootstrap_key(input_key, output_key, h, bits, ternary=ternary)

    q = out_scheme.rings[0].q_l
    size = 1 << (PRECISION - 1)
    table = [random.randrange(size) for _ in range(size)]  # noqa: S311 - test data
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in table])
    msg = [random.randrange(size) for _ in range(n)]  # noqa: S311 - test data
    m = Polynomial(in_ring).from_bigint_array(
        [mod_switch(x, 1 << PRECISION, in_ring.q_l) for x in msg]
    )
    rlwe_in = in_scheme.sample(m, input_key)
    expected = [0] * N
    for k, x in enumerate(msg):
        expected[k * (N // n)] = table[x]
    return gp25, key, tv, rlwe_in, output_key, out_scheme, expected


def _decrypt_messages(out_scheme, out, output_key):
    q = out_scheme.rings[0].q_l
    d = out_scheme.linear_decrypt(out, output_key).get_polynomial()
    return [mod_switch(v, q, 1 << PRECISION) for v in d]


@pytest.mark.parametrize("case", CASES)
def test_bootstrap(deterministic_prng, case):
    deterministic_prng(0x5AB00002)
    random.seed(0x5AB00002)
    gp25, key, tv, rlwe_in, output_key, out_scheme, expected = _bootstrap_case(
        **CASES[case]
    )
    out = MLWE(out_scheme)
    gp25.bootstrap(out, rlwe_in, tv, key)
    assert _decrypt_messages(out_scheme, out, output_key) == expected


def test_bootstrap_does_not_depend_on_the_threads(deterministic_prng):
    deterministic_prng(0x5AB00003)
    random.seed(0x5AB00003)
    gp25, key, tv, rlwe_in, _output_key, out_scheme, _ = _bootstrap_case(in_rank=2)
    results = []
    for threads in (1, 4):
        engine.set_num_threads(threads)
        try:
            out = MLWE(out_scheme)
            gp25.bootstrap(out, rlwe_in, tv, key)
        finally:
            engine.set_num_threads()
        out.to_coeff()
        results.append(
            [out.get_a_poly(0).get_coeff_matrix(), out.get_b_poly().get_coeff_matrix()]
        )
    assert results[0] == results[1]


@pytest.mark.complete
@pytest.mark.parametrize("threads", [1, 4])
def test_bootstrap_at_256(deterministic_prng, threads):
    deterministic_prng(0x5AB00001)
    random.seed(0x5AB00001)
    gp25, key, tv, rlwe_in, output_key, out_scheme, expected = _bootstrap_case(
        N=256, n=256, h=17, out_rank=4
    )
    engine.set_num_threads(threads)
    try:
        out = MLWE(out_scheme)
        gp25.bootstrap(out, rlwe_in, tv, key)
    finally:
        engine.set_num_threads()
    assert _decrypt_messages(out_scheme, out, output_key) == expected
