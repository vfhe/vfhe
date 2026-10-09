# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The GP25 sparse-amortized bootstrap: input keys and their gaps, the
building blocks (rotation, signed monomials, extraction, packing) and the
whole bootstrap over the options it takes, composed with itself."""

import random

import pytest
from vfhe import engine
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import GP25, mod_switch
from vfhe.mlwe import LWE, LWE_Key, MLWE_Key, MLWE_Scheme, MLWE_Set


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


def _schemes(
    n,
    N,
    primes=(50, 50, 50),
    special=1,
    io_rank=1,
    rotation_rank=1,
    balanced=True,
    rotation_levels=1,
):
    """The input and output keys' scheme over ``R_n`` and the rotation scheme
    over ``R_N``, on the same primes: drawn for the larger dimension (an NTT
    prime for 2 max(n, N) serves both) and given to the other."""
    big, small = (n, N) if n >= N else (N, n)
    first = Ring(big, prime_size=list(primes), split_degree=1)
    second = Ring(small, primes=list(first.primes), split_degree=1)
    io_ring, rotation_ring = (first, second) if n >= N else (second, first)
    io = MLWE_Scheme(io_ring, special_primes=special, module_rank=io_rank)
    rotation = MLWE_Scheme(
        rotation_ring,
        special_primes=special,
        module_rank=rotation_rank,
        max_lvl=rotation_levels,
        balanced=balanced,
    )
    return io, rotation


def _keys(io, rotation, h=4, gap_bits=None, ternary=True, same_key=False):
    gap_bits = io.N.bit_length() if gap_bits is None else gap_bits
    input_key = GP25.sample_input_key(io, h, gap_bits, 3.2, ternary=ternary)
    output_key = io.key_gen_sparse(16 * io.r, 3.2)
    rotation_key = (
        MLWE_Key(output_key.key, 3.2, rotation)
        if same_key
        else rotation.key_gen_sparse(16 * rotation.r, 3.2)
    )
    return input_key, rotation_key, output_key, gap_bits


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
    io, rotation = _schemes(64, 64)
    _, rotation_key, output_key, _ = _keys(io, rotation)
    gp25 = GP25(rotation)
    for positions in ([40, 33], [30, 3]):
        with pytest.raises(ValueError, match="a gap of the input key"):
            gp25.generate_bootstrap_key(
                _key_with(io, positions), rotation_key, output_key, 2, 5
            )


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


def test_keys_are_checked():
    io, rotation = _schemes(64, 64)
    _, rotation_key, output_key, _ = _keys(io, rotation)
    gp25 = GP25(rotation)
    key = _key_with(io, [40, 33, 20])
    with pytest.raises(ValueError, match="h = 2 non-zero"):
        gp25.generate_bootstrap_key(key, rotation_key, output_key, 2, 7)
    with pytest.raises(ValueError, match="binary"):
        gp25.generate_bootstrap_key(key, rotation_key, output_key, 3, 7, ternary=False)
    with pytest.raises(ValueError, match="gap_bits must be in"):
        gp25.generate_bootstrap_key(key, rotation_key, output_key, 3, 0)
    with pytest.raises(ValueError, match="log2"):
        gp25.generate_bootstrap_key(key, rotation_key, output_key, 3, 8)
    with pytest.raises(ValueError, match="one scheme"):
        gp25.generate_bootstrap_key(key, rotation_key, rotation_key, 3, 7)
    with pytest.raises(ValueError, match="this GP25's scheme"):
        gp25.generate_bootstrap_key(key, output_key, output_key, 3, 7)


def test_the_rotation_primes_must_be_a_level_of_the_output():
    io = MLWE_Scheme(
        Ring(64, prime_size=[50, 50, 50], split_degree=1), special_primes=1
    )
    rotation = MLWE_Scheme(
        Ring(64, prime_size=[45, 45, 45], split_degree=1), special_primes=1, max_lvl=1
    )
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    with pytest.raises(ValueError, match="primes of a level"):
        GP25(rotation).generate_bootstrap_key(
            input_key, rotation_key, output_key, 4, bits
        )


def test_trace_repacking_needs_equal_dimensions():
    io, rotation = _schemes(16, 64)
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    with pytest.raises(ValueError, match="trace repacking needs"):
        GP25(rotation, trace_repack=True).generate_bootstrap_key(
            input_key, rotation_key, output_key, 4, bits
        )


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
    automorphism_key = scheme.gen_ksk_automorphism(key, key, 2 * N - 1, lvl=0)
    assert isinstance(automorphism_key, MLWE_Set)
    acc, delta = _accumulators(scheme, key, n)
    bits = gp25.mgsw_scheme.encrypt_constants([(d >> b) & 1 for b in range(5)], key)
    gp25.rotate(acc, bits, automorphism_key)
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
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = scheme.extract_lwe(rlwe_sample, idx)
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
    lwe_key = key.extract_lwe_key()

    msg_coeffs = [((i + 1) * 123) % Rp.primes[0] for i in range(N)]
    msg = Polynomial(Rp).from_array(msg_coeffs)
    delta = Rq.modulus_ratio(Rp, return_pointer=True)
    rlwe_sample = scheme.sample(msg.scaled_lift(Rq, delta=delta), key)

    for idx in [0, 1, N // 2, N - 1]:
        lwe_sample = scheme.extract_lwe(rlwe_sample, idx)
        decryption = lwe_sample.linear_decrypt(lwe_key, recompose=True)
        res = mod_switch(decryption, Rq.q_l, Rp.primes[0])
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
        m_i = mod_switch((i * 137), 2000, Rq.q_l)
        extracted.append(LWE(ring=Rq, m=[m_i % q for q in Rq.primes], key=lwe_key))

    out_repacked = out_scheme.packing_keyswitch(extracted, packing_key)
    out_coeffs = out_scheme.linear_decrypt(out_repacked, output_key).get_polynomial()
    for j in range(Rq.N):
        expected = (j * 137) % 2000 if j < count else 0
        m_j = mod_switch(out_coeffs[j], Rq.q_l, 2000)
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
    out_scheme = _scheme_over_a_populated_base(256, module_rank=4)
    output_key = out_scheme.key_gen_sparse(64, 3.2, ternary=True)
    _check_packing(out_scheme, output_key, 256)


@pytest.mark.parametrize("n", [16, 128])
def test_packing_samples_of_another_dimension(n):
    # Samples extracted from R_N packed into R_n: the two rings have their own
    # bases, so the primes are matched by value.
    N = 64
    io, rotation = _schemes(n, N, rotation_rank=2)
    _, rotation_key, output_key, _ = _keys(io, rotation)
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
    "trace_repack": {"trace": True},
    "trace_repack_one_key": {"trace": True, "same_key": True},
    "binary": {"ternary": False},
    "input_rank_2": {"io_rank": 2},
    "rotation_rank_2": {"rotation_rank": 2},
    "unbalanced": {"balanced": False},
    "smaller_input": {"n": 16},
    "larger_input": {"n": 128},
    "radix_without_special_primes": {"radix": 10, "primes": (50, 50), "special": 0},
    "several_levels": {"primes": (50, 50, 50, 50)},
}


def _bootstrap_case(
    N=64,
    n=64,
    h=4,
    ternary=True,
    trace=False,
    same_key=False,
    radix=None,
    hybrid=True,
    **scheme_options,
):
    """A GP25 with its keys, a test vector, an input under the output key at
    the lowest level, and what its bootstrap must decrypt to."""
    io, rotation = _schemes(n, N, **scheme_options)
    input_key, rotation_key, output_key, bits = _keys(
        io, rotation, h=h, ternary=ternary, same_key=same_key
    )
    gp25 = GP25(rotation, radix_log_base=radix, trace_repack=trace)
    key = gp25.generate_bootstrap_key(
        input_key,
        rotation_key,
        output_key,
        h,
        bits,
        ternary=ternary,
        hw_reducing_hybrid=hybrid,
    )
    q = rotation.rings[0].q_l
    size = 1 << (PRECISION - 1)
    table = [random.randrange(size) for _ in range(size)]  # noqa: S311 - test data
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in table])
    msg = [random.randrange(size) for _ in range(n)]  # noqa: S311 - test data
    rlwe_in = _encrypt_at_the_bottom(io, msg, output_key)
    return gp25, key, tv, rlwe_in, io, output_key, table, msg


def _encrypt_at_the_bottom(io, msg, output_key):
    q = io.rings[0].q_l
    c = io.sample(
        Polynomial(io.rings[0]).from_bigint_array(
            [mod_switch(x, 1 << PRECISION, q) for x in msg]
        ),
        output_key,
    )
    last = len(io.rings) - 1
    return c.round_division(lvl=last) if last else c


def _decrypt_messages(scheme, out, key):
    q = out.ring.q_l
    d = scheme.linear_decrypt(out, key).get_polynomial()
    return [mod_switch(v, q, 1 << PRECISION) for v in d]


@pytest.mark.parametrize("case", CASES)
def test_bootstrap(deterministic_prng, case):
    deterministic_prng(0x5AB00002)
    random.seed(0x5AB00002)
    gp25, key, tv, rlwe_in, io, output_key, table, msg = _bootstrap_case(**CASES[case])
    out = gp25.bootstrap(rlwe_in, tv, key)
    assert out.lvl == key.output_lvl
    assert _decrypt_messages(io, out, output_key) == [table[x] for x in msg]


@pytest.mark.parametrize(
    ("n", "hybrid", "radix"),
    [(16, True, None), (64, True, None), (128, True, None), (64, False, 10)],
)
def test_bootstraps_compose(deterministic_prng, n, hybrid, radix):
    # The output is under the output key, as the input was: brought down to
    # the lowest level, it bootstraps again.
    deterministic_prng(0x5AB00005)
    random.seed(0x5AB00005)
    gp25, key, tv, rlwe_in, io, output_key, table, msg = _bootstrap_case(
        n=n, hybrid=hybrid, radix=radix, primes=(50, 50, 50, 50)
    )
    last = len(io.rings) - 1
    assert key.hw_reducing_lvl == last
    assert io.rings[last].ell == 1
    c = rlwe_in
    for expected in ([table[x] for x in msg], [table[table[x]] for x in msg]):
        out = gp25.bootstrap(c, tv, key)
        assert _decrypt_messages(io, out, output_key) == expected
        c = out.round_division(lvl=last)


def test_weight_reduction_takes_inputs_at_its_level_only():
    gp25, key, tv, _, io, output_key, _, _ = _bootstrap_case()
    c = io.sample(Polynomial(io.rings[0]).from_array([0]), output_key)
    with pytest.raises(ValueError, match=f"level {len(io.rings) - 1}"):
        gp25.bootstrap(c, tv, key)


def test_bootstrap_does_not_depend_on_the_threads(deterministic_prng):
    deterministic_prng(0x5AB00003)
    random.seed(0x5AB00003)
    gp25, key, tv, rlwe_in, *_ = _bootstrap_case(io_rank=2)
    results = []
    for threads in (1, 4):
        engine.set_num_threads(threads)
        try:
            out = gp25.bootstrap(rlwe_in.copy(), tv, key)
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
    gp25, key, tv, rlwe_in, io, output_key, table, msg = _bootstrap_case(
        N=256, n=256, h=17, rotation_rank=4
    )
    engine.set_num_threads(threads)
    try:
        out = gp25.bootstrap(rlwe_in, tv, key)
    finally:
        engine.set_num_threads()
    assert _decrypt_messages(io, out, output_key) == [table[x] for x in msg]
