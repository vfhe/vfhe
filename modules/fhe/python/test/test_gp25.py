# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The GP25 sparse-amortized bootstrap: input keys and their gaps, the
building blocks (rotation, signed monomials) and the whole bootstrap over the
options it takes, composed with itself."""

import random

import pytest
from vfhe import engine
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import GP25, mod_switch
from vfhe.mlwe import MLWE_Key, MLWE_Scheme, MLWE_Set


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
    "packing_key_switch": {"trace": False},
    "packing_key_switch_smaller_input": {"trace": False, "n": 16},
    "packing_key_switch_larger_input": {"trace": False, "n": 128},
    "one_key": {"same_key": True},
    "binary": {"ternary": False},
    "input_rank_2": {"io_rank": 2},
    "rotation_rank_2": {"rotation_rank": 2},
    "unbalanced": {"balanced": False},
    "smaller_input": {"n": 16},
    "larger_input": {"n": 128},
    "radix_without_special_primes": {"radix": 10, "primes": (50, 50), "special": 0},
    "radix_without_special_primes_smaller_input": {
        "n": 16,
        "radix": 10,
        "primes": (50, 50),
        "special": 0,
    },
    "several_levels": {"primes": (50, 50, 50, 50)},
}


def _bootstrap_case(
    N=64,
    n=64,
    h=4,
    ternary=True,
    trace=True,
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
    ("n", "hybrid", "radix", "trace"),
    [
        (16, True, None, False),
        (64, True, None, False),
        (128, True, None, False),
        (64, False, 10, False),
        (16, True, None, True),
        (128, True, None, True),
    ],
)
def test_bootstraps_compose(deterministic_prng, n, hybrid, radix, trace):
    # The output is under the output key, as the input was: brought down to
    # the lowest level, it bootstraps again.
    deterministic_prng(0x5AB00005)
    random.seed(0x5AB00005)
    gp25, key, tv, rlwe_in, io, output_key, table, msg = _bootstrap_case(
        n=n, hybrid=hybrid, radix=radix, trace=trace, primes=(50, 50, 50, 50)
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


# --- without an output key ---------------------------------------------------


def _bare_case(n, N=64, **scheme_options):
    """A key without an output key, and an input under the input key."""
    io, rotation = _schemes(n, N, **scheme_options)
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(rotation)
    key = gp25.generate_bootstrap_key(input_key, rotation_key, None, 4, bits)
    q = rotation.rings[0].q_l
    size = 1 << (PRECISION - 1)
    table = [random.randrange(size) for _ in range(size)]  # noqa: S311 - test data
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in table])
    msg = [random.randrange(size) for _ in range(n)]  # noqa: S311 - test data
    rlwe_in = _encrypt_at_the_bottom(io, msg, input_key)
    return gp25, key, tv, rlwe_in, io, rotation, rotation_key, output_key, table, msg


@pytest.mark.parametrize("n", [16, 64, 128])
def test_without_an_output_key_the_result_stays_in_the_rotation_ring(
    deterministic_prng, n
):
    deterministic_prng(0x5AB00007)
    random.seed(0x5AB00007)
    N = 64
    gp25, key, tv, rlwe_in, _, rotation, rotation_key, _, table, msg = _bare_case(n)
    assert key.hw_reducing_key is None and key.output_switch_key is None
    assert key.hw_reducing_lvl is None and key.output_lvl is None
    packed = gp25.repack(gp25.blind_rotate(rlwe_in, tv, key), key)
    groups = max(n // N, 1)
    assert len(packed) == groups
    for i, c in enumerate(packed):
        assert c.scheme is rotation
        got = _decrypt_messages(rotation, c, rotation_key)
        if n <= N:
            # The other coefficients are not cleared: a switch down drops them.
            assert got[:: N // n] == [table[x] for x in msg]
        else:
            assert got == [table[msg[i + groups * j]] for j in range(N)]


def test_the_caller_switches_rings_at_the_lowest_level(deterministic_prng):
    # What the bare mode is for: the switch key from the rotation key to the
    # output key is generated at the lowest modulus only.
    deterministic_prng(0x5AB00008)
    random.seed(0x5AB00008)
    gp25, key, tv, rlwe_in, io, rotation, rotation_key, output_key, table, msg = (
        _bare_case(16, primes=(50, 50, 50, 50), rotation_levels=None)
    )
    (packed,) = gp25.repack(gp25.blind_rotate(rlwe_in, tv, key), key)
    last = len(io.rings) - 1
    assert io.rings[last].ell == 1
    down = packed.round_division(lvl=len(rotation.rings) - 1)
    switch_key = io.gen_ring_switch_key(output_key, rotation_key, last)
    out = io.ring_switch(down, switch_key)
    assert out.lvl == last
    assert _decrypt_messages(io, out, output_key) == [table[x] for x in msg]


def test_blind_rotate_exponents_is_the_rotation_alone(deterministic_prng):
    deterministic_prng(0x5AB00009)
    random.seed(0x5AB00009)
    gp25, key, tv, rlwe_in, *_ = _bare_case(64, io_rank=2)
    rlwe_in.to_coeff()
    q = rlwe_in.ring.q_l
    b = gp25._exponents(rlwe_in.get_b_poly().get_polynomial(), q)
    a = [gp25._exponents(rlwe_in.get_a_poly(j).get_polynomial(), q) for j in range(2)]
    expected = gp25.blind_rotate(rlwe_in, tv, key)
    got = gp25.blind_rotate_exponents(a, b, tv, key)
    for x, y in zip(expected, got, strict=True):
        assert x.get_b_poly().get_coeff_matrix() == y.get_b_poly().get_coeff_matrix()
        assert x.get_a_poly(0).get_coeff_matrix() == y.get_a_poly(0).get_coeff_matrix()
    with pytest.raises(ValueError, match=r"\[0, 2N\)"):
        gp25.blind_rotate_exponents(a, [128, *b[1:]], tv, key)
    with pytest.raises(ValueError, match="mask components"):
        gp25.blind_rotate_exponents(a[:1], b, tv, key)


def test_bootstrap_needs_an_output_key():
    gp25, key, tv, rlwe_in, *_ = _bare_case(64)
    with pytest.raises(ValueError, match="no output key"):
        gp25.bootstrap(rlwe_in, tv, key)
    io, rotation = _schemes(64, 64)
    input_key, rotation_key, _, bits = _keys(io, rotation)
    with pytest.raises(ValueError, match="repack with the trace"):
        GP25(rotation, trace_repack=False).generate_bootstrap_key(
            input_key, rotation_key, None, 4, bits
        )


@pytest.mark.parametrize(("gp25_radix", "own", "log_base"), [(10, 0, 0), (None, 4, 4)])
def test_the_weight_reduction_has_its_own_radix(
    deterministic_prng, gp25_radix, own, log_base
):
    deterministic_prng(0x5AB0000A)
    random.seed(0x5AB0000A)
    io, rotation = _schemes(64, 64, primes=(50, 50, 50, 50))
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(rotation, radix_log_base=gp25_radix)
    key = gp25.generate_bootstrap_key(
        input_key,
        rotation_key,
        output_key,
        4,
        bits,
        hw_reducing_hybrid=log_base == 0,
        hw_reducing_radix_log_base=own,
    )
    assert key.hw_reducing_key is not None
    assert key.hw_reducing_key.log_base == log_base
    q = rotation.rings[0].q_l
    table = list(range(8))
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in table])
    msg = [random.randrange(8) for _ in range(64)]  # noqa: S311 - test data
    out = gp25.bootstrap(_encrypt_at_the_bottom(io, msg, output_key), tv, key)
    assert _decrypt_messages(io, out, output_key) == msg


def test_fewer_digits_in_the_weight_reduction(deterministic_prng):
    # Radix 4 at the one-prime lowest level, 8 of its 13 digits kept.
    deterministic_prng(0x5AB0000C)
    random.seed(0x5AB0000C)
    io, rotation = _schemes(64, 64, primes=(50, 50, 50, 50))
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(rotation)
    key = gp25.generate_bootstrap_key(
        input_key,
        rotation_key,
        output_key,
        4,
        bits,
        hw_reducing_hybrid=False,
        hw_reducing_radix_log_base=4,
        hw_reducing_digits=8,
    )
    assert key.hw_reducing_key is not None
    component = key.hw_reducing_key.mlwe[0]
    assert component is not None
    assert len(component) == 13 - 5
    q = rotation.rings[0].q_l
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in range(8)])
    msg = [random.randrange(8) for _ in range(64)]  # noqa: S311 - test data
    out = gp25.bootstrap(_encrypt_at_the_bottom(io, msg, output_key), tv, key)
    assert _decrypt_messages(io, out, output_key) == msg


def test_fewer_digits_in_the_rotation(deterministic_prng):
    # A rotation ring whose level 0 is one prime: its MGSW, automorphism and
    # trace keys keep 4 of their 5 digits.
    deterministic_prng(0x5AB0000D)
    random.seed(0x5AB0000D)
    io, rotation = _schemes(64, 64, primes=(50, 50))
    assert rotation.rings[0].ell == 1
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(rotation, radix_log_base=10, digits=4)
    key = gp25.generate_bootstrap_key(input_key, rotation_key, output_key, 4, bits)
    assert gp25.mgsw_scheme.gadget_size(0) == 4
    q = rotation.rings[0].q_l
    tv = gp25.test_vector([mod_switch(t, 1 << PRECISION, q) for t in range(8)])
    msg = [random.randrange(8) for _ in range(64)]  # noqa: S311 - test data
    out = gp25.bootstrap(_encrypt_at_the_bottom(io, msg, output_key), tv, key)
    assert _decrypt_messages(io, out, output_key) == msg


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
