# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The GP25 sparse-amortized bootstrap: input keys and their gaps, the
building blocks (rotation, signed monomials) and the whole bootstrap over the
options it takes, composed with itself."""

import random

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import GP25
from vfhe.mlwe import GadgetParams, MLWE_Key, MLWE_Scheme, MLWE_Set
from vfhe.util import parallel


def _mod_switch(v: int, q: int, p: int) -> int:
    """``v`` scaled from ``Z_q`` to ``Z_p``, rounded half up."""
    return ((2 * v * p + q) // (2 * q)) % p


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


def _rotation_setup(N=64, primes=(50, 50, 50), bits_per_layer=1):
    scheme = MLWE_Scheme(
        Ring(N, prime_size=list(primes), split_degree=1), special_primes=1
    )
    key = scheme.key_gen_sparse(8, 3.2)
    gp25 = GP25(scheme, bits_per_layer=bits_per_layer)
    automorphism_key = scheme.gen_ksk_automorphism(key, key, 2 * N - 1, lvl=0)
    assert isinstance(automorphism_key, MLWE_Set)
    return scheme, key, gp25, automorphism_key


def _rotated(d, n):
    """What `_constant_or_reflected` reads after a rotation by ``d``: ``acc[k]`` takes
    ``acc[k - d]``, through ``sigma`` once per wrap (``sigma`` twice is the identity)."""
    out = []
    for k in range(n):
        j, wraps = k - d, 0
        while j < 0:
            j, wraps = j + n, wraps + 1
        out.append(j + 1 if wraps % 2 == 0 else -(j + 1))
    return out


@pytest.mark.parametrize("bits_per_layer", [1, 2])
@pytest.mark.parametrize("d", [0, 1, 5, 8, 15, 16])
def test_rotate(d, bits_per_layer):
    n = 16
    scheme, key, gp25, automorphism_key = _rotation_setup(bits_per_layer=bits_per_layer)
    acc, delta = _accumulators(scheme, key, n)
    (keys,) = gp25.encrypt_rotations([d], 5, n, key)
    gp25.rotate(acc, keys, automorphism_key)
    got = [_constant_or_reflected(scheme, key, c, delta) for c in acc]
    assert got == _rotated(d, n)


def test_the_bits_of_a_one_bit_rotation_are_its_keys():
    n, d = 16, 13
    scheme, key, gp25, automorphism_key = _rotation_setup()
    acc, delta = _accumulators(scheme, key, n)
    bits = gp25.mgsw_scheme.encrypt_constants([(d >> b) & 1 for b in range(5)], key)
    gp25.rotate(acc, bits, automorphism_key)
    assert [_constant_or_reflected(scheme, key, c, delta) for c in acc] == _rotated(
        d, n
    )


@pytest.mark.parametrize(
    ("n", "d", "n_bits"),
    [(12, 11, 4), (12, 12, 4), (5, 3, 3), (4, 4, 3), (40, 37, 6), (3, 2, 2)],
)
def test_two_bit_layers_rotate_any_number_of_accumulators(n, d, n_bits):
    # n not a power of two, rotations by n (every accumulator wraps), and pairs
    # whose three inputs back wrap more than once along a chain.
    scheme, key, gp25, automorphism_key = _rotation_setup(bits_per_layer=2)
    acc, delta = _accumulators(scheme, key, n)
    (keys,) = gp25.encrypt_rotations([d], n_bits, n, key)
    gp25.rotate(acc, keys, automorphism_key)
    assert [_constant_or_reflected(scheme, key, c, delta) for c in acc] == _rotated(
        d, n
    )


def test_rotation_layers():
    scheme = MLWE_Scheme(
        Ring(64, prime_size=[50, 50], split_degree=1), special_primes=1
    )
    one, two = GP25(scheme), GP25(scheme, bits_per_layer=2)
    assert one.rotation_layers(16, 5) == [1, 1, 1, 1, 1]
    # pairs while 4 * 2^b <= n
    assert two.rotation_layers(16, 5) == [2, 2, 1]
    assert two.rotation_layers(8, 4) == [2, 1, 1]
    assert two.rotation_layers(2048, 12) == [2, 2, 2, 2, 2, 1, 1]
    assert two.rotation_layers(3, 2) == [1, 1]
    assert two.rotation_layers(64, 0) == []
    with pytest.raises(ValueError, match="bits_per_layer"):
        GP25(scheme, bits_per_layer=3)


def test_rotation_keys_are_checked():
    n = 16
    scheme, key, gp25, automorphism_key = _rotation_setup(bits_per_layer=2)
    acc, _ = _accumulators(scheme, key, n)
    (keys,) = gp25.encrypt_rotations([7], 5, n, key)
    assert len(keys) == 7  # two pairs and a single bit
    with pytest.raises(ValueError, match="not the keys of a rotation"):
        gp25.rotate(acc, keys[:5], automorphism_key)
    with pytest.raises(ValueError, match="rotation values"):
        gp25.encrypt_rotations([32], 5, n, key)
    with pytest.raises(ValueError, match="gap_bits"):
        gp25.encrypt_rotations([1], 6, n, key)


@pytest.mark.parametrize("N", [64, 4096])
def test_two_bit_layers_do_not_depend_on_the_threads(N):
    # The chains are cut into segments by the thread count, and at N = 4096 a
    # sequence's rows run on threads of their own.
    n, d = 64, 45
    scheme, key, gp25, automorphism_key = _rotation_setup(N=N, bits_per_layer=2)
    acc, _ = _accumulators(scheme, key, n)
    (keys,) = gp25.encrypt_rotations([d], 7, n, key)
    results = []
    for threads in (1, 3, 8):
        out = [c.copy() for c in acc]
        parallel.set_num_threads(threads)
        try:
            gp25.rotate(out, keys, automorphism_key)
        finally:
            parallel.set_num_threads()
        for c in out:
            c.to_coeff()
        results.append([c.get_b_poly().get_coeff_matrix() for c in out])
    assert results[0] == results[1] == results[2]


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
    "grouped_digits": {"primes_per_digit": 2, "primes": (50, 50, 50, 50), "special": 2},
    "two_bit_layers": {"bits_per_layer": 2},
    "two_bit_layers_binary": {"bits_per_layer": 2, "ternary": False},
    "two_bit_layers_input_rank_2": {"bits_per_layer": 2, "io_rank": 2},
    "two_bit_layers_rotation_rank_2": {"bits_per_layer": 2, "rotation_rank": 2},
    "two_bit_layers_unbalanced": {"bits_per_layer": 2, "balanced": False},
    "two_bit_layers_smaller_input": {"bits_per_layer": 2, "n": 16},
    "two_bit_layers_larger_input": {"bits_per_layer": 2, "n": 128},
    "two_bit_layers_radix_without_special_primes": {
        "bits_per_layer": 2,
        "radix": 10,
        "primes": (50, 50),
        "special": 0,
    },
    "two_bit_layers_grouped_digits": {
        "bits_per_layer": 2,
        "primes_per_digit": 2,
        "primes": (50, 50, 50, 50),
        "special": 2,
    },
}


def _bootstrap_case(
    N=64,
    n=64,
    h=4,
    ternary=True,
    same_key=False,
    radix=None,
    primes_per_digit=1,
    hybrid=True,
    bits_per_layer=1,
    **scheme_options,
):
    """A GP25 with its keys, a test vector, an input under the output key at
    the lowest level, and what its bootstrap must decrypt to."""
    io, rotation = _schemes(n, N, **scheme_options)
    input_key, rotation_key, output_key, bits = _keys(
        io, rotation, h=h, ternary=ternary, same_key=same_key
    )
    gp25 = GP25(
        rotation,
        gadget_params=GadgetParams(radix, primes_per_digit=primes_per_digit),
        bits_per_layer=bits_per_layer,
    )
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
    tv = gp25.test_vector([_mod_switch(t, 1 << PRECISION, q) for t in table])
    msg = [random.randrange(size) for _ in range(n)]  # noqa: S311 - test data
    rlwe_in = _encrypt_at_the_bottom(io, msg, output_key)
    return gp25, key, tv, rlwe_in, io, output_key, table, msg


def _encrypt_at_the_bottom(io, msg, output_key):
    q = io.rings[0].q_l
    c = io.sample(
        Polynomial(io.rings[0]).from_bigint_array(
            [_mod_switch(x, 1 << PRECISION, q) for x in msg]
        ),
        output_key,
    )
    last = len(io.rings) - 1
    return c.round_division(lvl=last) if last else c


def _decrypt_messages(scheme, out, key):
    q = out.ring.q_l
    d = scheme.linear_decrypt(out, key).get_polynomial()
    return [_mod_switch(v, q, 1 << PRECISION) for v in d]


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
        parallel.set_num_threads(threads)
        try:
            out = gp25.bootstrap(rlwe_in.copy(), tv, key)
        finally:
            parallel.set_num_threads()
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
    tv = gp25.test_vector([_mod_switch(t, 1 << PRECISION, q) for t in table])
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


@pytest.mark.parametrize("n", [16, 128])
def test_a_repacking_key_alone_repacks(deterministic_prng, n):
    # The rotation is the caller's; the repacking key has only the keys
    # `rotate` and `repack` read, from the same generators.
    deterministic_prng(0x5AB0000A)
    random.seed(0x5AB0000A)
    N = 64
    gp25, key, tv, rlwe_in, _, rotation, rotation_key, _, table, msg = _bare_case(n)
    repacking = gp25.generate_repacking_key(rotation_key, n)
    assert repacking.n == n and not repacking.gaps and repacking.signs is None
    assert repacking.hw_reducing_key is None and repacking.output_switch_key is None
    assert repacking.automorphism_key is not None
    assert repacking.trace_repack_key is not None and key.trace_repack_key is not None
    assert len(repacking.trace_repack_key.mlwe) == len(key.trace_repack_key.mlwe)
    packed = gp25.repack(gp25.blind_rotate(rlwe_in, tv, key), repacking)
    groups = max(n // N, 1)
    for i, c in enumerate(packed):
        got = _decrypt_messages(rotation, c, rotation_key)
        if n <= N:
            assert got[:: N // n] == [table[x] for x in msg]
        else:
            assert got == [table[msg[i + groups * j]] for j in range(N)]


def test_a_repacking_key_needs_a_power_of_two():
    _, rotation = _schemes(64, 64)
    rotation_key = rotation.key_gen_sparse(16, 3.2)
    with pytest.raises(ValueError, match="power of two"):
        GP25(rotation).generate_repacking_key(rotation_key, 48)


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


@pytest.mark.parametrize(
    ("gp25_radix", "own"), [(10, GadgetParams()), (None, GadgetParams(4))]
)
def test_the_weight_reduction_has_its_own_gadget(deterministic_prng, gp25_radix, own):
    deterministic_prng(0x5AB0000A)
    random.seed(0x5AB0000A)
    io, rotation = _schemes(64, 64, primes=(50, 50, 50, 50))
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(rotation, gadget_params=GadgetParams(gp25_radix))
    key = gp25.generate_bootstrap_key(
        input_key,
        rotation_key,
        output_key,
        4,
        bits,
        hw_reducing_hybrid=own.log_base is None,
        hw_reducing_gadget_params=own,
    )
    assert key.hw_reducing_key is not None
    assert key.hw_reducing_key.gadget_params == own
    q = rotation.rings[0].q_l
    table = list(range(8))
    tv = gp25.test_vector([_mod_switch(t, 1 << PRECISION, q) for t in table])
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
        hw_reducing_gadget_params=GadgetParams(4, 8),
    )
    assert key.hw_reducing_key is not None
    component = key.hw_reducing_key.mlwe[0]
    assert component is not None
    assert len(component) == 13 - 5
    q = rotation.rings[0].q_l
    tv = gp25.test_vector([_mod_switch(t, 1 << PRECISION, q) for t in range(8)])
    msg = [random.randrange(8) for _ in range(64)]  # noqa: S311 - test data
    out = gp25.bootstrap(_encrypt_at_the_bottom(io, msg, output_key), tv, key)
    assert _decrypt_messages(io, out, output_key) == msg


@pytest.mark.parametrize("bits_per_layer", [1, 2])
def test_fewer_digits_in_the_rotation(deterministic_prng, bits_per_layer):
    # A rotation ring whose level 0 is one prime: its MGSW, automorphism and
    # trace keys keep 4 of their 5 digits.
    deterministic_prng(0x5AB0000D)
    random.seed(0x5AB0000D)
    io, rotation = _schemes(64, 64, primes=(50, 50))
    assert rotation.rings[0].ell == 1
    input_key, rotation_key, output_key, bits = _keys(io, rotation)
    gp25 = GP25(
        rotation, gadget_params=GadgetParams(10, 4), bits_per_layer=bits_per_layer
    )
    key = gp25.generate_bootstrap_key(input_key, rotation_key, output_key, 4, bits)
    assert gp25.mgsw_scheme.gadget_size(0) == 4
    q = rotation.rings[0].q_l
    tv = gp25.test_vector([_mod_switch(t, 1 << PRECISION, q) for t in range(8)])
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
    parallel.set_num_threads(threads)
    try:
        out = gp25.bootstrap(rlwe_in, tv, key)
    finally:
        parallel.set_num_threads()
    assert _decrypt_messages(io, out, output_key) == [table[x] for x in msg]
