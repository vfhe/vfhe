# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.arith over the cffi boundary.

Python big ints are the exact oracle: negacyclic schoolbook for the ring
product, CRT for reconstruction. Also covers domain conversion, automorphism,
slot inversion, the CKKS complex FFT roundtrip, and the multiprecision bridge.
"""

import math
import random
from fractions import Fraction

import pytest
from vfhe.arith import (
    ComplexPolynomial,
    ComplexRing,
    Multiprecision,
    Polynomial,
    Ring,
    repr,
)
from vfhe.engine import ffi

N = 16
rng = random.Random(0xC0FFEE)  # noqa: S311 - test data, not a key


@pytest.fixture
def ring():
    return Ring(N, prime_size=[30, 30], split_degree=1)


def negacyclic_mul(a, b, q, n):
    out = [0] * n
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            s = (a[i] * b[j]) % q
            out[k] = (out[k] + (q - s if i + j >= n else s)) % q
    return out


def test_ring_and_roundtrip(ring):
    assert ring.ell == 2
    v = [i * 7 + 1 for i in range(N)]
    p = Polynomial(ring).from_array(v)
    assert p.get_polynomial() == [x % ring.q_l for x in v]


def test_multiply_matches_schoolbook(ring):
    a_c = [i + 1 for i in range(N)]
    b_c = [3 * i + 2 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)
    assert (a * b).get_polynomial() == negacyclic_mul(a_c, b_c, ring.q_l, N)


def test_add_sub_negate_scale(ring):
    a_c = [i * 7 + 1 for i in range(N)]
    b_c = [i * i + 3 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)
    assert (a + b).get_polynomial() == [
        (x + y) % ring.q_l for x, y in zip(a_c, b_c, strict=True)
    ]
    assert (a - b).get_polynomial() == [
        (x - y) % ring.q_l for x, y in zip(a_c, b_c, strict=True)
    ]
    assert (-a).get_polynomial() == [(-x) % ring.q_l for x in a_c]
    assert (a * 5).get_polynomial() == [(5 * x) % ring.q_l for x in a_c]
    assert (a + 9).get_polynomial() == [(a_c[0] + 9) % ring.q_l] + [
        x % ring.q_l for x in a_c[1:]
    ]


def test_ntt_roundtrip(ring):
    v = [rng.randrange(ring.q_l) for _ in range(N)]
    p = Polynomial(ring).from_array(v)  # ends in NTT form
    p.to_coeff()
    p.to_NTT()
    p.to_coeff()
    assert p.get_polynomial() == [x % ring.q_l for x in v]


def test_automorphism_composition(ring):
    v = [rng.randrange(ring.q_l) for _ in range(N)]
    p = Polynomial(ring).from_array(v)
    # gen=3, gen^-1=11 (3*11=33==1 mod 2N=32) compose to identity
    s = p.automorphism(3).automorphism(11)
    assert s.get_polynomial() == p.get_polynomial()


def test_copy_and_eq(ring):
    a = ring.random_element()
    b = a.copy()
    assert a == b


def test_fast_inverse():
    r = Ring(N, prime_size=[30], split_degree=1)
    a = r.random_element()  # NTT form, uniform slots (nonzero w.h.p.)
    inv = a.fast_inverse()
    one = a * inv
    one.to_coeff()
    # a * a^-1 == 1 in every eval slot -> constant polynomial 1
    a.to_NTT()
    # in NTT/eval domain all slots are 1; check the product equals the all-ones poly
    prod_ntt = a * inv
    prod_ntt.to_coeff()
    # constant term 1, rest 0 (identity element)
    coeffs = prod_ntt.get_polynomial()
    assert coeffs[0] == 1 and all(c == 0 for c in coeffs[1:])


def test_fast_inverse_test_vectors():
    r = Ring(N, prime_size=[30], split_degree=1)
    q = r.primes[0]
    # Specific test vector of slots: [1, 2, 3, ..., N]
    slots = list(range(1, N + 1))
    a = Polynomial(r)
    a.from_coeff_matrix([slots], repr=repr.ntt)

    inv = a.fast_inverse()
    inv_slots = inv.get_coeff_matrix(repr=repr.ntt)[0]

    # Verify that inv_slots are the modular inverses of slots
    for x, y in zip(slots, inv_slots, strict=True):
        assert (x * y) % q == 1


def test_fast_inverse_zero_slot():
    r = Ring(N, prime_size=[30], split_degree=1)
    a = Polynomial(r)
    # Put a zero in one of the slots (e.g. index 5)
    slots = [i + 1 for i in range(N)]
    slots[5] = 0
    a.from_coeff_matrix([slots], repr=repr.ntt)

    with pytest.raises(ValueError, match="zero slot is not invertible"):
        a.fast_inverse()


def test_fast_inverse_multi_prime():
    r = Ring(N, prime_size=[30, 30, 30], split_degree=1)
    a = r.random_element()
    inv = a.fast_inverse()
    one = a * inv
    one.to_coeff()
    coeffs = one.get_polynomial()
    assert coeffs[0] == 1 and all(c == 0 for c in coeffs[1:])


def test_complex_fft_roundtrip():
    cN = 8
    cring = ComplexRing(cN)
    slots = [complex(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(cN)]
    cp = ComplexPolynomial(cring).from_array(slots)
    cp.IFFT()
    cp.FFT()
    out = list(cp)
    assert all(abs(out[i] - slots[i]) < 1e-6 for i in range(cN))


@pytest.mark.parametrize("n", [8, 64, 1024])
@pytest.mark.parametrize(
    "gen_of", [lambda _n: 3, lambda _n: 5, lambda n: 25 % (2 * n), lambda n: 2 * n - 1]
)
def test_ntt_domain_automorphism_matches_the_coefficient_one(n, gen_of):
    # Narrow and wide primes, and a ring that does not start at base index 0,
    # across the vectorized transform's minimum length.
    Rq = Ring(n, prime_size=[28, 50, 30, 45, 60], split_degree=1)
    mask = 0
    for idx in Rq.prime_indices[1:]:
        mask |= 1 << idx
    R = Rq.quotient_ring(mask=mask)
    gen = gen_of(n)
    a = R.random_element(ntt=False)
    expected = a.automorphism(gen)
    expected.to_NTT()

    a.to_NTT()
    idx = ffi.new("uint32_t[]", n)
    R.lib.polynomial_RNS_automorphism_index(idx, n, gen)
    out = Polynomial(R)
    R.lib.polynomial_RNS_permute(out.obj, a.obj, idx)
    out.repr = repr.ntt
    assert out.get_coeff_matrix(repr=repr.ntt) == expected.get_coeff_matrix(
        repr=repr.ntt
    )


@pytest.mark.parametrize("gen", [5, 2 * N - 1])
def test_automorphism_stays_in_the_ntt_domain(ring, gen):
    a = ring.random_element(ntt=False)
    in_coeff = a.automorphism(gen)
    assert in_coeff.repr == repr.coeff
    a.to_NTT()
    in_ntt = a.automorphism(gen)
    assert in_ntt.repr == repr.ntt and a.repr == repr.ntt
    assert in_ntt.get_coeff_matrix(repr=repr.coeff) == in_coeff.get_coeff_matrix(
        repr=repr.coeff
    )


@pytest.mark.parametrize("n", [4, 16])
@pytest.mark.parametrize("sizes", [[60], [28], [28, 50, 30, 45, 60], [60] * 12])
def test_centered_doubles_match_the_exact_crt(sizes, n):
    # One prime takes a vectorized path from 8 coefficients on (narrow and
    # wide rows); more take Garner's.
    Rq = Ring(n, prime_size=[40, *sizes], split_degree=1)
    mask = 0
    for idx in Rq.prime_indices[1:]:
        mask |= 1 << idx
    R = Rq.quotient_ring(mask=mask)
    Q = math.prod(R.primes)
    half = (Q - 1) // 2
    values = [
        rng.randrange(Q),  # anywhere modulo Q
        rng.randrange(-(2**40), 2**40) % Q,  # small, of either sign
        half,  # the largest positive value
        half + 1,  # the most negative one
        0,
        Q - 1,
        *(rng.randrange(Q) for _ in range(10)),
    ][:n]
    p = Polynomial(R).from_bigint_array(values)
    p.to_coeff()
    out = ffi.new("double[]", R.N)
    R.lib.polynomial_RNSc_to_centered_doubles(out, p.obj, 0.5)
    for k, v in enumerate(values):
        centered = v - Q if v > half else v
        assert out[k] == pytest.approx(centered / 2, rel=1e-14, abs=0)


def test_short_arrays_are_padded_with_zeros():
    R = Ring(64, prime_size=[50, 50], split_degree=1)
    big = (1 << 90) + 7
    zeros = [0] * (R.N - 2)
    for _ in range(8):  # freed rows full of non-zero values to land on
        Polynomial(R).from_bigint_array([R.q_l - 1] * R.N)
        assert Polynomial(R).from_bigint_array([0, big]).get_polynomial() == [
            0,
            big,
            *zeros,
        ]
        assert Polynomial(R).from_array([0, 5]).get_polynomial() == [0, 5, *zeros]


def test_encode_rounding_is_round_half_to_even():
    # The native rounding (vectorized on avx512ifma) against Python's round.
    cring = ComplexRing(32)
    R = Ring(64, prime_size=[60, 60], split_degree=1)
    values = [rng.uniform(-(2**50), 2**50) for _ in range(40)]
    values += [k + 0.5 for k in range(-12, 12)]  # ties go to the even neighbour
    cp = ComplexPolynomial(cring)
    for i in range(32):
        cp.obj[i] = values[i]
        cp.obj[i + 32] = values[i + 32]
    native = cp.round_to_RNS_cpp(R)
    assert native.get_polynomial(signed=True) == [round(v) for v in values]


def _brv(x, bits):
    return int(bin(x)[2:].rjust(bits, "0")[::-1], 2) if bits else 0


def _complex_fft_oracle(v, n, inverse):
    """The scalar CT_NR (forward) / GS_RN (inverse) transforms, in Python."""
    rous = ComplexRing.gen_special_rous_hp(2 * n)
    ws = [r**-1 for r in rous] if inverse else rous
    logn = int(math.log2(n))
    x = list(v)
    if inverse:
        x = [x[_brv(i, logn)] for i in range(n)]
        t, m = 1, n
        while m > 1:
            h = m >> 1
            for i in range(h):
                for j in range(2 * i * t, 2 * i * t + t):
                    u, w = x[j], x[j + t]
                    x[j], x[j + t] = u + w, (u - w) * ws[h + i]
            t, m = t << 1, h
        return [c / n for c in x]
    t, m = n, 1
    while m < n:
        t >>= 1
        for i in range(m):
            for j in range(2 * i * t, 2 * i * t + t):
                u, w = x[j], x[j + t] * ws[m + i]
                x[j], x[j + t] = u + w, u - w
        m <<= 1
    return [x[_brv(i, logn)] for i in range(n)]


@pytest.mark.parametrize("cN", [1, 2, 4, 8, 16, 32])
def test_complex_fft_matches_oracle(cN):
    # Below 8 values the vectorized transform cannot run and its table
    # loader cannot be built, so this length range crosses that cutover.
    cring = ComplexRing(cN)
    slots = [complex(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(cN)]
    cp = ComplexPolynomial(cring).from_array(slots)
    cp.IFFT()
    coeffs = list(cp)
    expected = _complex_fft_oracle(slots, cN, inverse=True)
    assert all(abs(coeffs[i] - expected[i]) < 1e-9 for i in range(cN))
    cp *= 3.0
    cp.FFT()
    out = list(cp)
    expected = _complex_fft_oracle([3.0 * c for c in coeffs], cN, inverse=False)
    assert all(abs(out[i] - expected[i]) < 1e-9 for i in range(cN))
    assert all(abs(out[i] - 3.0 * slots[i]) < 1e-9 for i in range(cN))


def test_complex_polynomial_accepts_any_number():
    # Number types beyond the builtins: a subclass of complex (as array
    # libraries' complex scalars are) and a Real that is not a float.
    class Sub(complex):
        pass

    cring = ComplexRing(8)
    cp = ComplexPolynomial(cring).from_array([Fraction(1, 2), Sub(1, 2), 3, -2.5])
    cp[4] = Sub(0, -1)
    cp *= Fraction(2)
    assert list(cp)[:5] == [1, complex(2, 4), 6, -5, complex(0, -2)]
    with pytest.raises(NotImplementedError):
        cp[0] = "1"  # type: ignore[assignment]
    with pytest.raises(NotImplementedError):
        cp.from_array(["1"])  # type: ignore[list-item]


def test_ring_handles_stay_interned_past_hundreds_of_rings():
    # Native structures compare rings by handle, so one ring must keep one
    # handle however many distinct rings the process has used.
    Rq = Ring(16, prime_size=[30] * 9, split_degree=1)
    rings = []
    for subset in range(1, 1 << Rq.ell):
        mask = 0
        for i, idx in enumerate(Rq.prime_indices):
            if subset >> i & 1:
                mask |= 1 << idx
        rings.append(Rq.quotient_ring(mask=mask))
    handles = [r.arith_ring for r in rings]
    assert len({int(ffi.cast("uintptr_t", h)) for h in handles}) == len(rings)
    assert all(r.arith_ring == h for r, h in zip(rings, handles, strict=True))


def test_multiprecision_scalar_ops():
    mp = Multiprecision()
    a = rng.randrange(2**200)
    b = rng.randrange(2**180)
    a_mp = mp.load(a)
    b_mp = mp.load(b)
    mp.lib.mp_sub(a_mp, a_mp, b_mp)
    assert mp.scalar_digits(a_mp) == mp.scalar_digits(mp.load(a - b))

    a = rng.randrange(2**180)
    scale = rng.randrange(2**51)
    a_mp = mp.load(a)
    out_mp = mp.load(rng.randrange(2**250))
    mp.lib.mp_scale(out_mp, a_mp, mp.load_small(scale))
    assert mp.scalar_digits(out_mp) == mp.scalar_digits(mp.load(scale * a))


def test_multiprecision_from_rns():
    r = Ring(2**12, 200, split_degree=1)
    a = r.random_element()
    mp = Multiprecision()
    crt = mp.compute_crt_consts(r.primes)
    a_mp = mp.from_polynomial(a, crt)
    assert mp.poly_to_list(a_mp) == a.get_polynomial()


# The RNS base is shared by every ring of the same (N, split_degree) and grows
# as rings are built, so a ring's primes sit at arbitrary base indices and the
# rows between them belong to other rings. Reconstruction has to follow the
# polynomial's own mask; walking the base instead read rows it does not own.
@pytest.mark.parametrize(
    "sizes",
    [
        [30, 30],  # narrow rows, and primes far below a 52-bit digit
        [20, 50, 20, 50],  # narrow and wide rows in one ring
        [49] * 9,  # more primes than a 52-bit CRT quotient would allow
    ],
)
def test_multiprecision_from_rns_off_the_start_of_the_base(sizes):
    N_mp = 2**10
    Ring(N_mp, prime_size=[31, 31], split_degree=1)  # claims the first indices
    r = Ring(N_mp, prime_size=sizes, split_degree=1)
    mp = Multiprecision()
    a = r.random_element()
    a_mp = mp.from_polynomial(a, mp.compute_crt_consts(r.primes))
    assert mp.poly_to_list(a_mp) == a.get_polynomial()


def test_multiprecision_from_rns_over_a_non_contiguous_quotient():
    r = Ring(2**10, prime_size=[35, 40, 45, 50], split_degree=1)
    sub = r.quotient_ring(mask=(1 << r.prime_indices[0]) | (1 << r.prime_indices[3]))
    mp = Multiprecision()
    a = sub.random_element()
    a_mp = mp.from_polynomial(a, mp.compute_crt_consts(sub.primes))
    assert mp.poly_to_list(a_mp) == a.get_polynomial()


def test_compute_crt_consts_rejects_primes_wider_than_a_digit():
    r = Ring(2**10, prime_size=[60, 60], split_degree=1)
    with pytest.raises(ValueError, match="52 bits"):
        Multiprecision().compute_crt_consts(r.primes)


def _round_scaled(c, q, k):
    """``round(2**k * c / q)`` mod 2**k on the centered representative, in
    exact integer arithmetic -- the oracle for the native rescale."""
    c = c - q if c > q // 2 else c
    num = c << k
    t = (2 * num + q) // (2 * q) if num >= 0 else -((-2 * num + q) // (2 * q))
    return t % (1 << k)


@pytest.mark.parametrize("sizes", [[40], [30, 30], [32, 32, 32], [20, 50, 20, 50]])
@pytest.mark.parametrize("k", [1, 8, 32, 52, 63, 64])
def test_rescale_to_power_of_two(sizes, k):
    Ring(2**9, prime_size=[31, 31], split_degree=1)  # not the first of its base
    r = Ring(2**9, prime_size=sizes, split_degree=1)
    q = math.prod(r.primes)
    # The halves are where the rounding and the centering both turn over.
    values = [0, 1, q - 1, q // 2, q // 2 + 1] + [
        rng.randrange(q) for _ in range(2**9 - 5)
    ]
    poly = Polynomial(r).from_bigint_array(values)

    got = list(poly.rescale_to_power_of_two(k))
    assert got == [_round_scaled(c, q, k) for c in values]
    # The rescale reads; it must not have changed the polynomial's value.
    assert poly.get_polynomial() == values


@pytest.mark.parametrize("k", [8, 64])
def test_rescale_to_power_of_two_at_the_exact_half(k):
    # The rescale rounds on `2**k * c mod q >= (q+1)/2`, and the tie is the one
    # case an oracle over random coefficients will not reach: it needs the
    # comparison to run out of digits with every one of them equal.
    r = Ring(2**9, prime_size=[30, 30, 30], split_degree=1)
    q = math.prod(r.primes)
    tie = (q + 1) // 2 * pow(pow(2, k, q), -1, q) % q
    assert (tie << k) % q == (q + 1) // 2  # the tie really is one
    values = [tie, (tie + 1) % q, (tie - 1) % q] + [0] * (2**9 - 3)
    got = list(Polynomial(r).from_bigint_array(values).rescale_to_power_of_two(k))
    assert got == [_round_scaled(c, q, k) for c in values]


def test_rescale_to_power_of_two_rejects_a_width_over_a_word():
    r = Ring(2**9, prime_size=[30, 30], split_degree=1)
    with pytest.raises(ValueError, match="between 1 and 64"):
        r.random_element().rescale_to_power_of_two(65)


def test_quotient_ring_rejects_mask_bits_outside_its_primes():
    big = Ring(2**10, prime_size=[17, 17, 17, 17, 22], split_degree=1)
    small = Ring(2**10, prime_size=[32, 32], split_degree=1)
    with pytest.raises(ValueError, match="not among the given primes"):
        big.quotient_ring(mask=big.mask | small.mask)

    # `union` is the way to name both rings' primes at once.
    both = big.union(small)
    assert both.mask == big.mask | small.mask
    assert both.quotient_ring(mask=small.mask).primes == small.primes


@pytest.mark.parametrize("split_degree", [2, 4, 8])
def test_fast_inverse_generic(split_degree):
    r = Ring(128, prime_size=[30], split_degree=split_degree)
    a = r.random_element()
    inv = a.fast_inverse()
    one = a * inv
    one.to_coeff()
    coeffs = one.get_polynomial()
    assert coeffs[0] == 1 and all(c == 0 for c in coeffs[1:])


# --------------------------------------------------------------------------
# Short transforms and short element-wise lengths
#
# The vectorized kernels need two AVX512 lane groups per NTT butterfly stage
# and one per element-wise step, so below those the transforms used to skip
# every stage (forward) or walk off the buffer (inverse), and the element-wise
# kernels computed nothing at all. Both now fall back to the size-generic
# scalar path, so the results must match the big-int oracle at every length --
# and, since only one engine is loaded per process, the oracle is what pins
# this down rather than a cross-engine comparison.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "split_degree"),
    [(4, 1), (8, 1), (16, 1), (16, 4), (32, 4), (64, 16), (64, 4)],
)
def test_short_transform_matches_oracle(n, split_degree):
    r = Ring(n, prime_size=[30], split_degree=split_degree)
    q = r.primes[0]
    a_c = [rng.randrange(q) for _ in range(n)]
    b_c = [rng.randrange(q) for _ in range(n)]
    a = Polynomial(r).from_array(a_c)
    b = Polynomial(r).from_array(b_c)
    assert (a * b).get_polynomial() == negacyclic_mul(a_c, b_c, q, n)
    # forward then inverse is the identity (the inverse used to segfault here)
    round_trip = a.copy()
    round_trip.to_coeff()
    round_trip.to_NTT()
    assert round_trip.get_polynomial() == a_c


@pytest.mark.parametrize(("n", "split_degree"), [(16, 4), (64, 16), (8, 1)])
def test_short_elementwise_matches_oracle(n, split_degree):
    """Element-wise ops on a ring whose per-block length is under one vector."""
    r = Ring(n, prime_size=[30], split_degree=split_degree)
    q = r.primes[0]
    coeffs = [rng.randrange(q) for _ in range(n)]
    for as_ntt in (True, False):
        a = Polynomial(r).from_array(coeffs)
        if not as_ntt:
            a.to_coeff()
        # adding a constant touches only the first N/split_degree slots, which
        # is where a dropped vector tail used to make it a silent no-op
        assert (a + 5).get_polynomial() == [(coeffs[0] + 5) % q, *coeffs[1:]]
        assert (a * 3).get_polynomial() == [(c * 3) % q for c in coeffs]
        assert (-a).get_polynomial() == [(q - c) % q for c in coeffs]
        assert (a + a).get_polynomial() == [(2 * c) % q for c in coeffs]
        assert (a - a) == 0


# --------------------------------------------------------------------------
# Representation hygiene
# --------------------------------------------------------------------------


def test_reading_a_polynomial_preserves_its_representation(ring):
    """Readers convert a copy, never the object being read.

    A reader that converted in place left a table in mixed representations,
    and the next C kernel over it folded the wrong data and returned silent
    garbage with no error anywhere.
    """
    for target in (repr.ntt, repr.coeff):
        a = ring.random_element()
        a.to_repr(target)
        expected = a.get_polynomial()

        a.get_polynomial()
        list(a)
        a.get_coeff_matrix()
        a.get_coeff_matrix(repr=repr.ntt)
        a.get_hash()
        # the comparisons matter for their side effect, not their value
        _ = a == 0
        _ = a == [0, 0]
        _ = a == ring.random_element()

        assert a.repr == target
        # and the value is untouched, not just the flag
        assert a.get_polynomial() == expected


def test_to_repr_rejects_a_non_representation(ring):
    a = ring.random_element()
    with pytest.raises(ValueError, match="not a data representation"):
        a.to_repr(repr.empty)


# --------------------------------------------------------------------------
# Integer operands
# --------------------------------------------------------------------------


def test_scaling_by_zero_gives_a_polynomial(ring):
    a = ring.random_element()
    scaled = a * 0
    assert isinstance(scaled, Polynomial)
    assert scaled.ring is ring and scaled == 0
    in_place = ring.random_element()
    in_place *= 0
    assert isinstance(in_place, Polynomial) and in_place == 0


def test_integer_operands_are_symmetric(ring):
    """+, -, += and -= all take any int, not just 0."""
    coeffs = [rng.randrange(1 << 20) for _ in range(N)]
    for target in (repr.ntt, repr.coeff):

        def fresh(target=target):
            p = Polynomial(ring).from_array(coeffs)
            p.to_repr(target)
            return p

        assert fresh().__add__(7).get_polynomial()[0] == coeffs[0] + 7
        assert fresh().__sub__(7).get_polynomial()[0] == coeffs[0] - 7
        assert (7 + fresh()).get_polynomial()[0] == coeffs[0] + 7
        assert (7 - fresh()).get_polynomial(signed=True)[0] == 7 - coeffs[0]
        assert fresh().__add__(-7).get_polynomial()[0] == coeffs[0] - 7

        acc = fresh()
        acc += 7
        assert acc.get_polynomial()[0] == coeffs[0] + 7
        acc -= 7
        assert acc.get_polynomial()[:2] == coeffs[:2]

        # the int operand leaves the representation alone
        assert (fresh() + 7).repr == target
        # and the other coefficients with it
        assert fresh().__sub__(7).get_polynomial()[1:] == coeffs[1:]

    with pytest.raises(ValueError, match="int64"):
        _ = Polynomial(ring).from_array(coeffs) + 2**63


# --------------------------------------------------------------------------
# The shared RNS base
# --------------------------------------------------------------------------


def test_rns_rows_is_stable_when_the_incntt_grows():
    """A ring's row count must not follow the shared RNS base's prime count.

    The RNS base of an (N, split_degree) pair is process-global and grows in
    place when a ring introduces a new prime, so `_base_l()` increases under
    rings built earlier. Anything an allocation was sized with has to come
    from the ring's own mask instead: re-reading the live count and freeing
    with it walks off the end of the array.
    """
    # Own the (N, split_degree) key so the growth below is ours to observe.
    first = Ring(64, prime_size=[30], split_degree=1)
    rows = first.rns_rows
    assert rows == first._base_l() == 1

    second = Ring(64, prime_size=[30, 31], split_degree=1)
    assert first._base_l() == 2  # the shared count grew under `first`
    assert first.rns_rows == rows  # the ring's own row count did not
    assert second.rns_rows == 2

    # and `first` still allocates and computes correctly afterwards
    a = first.random_element()
    assert (a - a) == 0
    assert len(first.scalar_array(3)) == rows


def test_ring_follows_a_replaced_registry(monkeypatch):
    """A `Ring` must resolve the RNS base registry at use, not at import.

    A dynamic-extension reload cannot patch the registry's `lib` (an instance
    attribute, unlike the module-level `lib`/`ffi` `update_cffi_references`
    rewrites), so the implementation's `state.register_rebind` handler swaps
    the whole instance. A name bound at import time would keep pointing at the
    retired one and go on building RNS bases in the unloaded library.
    """
    from vfhe.arith.impl.rns import rns_base

    retired = rns_base.registry()
    monkeypatch.setattr(rns_base, "rns_base_registry", rns_base.RNS_Base_Registry())
    fresh = rns_base.registry()
    assert fresh is not retired

    key = (32, 1)
    assert key not in retired.bases  # nothing has claimed it yet
    ring = Ring(32, prime_size=[30], split_degree=1)

    # the ring registered with the current registry, not the retired one
    assert key in fresh.bases
    assert key not in retired.bases
    assert ring.base == fresh.bases[key]
    # and it is a working ring, not just a bookkeeping entry
    a = ring.random_element()
    assert (a - a) == 0


def _cross_base_rings(n, plaintext_bits, other_bits, aux_bits):
    """A source and a destination ring sharing only their first prime.

    Neither contains the other, so converting between them is what
    `base_extend` cannot express -- the shape a scaled ciphertext product
    lands in, divided by the very primes the destination keeps.
    """
    from vfhe.arith.impl.rns.polynomial import RNSRing

    dst = Ring(n, prime_size=[plaintext_bits, *other_bits], split_degree=1)
    aux = []
    for bits in aux_bits:
        aux.append(RNSRing.gen_prime(2 * n, bits, exclude_list=dst.primes + aux))
    src = Ring(n, primes=aux, prime_size=aux_bits, split_degree=1).union(
        dst.quotient_ring(ell=1)
    )
    return src, dst


def test_convert_base_exact_removes_the_overflow():
    """`exact=True` writes the value; the default writes it plus a multiple of M.

    Both are congruent to the value modulo ``M``. Where the destination is a
    superset that is enough, since reducing back recovers it; here the two
    rings share one prime, so the term lands in the answer at full size.
    """
    src, dst = _cross_base_rings(N, 17, [30, 30], [30, 30, 30])
    M, q, half = src.q_l, dst.q_l, src.q_l // 2
    assert not dst.is_quotient_ring(src)  # a genuine cross-base move

    # An eighth of the modulus either side of zero, then shifted to the middle
    # of [0, M): the preparation the exact conversion asks of a caller.
    values = [rng.randrange(-(M // 8), M // 8) for _ in range(N)]
    source = Polynomial(src).from_bigint_array([(v + half) % M for v in values])
    source.to_coeff()
    expected = [(v + half) % q for v in values]

    assert source.copy().convert_base(dst, exact=True).get_polynomial() == expected

    fast = source.copy().convert_base(dst).get_polynomial()
    assert fast != expected
    # and what it is off by is one of the first `ell` multiples of M
    reachable = {u * M % q for u in range(src.ell)}
    assert {(f - e) % q for f, e in zip(fast, expected, strict=True)} <= reachable


@pytest.mark.parametrize("n_source", [6, 10, 18])
def test_convert_base_exact_over_every_lookup_width(n_source):
    """Exact for a source base of any width.

    The overflow table is one entry per input prime plus one, and the kernel
    picks it out of registers while it fits in one vector, then two, and falls
    back to a gather beyond that. These three widths cross both boundaries.
    """
    src, dst = _cross_base_rings(N, 17, [30, 30], [30] * n_source)
    M, q, half = src.q_l, dst.q_l, src.q_l // 2
    values = [rng.randrange(-(M // 8), M // 8) for _ in range(N)]
    source = Polynomial(src).from_bigint_array([(v + half) % M for v in values])
    source.to_coeff()
    assert source.convert_base(dst, exact=True).get_polynomial() == [
        (v + half) % q for v in values
    ]


def test_convert_base_default_matches_base_extend():
    """The default path is untouched: same result as `base_extend`."""
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    wider = Ring(N, prime_size=[30, 30, 30], split_degree=1)
    a = ring.random_element()
    a.to_coeff()
    assert (
        a.copy().convert_base(wider).get_polynomial()
        == a.copy().base_extend(wider).get_polynomial()
    )


def test_convert_base_exact_where_base_extend_also_applies():
    """Extending into a superset, only `exact=True` gives the integer itself.

    `base_extend` lands a multiple of the source modulus away, which is why
    reducing back into the source ring recovers the value either way.
    """
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    wider = Ring(N, prime_size=[30, 30, 30], split_degree=1)
    a = ring.random_element()
    a.to_coeff()
    value = a.get_polynomial()

    assert a.copy().convert_base(wider, exact=True).get_polynomial() == value

    extended = a.copy().base_extend(wider)
    lifted = extended.get_polynomial()
    # congruent modulo the source modulus, but not the value itself
    assert all((v - x) % ring.q_l == 0 for v, x in zip(lifted, value, strict=True))
    assert lifted != value
    assert extended.copy().mod_reduce(ring).get_polynomial() == value


def _lift_residue(poly, idx, centered):
    lib = poly.ring.lib
    lift = (
        lib.polynomial_RNSc_mod_reduce_lifted_centered
        if centered
        else lib.polynomial_RNSc_mod_reduce_lifted
    )
    out = Polynomial(poly.ring, repr.coeff)
    lift(out.obj, poly.obj, idx)
    return out


def test_centered_lift_is_the_rns_gadget_digit():
    """Residue ``j`` as one integer in ``(-p_j/2, p_j/2]``, in every prime.

    That integer is the residue mod ``p_j``, so the digits recombine through
    the CRT idempotents to the value mod ``Q``. Two narrow primes and two wide
    ones put every pair of row widths on the lift's path, and the first
    coefficients of each row sit where the centered representative turns
    negative.
    """
    ring = Ring(N, prime_size=[42, 25, 52, 28], split_degree=1)
    x = ring.random_element(ntt=False)
    rows = x.get_coeff_matrix()
    for row, p in zip(rows, ring.primes, strict=True):
        row[:4] = [0, (p - 1) // 2, (p + 1) // 2, p - 1]
    x.from_coeff_matrix(rows)

    Q = ring.q_l
    recombined = [0] * N
    for idx, p, row in zip(ring.prime_indices, ring.primes, rows, strict=True):
        digit = _lift_residue(x, idx, centered=True).get_polynomial(signed=True)
        assert digit == [r - p if r > p // 2 else r for r in row]
        assert digit[:4] == [0, (p - 1) // 2, -(p - 1) // 2, -1]
        idempotent = Q // p * pow(Q // p, -1, p)
        recombined = [
            s + d * idempotent for s, d in zip(recombined, digit, strict=True)
        ]

        assert _lift_residue(x, idx, centered=False).get_polynomial() == row
    assert [s % Q for s in recombined] == x.get_polynomial()
