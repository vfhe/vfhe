# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the (reverted) vfhe.arith over the cffi boundary.

Python big ints are the exact oracle: negacyclic schoolbook for the ring
product, CRT for reconstruction. Also covers domain conversion, automorphism,
slot inversion, the CKKS complex FFT roundtrip, and the multiprecision bridge.
"""

import random

import pytest
from vfhe.arith import (
    ComplexPolynomial,
    ComplexRing,
    Multiprecision,
    Polynomial,
    Ring,
    repr,
)

N = 16
rng = random.Random(0xC0FFEE)


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

    with pytest.raises(AssertionError, match="int64"):
        Polynomial(ring).from_array(coeffs) + 2**63


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
