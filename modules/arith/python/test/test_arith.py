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
    assert (a + b).get_polynomial() == [(x + y) % ring.q_l for x, y in zip(a_c, b_c)]
    assert (a - b).get_polynomial() == [(x - y) % ring.q_l for x, y in zip(a_c, b_c)]
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
    got = one
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
    for x, y in zip(slots, inv_slots):
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
