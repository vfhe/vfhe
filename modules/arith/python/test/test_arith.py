# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for vfhe.arith against Python big-int oracles.

Python integers are exact, so every native result is checked against a
straightforward reimplementation: negacyclic schoolbook multiplication for
the ring product, CRT for reconstruction, exact integer division for the
tower operations.
"""

import pytest
from vfhe.arith import (
    ComplexPolynomial,
    ComplexRing,
    Domain,
    Multiprecision,
    NotInvertibleError,
    Polynomial,
    Ring,
    crt,
    is_prime,
    mp_polynomial_to_list,
)

N = 16
PRIME_SIZE = [30, 30, 30]


@pytest.fixture(scope="module")
def ring():
    """A 3-level ring with a complete NTT (split_degree = 1)."""
    return Ring(N, prime_size=PRIME_SIZE, split_degree=1)


@pytest.fixture(scope="module")
def ring2(ring):
    """The 2-level quotient ring of `ring`."""
    return ring.quotient_ring(ell=2)


def negacyclic_mul(a, b, q, n):
    """Schoolbook multiplication oracle in Z_q[X]/(X^n + 1)."""
    out = [0] * n
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            sign = -1 if i + j >= n else 1
            out[k] = (out[k] + sign * a[i] * b[j]) % q
    return out


# --- number theory ------------------------------------------------------------


def test_is_prime_and_crt():
    assert is_prime(2**31 - 1)
    assert not is_prime(2**32 - 1)
    moduli = [97, 101, 103]
    x = 123456
    assert crt([x % m for m in moduli], moduli) == x % (97 * 101 * 103)


def test_ring_primes_are_ntt_friendly(ring):
    for p, size in zip(ring.primes, PRIME_SIZE):
        assert is_prime(p)
        assert p.bit_length() <= size
        assert p % (2 * N // ring.split_degree) == 1


# --- loading / representation ---------------------------------------------------


def test_from_array_roundtrip(ring):
    coeffs = list(range(1, N + 1))
    p = Polynomial(ring).from_array(coeffs)
    assert p.domain == Domain.EVAL
    assert p.get_polynomial() == coeffs
    p.to_coeff()
    assert p.domain == Domain.COEFF
    p.to_eval()
    assert p.get_polynomial() == coeffs


def test_from_bigint_array_roundtrip(ring):
    big = [(ring.q_l - 1) // (i + 1) for i in range(N)]
    p = Polynomial(ring).from_bigint_array(big)
    assert p.get_polynomial() == [b % ring.q_l for b in big]


# --- arithmetic ------------------------------------------------------------------


def test_add_sub_negate_scale(ring):
    a_c = [i * 7 + 1 for i in range(N)]
    b_c = [i * i + 3 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)

    assert (a + b).get_polynomial() == [(x + y) % ring.q_l for x, y in zip(a_c, b_c)]
    assert (a - b).get_polynomial() == [(x - y) % ring.q_l for x, y in zip(a_c, b_c)]
    assert (-a).get_polynomial() == [(-x) % ring.q_l for x in a_c]
    assert (a * 5).get_polynomial() == [(5 * x) % ring.q_l for x in a_c]
    assert (a + 9).get_polynomial() == [(a_c[0] + 9) % ring.q_l] + a_c[1:]


def test_mul_against_schoolbook(ring):
    a_c = [i + 1 for i in range(N)]
    b_c = [3 * i + 2 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)
    expect = negacyclic_mul(a_c, b_c, ring.q_l, N)
    assert (a * b).get_polynomial() == expect


def test_mul_split_degree_against_schoolbook():
    ring = Ring(N, prime_size=[30, 30], split_degree=2)
    a_c = [i + 1 for i in range(N)]
    b_c = [2 * i + 5 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)
    expect = negacyclic_mul(a_c, b_c, ring.q_l, N)
    assert (a * b).get_polynomial() == expect


def test_imul_matches_mul(ring):
    a_c = [i + 2 for i in range(N)]
    b_c = [5 * i + 1 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    b = Polynomial(ring).from_array(b_c)
    a *= b
    assert a.get_polynomial() == negacyclic_mul(a_c, b_c, ring.q_l, N)


def test_fast_inverse(ring):
    p = ring.random_element()
    inv = p.fast_inverse()
    one = Polynomial(ring).from_array([1])
    assert (p * inv) == one


def test_fast_inverse_zero_raises(ring):
    zero = Polynomial(ring).from_array([0])
    with pytest.raises(NotInvertibleError):
        zero.fast_inverse()


def test_automorphism(ring):
    gen = 3
    a_c = [i + 1 for i in range(N)]
    a = Polynomial(ring).from_array(a_c)
    out = a.automorphism(gen)
    # Oracle: X^i -> X^(i*gen) with negacyclic sign wrap.
    expect = [0] * N
    for i in range(N):
        idx = (i * gen) % (2 * N)
        sign = -1 if idx >= N else 1
        expect[idx % N] = (expect[idx % N] + sign * a_c[i]) % ring.q_l
    assert out.get_polynomial() == expect


# --- sampling ------------------------------------------------------------------------


def test_sample_uniform_and_hash(ring):
    a = ring.random_element()
    b = ring.random_element()
    assert a.get_hash() == a.get_hash()
    assert a.get_hash() != b.get_hash()


def test_sample_gaussian_is_small(ring):
    p = ring.random_gaussian_element(sigma=3.2)
    signed = p.get_polynomial(signed=True)
    assert all(abs(c) < 64 for c in signed)


def test_sample_exceptional_constant_slots(ring):
    p = ring.random_exceptional()
    rows = p.get_coeff_matrix(domain=Domain.EVAL)
    for row in rows:
        assert len(set(row)) == 1


# --- tower ---------------------------------------------------------------------------


def test_mod_reduce(ring, ring2):
    val = [ring.q_l // (i + 2) for i in range(N)]
    p = Polynomial(ring).from_bigint_array(val)
    reduced = p.mod_reduce(ring2)
    assert reduced.get_polynomial() == [v % ring2.q_l for v in val]


def test_scaled_lift_then_divide_roundtrip(ring, ring2):
    val = [ring2.q_l // (i + 3) for i in range(N)]
    p = Polynomial(ring2).from_bigint_array(val)
    # v -> v * Delta in the big ring (exact), then round-divide by Delta.
    lifted = p.scaled_lift(ring)
    assert lifted.get_polynomial() == [
        (v * (ring.q_l // ring2.q_l)) % ring.q_l for v in val
    ]
    lifted.round_division(ring2)
    assert lifted.get_polynomial() == [v % ring2.q_l for v in val]


def test_base_extend_congruence(ring, ring2):
    val = [ring2.q_l // (i + 2) for i in range(N)]
    p = Polynomial(ring2).from_bigint_array(val)
    lifted = p.lift_to(ring=ring)
    # Fast base extension is exact up to a small multiple of the source
    # modulus: check congruence and the overshoot bound.
    for got, v in zip(lifted.get_polynomial(), val):
        assert (got - v) % ring2.q_l == 0
        assert got < ring2.q_l * (ring2.ell + 1)


def test_round_division_of_scaled_value(ring, ring2):
    scale = ring.q_l // ring2.q_l  # the dropped prime
    val = [(i + 1) * 1000 for i in range(N)]
    p = Polynomial(ring).from_bigint_array([v * scale for v in val])
    p.round_division(ring2)
    assert p.get_polynomial() == [v % ring2.q_l for v in val]


def test_mod_lift_residue(ring):
    val = [i * 17 + 5 for i in range(N)]
    p = Polynomial(ring).from_array(val)
    q0 = ring.primes[0]
    lifted = p % q0
    assert lifted.get_polynomial() == [v % q0 for v in val]


# --- multiprecision bridge --------------------------------------------------------------


def test_mp_bridge_reconstruction():
    # The bridge's digit/Barrett schedule (d = ell + 1 digits, k = ~52 * d)
    # is sized for production primes near 52 bits, so use those here.
    mp_ring = Ring(N, prime_size=[49, 49, 49], split_degree=1)
    val = [mp_ring.q_l // (i + 2) for i in range(N)]
    p = Polynomial(mp_ring).from_bigint_array(val)
    mp = Multiprecision()
    consts = mp.compute_crt_consts(mp_ring.primes)
    mp_poly = mp.from_polynomial(p, consts)
    assert mp_polynomial_to_list(mp_poly) == [v % mp_ring.q_l for v in val]


# --- encoding ------------------------------------------------------------------------------


def test_cfft_roundtrip():
    cring = ComplexRing(16)
    values = [complex(i, -2 * i) for i in range(16)]
    cp = ComplexPolynomial(cring).from_array(values)
    cp.FFT()
    cp.IFFT()
    for got, want in zip(list(cp), values):
        assert abs(got - want) < 1e-6


def test_encode_round_to_rns(ring):
    cring = ComplexRing(N // 2)  # N/2 complex slots <-> N real coordinates
    cp = ComplexPolynomial(cring)
    coords = [float(i + 1) for i in range(N)]
    for i in range(N // 2):
        cp.obj[i] = coords[i]
        cp.obj[i + N // 2] = coords[i + N // 2]
    p = cring and cp.round_to_RNS_native(ring)
    assert p.get_polynomial() == [round(c) % ring.q_l for c in coords]
