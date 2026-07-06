# SPDX-License-Identifier: Apache-2.0
"""Battle tests for vfhe.arith: seeded randomized rounds against exact
Python big-int oracles, plus the corners the basic suite does not reach
(non-contiguous limb masks, shared prime pools, slot operations, operator
protocol, statistical sanity of the samplers).
"""

import random

import pytest
from vfhe.arith import (
    ComplexPolynomial,
    ComplexRing,
    Domain,
    Multiprecision,
    Polynomial,
    Ring,
    UnsupportedError,
    mp_polynomial_to_list,
)

rng = random.Random(0xC0FFEE)  # deterministic: failures reproduce


def negacyclic_mul(a, b, q, n):
    """Schoolbook oracle in Z_q[X]/(X^n + 1)."""
    out = [0] * n
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            sign = -1 if i + j >= n else 1
            out[k] = (out[k] + sign * a[i] * b[j]) % q
    return out


def rand_coeffs(n, bound):
    return [rng.randrange(bound) for _ in range(n)]


# --- randomized composite expressions -------------------------------------------


def test_randomized_expression_rounds():
    """(a*b + c - d) * e over 6 seeded rounds vs the big-int oracle, N=64."""
    N = 64
    ring = Ring(N, prime_size=[30, 30, 30], split_degree=1)
    Q = ring.q_l
    for _ in range(6):
        coeffs = [rand_coeffs(N, Q) for _ in range(5)]
        a, b, c, d, e = (Polynomial(ring).from_bigint_array(v) for v in coeffs)
        got = ((a * b + c - d) * e).get_polynomial()

        ab = negacyclic_mul(coeffs[0], coeffs[1], Q, N)
        inner = [(x + y - z) % Q for x, y, z in zip(ab, coeffs[2], coeffs[3])]
        want = negacyclic_mul(inner, coeffs[4], Q, N)
        assert got == want


@pytest.mark.parametrize("split_degree", [2, 4])
def test_randomized_mul_split_degrees(split_degree):
    """Randomized products on incomplete-NTT rings (cross-term kernel)."""
    N = 32
    ring = Ring(N, prime_size=[30, 30], split_degree=split_degree)
    Q = ring.q_l
    for _ in range(4):
        a_c, b_c = rand_coeffs(N, Q), rand_coeffs(N, Q)
        a = Polynomial(ring).from_bigint_array(a_c)
        b = Polynomial(ring).from_bigint_array(b_c)
        assert (a * b).get_polynomial() == negacyclic_mul(a_c, b_c, Q, N)


# --- masks and shared pools --------------------------------------------------------


def test_non_contiguous_limb_mask():
    """A quotient ring skipping a middle prime: arithmetic and rescaling."""
    N = 16
    base = Ring(N, prime_size=[30, 30, 30], split_degree=1)
    p0, p1, p2 = base.primes
    # Select limbs {0, 2} of the pool -- a hole at limb 1.
    sub = base.quotient_ring(mask=(base.mask & ~(1 << base.prime_indices[1])))
    assert sub.ell == 2 and sub.primes == [p0, p2]
    assert sub.is_quotient_ring(base)

    val = rand_coeffs(N, sub.q_l)
    a = Polynomial(sub).from_bigint_array(val)
    b = Polynomial(sub).from_bigint_array(val)
    assert (a + b).get_polynomial() == [2 * v % sub.q_l for v in val]
    assert (a * b).get_polynomial() == negacyclic_mul(val, val, sub.q_l, N)

    # Divide the middle prime out of a full-ring element: v * p1 -> v.
    full_val = [v * p1 for v in val]
    p = Polynomial(base).from_bigint_array(full_val)
    p.round_division(sub)
    assert p.get_polynomial() == [v % sub.q_l for v in val]


def test_pool_extension_keeps_old_polynomials_valid():
    """Growing the shared pool must not disturb live polynomials."""
    N = 16
    r1 = Ring(N, prime_size=[30, 30], split_degree=1)
    val = rand_coeffs(N, r1.q_l)
    p = Polynomial(r1).from_bigint_array(val)

    # This registers new primes into the same (N, split_degree) pool.
    r2 = Ring(N, prime_size=[31, 31], split_degree=1)
    assert set(r2.prime_indices).isdisjoint(set(r1.prime_indices))

    assert p.get_polynomial() == [v % r1.q_l for v in val]
    q = Polynomial(r2).from_bigint_array(val)
    assert q.get_polynomial() == [v % r2.q_l for v in val]


def test_lift_between_pool_generations():
    """A polynomial allocated before a pool extension lifts into a ring that
    includes the newer primes (the output carries the headroom)."""
    N = 16
    r_small = Ring(N, prime_size=[30, 30], split_degree=1)
    val = [v % r_small.q_l for v in rand_coeffs(N, r_small.q_l)]
    p = Polynomial(r_small).from_bigint_array(val)

    r_big = Ring(
        N,
        prime_size=r_small.prime_size + [31],
        split_degree=1,
        primes=r_small.primes + Ring(N, prime_size=[31], split_degree=1).primes,
    )
    lifted = p.scaled_lift(r_big)
    delta = r_big.q_l // r_small.q_l
    assert lifted.get_polynomial() == [(v * delta) % r_big.q_l for v in val]


# --- slot operations ------------------------------------------------------------------


def test_slot_rotation_and_broadcast():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=2)
    ps = N // ring.split_degree
    p = ring.random_element()
    rows = p.get_coeff_matrix(domain=Domain.EVAL)

    rot = p.rotate_slots(3)
    rot_rows = rot.get_coeff_matrix(domain=Domain.EVAL)
    # get_coeff_matrix interleaves blocks; undo per (block, slot) coordinates.
    for level in range(ring.ell):
        for blk in range(ring.split_degree):
            orig = [rows[level][j] for j in range(N) if j % ring.split_degree == blk]
            got = [rot_rows[level][j] for j in range(N) if j % ring.split_degree == blk]
            assert got == [orig[(k + 3) % ps] for k in range(ps)]

    # Rotating back is the identity.
    assert rot.rotate_slots(ps - 3) == p

    bc = p.broadcast_slot(0)
    bc_rows = bc.get_coeff_matrix(domain=Domain.EVAL)
    for level in range(ring.ell):
        for blk in range(ring.split_degree):
            got = {bc_rows[level][j] for j in range(N) if j % ring.split_degree == blk}
            assert len(got) == 1

    cp = p.copy_slot(5, 1)
    assert cp.rns_mask == p.rns_mask


# --- operator protocol ------------------------------------------------------------------


def test_operator_protocol():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    Q = ring.q_l
    val = rand_coeffs(N, Q)
    p = Polynomial(ring).from_bigint_array(val)

    # Reflected / scalar forms. (7 - p) negates and adds 7 to the constant.
    assert (3 * p).get_polynomial() == [(3 * v) % Q for v in val]
    assert (p + 0).get_polynomial() == val
    assert (0 + p).get_polynomial() == val
    want_rsub = [(-v) % Q for v in val]
    want_rsub[0] = (7 - val[0]) % Q
    assert (7 - p).get_polynomial() == want_rsub

    # In-place forms.
    q = p.copy()
    q += p
    assert q.get_polynomial() == [(2 * v) % Q for v in val]
    q -= p
    assert q == p
    q *= 2
    assert q.get_polynomial() == [(2 * v) % Q for v in val]

    # Per-level scalar vector multiply.
    scal = [rng.randrange(pr) for pr in ring.primes]
    got = (p * scal).get_coeff_matrix()
    plain = p.get_coeff_matrix()
    for lvl in range(ring.ell):
        assert got[lvl] == [(x * scal[lvl]) % ring.primes[lvl] for x in plain[lvl]]

    # Equality against ints and per-level constant lists.
    assert Polynomial(ring).from_array([1]) == 1
    lvl_vals = [(5 % pr) for pr in ring.primes]
    assert Polynomial(ring).from_array([5]) == lvl_vals


def test_mod_and_itruediv_operators():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    p0 = ring.primes[0]
    val = rand_coeffs(N, p0)
    p = Polynomial(ring).from_bigint_array(val)

    lifted = p % p0
    assert lifted.get_polynomial() == [v % p0 for v in val]

    small = ring.quotient_ring(ell=1)
    scaled = Polynomial(ring).from_bigint_array([v * ring.primes[1] for v in val])
    scaled /= ring.primes[1]
    scaled.ring = small  # the divide dropped limb 1; rebind the Python view
    assert scaled.get_polynomial() == [v % small.q_l for v in val]


def test_decompose_reconstruction():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    Q = ring.q_l
    base_bits = 10
    val = rand_coeffs(N, Q)
    p = Polynomial(ring).from_bigint_array(val)
    digits = p.decompose(base_bits)

    recon = [0] * N
    for level, d in enumerate(digits):
        dv = d.get_polynomial()
        for i in range(N):
            recon[i] += dv[i] << (base_bits * level)
    assert [r % Q for r in recon] == val


# --- sampling statistics -------------------------------------------------------------------


def test_gaussian_moments():
    N = 1024
    ring = Ring(N, prime_size=[30], split_degree=1, exceptional_set_size=16)
    sigma = 3.2
    samples = ring.random_gaussian_element(sigma).get_polynomial(signed=True)
    mean = sum(samples) / N
    var = sum(s * s for s in samples) / N - mean * mean
    assert abs(mean) < 1.0  # sd of the mean ~ 0.1: 10-sigma bound
    assert 6.0 < var < 16.0  # sigma^2 = 10.24


def test_uniform_is_full_range():
    N = 64
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    rows = ring.random_element().get_coeff_matrix(domain=Domain.EVAL)
    for lvl, row in enumerate(rows):
        assert all(0 <= x < ring.primes[lvl] for x in row)
        assert max(row) > ring.primes[lvl] // 8  # not stuck near zero


# --- hashing ----------------------------------------------------------------------------------


def test_digest_mask_sensitivity():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    sub = ring.quotient_ring(ell=1)
    val = rand_coeffs(N, sub.q_l)
    full = Polynomial(ring).from_bigint_array(val)
    reduced = full.mod_reduce(sub)
    # Same shared limb data, different mask -> different transcript digest.
    assert full.get_hash() != reduced.get_hash()


# --- error surface ------------------------------------------------------------------------------


def test_unsupported_inverse_on_split_ring():
    ring = Ring(16, prime_size=[30, 30], split_degree=2)
    p = ring.random_element()
    with pytest.raises(UnsupportedError):
        p.fast_inverse()


# --- multiprecision + encoding stress -----------------------------------------------------------


def test_mp_bridge_randomized_extremes():
    N = 16
    ring = Ring(N, prime_size=[49, 49, 49], split_degree=1)
    Q = ring.q_l
    mp = Multiprecision()
    consts = mp.compute_crt_consts(ring.primes)
    for vals in ([0] * N, [Q - 1] * N, rand_coeffs(N, Q)):
        p = Polynomial(ring).from_bigint_array(vals)
        assert mp_polynomial_to_list(mp.from_polynomial(p, consts)) == [
            v % Q for v in vals
        ]


def test_encode_batch_matches_single():
    N = 16
    ring = Ring(N, prime_size=[30, 30], split_degree=1)
    cring = ComplexRing(N // 2)
    delta = 2.0**10

    slots = [[complex(i + 1, -j) for i in range(N // 2)] for j in range(3)]
    # Single-polynomial path.
    singles = []
    for s in slots:
        cp = ComplexPolynomial(cring).from_array(s)
        cp.IFFT()
        cp *= delta
        singles.append(cp.round_to_RNS_native(ring).get_polynomial())
    # Batched native pipeline: same math (bit-reverse, inverse FFT, scale by
    # delta/N, round), fanned out over native threads.
    batch_in = [ComplexPolynomial(cring).from_array(s) for s in slots]
    batched = cring.exp_complex_polys_ifft_scale_round_to_RNS_batch(
        batch_in, ring, delta
    )
    for got, want in zip(batched, singles):
        assert got.get_polynomial() == want
