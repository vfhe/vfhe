# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The SHIP bootstrap [CHKS25]: Lemma 1's masks in the clear, the windowed
input keys, the blind rotations against direct evaluation, and the half and
full bootstrap."""

from __future__ import annotations

import cmath
import math
import random

import pytest
from vfhe import engine
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import SHIP, CKKS_Ciphertext, CKKS_Scheme
from vfhe.fhe.ship import _lemma1_masks
from vfhe.mlwe import MLWE_Key

N = 256
M = N // 2
H = 4
GAMMA = 32.0
# The sin approximation's error, twice, and the rounding of the coefficients.
TOLERANCE = 2 * (2 * math.pi) ** 2 / (6 * GAMMA**2) + 2 / GAMMA


def _scheme(depth: int = H.bit_length()) -> CKKS_Scheme:
    """Table 2's LL13 chain at N = 256 [CHKS25]: q0, SlotToCoeff, one level for
    the application and ``depth`` for the product, then the special prime."""
    ring = Ring(N, prime_size=[25, 20, 20] + [25] * depth + [28], split_degree=1)
    return CKKS_Scheme(ring, scaling_factor=2.0**25, special_primes=1)


def _ship(scheme: CKKS_Scheme, theta: int = 6, window: int | None = 16) -> SHIP:
    return SHIP(scheme, h=H, theta=theta, base=4, gamma=GAMMA, window=window)


@pytest.fixture(scope="module")
def setup():
    scheme = _scheme()
    ship = _ship(scheme)
    dense = scheme.key_gen_sparse(M, 3.2)
    scheme.rlk = scheme.gen_rlk(dense, dense)
    sparse = ship.sample_input_key(3.2)
    return scheme, ship, dense, sparse, ship.generate_bootstrap_key(dense, sparse)


def _values(seed: int) -> list[float]:
    """N/2 real slot values in [-1, 1]."""
    rng = random.Random(seed)  # noqa: S311 - test data, not a key
    return [rng.uniform(-1, 1) for _ in range(M)]


def _centered(x: int, q: int) -> int:
    x %= q
    return x - q if x > q // 2 else x


def _factor(a: list[int], j: int, sign: int, q: int) -> list[complex]:
    """omega^((sign X^j a)_i) for i < N/2, evaluated directly."""
    n = len(a)
    out = []
    for i in range(n // 2):
        value = sign * a[i - j] if i >= j else -sign * a[n + i - j]
        out.append(cmath.exp(2j * math.pi * _centered(value, q) / q))
    return out


def _rotate(v: list[complex], k: int) -> list[complex]:
    """The paper's Rot_k, to the right: (Rot_k v)_i = v_(i - k)."""
    return [v[(i - k) % len(v)] for i in range(len(v))]


def _max_error(x, y) -> float:
    return max(abs(u - v) for u, v in zip(x, y, strict=True))


# --- Lemma 1, in the clear --------------------------------------------------


@pytest.mark.parametrize("n", [16, 64])
def test_lemma1_masks(n):
    # sum_k Rot_j(pi_k) * mask_k is omega^((s_j X^j a)_i): both halves of the
    # key's support, both signs, and the edges of each half.
    q = 1 << 16
    rng = random.Random(n)  # noqa: S311 - test data, not a key
    a = [rng.randrange(q) for _ in range(n)]
    half = n // 2
    w = [cmath.exp(2j * math.pi * _centered(x, q) / q) for x in a]
    pis = [w[:half], [x.conjugate() for x in w[:half]], w[half:]]
    pis.append([x.conjugate() for x in pis[2]])
    edges = [0, 1, half - 1, half, half + 3, n - 1]
    for j in edges + [rng.randrange(n) for _ in range(8)]:
        for sign in (-1, 1):
            masks = _lemma1_masks(j, sign, half)
            got = [
                sum(
                    _rotate(pi, j % half)[i] * mask[i]
                    for pi, mask in zip(pis, masks, strict=True)
                )
                for i in range(half)
            ]
            assert _max_error(got, _factor(a, j, sign, q)) < 1e-9


# --- input keys --------------------------------------------------------------


def test_input_keys_have_one_coefficient_per_window():
    scheme = _scheme()
    ship = _ship(scheme)
    for _ in range(10):
        key = ship.sample_input_key(3.2)
        positions = ship._assign_windows(key)
        assert sorted(positions) == [j for j, c in enumerate(key.key[0]) if c]
        for k, j in enumerate(positions):
            assert (j - ((k * N) // H - ship.window)) % N < 2 * ship.window
            assert key.key[0][j] in (-1, 1)


def test_input_keys_are_checked():
    scheme = _scheme()
    ship = _ship(scheme)
    dense = scheme.key_gen_sparse(M, 3.2)

    def key_at(positions, value=1):
        poly = [0] * N
        for j in positions:
            poly[j] = value
        return MLWE_Key([poly], 3.2, scheme)

    # Windows of 32 coefficients around 0, 64, 128 and 192.
    ship._assign_windows(key_at([250, 70, 130, 200]))
    with pytest.raises(ValueError, match="one non-zero coefficient per window"):
        ship._assign_windows(key_at([5, 10, 130, 200]))
    with pytest.raises(ValueError, match="h = 4"):
        ship.generate_bootstrap_key(dense, key_at([250, 70, 130]))
    with pytest.raises(ValueError, match="-1, 1"):
        ship.generate_bootstrap_key(dense, key_at([250, 70, 130, 200], value=2))


def test_overlapping_windows_take_any_matching():
    # With w > N / (2h) the windows overlap, and coefficients may need
    # matching to windows out of order.
    scheme = _scheme()
    ship = SHIP(scheme, h=H, theta=6, window=40)
    poly = [0] * N
    for j in (30, 20, 140, 200):  # 30 is in windows 0 and 1, 20 only in 0
        poly[j] = 1
    positions = ship._assign_windows(MLWE_Key([poly], 3.2, scheme))
    assert positions[:2] == [20, 30]


def test_parameters_are_checked():
    with pytest.raises(ValueError, match="special primes"):
        SHIP(
            CKKS_Scheme(Ring(N, prime_size=[25, 20, 20, 25, 25, 25], split_degree=1)),
            h=H,
            theta=6,
        )
    with pytest.raises(ValueError, match="levels"):
        SHIP(_scheme(depth=1), h=H, theta=6)
    with pytest.raises(ValueError, match="theta"):
        SHIP(_scheme(), h=H, theta=M + 1)


# --- blind rotations ---------------------------------------------------------


@pytest.mark.parametrize(
    "theta, window",
    [
        (6, 16),  # the windows' alignments, and two mux digits
        (6, None),  # the paper's w = 175 covers the ring: no alignment
        (M, 16),  # the column step alone
        (1, None),  # the mux step alone, four digits
    ],
)
def test_blind_rotate(theta, window):
    scheme = _scheme()
    ship = _ship(scheme, theta, window)
    dense = scheme.key_gen_sparse(M, 3.2)
    sparse_key = ship.sample_input_key(3.2)
    key = ship.generate_bootstrap_key(dense, sparse_key)
    lowest = ship.lowest_lvl
    rng = random.Random(theta)  # noqa: S311 - test data, not a key
    sample = scheme.sample(
        Polynomial(scheme.rings[lowest]).from_array(
            [rng.randrange(1 << 20) for _ in range(N)]
        ),
        sparse_key,
        lvl=lowest,
    )
    factors = ship.blind_rotate(sample, key)
    sample.to_coeff()
    a = [-x for x in sample.get_a_poly(0).get_polynomial(signed=True)]
    for f, j in zip(factors, ship._assign_windows(sparse_key), strict=True):
        assert f.lvl == 0 and f.delta == scheme.scaling_factor
        got = scheme.decode(scheme.decrypt(f, dense), f.delta)
        assert _max_error(got, _factor(a, j, sparse_key.key[0][j], ship.q0)) < 1e-3


# --- the bootstrap -----------------------------------------------------------


def _coefficient_input(scheme, ship, dense, mu, scale):
    """mu in the coefficients t < N/2 at the lowest level, at ``scale``."""
    lowest = ship.lowest_lvl
    coeffs = [round(scale * x) for x in mu] + [0] * M
    ct = CKKS_Ciphertext(scheme, lvl=lowest)
    scheme.sample(
        Polynomial(scheme.rings[lowest]).from_bigint_array(coeffs), dense, out=ct
    )
    ct.delta = scale
    return ct


def _slots(scheme, ct, key) -> list[float]:
    return [z.real for z in scheme.decode(scheme.decrypt(ct, key), ct.delta)]


@pytest.mark.parametrize("scale_factor", [1.0, 0.5])
def test_half_bootstrap(setup, scale_factor):
    # gamma comes from the input's actual scale, so other scales work too.
    scheme, ship, dense, _sparse, key = setup
    mu = _values(1)
    ct = _coefficient_input(scheme, ship, dense, mu, ship.input_scale * scale_factor)
    out = ship.half_bootstrap(ct, key)
    assert out.lvl == ship.output_lvl
    assert _max_error(_slots(scheme, out, dense), mu) < TOLERANCE


def test_bootstrap(setup):
    scheme, ship, dense, _sparse, key = setup
    mu = _values(2)
    lvl = ship.slot_to_coeff_lvl
    ct = CKKS_Ciphertext(scheme, lvl=lvl)
    scheme.sample(
        scheme.encode(mu, ring=scheme.rings[lvl], scale=ship.input_scale), dense, out=ct
    )
    ct.delta = ship.input_scale

    # SlotToCoeff puts round(q0 mu_t / gamma) in coefficient t < N/2.
    coefficients = scheme.rescale(ship.slot_to_coeff.apply(ct, key.slot_to_coeff_keys))
    assert coefficients.lvl == ship.lowest_lvl
    plain = scheme.linear_decrypt(coefficients, dense)
    plain.to_coeff()
    values = plain.get_polynomial(signed=True)
    assert max(abs(values[t] - ship.q0 * mu[t] / GAMMA) for t in range(M)) < 1 << 10

    out = ship.bootstrap(ct, key)
    assert out.lvl == ship.output_lvl
    assert _max_error(_slots(scheme, out, dense), mu) < TOLERANCE
    # The output is an ordinary ciphertext: one level is left to multiply.
    product = out * out
    assert (
        _max_error(_slots(scheme, product, dense), [x * x for x in mu]) < 3 * TOLERANCE
    )


def test_bootstrap_does_not_depend_on_the_thread_count(setup):
    scheme, ship, dense, _sparse, key = setup
    mu = _values(3)
    ct = _coefficient_input(scheme, ship, dense, mu, ship.input_scale)
    one = ship.half_bootstrap(ct, key, n_threads=1)
    engine.set_num_threads(8)
    try:
        many = ship.half_bootstrap(ct, key, n_threads=0)
    finally:
        engine.set_num_threads()
    for c in (one, many):
        c.to_coeff()
    assert one.get_a_poly(0) == many.get_a_poly(0)
    assert one.get_b_poly() == many.get_b_poly()


def test_special_prime_of_a_ring_that_is_not_the_first_of_its_base():
    # P is the special prime even when the ring's primes are not the first of
    # the shared RNS base.
    Ring(N, prime_size=[30, 30], split_degree=1)
    scheme = _scheme()
    assert scheme.special_primes == 1
    assert (
        int(scheme.special_rings[0].modulus_ratio(scheme.rings[0])).bit_length() == 28
    )
    ship = _ship(scheme)
    dense = scheme.key_gen_sparse(M, 3.2)
    scheme.rlk = scheme.gen_rlk(dense, dense)
    key = ship.generate_bootstrap_key(dense, ship.sample_input_key(3.2))
    mu = _values(4)
    out = ship.half_bootstrap(
        _coefficient_input(scheme, ship, dense, mu, ship.input_scale), key
    )
    assert _max_error(_slots(scheme, out, dense), mu) < TOLERANCE


def test_inputs_are_checked(setup):
    scheme, ship, dense, _sparse, key = setup
    ct = scheme.encrypt(scheme.encode([0.5] * M), dense)
    with pytest.raises(ValueError, match="SlotToCoeff level"):
        ship.bootstrap(ct, key)
    with pytest.raises(ValueError, match="lowest level"):
        ship.half_bootstrap(ct, key)
    other = SHIP(scheme, h=H - 1, theta=6, window=16)
    low = _coefficient_input(scheme, ship, dense, [0.5] * M, ship.input_scale)
    with pytest.raises(ValueError, match="h = 3"):
        other.blind_rotate(scheme.keyswitch(low, key.sparse_switch_key), key)
