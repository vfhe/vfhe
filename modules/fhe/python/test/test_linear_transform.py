# SPDX-FileCopyrightText: 2026 Robin Koestler
# SPDX-License-Identifier: Apache-2.0
"""CKKS linear transforms: BSGS on slot matrices, SlotToCoeff, CoeffToSlot."""

import random

import pytest
import vfhe.engine as engine
from vfhe.arith import Polynomial, Ring
from vfhe.fhe import CKKS_LinearTransform, CKKS_Scheme, linear_transform

N = 64
M = N // 2
DELTA = 2**40
rng = random.Random(0x1A7)  # noqa: S311 - test data, not a key


@pytest.fixture
def threads():
    """Lets the test's parallel calls use up to 8 threads (vfhe defaults to 1)."""
    engine.set_num_threads(8)
    yield
    engine.set_num_threads()


def _values(n):
    return [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


@pytest.fixture(scope="module")
def setup():
    scheme = CKKS_Scheme(
        Ring(N, prime_size=[60, 40, 40, 40, 60], split_degree=1),
        scaling_factor=DELTA,
        special_primes=1,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    keys: dict[int, object] = {}
    return scheme, key, keys


def _keys(scheme, key, keys, transform):
    for k in transform.rotations:
        if k not in keys:
            keys[k] = scheme.gen_rotation_key(key, k)
    return keys


def _apply_matrix(diagonals, n, z):
    """A z for the n x n matrix with these diagonals, on every block of n."""
    return [
        sum(diag[t % n] * z[(t + d) % len(z)] for d, diag in diagonals.items())
        for t in range(len(z))
    ]


def _close(a, b, tol=1e-3):
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) < tol


@pytest.mark.parametrize("n_threads", [1, 3])
@pytest.mark.parametrize("baby_steps", [None, 1, 5])
@pytest.mark.usefixtures("threads")
def test_sparse_matrix_on_the_slots(setup, n_threads, baby_steps):
    scheme, key, keys = setup
    diagonals = {d: _values(M) for d in (0, 1, 6, 17, 31)}
    lt = CKKS_LinearTransform(scheme, diagonals, scale=2**30, baby_steps=baby_steps)
    z = _values(M)
    ct = scheme.encrypt(scheme.encode(z), key)
    out = lt.apply(ct, _keys(scheme, key, keys, lt), n_threads=n_threads)
    assert out.lvl == 0 and out.delta == DELTA * 2**30
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    assert _close(dec, _apply_matrix(diagonals, M, z))


@pytest.mark.usefixtures("threads")
def test_threads_share_the_baby_rotations():
    # Every giant step reads all the baby rotations; a thread must never see
    # one while another is still transforming it.
    big_n = 256
    scheme = CKKS_Scheme(
        Ring(big_n, prime_size=[60, 50, 50, 60], split_degree=1),
        scaling_factor=2**50,
        special_primes=1,
    )
    key = scheme.key_gen_sparse(big_n // 8, 3.2)
    lt = CKKS_LinearTransform.slot_to_coeff(scheme, scale=2**50)
    keys = {k: scheme.gen_rotation_key(key, k) for k in lt.rotations}
    slots = big_n // 2
    z = _values(slots)
    for n_threads in (2, 4, 8, 2, 4, 8):
        ct = scheme.encrypt(scheme.encode(z), key)
        out = lt.apply(ct, keys, n_threads=n_threads)
        m = scheme.decrypt(out, key).get_polynomial(signed=True)
        got = [complex(m[t], m[t + slots]) / out.delta for t in range(slots)]
        assert _close(got, z)


def test_a_matrix_on_every_block_of_slots(setup):
    # n = 8 of N/2 = 32: the slots hold an 8-periodic vector, and the same
    # matrix acts on each block.
    scheme, key, keys = setup
    n = 8
    diagonals = {d: _values(n) for d in range(n)}
    lt = CKKS_LinearTransform(scheme, diagonals, scale=2**30)
    w = _values(n)
    z = w * (M // n)
    ct = scheme.encrypt(scheme.encode(z), key)
    out = lt.apply(ct, _keys(scheme, key, keys, lt))
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    assert _close(dec, _apply_matrix(diagonals, n, z))


@pytest.mark.parametrize("slots", [M, 8])
def test_slot_to_coeff_puts_the_slots_in_the_coefficients(setup, slots):
    scheme, key, keys = setup
    n, R = slots, M // slots
    lt = CKKS_LinearTransform.slot_to_coeff(scheme, slots=slots, scale=2**30)
    w = _values(n)
    ct = scheme.encrypt(scheme.encode(w * (M // n)), key)
    out = lt.apply(ct, _keys(scheme, key, keys, lt))
    m = scheme.decrypt(out, key).get_polynomial(signed=True)
    got = [complex(m[R * t], m[R * t + M]) / out.delta for t in range(n)]
    assert _close(got, w)
    others = [m[k] for k in range(N) if k % R]
    assert all(abs(c) / out.delta < 1e-3 for c in others)


def test_coeff_to_slot_reads_the_coefficients(setup):
    scheme, key, keys = setup
    lt = CKKS_LinearTransform.coeff_to_slot(scheme, scale=2**30)
    coeffs = [rng.randrange(-(2**40), 2**40) for _ in range(N)]
    pt = Polynomial(scheme.ring).from_bigint_array(coeffs)
    ct = scheme.encrypt(pt, key)
    out = lt.apply(ct, _keys(scheme, key, keys, lt))
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    expected = [complex(coeffs[k], coeffs[k + M]) / DELTA for k in range(M)]
    assert _close(dec, expected)


@pytest.mark.parametrize("slots", [M, 8])
def test_coeff_to_slot_undoes_slot_to_coeff(setup, slots):
    scheme, key, keys = setup
    q1 = scheme.rings[0].primes[-1]
    s2c = CKKS_LinearTransform.slot_to_coeff(scheme, slots=slots, scale=q1)
    w = _values(slots) * (M // slots)
    ct = scheme.encrypt(scheme.encode(w), key)
    mid = scheme.rescale(s2c.apply(ct, _keys(scheme, key, keys, s2c)))
    c2s = CKKS_LinearTransform.coeff_to_slot(
        scheme, slots=slots, lvl=mid.lvl, scale=scheme.rings[1].primes[-1]
    )
    keys_1 = {k: scheme.gen_rotation_key(key, k) for k in c2s.rotations}
    out = scheme.rescale(c2s.apply(mid, keys_1))
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    assert _close(dec, w)


def test_conjugate(setup):
    scheme, key, _ = setup
    z = _values(M)
    ct = scheme.encrypt(scheme.encode(z), key)
    out = scheme.conjugate(ct, scheme.gen_conjugation_key(key))
    dec = scheme.decode(scheme.decrypt(out, key))
    assert _close(dec, [v.conjugate() for v in z])


def test_linear_combination_matches_term_by_term_products(setup):
    scheme, key, _ = setup
    zs = [_values(M) for _ in range(3)]
    ws = [_values(M) for _ in range(3)]
    cts = [scheme.encrypt(scheme.encode(z), key) for z in zs]
    pts = [scheme.encode(w) for w in ws]
    out = scheme.linear_combination(cts, pts)
    assert out.delta == DELTA * scheme.scaling_factor
    dec = scheme.decode(scheme.decrypt(out, key), scaling_factor=out.delta)
    expected = [sum(z[t] * w[t] for z, w in zip(zs, ws, strict=True)) for t in range(M)]
    assert _close(dec, expected)


@pytest.mark.usefixtures("threads")
def test_diagonals_are_encoded_rotated(setup, monkeypatch):
    # rot(diag_(j*b+i), -j*b), encoded in chunks smaller than the input.
    monkeypatch.setattr(linear_transform, "_ENCODE_CHUNK", 2)
    scheme, _, _ = setup
    n, b = 16, 3
    diagonals = {d: _values(n) for d in (0, 1, 5, 6, 10, 15)}
    lt = CKKS_LinearTransform(
        scheme, iter(diagonals.items()), baby_steps=b, n_threads=3
    )
    for d, diag in diagonals.items():
        j, i = divmod(d, b)
        cut = n - j * b
        expected = scheme.encode([*diag[cut:], *diag[:cut]])
        assert lt.plaintexts[j][i].get_polynomial(
            signed=True
        ) == expected.get_polynomial(signed=True)


def test_refusals(setup, monkeypatch):
    scheme, key, _ = setup
    with pytest.raises(ValueError, match="power of two"):
        CKKS_LinearTransform(scheme, {0: _values(6)})
    with pytest.raises(ValueError, match="twice"):
        CKKS_LinearTransform(scheme, [(1, _values(M)), (1, _values(M))])
    monkeypatch.setattr(linear_transform, "_ENCODE_CHUNK", 1)
    with pytest.raises(ValueError, match="twice"):
        CKKS_LinearTransform(scheme, [(1, _values(M)), (1, _values(M))])
    lt = CKKS_LinearTransform(scheme, {0: _values(M), 3: _values(M)})
    ct = scheme.encrypt(scheme.encode(_values(M)), key)
    with pytest.raises(ValueError, match="missing rotation keys"):
        lt.apply(ct, {})
    lower = scheme.rescale(scheme.encrypt(scheme.encode(_values(M)), key))
    with pytest.raises(ValueError, match="level 0"):
        lt.apply(lower, {3: scheme.gen_rotation_key(key, 3)})
