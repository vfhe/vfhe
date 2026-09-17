# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the foldable codes over R_q.

Both instantiations share the decoder (round trip plus the degree check)
and the fold identity -- folding a codeword equals encoding the folded
message. The general code adds what makes it [ZCF24]'s: distinct, seeded
tables, a Vandermonde base past the primes' root orders, and the paper's
distance bound. The Reed-Solomon option keeps the tests of its structure,
backed by the rs_* C kernels: the roots the kernels transform with (order,
and the tower psi_{l-1} = psi_l^2 the fold depends on), agreement with naive
per-prime evaluation at the bit-reversed negacyclic points, and the
`(P(x), P(-x))` pairs."""

import math

import pytest
from vfhe.arith import Ring, RNSRing
from vfhe.polycom import (
    INSTANTIATIONS,
    FoldableRS,
    bit_reverse,
    foldable_relative_distance,
)
from vfhe.polycom import code as code_module


def _ring() -> RNSRing:
    return Ring(1024, prime_size=[49], split_degree=4)


@pytest.fixture(params=INSTANTIATIONS)
def instantiation(request):
    return request.param


def _code(ring: RNSRing, k0: int = 4, c: int = 4, d: int = 2, **kwargs) -> FoldableRS:
    return FoldableRS(ring, k0=k0, c=c, d=d, **kwargs)  # n0 = 16, n_d = 64, k_d = 16


def _rs(ring: RNSRing, **kwargs) -> FoldableRS:
    return _code(ring, instantiation="rs", **kwargs)


def _naive(message: list, x: int, p: int):
    """sum_m message[m] * x^m at the first prime."""
    total = None
    for m, coeff in enumerate(message):
        term = coeff * [pow(x, m, p)]
        total = term if total is None else total + term
    return total


def _equal(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=True))


# --- both instantiations ---


def test_decode_round_trip_and_degree_check(instantiation):
    ring = _ring()
    code = _code(ring, instantiation=instantiation)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    assert len(word) == code.n_d
    ok, decoded = code.decode(word)
    assert ok and _equal(decoded, message)
    # A perturbed word is (with overwhelming probability) outside the code.
    tampered = list(word)
    tampered[0] = tampered[0] + ring.random_element()
    assert not code.decode(tampered)[0]


def test_fold_commutes_with_message_fold(instantiation):
    ring = _ring()
    code = _code(ring, instantiation=instantiation)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    r = ring.random_exceptional()
    folded_word = code.fold(word, r, level=code.d)
    folded_message = [
        message[2 * i] + r * message[2 * i + 1] for i in range(code.k_d // 2)
    ]
    assert _equal(folded_word, code.encode(folded_message))
    for i in range(len(folded_word)):
        assert code.fold_at(word, r, code.d, i) == folded_word[i]


def test_fold_all_the_way_down_stays_a_codeword(instantiation):
    # Folding d times lands on the base code, and the result decodes.
    ring = _ring()
    code = _code(ring, instantiation=instantiation)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    for level in range(code.d, 0, -1):
        r = ring.random_exceptional()
        word = code.fold(word, r, level=level)
        message = [
            message[2 * i] + r * message[2 * i + 1] for i in range(len(message) // 2)
        ]
    assert len(word) == code.n0
    ok, decoded = code.decode(word)
    assert ok and _equal(decoded, message)


def test_parameter_validation(instantiation):
    ring = _ring()
    with pytest.raises(ValueError, match="power of two"):
        FoldableRS(ring, k0=3, c=8, d=1, instantiation=instantiation)
    with pytest.raises(ValueError, match="d must be at least 1"):
        FoldableRS(ring, k0=4, c=4, d=0, instantiation=instantiation)
    code = _code(ring, instantiation=instantiation)
    with pytest.raises(ValueError, match="not k0"):
        code.encode([ring.random_element() for _ in range(6)])
    with pytest.raises(ValueError, match="exceeds the level"):
        code.encode([ring.random_element() for _ in range(2 * code.k_d)])


def test_short_codeword_round_trips(instantiation):
    """A base codeword below one AVX512 lane group still encodes and folds:
    n0 = 8 puts the base transform on arith's scalar NTT path."""
    ring = _ring()
    code = FoldableRS(ring, k0=2, c=4, d=2, instantiation=instantiation)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    assert len(word) == code.n_d == 32
    ok, decoded = code.decode(word)
    assert ok and _equal(decoded, message)
    for level in range(code.d, 0, -1):
        word = code.fold(word, ring.random_exceptional(), level=level)
    assert len(word) == code.n0
    assert code.decode(word)[0]


# --- the general code ---


def test_the_default_is_the_general_code():
    ring = _ring()
    code = _code(ring)
    assert code.instantiation == "general"
    assert len(code.roots) == 1  # the base transform's, and no other
    with pytest.raises(ValueError, match="instantiation"):
        _code(ring, instantiation="fri")


def test_twist_tables_are_nonzero_negated_and_seeded():
    ring = _ring()
    code = _code(ring)
    for level in range(code.d):
        assert len(code.twists[level]) == code.n0 << level
        for row, row_odd in zip(
            code.twists[level], code.twists_odd[level], strict=True
        ):
            assert len(row) == len(ring.primes)
            for t, u, p in zip(row, row_odd, ring.primes, strict=True):
                assert t != 0 and u == (p - t) % p  # T' = -T, as in the paper
    message = [ring.random_element() for _ in range(code.k_d)]
    same = _code(ring, seed=code.seed)
    assert same.twists == code.twists and same.twists_odd == code.twists_odd
    assert _equal(same.encode(message), code.encode(message))
    other = _code(ring, seed=b"another public parameter")
    assert other.twists != code.twists
    assert not _equal(other.encode(message), code.encode(message))


def test_a_codeword_longer_than_the_primes_root_orders_is_served():
    """n_d = 2048 > N/split_degree = 256, which the option cannot encode; the
    general code needs nothing above its base, and here not even that --
    n0 = 512 has no transform either, so the base is a Vandermonde product."""
    ring = _ring()
    with pytest.raises(ValueError, match="N/split_degree"):
        _rs(ring, k0=256, c=2, d=2)
    code = _code(ring, k0=256, c=2, d=2)
    assert code.roots == []
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    ok, decoded = code.decode(word)
    assert ok and _equal(decoded, message)
    r = ring.random_exceptional()
    folded = [message[2 * i] + r * message[2 * i + 1] for i in range(code.k_d // 2)]
    assert _equal(code.fold(word, r, level=code.d), code.encode(folded))


def test_relative_distance_per_instantiation():
    ring = _ring()
    rs, general = _rs(ring, d=4), _code(ring, d=4)
    assert rs.relative_distance() == pytest.approx(1 - 1 / rs.c + 1 / rs.n_d)
    expected = foldable_relative_distance(4, 4, 4, math.log2(min(ring.primes)), 128)
    assert general.relative_distance() == pytest.approx(expected)
    assert 0 < general.relative_distance() < rs.relative_distance()
    assert general.relative_distance(64) > general.relative_distance(128)
    assert general.relative_distance(bound="zcf24") <= general.relative_distance()


# --- the Reed-Solomon option ---


def test_rs_root_orders_and_level_consistency():
    ring = _ring()
    code = _rs(ring)
    for level, roots in enumerate(code.roots):
        n = code.n0 << level
        for psi, p in zip(roots, ring.primes, strict=True):
            # The negacyclic transform's root has order exactly 2n.
            assert pow(psi, 2 * n, p) == 1
            assert pow(psi, n, p) == p - 1
    # Successive levels share a root tower (psi_{l-1} = psi_l^2), which is
    # what makes the squared fold points the next level's evaluation points.
    for level in range(code.d, 0, -1):
        for below, above, p in zip(
            code.roots[level - 1], code.roots[level], ring.primes, strict=True
        ):
            assert below == pow(above, 2, p)


def test_rs_encode_matches_naive_evaluation():
    ring = _ring()
    code = _rs(ring)
    p = ring.primes[0]
    message = [ring.random_element() for _ in range(8)]  # k0 = 4: level 1
    level = code.level_of(message)
    word = code.encode(message)
    n = code.n0 << level
    assert len(word) == n
    psi = code.roots[level][0]
    bits = n.bit_length() - 1
    for j in range(n):
        # Position j evaluates at psi^(2*brv(j)+1) — bit-reversed output.
        x = pow(psi, 2 * bit_reverse(j, bits) + 1, p)
        assert word[j] == _naive(message, x, p)


def test_rs_pairs_are_plus_minus():
    # The bit-reversal puts the +/- pairs adjacent: word[2i] = P(x_i) and
    # word[2i+1] = P(-x_i), which is what fold_at reads.
    ring = _ring()
    code = _rs(ring)
    p = ring.primes[0]
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    psi = code.roots[code.d][0]
    half_bits = (len(word) // 2).bit_length() - 1
    for i in range(len(word) // 2):
        x = pow(psi, 2 * bit_reverse(i, half_bits) + 1, p)
        assert code.twists[code.d - 1][i][0] == x
        assert code.twists_odd[code.d - 1][i][0] == p - x
        assert word[2 * i] == _naive(message, x, p)
        assert word[2 * i + 1] == _naive(message, p - x, p)


def test_plans_freed_with_their_allocation_length(monkeypatch):
    """A code may outlive an extension of the RNS base it was built against.

    The RNS base of an (N, split_degree) pair is shared process-wide, so a ring
    introducing a new prime extends it in place and every existing ring's
    `_base_l()` grows. The plan arrays are sized by the ring's own mask, which
    does not, and the free path must use that — following the shared count
    walks off the end of the array and corrupts the heap.
    """
    # Own the (N, split_degree) key, so no other test's rings have already
    # extended this base: the growth below has to be ours to observe.
    ring = Ring(512, prime_size=[49], split_degree=2)
    code = FoldableRS(ring, k0=4, c=4, d=2, instantiation="rs")
    allocated_l = ring.rns_rows
    assert allocated_l == ring._base_l()

    # A second ring over the same key, with a prime the first one lacks.
    Ring(512, prime_size=[49, 50], split_degree=2)
    assert ring._base_l() > allocated_l  # the shared count grew under `code`
    assert ring.rns_rows == allocated_l  # but the ring's own row count did not

    # The free path must pass the recorded length, not the ring's current one.
    freed: list[int] = []
    module_lib = code_module.lib

    class _RecordingLib:
        def __getattr__(self, name):
            return getattr(module_lib, name)

        def rs_free_plans(self, plans, count):
            freed.append(count)
            return module_lib.rs_free_plans(plans, count)

    monkeypatch.setattr(code_module, "lib", _RecordingLib())
    levels = code.d + 1  # one plan array per level
    del code
    assert freed == [allocated_l] * levels
