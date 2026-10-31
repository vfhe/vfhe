# SPDX-FileCopyrightText: 2026 The vFHE Authors
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
from vfhe.arith import Ring, RNSRing, repr
from vfhe.polycom import (
    INSTANTIATIONS,
    FoldableRS,
    bit_reverse,
    foldable_relative_distance,
)
from vfhe.polycom import code as code_module
from vfhe.util import parallel


def _ring() -> RNSRing:
    return Ring(1024, prime_size=[49], split_degree=4)


@pytest.fixture(params=INSTANTIATIONS)
def instantiation(request):
    return request.param


def _code(ring: RNSRing, k0: int = 4, c: int = 4, d: int = 2, **kwargs) -> FoldableRS:
    """The code with the per-prime integer twists this file tests, unless
    `twist_field` is given (`test_residue_code.py` covers ``"field"``)."""
    kwargs.setdefault("twist_field", "prime")
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


def test_fold_fibers_agrees_with_the_fold_chain(instantiation):
    """Refolding the fiber above a position gives the value the whole-codeword
    fold chain holds there, for every number of folds from the top; a fiber's
    leaf digest is the one `leaf_digests` gives it, and at two symbols the
    pair's."""
    ring = _ring()
    code = _code(ring, d=3, instantiation=instantiation)
    words = {code.d: code.encode([ring.random_element() for _ in range(code.k_d)])}
    challenges = [ring.random_exceptional() for _ in range(code.d)]
    for level in range(code.d, 0, -1):
        words[level - 1] = code.fold(words[level], challenges[code.d - level], level)

    for steps in range(1, code.d + 1):
        size = 1 << steps
        below = words[code.d - steps]
        indices = list(range(0, len(below), 3))
        fibers = [code.fiber_at(words[code.d], j, size) for j in indices]
        block = code.join_fibers(fibers, size)
        folded = code.fold_fibers(block, challenges[:steps], code.d, indices)
        assert _equal(folded, [below[j] for j in indices]), f"{steps} folds"
        digests = code.leaf_digests(words[code.d], size)
        assert code.leaf_digests(block, size) == b"".join(
            digests[32 * j : 32 * (j + 1)] for j in indices
        )
    assert code.leaf_digests(words[code.d]) == b"".join(
        code_module.pair_digest(pair) for pair in code.pair_leaves(words[code.d])
    )
    with pytest.raises(ValueError, match="tuple of 4"):
        code.join_fibers([tuple(words[code.d][:2])], 4)


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


@pytest.mark.parametrize(("split_degree", "twist_field"), [(4, "field"), (1, "prime")])
def test_the_default_twists_live_in_the_residue_fields(split_degree, twist_field):
    ring = Ring(1024, prime_size=[49], split_degree=split_degree)
    code = FoldableRS(ring, k0=4, c=4, d=2)
    assert code.twist_field == twist_field
    assert bool(code.residue_codes) == (twist_field == "field")
    # The residue codes' base transform has its root in F_p.
    assert all(residue.roots for residue in code.residue_codes)


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


# --- truncation ---

# (instantiation, k0, c, d, twist_field) over N = 64 and split degree 4, so
# k = 16 slots: n0 = 8 has a base transform, n0 = 32 a Vandermonde base, and
# the "rs" code needs n_d <= 16.
_TRUNCATED = [
    ("general", 2, 4, 3, "prime"),
    ("general", 8, 4, 2, "prime"),
    ("rs", 2, 2, 2, "prime"),
    ("general", 2, 4, 3, "field"),
]


@pytest.mark.parametrize(
    ("instantiation", "k0", "c", "d", "twist_field"), _TRUNCATED, ids=str
)
def test_truncated_is_the_code_built_with_that_depth(
    instantiation, k0, c, d, twist_field
):
    ring = Ring(64, prime_size=[28, 50], split_degree=4)
    options = {"instantiation": instantiation, "twist_field": twist_field}
    deep = FoldableRS(ring, k0=k0, c=c, d=d, **options)
    for depth in range(1, d + 1):
        code = deep.truncated(depth)
        built = FoldableRS(ring, k0=k0, c=c, d=depth, **options)
        assert (code.d, code.k_d, code.n_d) == (built.d, built.k_d, built.n_d)
        assert code.twists == built.twists and code.roots == built.roots
        assert code.base_points == built.base_points
        assert code.relative_distance() == built.relative_distance()
        for level in range(depth + 1):
            message = [ring.random_element() for _ in range(k0 << level)]
            word = code.encode(message)
            assert _equal(word, built.encode(message))
            assert code.leaf_digests(word) == built.leaf_digests(word)
            ok, decoded = code.decode(word)
            assert ok and _equal(decoded, message)
            if level:
                r = ring.random_exceptional()
                assert _equal(code.fold(word, r, level), built.fold(word, r, level))
        with pytest.raises(ValueError, match="exceeds"):
            code.encode([ring.random_element() for _ in range(k0 << (depth + 1))])
    for depth in (0, d + 1):
        with pytest.raises(ValueError, match="depth"):
            deep.truncated(depth)


def test_truncated_code_borrows_its_plans(monkeypatch):
    """The plans of a truncated code are its parent's: freed once, by the
    parent, after both are gone, however deep the chain."""
    ring = Ring(512, prime_size=[49], split_degree=2)
    code = FoldableRS(ring, k0=4, c=4, d=3, instantiation="rs")
    rows = ring.rns_rows
    freed: list[int] = []
    module_lib = code_module.lib

    class _RecordingLib:
        def __getattr__(self, name):
            return getattr(module_lib, name)

        def rs_free_plans(self, plans, count):
            freed.append(count)
            return module_lib.rs_free_plans(plans, count)

    monkeypatch.setattr(code_module, "lib", _RecordingLib())
    shorter = code.truncated(2)
    shortest = shorter.truncated(1)
    del code, shorter
    assert freed == []
    message = [ring.random_element() for _ in range(8)]
    built = FoldableRS(ring, k0=4, c=4, d=1, instantiation="rs")
    assert _equal(shortest.encode(message), built.encode(message))
    del built
    freed.clear()
    del shortest
    assert freed == [rows] * 4  # the four levels of the depth-3 code


# --- thread limits ---

# A narrow and a wide prime, on a plain and on a split ring.
_TWO_PRIME_RINGS = [([30, 49], 1), ([49, 30], 4)]


def _at_thread_limits(compute) -> list:
    """`compute()` at the default limit of one thread, then at 5 and 8."""
    results = []
    try:
        for limit in (1, 5, 8):
            parallel.set_num_threads(limit)
            results.append(compute())
    finally:
        parallel.set_num_threads()
    return results


def _with_one_slot_changed(p, row: int, slot: int):
    """A copy of `p` whose NTT form differs from `p`'s at one slot of one
    prime only."""
    out = p.copy()
    rows = out.get_coeff_matrix(repr.ntt)
    rows[row][slot] = (rows[row][slot] + 1) % out.ring.primes[row]
    return out.from_coeff_matrix(rows, repr.ntt)


@pytest.mark.parametrize(("prime_size", "split_degree"), _TWO_PRIME_RINGS)
def test_rs_kernels_agree_at_any_thread_limit(prime_size, split_degree):
    """The transforms run their (prime, slot) columns in chunks once the
    limit is raised: 2048 columns of 256 words are above the chunk grain, and
    at five threads the chunks neither divide the columns nor stop at a
    prime's last slot. The codewords, the decoded messages and the degree
    check are the same at every limit, and a failure confined to one column
    -- in the first chunk or in the last -- is still found."""
    ring = Ring(1024, prime_size=prime_size, split_degree=split_degree)
    code = _rs(ring, k0=8, c=4, d=3)  # n_d = 256
    message = [ring.random_element() for _ in range(code.k_d)]
    words = _at_thread_limits(lambda: code.encode(message))
    for word in words[1:]:
        assert _equal(word, words[0])
    word = words[0]
    for ok, decoded in _at_thread_limits(lambda: code.decode(word)):
        assert ok and _equal(decoded, message)
    for row, slot in ((0, 0), (-1, -1)):
        tampered = list(word)
        tampered[5] = _with_one_slot_changed(word[5], row, slot)
        decoded = _at_thread_limits(lambda t=tampered: code.decode(t))
        assert not any(ok for ok, _ in decoded)


@pytest.mark.parametrize(("prime_size", "split_degree"), _TWO_PRIME_RINGS)
def test_leaf_digests_are_the_digests_of_the_entries(prime_size, split_degree):
    """A leaf's digest is BLAKE3 over its entries' `get_hash` digests (the
    pair digest at two symbols), whatever form the entries are in, and the
    entries keep their form. Fibers whose digest input is past one BLAKE3
    chunk (64 symbols) and words large enough to be digested in chunks once
    the limit is raised give the same bytes at every limit. A word that is
    not a whole number of leaves is refused."""
    ring = Ring(1024, prime_size=prime_size, split_degree=split_degree)
    code = _code(ring)
    word = [ring.random_element() for _ in range(192)]
    for p in word[::3]:
        p.to_coeff()
    forms = [p.repr for p in word]
    assert len(set(forms)) == 2
    for size in (1, 2, 4, 64):
        expected = b"".join(
            code_module._fiber_digest(word[i : i + size])
            for i in range(0, len(word), size)
        )
        for digests in _at_thread_limits(lambda s=size: code.leaf_digests(word, s)):
            assert bytes(digests) == expected, f"size {size}"
    assert [p.repr for p in word] == forms
    pairs = code.pair_leaves(word)
    assert bytes(code.leaf_digests_of(pairs[:7])) == b"".join(
        code_module.pair_digest(pair) for pair in pairs[:7]
    )
    assert bytes(code.leaf_digests(word[:1], 1)) == code_module._fiber_digest(word[:1])
    assert bytes(code.leaf_digests([])) == b""
    with pytest.raises(ValueError, match="whole leaves"):
        code.leaf_digests(word[:191], 2)
    with pytest.raises(ValueError, match="whole leaves"):
        code.leaf_digests(word, 0)
