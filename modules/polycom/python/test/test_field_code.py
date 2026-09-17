# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the foldable codes over a field.

Both instantiations share the decoder (round trip plus the degree check),
the fold identity -- folding a codeword equals encoding the folded message,
whole-codeword and per pair -- leaf digests, and parameter validation. The
general code adds what makes it [ZCF24]'s: the encoder is the recursion in
its tables, the tables are distinct and a pure function of the seed, a field
with no 2-adicity is served, the two base encoders agree, and the reported
distance is the paper's recurrence. The Reed-Solomon option keeps the tests
of its structure: the roots' orders and the tower psi_{l-1} = psi_l^2,
agreement with naive evaluation at the bit-reversed negacyclic points, the
`(P(x), P(-x))` pairs.

Every case runs over four fields, one per transform kernel a codeword can go
through: extension fields over a prime below 2^50 and one above it, which
arith's NTT transforms with different kernels, and pseudo-Mersenne fields of
five and six limbs, transformed by arith's `PseudoMersenneNTT`."""

import math

import pytest
from vfhe.arith import Field, FieldVector, PseudoMersenneField
from vfhe.polycom import (
    INSTANTIATIONS,
    FieldFoldableRS,
    bit_reverse,
    foldable_relative_distance,
)

# Extension fields over primes with 2-adicity 20 and 22 (x^2 - w irreducible
# for the given w), and pseudo-Mersenne fields of five and six limbs.
_EXTENSION = {"p50": (562949948178433, 5), "p61": (1152921504577486849, 11)}
_PSEUDO_MERSENNE = {"pmf260": 260, "pmf312": 312}
# 2^61 - 1 has 2-adicity 1: no transform of any length.
_MERSENNE = ((1 << 61) - 1, 3)


def _field(name: str) -> Field:
    if name in _EXTENSION:
        prime, w = _EXTENSION[name]
        return Field(prime, 2, w)
    return PseudoMersenneField.generate(_PSEUDO_MERSENNE[name], two_adicity=10)


@pytest.fixture(params=[*_EXTENSION, *_PSEUDO_MERSENNE])
def field(request):
    return _field(request.param)


@pytest.fixture(params=INSTANTIATIONS)
def instantiation(request):
    return request.param


def _code(field, k0: int = 4, c: int = 4, d: int = 2, **kwargs) -> FieldFoldableRS:
    return FieldFoldableRS(field, k0=k0, c=c, d=d, **kwargs)  # n0 = 16, n_d = 64


def _rs(field, **kwargs) -> FieldFoldableRS:
    return _code(field, instantiation="rs", **kwargs)


def _element(field, value: int):
    """`value` as an element of `field`, whichever implementation it is."""
    return type(field.one)(field, value)


def _random(field, n: int, seed: bytes) -> FieldVector:
    vec = FieldVector(field, n)
    vec.sample_random(seed)
    return vec


def _naive(field, message: FieldVector, x: int):
    """sum_m message[m] * x^m, for an evaluation point x in F_p."""
    p = field.prime
    total = None
    for m, coeff in enumerate(message):
        term = coeff * _element(field, pow(x, m, p))
        total = term if total is None else total + term
    return total


def _folded(field, message: FieldVector, r) -> FieldVector:
    """The message fold the codeword fold must commute with."""
    return FieldVector(
        field,
        [message[2 * i] + r * message[2 * i + 1] for i in range(len(message) // 2)],
    )


# --- both instantiations ---


def test_decode_round_trip_and_degree_check(field, instantiation):
    code = _code(field, instantiation=instantiation)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    assert isinstance(word, FieldVector) and len(word) == code.n_d
    ok, decoded = code.decode(word)
    assert ok and decoded == message
    tampered = word.copy()
    tampered[0] = tampered[0] + field.one
    assert not code.decode(tampered)[0]
    # A list of elements encodes the same as the vector holding them, and
    # encoding does not consume its operand: the same message encodes twice.
    assert code.encode(message.to_list()) == word
    assert code.encode(message) == word


def test_rate_one_code_round_trips(field, instantiation):
    # c = 1: the message already fills the codeword, so nothing is padded and
    # nothing is cut off by the degree check.
    code = _code(field, k0=8, c=1, d=1, instantiation=instantiation)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    assert len(word) == code.n_d == code.k_d
    assert code.decode(word) == (True, message)


def test_fold_commutes_with_message_fold(field, instantiation):
    code = _code(field, instantiation=instantiation)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    r = field.random_element(b"r")
    folded_word = code.fold(word, r, level=code.d)
    assert folded_word == code.encode(_folded(field, message, r))
    # fold_at / fold_pair (the verifier's per-position form) agree with fold.
    for i in range(len(folded_word)):
        assert code.fold_at(word, r, code.d, i) == folded_word[i]


def test_fold_all_the_way_down_stays_a_codeword(field, instantiation):
    code = _code(field, instantiation=instantiation)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    for level in range(code.d, 0, -1):
        r = field.random_element(bytes([level]))
        word = code.fold(word, r, level=level)
        message = _folded(field, message, r)
        assert word == code.encode(message)
    assert len(word) == code.n0
    assert code.decode(word) == (True, message)


def test_leaf_digests(field, instantiation):
    code = _code(field, instantiation=instantiation)
    word = code.encode(_random(field, code.k_d, b"m"))
    digests = code.leaf_digests(word)
    assert len(digests) == 32 * (len(word) // 2)
    for i in range(len(word) // 2):
        assert digests[32 * i : 32 * (i + 1)] == code.leaf_digest(code.pair_at(word, i))
    lo, hi = code.pair_at(word, 0)
    assert code.leaf_digest((hi, lo)) != digests[:32]  # an ordered pair


def test_parameter_validation(field, instantiation):
    with pytest.raises(ValueError, match="power of two"):
        FieldFoldableRS(field, k0=3, c=8, d=1, instantiation=instantiation)
    with pytest.raises(ValueError, match="d must be at least 1"):
        FieldFoldableRS(field, k0=4, c=4, d=0, instantiation=instantiation)
    code = _code(field, instantiation=instantiation)
    with pytest.raises(ValueError, match="not k0"):
        code.encode(_random(field, 6, b"m"))
    with pytest.raises(ValueError, match="exceeds the level"):
        code.encode(_random(field, 2 * code.k_d, b"m"))
    with pytest.raises(ValueError, match="different field"):
        code.encode(_random(_field("p50"), code.k_d, b"m"))


def test_short_codeword_round_trips(field, instantiation):
    # n0 = 8 puts the base transform on arith's scalar NTT path, and every
    # level below the vector kernels' padding unit.
    code = FieldFoldableRS(field, k0=2, c=4, d=2, instantiation=instantiation)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    assert len(word) == code.n_d == 32
    assert code.decode(word) == (True, message)
    for level in range(code.d, 0, -1):
        word = code.fold(word, field.random_element(bytes([level])), level=level)
    assert code.decode(word)[0]


def test_fold_agrees_with_the_per_pair_form(field, instantiation):
    """`fold` gives, at every position, what `fold_pair` and `fold_pairs`
    give for that position's pair alone.

    The whole-codeword fold and the one a verifier runs on a single
    authenticated pair are different code paths -- one a kernel over the
    vector, the other element arithmetic -- and a prover and a verifier
    disagreeing about a fold is the failure this catches.
    """
    code = FieldFoldableRS(field, k0=4, c=2, d=6, instantiation=instantiation)
    message = FieldVector(field, code.k_d)
    message.sample_random(b"fold-agreement")
    word = code.encode(message)
    r = field.random_element(b"r")

    for level in range(code.d, 0, -1):
        folded = code.fold(word, r, level=level)
        for i in range(len(folded)):
            lo, hi = code.pair_at(word, i)
            assert folded[i] == code.fold_pair(lo, hi, r, level, i), (
                f"level {level}, {i}"
            )
        positions = list(range(0, len(folded), 3))
        pairs = [code.pair_at(word, i) for i in positions]
        assert code.fold_pairs(
            [lo for lo, _ in pairs], [hi for _, hi in pairs], r, level, positions
        ) == [folded[i] for i in positions]
        word = folded


def test_leaf_digests_go_straight_into_a_tree(field, instantiation):
    """The digests are packed, which is exactly what `Merkle.from_digests`
    reads -- the codeword reaches a root with no object per leaf."""
    from vfhe.crypto import Merkle

    code = FieldFoldableRS(field, k0=4, c=2, d=2, instantiation=instantiation)
    word = code.encode(FieldVector(field, [field(i + 1) for i in range(code.k_d)]))
    digests = code.leaf_digests(word)
    tree = Merkle.from_digests(digests)
    leaves = [digests[i : i + 32] for i in range(0, len(digests), 32)]
    assert len(tree) == len(word) // 2
    assert tree.root == Merkle(leaves, hash=lambda leaf: leaf).root


# --- the general code ---


def test_the_default_is_the_general_code(field):
    code = _code(field)
    assert code.instantiation == "general"
    assert len(code.roots) == 1  # the base transform's, and no other
    with pytest.raises(ValueError, match="instantiation"):
        _code(field, instantiation="fri")


def test_encode_is_the_recursion_in_the_tables(field):
    """`encode_l(m)[2j] = encode_{l-1}(m_even)[j] + encode_{l-1}(m_odd)[j] T_l[j]`
    and the same with `T'_l` at `2j + 1`, written out from the public tables
    against the whole-vector encoder."""
    code = _code(field)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    even = code.encode(message.query(range(0, code.k_d, 2)))
    odd = code.encode(message.query(range(1, code.k_d, 2)))
    twists, twists_odd = code.twists[code.d - 1], code.twists_odd[code.d - 1]
    for j in range(len(even)):
        assert word[2 * j] == even[j] + odd[j] * twists[j]
        assert word[2 * j + 1] == even[j] + odd[j] * twists_odd[j]


def test_twist_tables_are_distinct_and_seeded(field):
    code = _code(field)
    for level in range(code.d):
        assert len(code.twists[level]) == code.n0 << level
        for t, u in zip(code.twists[level], code.twists_odd[level], strict=True):
            assert t != u
    message = _random(field, code.k_d, b"m")
    same = _code(field, seed=code.seed)
    assert same.twists == code.twists and same.twists_odd == code.twists_odd
    assert same.encode(message) == code.encode(message)
    other = _code(field, seed=b"another public parameter")
    assert other.twists != code.twists
    assert other.encode(message) != code.encode(message)
    # The option has no seeded tables, so the seed changes nothing there.
    assert _rs(field, seed=b"x").encode(message) == _rs(field, seed=b"y").encode(
        message
    )


def test_a_field_without_two_adicity_is_served():
    """Field-agnosticism: with no root of unity of any order the general
    code still encodes, folds and decodes, on a Vandermonde base."""
    field = Field(*_MERSENNE[:1], 2, _MERSENNE[1])
    code = _code(field)
    assert code.roots == []
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    assert code.decode(word) == (True, message)
    r = field.random_element(b"r")
    assert code.fold(word, r, level=code.d) == code.encode(_folded(field, message, r))
    with pytest.raises(ValueError, match="2-adicity"):
        _rs(field)


def test_the_two_base_encoders_agree(field):
    """The transform and the Vandermonde product are encoders of one code:
    a base message encodes to the same codeword through either."""
    code = _code(field)
    message = _random(field, code.k0, b"m")
    assert code.encode(message) == code._base_encode_all(message, 0)


def test_relative_distance_recurrence_matches_the_paper():
    """[ZCF24, Table 1], every row, at the security parameter each row was
    computed with (the first at 100 bits, the others at 128)."""
    rows = [
        (2**5, 2**20, 16, 31, 100, 0.5044),
        (1, 2**20, 16, 61, 128, 0.484),
        (1, 2**25, 8, 128, 128, 0.557),
        (1, 2**25, 8, 256, 128, 0.728),
    ]
    for k0, k_d, c, field_bits, security_bits, expected in rows:
        d = (k_d // k0).bit_length() - 1
        assert foldable_relative_distance(k0, c, d, field_bits, security_bits) == (
            pytest.approx(expected, abs=1e-3)
        )
    assert foldable_relative_distance(1, 1, 1, 61) == 0.0  # rate one: nothing
    with pytest.raises(ValueError, match="field"):
        foldable_relative_distance(4, 4, 2, 1)


def test_relative_distance_per_instantiation(field):
    """The option reports the exact Reed-Solomon distance of its longest
    level; the general code the recurrence at this field's size, which is
    smaller, and smaller still at a higher security parameter."""
    rs, general = _rs(field, d=5), _code(field, d=5)
    assert rs.relative_distance() == pytest.approx(1 - 1 / rs.c + 1 / rs.n_d)
    assert rs.relative_distance(64) == rs.relative_distance()
    expected = foldable_relative_distance(4, 4, 5, math.log2(field.order), 128)
    assert general.relative_distance() == pytest.approx(expected)
    assert 0 < general.relative_distance() < rs.relative_distance()
    assert general.relative_distance(64) > general.relative_distance(128)


# --- the Reed-Solomon option ---


def test_rs_root_orders_and_level_consistency(field):
    code = _rs(field)
    p = field.prime
    for level, psi in enumerate(code.roots):
        n = code.n0 << level
        assert pow(psi, 2 * n, p) == 1
        assert pow(psi, n, p) == p - 1
    for level in range(code.d, 0, -1):
        assert code.roots[level - 1] == pow(code.roots[level], 2, p)


def test_rs_encode_matches_naive_evaluation(field):
    code = _rs(field)
    message = _random(field, 8, b"m")  # k0 = 4: level 1
    level = code.level_of(message)
    word = code.encode(message)
    n = code.n0 << level
    psi = code.roots[level]
    bits = n.bit_length() - 1
    for j in range(n):
        x = pow(psi, 2 * bit_reverse(j, bits) + 1, field.prime)
        assert word[j] == _naive(field, message, x)


def test_rs_pairs_are_plus_minus(field):
    code = _rs(field)
    p = field.prime
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    psi = code.roots[code.d]
    half_bits = (len(word) // 2).bit_length() - 1
    for i in range(len(word) // 2):
        x = pow(psi, 2 * bit_reverse(i, half_bits) + 1, p)
        assert code.twists[code.d - 1][i] == _element(field, x)
        assert code.twists_odd[code.d - 1][i] == _element(field, p - x)
        assert code.pair_at(word, i) == (
            _naive(field, message, x),
            _naive(field, message, p - x),
        )


def test_rs_points_are_distinct_at_every_level(field):
    """The exact distance the option reports holds only because every level
    is an RS code on distinct points: if a level's evaluation points ever
    collided, the code would not be MDS there and the number would be an
    overstatement rather than a bound."""
    code = _rs(field, d=5)
    p = field.prime
    for level in range(code.d + 1):
        n, k = code.n0 << level, code.k0 << level
        points = {pow(code.roots[level], 2 * i + 1, p) for i in range(n)}
        assert len(points) == n, f"level {level} repeats an evaluation point"
        assert code.relative_distance() <= 1 - k / n + 1 / n


def test_rs_needs_the_two_adicity(field):
    with pytest.raises(ValueError, match="2-adicity"):
        # n_d = 2^24 needs order 2^25; every test field is far below that.
        FieldFoldableRS(field, k0=1 << 22, c=2, d=1, instantiation="rs")


def test_rs_over_a_field_arith_cannot_transform_raises():
    class _Unsupported:  # an implementation with no NTT behind it
        two_adicity = 32

    with pytest.raises(NotImplementedError, match="no Reed-Solomon transform"):
        FieldFoldableRS(_Unsupported(), k0=4, c=4, d=2, instantiation="rs")  # pyright: ignore[reportArgumentType]
