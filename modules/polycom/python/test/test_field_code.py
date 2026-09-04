# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the foldable RS code over an extension field, backed by the
rs_field_* C kernels: the roots the kernels transform with (order, and the
cross-level relation psi_{l-1} = psi_l^2 the fold depends on), agreement of
the encoder with naive evaluation at its bit-reversed negacyclic points,
the decoder (round-trip plus degree check), the fold identity, leaf
digests, and parameter validation -- over a prime below 2^50 and one above
it, which arith's NTT transforms with different kernels."""

import pytest
from vfhe.arith import Field, FieldElement, FieldVector, PseudoMersenneField
from vfhe.polycom import FieldFoldableRS
from vfhe.polycom.code import _bit_reverse

# Primes with 2-adicity 20 and 22; x^2 - w irreducible for the given w.
_FIELDS = {
    "p50": (562949948178433, 5),
    "p61": (1152921504577486849, 11),
}


@pytest.fixture(params=list(_FIELDS))
def field(request):
    prime, w = _FIELDS[request.param]
    return Field(prime, 2, w)


def _code(field, k0: int = 4, c: int = 4, d: int = 2) -> FieldFoldableRS:
    return FieldFoldableRS(field, k0=k0, c=c, d=d)  # n0 = 16, n_d = 64, k_d = 16


def _random(field, n: int, seed: bytes) -> FieldVector:
    vec = FieldVector(field, n)
    vec.sample_random(seed)
    return vec


def _naive(field, message: FieldVector, x: int) -> FieldElement:
    """sum_m message[m] * x^m, for an evaluation point x in F_p."""
    p = field.prime
    total = None
    for m, coeff in enumerate(message):
        term = coeff * FieldElement(field, pow(x, m, p))
        total = term if total is None else total + term
    return total


def test_root_orders_and_level_consistency(field):
    code = _code(field)
    p = field.prime
    for level, psi in enumerate(code.roots):
        n = code.n0 << level
        assert pow(psi, 2 * n, p) == 1
        assert pow(psi, n, p) == p - 1
    for level in range(code.d, 0, -1):
        assert code.roots[level - 1] == pow(code.roots[level], 2, p)


def test_encode_matches_naive_evaluation(field):
    code = _code(field)
    message = _random(field, 8, b"m")  # k0 = 4: level 1
    level = code.level_of(message)
    word = code.encode(message)
    n = code.n0 << level
    assert isinstance(word, FieldVector) and len(word) == n
    psi = code.roots[level]
    bits = n.bit_length() - 1
    for j in range(n):
        x = pow(psi, 2 * _bit_reverse(j, bits) + 1, field.prime)
        assert word[j] == _naive(field, message, x)
    # A list of elements encodes the same as the vector holding them.
    assert code.encode(message.to_list()) == word


def test_encode_pairs_are_plus_minus(field):
    code = _code(field)
    p = field.prime
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    psi = code.roots[code.d]
    half_bits = (len(word) // 2).bit_length() - 1
    for i in range(len(word) // 2):
        x = code.twists[code.d - 1][i]
        assert x == pow(psi, 2 * _bit_reverse(i, half_bits) + 1, p)
        assert code.pair_at(word, i) == (
            _naive(field, message, x),
            _naive(field, message, p - x),
        )


def test_decode_round_trip_and_degree_check(field):
    code = _code(field)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    ok, decoded = code.decode(word)
    assert ok and decoded == message
    tampered = word.copy()
    tampered[0] = tampered[0] + field.one
    assert not code.decode(tampered)[0]


def test_fold_commutes_with_message_fold(field):
    code = _code(field)
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    r = FieldElement(field)
    r.sample_random(b"r")
    folded_word = code.fold(word, r, level=code.d)
    folded_message = FieldVector(
        field, [message[2 * i] + r * message[2 * i + 1] for i in range(code.k_d // 2)]
    )
    assert folded_word == code.encode(folded_message)
    # fold_at / fold_pair (the verifier's per-position form) agree with fold.
    for i in range(len(folded_word)):
        assert code.fold_at(word, r, code.d, i) == folded_word[i]


def test_fold_all_the_way_down_stays_a_codeword(field):
    code = _code(field)
    word = code.encode(_random(field, code.k_d, b"m"))
    for level in range(code.d, 0, -1):
        r = FieldElement(field)
        r.sample_random(bytes([level]))
        word = code.fold(word, r, level=level)
    assert len(word) == code.n0
    assert code.decode(word)[0]


def test_leaf_digests(field):
    code = _code(field)
    word = code.encode(_random(field, code.k_d, b"m"))
    digests = code.leaf_digests(word)
    assert len(digests) == len(word) // 2
    for i, digest in enumerate(digests):
        assert digest == code.leaf_digest(code.pair_at(word, i))
    lo, hi = code.pair_at(word, 0)
    assert code.leaf_digest((hi, lo)) != digests[0]  # an ordered pair


def test_parameter_validation(field):
    with pytest.raises(ValueError, match="power of two"):
        FieldFoldableRS(field, k0=3, c=8, d=1)
    with pytest.raises(ValueError, match="d must be at least 1"):
        FieldFoldableRS(field, k0=4, c=4, d=0)
    with pytest.raises(ValueError, match="2-adicity"):
        # n_d = 2^24 needs order 2^25; both primes have 2-adicity below that.
        FieldFoldableRS(field, k0=1 << 22, c=2, d=1)
    with pytest.raises(ValueError, match="2-adicity"):
        # 2^61 - 1 has 2-adicity 1: no transform of any length.
        FieldFoldableRS(Field((1 << 61) - 1, 2, 3), k0=1, c=2, d=1)
    code = _code(field)
    with pytest.raises(ValueError, match="not k0"):
        code.encode(_random(field, 6, b"m"))
    with pytest.raises(ValueError, match="exceeds the level"):
        code.encode(_random(field, 2 * code.k_d, b"m"))
    with pytest.raises(ValueError, match="different field"):
        code.encode(_random(Field(_FIELDS["p50"][0], 2, 5), code.k_d, b"m"))


def test_pseudo_mersenne_has_no_transform_yet():
    field = PseudoMersenneField.generate(260, two_adicity=8)
    with pytest.raises(NotImplementedError, match="no Reed-Solomon transform"):
        FieldFoldableRS(field, k0=4, c=4, d=2)


def test_short_codeword_round_trips(field):
    # n0 = 8 puts every level's transform on arith's scalar NTT path.
    code = FieldFoldableRS(field, k0=2, c=4, d=2)  # n0 = 8, n_d = 32, k_d = 8
    message = _random(field, code.k_d, b"m")
    word = code.encode(message)
    assert len(word) == code.n_d
    assert code.decode(word) == (True, message)
    for level in range(code.d, 0, -1):
        r = FieldElement(field)
        r.sample_random(bytes([level]))
        word = code.fold(word, r, level=level)
    assert code.decode(word)[0]
