# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the foldable RS code over R_q, backed by the rs_* C kernels:
the roots the kernels transform with (order, and the cross-level relation
psi_{l-1} = psi_l^2 the fold depends on), agreement of the encoder with
naive per-prime evaluation at its bit-reversed negacyclic points, the
decoder (round-trip plus degree check), and the fold identity — folding a
codeword equals encoding the folded message."""

import pytest
from vfhe.arith import Ring
from vfhe.polycom import FoldableRS
from vfhe.polycom.code import _bit_reverse


def _ring() -> Ring:
    return Ring(1024, prime_size=[49], split_degree=4)


def _code(ring: Ring, k0: int = 4, c: int = 4, d: int = 2) -> FoldableRS:
    return FoldableRS(ring, k0=k0, c=c, d=d)  # n0 = 16, n_d = 64, k_d = 16


def test_root_orders_and_level_consistency():
    ring = _ring()
    code = _code(ring)
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


def test_encode_matches_naive_evaluation():
    ring = _ring()
    code = _code(ring)
    p = ring.primes[0]
    message = [ring.random_element() for _ in range(8)]  # level 0? -> k0=4: level 1
    level = code.level_of(message)
    word = code.encode(message)
    n = code.n0 << level
    assert len(word) == n
    psi = code.roots[level][0]
    bits = n.bit_length() - 1
    for j in range(n):
        # Position j evaluates at psi^(2*brv(j)+1) — bit-reversed output.
        x = pow(psi, 2 * _bit_reverse(j, bits) + 1, p)
        naive = None
        for m, coeff in enumerate(message):
            term = coeff * [pow(x, m, p)]
            naive = term if naive is None else naive + term
        assert word[j] == naive


def test_encode_pairs_are_plus_minus():
    # The bit-reversal puts the +/- pairs adjacent: word[2i] = P(x_i) and
    # word[2i+1] = P(-x_i), which is what fold_at reads.
    ring = _ring()
    code = _code(ring)
    p = ring.primes[0]
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    psi = code.roots[code.d][0]
    for i in range(len(word) // 2):
        x = code.twists[code.d - 1][i][0]
        assert x == pow(
            psi, 2 * _bit_reverse(i, (len(word) // 2).bit_length() - 1) + 1, p
        )
        plus = minus = None
        for m, coeff in enumerate(message):
            t_plus = coeff * [pow(x, m, p)]
            t_minus = coeff * [pow(p - x, m, p)]
            plus = t_plus if plus is None else plus + t_plus
            minus = t_minus if minus is None else minus + t_minus
        assert word[2 * i] == plus and word[2 * i + 1] == minus


def test_decode_round_trip_and_degree_check():
    ring = _ring()
    code = _code(ring)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    ok, decoded = code.decode(word)
    assert ok
    assert all(a == b for a, b in zip(decoded, message, strict=True))
    # A perturbed word is (with overwhelming probability) outside the code:
    # its inverse transform has nonzero coefficients above the dimension.
    tampered = list(word)
    tampered[0] = tampered[0] + ring.random_element()
    assert not code.decode(tampered)[0]


def test_fold_commutes_with_message_fold():
    ring = _ring()
    code = _code(ring)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    r = ring.random_exceptional()
    folded_word = code.fold(word, r, level=code.d)
    folded_message = [
        message[2 * i] + r * message[2 * i + 1] for i in range(code.k_d // 2)
    ]
    expected = code.encode(folded_message)
    assert all(a == b for a, b in zip(folded_word, expected, strict=True))


def test_fold_all_the_way_down_stays_a_codeword():
    # Folding d times lands on the base code, and the result decodes.
    ring = _ring()
    code = _code(ring)
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    for level in range(code.d, 0, -1):
        word = code.fold(word, ring.random_exceptional(), level=level)
    assert len(word) == code.n0
    assert code.decode(word)[0]


def test_parameter_validation():
    ring = _ring()
    with pytest.raises(ValueError, match="power of two"):
        FoldableRS(ring, k0=3, c=8, d=1)
    with pytest.raises(ValueError, match="d must be at least 1"):
        FoldableRS(ring, k0=4, c=4, d=0)
    with pytest.raises(ValueError, match="N/split_degree"):
        # n_d = 2048 > 256: no root of unity of order 2 * n_d in the primes.
        FoldableRS(ring, k0=256, c=2, d=2)
    with pytest.raises(ValueError, match="below 16"):
        # arith's NTT kernels do not implement transforms shorter than that.
        FoldableRS(ring, k0=2, c=4, d=2)
    code = _code(ring)
    with pytest.raises(ValueError, match="not k0"):
        code.encode([ring.random_element() for _ in range(6)])
    with pytest.raises(ValueError, match="exceeds the level"):
        code.encode([ring.random_element() for _ in range(2 * code.k_d)])
