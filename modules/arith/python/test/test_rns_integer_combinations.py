# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`RNSRing.integer_linear_combinations`: an integer matrix times a vector of
ring elements, against the products computed coefficient by coefficient."""

import pytest
from vfhe.arith import IntegerMatrix, Polynomial, Ring, repr

# (N, split_degree, prime sizes): narrow and wide rows, both IFMA-family and
# wider primes, d = 1, 2, 4, and rows shorter than one column tile.
SHAPES = [(16, 4, [28, 30]), (64, 1, [50, 28]), (256, 2, [45, 60]), (1024, 4, [50, 50])]

#: Entries: negative, zero, the int64 extremes, and other values.
ROWS = [
    [3, -5, 0, 1],
    [-(2**63), 2**63 - 1, -1, 12345678901],
    [0, 0, 0, 0],
    [7, 7, -7, 2**40],
    [1, 0, 0, 0],
]


@pytest.fixture(params=SHAPES, ids=lambda s: f"N{s[0]}-d{s[1]}-{s[2]}")
def ring(request):
    N, d, sizes = request.param
    return Ring(N, prime_size=sizes, split_degree=d)


def _expected(ring, elements, rows, row):
    words = [e.get_coeff_matrix(e.repr) for e in elements]
    return [
        [
            sum(w * x[k][j] for w, x in zip(rows[row], words, strict=True)) % p
            for j in range(ring.N)
        ]
        for k, p in enumerate(ring.primes)
    ]


@pytest.mark.parametrize("domain", [repr.coeff, repr.ntt])
def test_matches_the_products(ring, domain):
    elements = [ring.random_element() for _ in range(len(ROWS[0]))]
    for e in elements:
        e.to_repr(domain)
    outs = ring.integer_linear_combinations(elements, IntegerMatrix(ROWS))
    assert len(outs) == len(ROWS)
    for r, out in enumerate(outs):
        assert out.repr == domain
        assert out.get_coeff_matrix(domain) == _expected(ring, elements, ROWS, r)


def test_mixed_domains_land_in_the_ntt_one(ring):
    elements = [ring.random_element() for _ in range(len(ROWS[0]))]
    elements[1].to_coeff()
    before = [e.repr for e in elements]
    outs = ring.integer_linear_combinations(elements, IntegerMatrix(ROWS))
    assert [e.repr for e in elements] == before
    ntt = [e.copy() for e in elements]
    for e in ntt:
        e.to_NTT()
    for r, out in enumerate(outs):
        assert out.repr == repr.ntt
        assert out.get_coeff_matrix(repr.ntt) == _expected(ring, ntt, ROWS, r)


def test_many_terms_and_a_buffer():
    # Past one block of terms at this ring's tiles, with the matrix in an int64 array.
    np = pytest.importorskip("numpy")
    ring = Ring(64, prime_size=[50, 50], split_degree=1)
    rng = np.random.default_rng(7)
    entries = rng.integers(-(2**40), 2**40, size=(3, 1100), dtype=np.int64)
    elements = [ring.random_element() for _ in range(1100)]
    outs = ring.integer_linear_combinations(elements, IntegerMatrix(entries))
    rows = entries.tolist()
    for r, out in enumerate(outs):
        assert out.get_coeff_matrix(out.repr) == _expected(ring, elements, rows, r)


def test_refusals():
    ring = Ring(16, prime_size=[30], split_degree=1)
    other = Ring(16, prime_size=[30, 30], split_degree=1)
    elements = [ring.random_element() for _ in range(2)]
    with pytest.raises(ValueError, match="one matrix column per element"):
        ring.integer_linear_combinations(elements, IntegerMatrix([[1, 2, 3]]))
    with pytest.raises(ValueError, match="over this ring"):
        ring.integer_linear_combinations(
            [elements[0], other.random_element()], IntegerMatrix([[1, 2]])
        )
    with pytest.raises(ValueError, match="no value"):
        ring.integer_linear_combinations(
            [elements[0], Polynomial(ring)], IntegerMatrix([[1, 2]])
        )
