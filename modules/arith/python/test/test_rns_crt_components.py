# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`RNSRing.crt_components` / `from_crt_components`: the CRT components of
ring elements at a prime, all in one residue field, and back.

Two oracles. The ring itself: component s must be a ring homomorphism into
the one field, send X to a root of X^N + 1 and of the slot's factor
X^d - w_s of X^N + 1 modulo p, fix the integers, and be inverted by
`from_crt_components`. And the definition, computed in Python from each
slot's coefficients (`test_matches_the_definition`).
"""

import pytest
from vfhe.arith import Polynomial, Ring, repr

# (N, split_degree, prime sizes): narrow and wide rows, and d = 1, 2, 4.
SHAPES = [(16, 4, [28, 30]), (32, 2, [40, 30]), (64, 1, [50, 28]), (256, 4, [28, 60])]


@pytest.fixture(params=SHAPES, ids=lambda s: f"N{s[0]}-d{s[1]}-{s[2]}")
def ring(request):
    N, d, sizes = request.param
    return Ring(N, prime_size=sizes, split_degree=d)


def _monomial(ring, k):
    coeffs = [0] * ring.N
    coeffs[k] = 1
    return Polynomial(ring).from_array(coeffs)


def test_every_slot_is_a_ring_homomorphism(ring):
    a, b = ring.random_element(), ring.random_element()
    elements = [a, b, a * b, a + b, a - b]
    count = len(elements)
    for index in range(ring.ell):
        v = ring.crt_components(elements, index)
        assert v.field is ring.residue_field(index)
        assert len(v) == count * (ring.N // ring.split_degree)
        for s in range(ring.N // ring.split_degree):
            pa, pb, pab, psum, pdiff = (v[s * count + t] for t in range(count))
            assert pa * pb == pab
            assert pa + pb == psum
            assert pa - pb == pdiff


def test_x_goes_to_a_root_of_each_slot_and_integers_are_fixed(ring):
    k = ring.N // ring.split_degree
    rou = ring.get_rou_matrix()
    for index in range(ring.ell):
        field = ring.residue_field(index)
        v = ring.crt_components(
            [_monomial(ring, 1), Polynomial(ring).from_array([7])], index
        )
        for s in range(k):
            x, seven = v[2 * s], v[2 * s + 1]
            assert x**ring.split_degree == field(rou[index][s])
            assert x**ring.N == -field.one
            assert seven == field(7)


@pytest.mark.parametrize(
    ("N", "d", "sizes", "count"),
    [(16, 4, [28, 50], 3), (4096, 4, [30, 50], 13), (2048, 2, [29, 45], 9)],
)
def test_matches_the_definition(N, d, sizes, count):
    """With P_s = (p, X^d - w_s) and w_s = w_0^(x_s), component s of an
    element is sum_j c_j y^j in R_q/P_0 = F_p[x]/(x^d - w_0), where c_j is the
    coefficient of x^j in the element's residue modulo P_s (its slot s) and
    y = x^(x_s) is the image of x under the isomorphism X -> X^(x_s)
    induces."""
    ring = Ring(N, prime_size=sizes, split_degree=d)
    k = N // d
    elements = [ring.random_element() for _ in range(count)]
    matrices = [a.get_coeff_matrix(repr=repr.ntt) for a in elements]
    rou = ring.get_rou_matrix()
    for index in range(ring.ell):
        field = ring.residue_field(index)
        p, w = ring.primes[index], rou[index]
        exponent = {pow(w[0], e, p): e for e in range(1, 2 * k, 2)}
        x = field([0, 1] + [0] * (d - 2)) if d > 1 else field(w[0])
        v = ring.crt_components(elements, index)
        for s in range(k):
            image = x ** exponent[w[s]]
            for b in range(count):
                # `get_coeff_matrix` lists each slot's d coefficients together.
                coefficients = matrices[b][index][s * d : (s + 1) * d]
                expected = field.zero
                for c in reversed(coefficients):
                    expected = expected * image + c
                assert v[s * count + b] == expected
        assert (
            ring.from_crt_components(
                [ring.crt_components(elements, i) for i in range(ring.ell)]
            )
            == elements
        )


def test_element_major_is_the_same_entries_transposed(ring):
    k = ring.N // ring.split_degree
    count = 11  # not a multiple of the kernels' tile
    elements = [ring.random_element() for _ in range(count)]
    for index in range(ring.ell):
        by_component = ring.crt_components(elements, index)
        by_element = ring.crt_components(elements, index, element_major=True)
        assert by_element.field is by_component.field
        for b in range(count):
            for s in range(k):
                assert by_element[b * k + s] == by_component[s * count + b]


def test_slot_zero_is_read_unchanged(ring):
    a = ring.random_element()
    # `get_coeff_matrix` lists each slot's d coefficients together.
    matrix = a.get_coeff_matrix(repr=repr.ntt)
    for index in range(ring.ell):
        field = ring.residue_field(index)
        expected = field(matrix[index][: ring.split_degree])
        assert ring.crt_components([a], index)[0] == expected


def test_from_crt_components_inverts_crt_components(ring):
    elements = [ring.random_element() for _ in range(3)]
    vectors = [ring.crt_components(elements, index) for index in range(ring.ell)]
    back = ring.from_crt_components(vectors)
    assert len(back) == len(elements)
    assert all(x == y for x, y in zip(back, elements, strict=True))


def test_crt_components_reads_the_canonical_domain_without_converting_it(ring):
    a = ring.random_element()
    c = a.copy()
    c.to_coeff()
    assert ring.crt_components([c], 0) == ring.crt_components([a], 0)
    assert c.repr == repr.coeff


def test_crt_components_takes_elements_of_a_ring_over_the_same_transform():
    big = Ring(32, prime_size=[30, 40, 50], split_degree=2)
    small = big.quotient_ring(mask=big.mask & ~(1 << big.prime_indices[0]))
    a = small.random_element()
    index = small.primes.index(big.primes[1])
    assert big.crt_components([a], 1) == small.crt_components([a], index)
    assert big.residue_field(1) is small.residue_field(index)
    with pytest.raises(ValueError, match="not over a ring"):
        big.crt_components([a], 0)  # `small` lacks the prime


def test_crt_components_of_no_elements_is_empty(ring):
    assert len(ring.crt_components([], 0)) == 0
    assert (
        ring.from_crt_components([ring.crt_components([], i) for i in range(ring.ell)])
        == []
    )


def test_from_crt_components_checks_its_vectors(ring):
    vectors = [ring.crt_components([ring.random_element()], i) for i in range(ring.ell)]
    with pytest.raises(ValueError, match="one vector per prime"):
        ring.from_crt_components(vectors[:-1])
    with pytest.raises(ValueError, match="residue_field"):
        ring.from_crt_components(vectors[::-1])
