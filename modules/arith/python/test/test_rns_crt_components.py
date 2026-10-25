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


def test_a_seeded_exceptional_element_is_its_coefficients_in_every_field(ring):
    """`Ring.random_exceptional(seed)` read at slot 0 of each prime is
    `Field.random_exceptional(seed, bound=min(primes))`: the same integer
    coefficients, below the smallest prime, in every residue field."""
    bound = min(ring.primes)
    element = ring.random_exceptional(seed=b"seed")
    assert element == ring.random_exceptional(seed=b"seed")
    assert element != ring.random_exceptional(seed=b"other")
    assert ring.random_exceptional(seed=b"seed", ntt=False) == element
    coefficients = element.get_polynomial()
    assert all(c < bound for c in coefficients[: ring.split_degree])
    assert all(c == 0 for c in coefficients[ring.split_degree :])
    for index in range(ring.ell):
        field = ring.residue_field(index)
        assert ring.crt_components([element], index)[0] == (
            field.random_exceptional(seed=b"seed", bound=bound)
        )


def test_the_native_exceptional_sampler_writes_the_drawn_integers(ring):
    """`sample_exceptional` writes, natively and with no transform, the
    element whose coefficients are the integers `vfhe.crypto.seeded` draws
    below the smallest prime; unseeded draws are in the same set."""
    from vfhe.crypto import seeded

    bound, d = min(ring.primes), ring.split_degree
    integers = seeded.below_many(d, bound, b"exceptional", b"seed")
    expected = Polynomial(ring).from_array(integers)
    assert ring.random_exceptional(seed=b"seed") == expected
    fresh = ring.random_exceptional(ntt=False).get_polynomial()
    assert all(c < bound for c in fresh[:d]) and not any(fresh[d:])


def test_crt_linear_combination_combines_the_components(ring):
    """Entry b is sum over c, s of coefficients[c k + s] times component s of
    parts[c][b], as computed from `crt_components` laid out element-major."""
    from vfhe.arith import FieldVector

    k = ring.N // ring.split_degree
    parts = [[ring.random_element() for _ in range(5)] for _ in range(3)]
    for index in range(ring.ell):
        field = ring.residue_field(index)
        coefficients = FieldVector(field, 3 * k)
        coefficients.sample_random(b"coefficients")
        combination = ring.crt_linear_combination(parts, index, coefficients)
        assert len(combination) == 5
        tables = [
            ring.crt_components(part, index, element_major=True) for part in parts
        ]
        for b in range(5):
            expected = field(0)
            for c, table in enumerate(tables):
                for s in range(k):
                    expected = expected + coefficients[c * k + s] * table[b * k + s]
            assert combination[b] == expected
    with pytest.raises(ValueError):
        ring.crt_linear_combination(parts, 0, FieldVector(ring.residue_field(0), k))


def test_components_agree_with_field_base_conversion(ring):
    """Component s is slot s carried into slot 0's field by the isomorphism
    `field_base_conversion` computes element by element (the one ring
    inversion uses): the tables of `crt_components` and that function are
    two forms of one map."""
    from vfhe.util.bindings import ffi, lib

    d, k = ring.split_degree, ring.N // ring.split_degree
    element = ring.random_element()
    # Per prime, slot s's d coefficients at s * d + j.
    slots = element.get_coeff_matrix(repr.ntt)
    base = ffi.cast("RNS_Base", ring.base)
    for index in range(ring.ell):
        row = ring.prime_indices[index]
        components = ring.crt_components([element], index)
        for s in range(k):
            source = ffi.new("uint64_t[]", slots[index][s * d : (s + 1) * d])
            out = ffi.new("uint64_t[]", d)
            lib.field_base_conversion(
                out, source, s, 0, d, k, base.w[row], base.mods[row]
            )
            component = components[s]
            expected = [int(c) for c in component.value] if d > 1 else [int(component)]
            assert [out[j] for j in range(d)] == expected
