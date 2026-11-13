# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""`Field.lagrange_interpolate`: Lagrange interpolation on the nodes 0, 1, ..., k.

Each implementation's native kernel and the generic default are checked
against the definition, with every denominator inverted in the field: at
random points, at the nodes themselves, from one value up to the kernels'
limit and past it (where the generic path takes over), and over a field
whose characteristic is below the number of nodes.
"""

from __future__ import annotations

import pytest
from vfhe.arith import ExtensionField, Field, PseudoMersenneField
from vfhe.util.bindings import lib

PRIME = (1 << 61) - 1
LIMIT = lib.ARITH_LAGRANGE_MAX_VALUES


@pytest.fixture(
    scope="module",
    params=["pmf-156", "pmf-260", "ext-d1", "ext-d4", "ext-small"],
)
def field(request):
    kind, _, size = request.param.partition("-")
    if kind == "pmf":
        return PseudoMersenneField.generate(int(size), two_adicity=8)
    if size == "small":
        return ExtensionField(97, 2, 5)
    return ExtensionField(PRIME, int(size[1:]), 3)


def _lagrange(field, values, x):
    """The definition: sum_t values[t] prod_{j != t} (x - j) / (t - j)."""
    total = field.zero
    for t in range(len(values)):
        numerator, denominator = field.one, field.one
        for j in range(len(values)):
            if j != t:
                numerator = numerator * (x - j)
                denominator = denominator * (t - j)
        total = total + values[t] * numerator * denominator.inverse()
    return total


def _values(field, count, tag):
    return [field.random_element(f"{tag}-{t}".encode()) for t in range(count)]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 9, LIMIT - 1, LIMIT, LIMIT + 1])
def test_lagrange_interpolate_matches_the_definition(field, count):
    if count > field.prime:
        pytest.skip("more nodes than the characteristic")
    values = _values(field, count, f"values-{count}")
    for x in (field.random_element(b"x"), field(count), field(count + 7)):
        want = _lagrange(field, values, x)
        assert field.lagrange_interpolate(values, x) == want
        assert Field.lagrange_interpolate(field, values, x) == want


def test_lagrange_interpolate_at_a_node_is_the_value_there(field):
    values = _values(field, 5, "nodes")
    for t in range(5):
        assert field.lagrange_interpolate(values, field(t)) == values[t]
        assert Field.lagrange_interpolate(field, values, field(t)) == values[t]


def test_lagrange_interpolate_reproduces_a_polynomial_of_lower_degree(field):
    # Values of x^2 + 3 at 0..5: the interpolant is that quadratic.
    values = [field(t * t + 3) for t in range(6)]
    x = field.random_element(b"quadratic")
    assert field.lagrange_interpolate(values, x) == x * x + 3


def test_lagrange_interpolate_leaves_its_inputs_unchanged(field):
    values = _values(field, 4, "inputs")
    x = field.random_element(b"inputs-x")
    copies = [v + 0 for v in values], x + 0
    field.lagrange_interpolate(values, x)
    assert values == copies[0]
    assert x == copies[1]


def test_lagrange_interpolate_refuses_no_values_and_repeated_nodes():
    field = ExtensionField(7, 1)
    with pytest.raises(ValueError):
        field.lagrange_interpolate([], field.one)
    with pytest.raises(ValueError):
        field.lagrange_interpolate([field.one] * 8, field.one)
    # Seven nodes are 0..6, distinct modulo 7.
    values = [field(t * t) for t in range(7)]
    assert field.lagrange_interpolate(values, field(3)) == field(9)


def test_lagrange_interpolate_refuses_elements_of_another_field(field):
    other = (
        ExtensionField(PRIME, 2, 3)
        if isinstance(field, PseudoMersenneField) or field.d != 2
        else ExtensionField(PRIME, 4, 3)
    )
    values = _values(field, 3, "mixed")
    with pytest.raises(TypeError):
        field.lagrange_interpolate([*values[:2], other.one], field.one)
    with pytest.raises(TypeError):
        field.lagrange_interpolate(values, other.one)
