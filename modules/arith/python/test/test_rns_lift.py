# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`RNSRing.lift_mod_2_128`: the coefficients as integers in [0, q), modulo
2^128, against the Python CRT of `get_polynomial`."""

import random

import pytest
from vfhe.arith import Polynomial, Ring

#: Prime sizes and splitting degrees: narrow and wide rows, q below and above
#: 2^128, split layouts, and many terms of words above 2^52.
RINGS = [
    ([30, 30], 1),
    ([45, 45, 45], 4),
    ([60, 60], 2),
    ([20, 40, 50, 61], 1),
    ([61] * 12, 1),
]


def _elements(ring, count, ntt=False, seed=1):
    rng = random.Random(seed)  # noqa: S311 - test data, not a key
    out = []
    for _ in range(count):
        values = [rng.randrange(ring.q_l) for _ in range(ring.N)]
        values[0], values[-1] = 0, ring.q_l - 1
        element = Polynomial(ring).from_bigint_array(values)
        if ntt:
            element.to_NTT()
        out.append(element)
    return out


@pytest.mark.parametrize(("sizes", "split"), RINGS)
@pytest.mark.parametrize("ntt", [False, True], ids=["coeff", "ntt"])
def test_lift_is_the_crt_modulo_2_128(sizes, split, ntt):
    ring = Ring(16, prime_size=sizes, split_degree=split)
    assert ring.split_degree == split
    elements = _elements(ring, 3, ntt)
    reprs = [e.repr for e in elements]
    low, high = ring.lift_mod_2_128(elements)
    for e, element in enumerate(elements):
        for k, c in enumerate(element.get_polynomial()):
            assert low[e * ring.N + k] | high[e * ring.N + k] << 64 == c % (1 << 128)
    assert [e.repr for e in elements] == reprs  # read through copies


def test_lift_of_many_elements_on_several_threads():
    """Enough coefficients for the polynomials to be lifted in parallel."""
    ring = Ring(1 << 12, prime_size=[45, 45, 45], split_degree=4)
    elements = _elements(ring, 12, seed=3)
    low, high = ring.lift_mod_2_128(elements)
    for e, element in enumerate(elements):
        expected = [c % (1 << 128) for c in element.get_polynomial()]
        got = [low[e * ring.N + k] | high[e * ring.N + k] << 64 for k in range(ring.N)]
        assert got == expected


def test_lift_of_nothing_and_of_another_ring():
    ring = Ring(16, prime_size=[30, 30], split_degree=1)
    low, high = ring.lift_mod_2_128([])
    assert len(low) == len(high) == 0
    other = Ring(16, prime_size=[31, 31], split_degree=1)
    with pytest.raises(ValueError, match="of this ring"):
        ring.lift_mod_2_128(_elements(other, 1))
