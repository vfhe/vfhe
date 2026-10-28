# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two tower operations the key switch needs, and the generic-only
switch: a digit of an element's residues, and the approximate round
division."""

import pytest
from vfhe import arith
from vfhe.arith import Ring, repr
from vfhe.util import trace

N = 64


@pytest.fixture
def ring():
    return Ring(N, prime_size=[45, 45, 45, 50], split_degree=1)


@pytest.fixture(autouse=True)
def defaults_restored():
    yield
    arith.set_generic_only(False)
    trace.enable().clear()


def test_radix_digits_sum_to_the_residue(ring):
    x = ring.random_element(ntt=False)
    p = ring.primes[1]
    residue = x.decompose_digit(ring, [p])
    assert residue.repr == repr.coeff
    total = None
    for level in range(6):
        digit = x.decompose_digit(ring, [p], log_base=8, level=level) * (
            1 << (8 * level)
        )
        total = digit if total is None else total + digit
    assert total == residue


def test_a_group_digit_keeps_the_group_residues(ring):
    x = ring.random_element(ntt=False)
    group = ring.primes[:2]
    digit = x.decompose_digit(ring, group, centered=True)
    small = ring.quotient_ring(ell=2)
    assert digit.mod_reduce(small) == x.mod_reduce(small)


def test_decompose_digit_refuses_foreign_primes_and_grouped_radix(ring):
    x = ring.random_element(ntt=False)
    with pytest.raises(ValueError, match="not a prime"):
        x.decompose_digit(ring, [7])
    with pytest.raises(ValueError, match="one prime"):
        x.decompose_digit(ring, ring.primes[:2], log_base=8)


def test_approximate_round_division_is_within_the_exact_one(ring):
    x = ring.random_element(ntt=False)
    small = ring.quotient_ring(ell=2)
    exact = x.copy().round_division(small)
    fast = x.copy().round_division_approximate(small)
    assert fast.ring == small and fast.repr == repr.coeff
    difference = fast - exact
    # Off by at most one in the two primes divided by, in either direction.
    assert all(abs(c) <= 1 for c in difference.get_polynomial(signed=True))


def test_the_two_operations_are_on_the_tape(ring):
    x = ring.random_element(ntt=False)
    small = ring.quotient_ring(ell=3)
    with trace.recording() as t:
        d = x.decompose_digit(
            ring, [ring.primes[2]], log_base=8, level=1, centered=True
        )
        x.round_division_approximate(small)
    events = [e for e in t.events() if e.kind != "mark"]
    assert [e.kind for e in events] == [
        "input",
        "decompose_digit",
        "round_division_approximate",
    ]
    assert events[1].primes == (ring.primes[2],)
    assert (
        events[1].log_base,
        events[1].level,
        events[1].dropped,
        events[1].centered,
    ) == (
        8,
        1,
        0,
        True,
    )
    assert events[1].value == d
    assert events[2].inputs == (events[0].sequence,) and events[2].ring == small
    assert events[2].value == x


def test_generic_only_is_a_switch():
    assert not arith.generic_only()
    arith.set_generic_only(True)
    assert arith.generic_only()
    arith.set_generic_only(False)
    assert not arith.generic_only()
