# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python front on the execution tape: the operations `Polynomial` makes
through the generic interface are recorded, and each equals its event."""

import pytest
from vfhe.arith import Ring
from vfhe.util import trace

N = 64


@pytest.fixture
def ring():
    return Ring(N, prime_size=[45, 45, 45], split_degree=1)


@pytest.fixture(autouse=True)
def a_clean_tape():
    trace.set_verify(True)
    yield
    t = trace.enable()
    untraced = t.untraced
    t.clear()
    assert untraced == 0


def test_each_front_operation_is_recorded_with_its_operands(ring):
    x = ring.random_element(ntt=False)
    y = ring.random_element(ntt=False)
    small = ring.quotient_ring(ell=2)
    with trace.recording() as t:
        total = x + y
        difference = x - y
        product = x * y
        scaled = x * 3
        residue = x % ring.primes[1]
        divided = x.copy().round_division(small)
        x.to_coeff()
        permuted = x.automorphism(5)
        scaled_in_place = y.copy()
        scaled_in_place *= 7
        noise = ring.random_gaussian_element(3.2, ntt=False)
    recorded = [e for e in t.events() if e.kind != "mark"]
    inputs = [e for e in recorded if e.kind == "input"]
    assert [e.value for e in inputs] == [x, y]
    events = [e for e in recorded if e.kind != "input"]
    assert [e.kind for e in events] == [
        "add",
        "sub",
        "mul",
        "scale_int",
        "decompose_digit",
        "copy",
        "round_division",
        "permute",
        "copy",
        "scale_int",
        "sample_gaussian",
    ]
    add, sub, mul, scale, digit, copy, division, permute, _, in_place, gaussian = events
    operands = (inputs[0].sequence, inputs[1].sequence)
    assert add.inputs == sub.inputs == mul.inputs == operands
    assert add.value == total
    assert sub.value == difference
    assert mul.value == product
    assert (scale.value, scale.param) == (scaled, 3)
    assert (digit.value, digit.primes) == (residue, (ring.primes[1],))
    assert division.inputs == (copy.sequence,)
    assert (division.value, division.ring) == (divided, small)
    assert (permute.value, permute.param) == (permuted, 5)
    assert (in_place.value, in_place.param) == (scaled_in_place, 7)
    assert (gaussian.value, gaussian.secret, gaussian.sigma) == (noise, True, 3.2)
    assert t.version_of(total) == add.sequence
    assert t.version_of(scaled_in_place) == in_place.sequence


def test_a_product_by_scalars_outside_a_word_is_recorded(ring):
    x = ring.random_element(ntt=False)
    factors = [5, -1, 2**65]
    scalar = ring.scalar(11)
    with trace.recording() as t:
        negated = x * -3
        wide = x.copy()
        wide *= 2**70 + 1
        per_prime = x * factors
        reused = x * scalar
    events = [e for e in t.events() if e.kind not in ("mark", "input")]
    assert [e.kind for e in events] == [
        "scale_by",
        "copy",
        "scale_by",
        "scale_by",
        "scale_by",
    ]
    negative, _, in_place, listed, prebuilt = events
    assert negative.value == negated
    assert negative.scalar == tuple((-3) % p for p in ring.primes)
    assert in_place.value == wide
    assert in_place.scalar == tuple((2**70 + 1) % p for p in ring.primes)
    assert listed.value == per_prime
    assert listed.scalar == tuple(
        f % p for f, p in zip(factors, ring.primes, strict=True)
    )
    assert (prebuilt.value, prebuilt.scalar) == (reused, scalar.residues)
