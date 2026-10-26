# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial operations at the CKKS shapes.

Every ring is fully split (``split_degree=1``), as CKKS's are, so the NTT
domain is the evaluation domain and an automorphism there is a permutation.
"""

from __future__ import annotations

from harness import Case, Shape, benchmark
from vfhe.arith import Polynomial, Ring, RNSPolynomial, RNSRing

SHAPES = ["ckks-n12-3p", "ckks-n13-7p", "ckks-n14-7p"]

# The Galois element of a one-slot rotation in CKKS.
ROTATION = 5


def chain(shape: Shape) -> RNSRing:
    """Returns the ring of the shape's full chain, special primes excluded."""
    return Ring(1 << shape["log_n"], prime_size=shape["prime_bits"], split_degree=1)


@benchmark(shapes=SHAPES, tier="headline")
def polynomial_product(shape: Shape) -> Case:
    """``a *= b`` in the NTT domain: the ring product."""
    ring = chain(shape)
    a, b = ring.random_element(), ring.random_element()
    return Case(lambda: a.__imul__(b))


@benchmark(shapes=SHAPES, tier="headline")
def ntt_round_trip(shape: Shape) -> Case:
    """The NTT domain to coefficients and back, on every prime."""
    a = chain(shape).random_element()
    expected = a.copy()

    def op() -> RNSPolynomial:
        a.to_coeff()
        a.to_NTT()
        return a

    def check(result: Polynomial) -> None:
        assert result == expected

    return Case(op, check=check)


@benchmark(shapes=SHAPES)
def polynomial_sum(shape: Shape) -> Case:
    """``a += b`` in the NTT domain: one pass over memory, no multiplication."""
    ring = chain(shape)
    a, b = ring.random_element(), ring.random_element()
    return Case(lambda: a.__iadd__(b))


@benchmark(shapes=SHAPES, tier="headline")
def rescale(shape: Shape) -> Case:
    """Dividing by the last prime with rounding, from the NTT domain back to it.

    This is what CKKS pays per multiplication: the inverse transform on every
    prime, the division, and the forward transform on the primes that remain.
    The input is copied on every call, since the division works in place; the
    copy is a small part of the two transforms.
    """
    ring = chain(shape)
    lower = ring.quotient_ring(ell=ring.ell - 1)
    source = ring.random_element()

    def op() -> RNSPolynomial:
        result = source.copy()
        result.round_division(lower)
        result.to_NTT()
        return result

    def check(result: Polynomial) -> None:
        assert result.ring is lower

    return Case(op, check=check)


@benchmark(shapes=SHAPES)
def base_extend(shape: Shape) -> Case:
    """The chain's value extended to its special primes, in the coefficient domain.

    The step that dominates key switching.
    """
    wide = Ring(
        1 << shape["log_n"],
        prime_size=shape["prime_bits"] + shape["special_bits"],
        split_degree=1,
    )
    narrow = wide.quotient_ring(ell=len(shape["prime_bits"]))
    source = narrow.random_element()
    source.to_coeff()
    out = Polynomial(wide)

    def check(result: Polynomial) -> None:
        assert result.mod_reduce(narrow) == source

    return Case(lambda: source.base_extend(out=out), check=check)


@benchmark(shapes=SHAPES)
def automorphism(shape: Shape) -> Case:
    """``X -> X^5`` in the NTT domain, a permutation, into a new polynomial."""
    a = chain(shape).random_element()
    n = a.ring.N
    inverse = pow(ROTATION, -1, 2 * n)

    def check(result: Polynomial) -> None:
        assert result.automorphism(inverse) == a

    return Case(lambda: a.automorphism(ROTATION), check=check)
