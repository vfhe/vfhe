# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial operations at the CKKS shapes.

Every ring is fully split (``split_degree=1``), as CKKS's are, so the NTT
domain is the evaluation domain and an automorphism there is a permutation.
"""

from __future__ import annotations

from harness import Case, Shape, benchmark
from vfhe.arith import IntegerMatrix, Polynomial, Ring, RNSPolynomial, RNSRing

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


#: An integer matrix times ring elements: ``outputs`` x ``inputs`` entries of up
#: to 2^10 in magnitude over a ring of degree 2^log_n split to degree ``split``.
#: The largest is a dense layer of 14,155 inputs over two primes, one component
#: of a 2^12 ring (about 0.9 GB of inputs); the 59-bit one takes the Shoup
#: kernel on the avx512ifma engine.
COMBINATION_SHAPES = [
    Shape("n10-2p-8x64", log_n=10, prime_bits=[50, 50], split=1, outputs=8, inputs=64),
    Shape(
        "n12-2p-138x2048",
        log_n=12,
        prime_bits=[50, 50],
        split=1,
        outputs=138,
        inputs=2048,
    ),
    Shape(
        "n12-59b-138x2048",
        log_n=12,
        prime_bits=[59, 59],
        split=1,
        outputs=138,
        inputs=2048,
    ),
    Shape(
        "n12-28b-138x2048",
        log_n=12,
        prime_bits=[28, 28],
        split=1,
        outputs=138,
        inputs=2048,
    ),
    Shape(
        "n12-2p-d4-138x14155",
        log_n=12,
        prime_bits=[52, 52],
        split=4,
        outputs=138,
        inputs=14155,
    ),
]


def _combination_setup(
    shape: Shape,
) -> tuple[RNSRing, list[RNSPolynomial], list[list[int]]]:
    ring = Ring(
        1 << shape["log_n"], prime_size=shape["prime_bits"], split_degree=shape["split"]
    )
    elements = [ring.random_element() for _ in range(shape["inputs"])]
    rows = [
        [(31 * r + 17 * j) % 2049 - 1024 for j in range(shape["inputs"])]
        for r in range(shape["outputs"])
    ]
    return ring, elements, rows


@benchmark(shapes=COMBINATION_SHAPES)
def integer_linear_combinations(shape: Shape) -> Case:
    """An integer matrix times ring elements in the NTT domain, the matrix prepared once."""
    ring, elements, rows = _combination_setup(shape)
    matrix = IntegerMatrix(rows)
    matrix.prepared(ring)

    def check(result: list[RNSPolynomial]) -> None:
        expected = elements[0] * rows[0][0]
        for w, e in zip(rows[0][1:], elements[1:], strict=True):
            expected = expected + e * w
        assert result[0] == expected

    return Case(
        lambda: ring.integer_linear_combinations(elements, matrix),
        check=check,
        items=shape["outputs"],
    )


@benchmark(shapes=COMBINATION_SHAPES)
def integer_matrix_prepare(shape: Shape) -> Case:
    """Preparing an integer matrix for a ring: entries reduced per prime, nonzero columns listed."""
    ring, _elements, rows = _combination_setup(shape)
    matrix = IntegerMatrix(rows)

    def op() -> object:
        matrix._prepared.clear()  # noqa: SLF001 - time the preparation, not the cache
        return matrix.prepared(ring)

    return Case(op)
