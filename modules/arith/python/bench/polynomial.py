# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial operations at the CKKS parameters.

Every ring is fully split (``split_degree=1``), as CKKS's are, so the NTT
domain is the evaluation domain and an automorphism there is a permutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import IntegerMatrix, Polynomial, Ring, RNSPolynomial, RNSRing

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: The CKKS chains: ring degree 2^log_n, the chain's prime sizes from level 0
#: up, the primes key switching adds, and the encoding scale. mlwe and fhe run
#: at the same.
CKKS = {
    "n12-3p": {
        "log_n": 12,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n13-7p": {
        "log_n": 13,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n14-7p": {
        "log_n": 14,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
}
chains = pytest.mark.parametrize("p", list(CKKS.values()), ids=list(CKKS))

# The Galois element of a one-slot rotation in CKKS.
ROTATION = 5


def _chain(p: dict) -> RNSRing:
    """The ring of the chain, special primes excluded."""
    return Ring(1 << p["log_n"], prime_size=p["prime_bits"], split_degree=1)


@chains
def polynomial_product(benchmark: BenchmarkFixture, p: dict) -> None:
    """``a *= b`` in the NTT domain: the ring product."""
    ring = _chain(p)
    a, b = ring.random_element(), ring.random_element()
    benchmark(a.__imul__, b)


@chains
def ntt_round_trip(benchmark: BenchmarkFixture, p: dict) -> None:
    """The NTT domain to coefficients and back, on every prime."""
    a = _chain(p).random_element()
    expected = a.copy()

    def op():
        a.to_coeff()
        a.to_NTT()

    benchmark(op)
    assert a == expected


@chains
def polynomial_sum(benchmark: BenchmarkFixture, p: dict) -> None:
    """``a += b`` in the NTT domain: one pass over memory, no multiplication."""
    ring = _chain(p)
    a, b = ring.random_element(), ring.random_element()
    benchmark(a.__iadd__, b)


@chains
def rescale(benchmark: BenchmarkFixture, p: dict) -> None:
    """Dividing by the last prime with rounding, from the NTT domain back to it.

    This is what CKKS pays per multiplication: the inverse transform on every
    prime, the division, and the forward transform on the primes that remain.
    The input is copied on every call, since the division works in place; the
    copy is a small part of the two transforms.
    """
    ring = _chain(p)
    lower = ring.quotient_ring(ell=ring.ell - 1)
    source = ring.random_element()

    def op():
        result = source.copy()
        result.round_division(lower)
        result.to_NTT()
        return result

    assert benchmark(op).ring is lower


@chains
def base_extend(benchmark: BenchmarkFixture, p: dict) -> None:
    """The chain's value extended to its special primes, in the coefficient domain.

    The step that dominates key switching.
    """
    wide = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    narrow = wide.quotient_ring(ell=len(p["prime_bits"]))
    source = narrow.random_element()
    source.to_coeff()
    out = Polynomial(wide)
    assert benchmark(source.base_extend, out=out).mod_reduce(narrow) == source


@chains
def automorphism(benchmark: BenchmarkFixture, p: dict) -> None:
    """``X -> X^5`` in the NTT domain, a permutation, into a new polynomial."""
    a = _chain(p).random_element()
    inverse = pow(ROTATION, -1, 2 * a.ring.N)
    assert benchmark(a.automorphism, ROTATION).automorphism(inverse) == a


#: Integer matrices times vectors of ring elements: ring degree 2^log_n, the
#: primes, the split degree, and the matrix's outputs x inputs.
COMBINATIONS = {
    "n10-2p-8x64": {
        "log_n": 10,
        "prime_bits": [50, 50],
        "split": 1,
        "outputs": 8,
        "inputs": 64,
    },
    "n12-2p-138x2048": {
        "log_n": 12,
        "prime_bits": [50, 50],
        "split": 1,
        "outputs": 138,
        "inputs": 2048,
    },
    "n12-59b-138x2048": {
        "log_n": 12,
        "prime_bits": [59, 59],
        "split": 1,
        "outputs": 138,
        "inputs": 2048,
    },
    "n12-28b-138x2048": {
        "log_n": 12,
        "prime_bits": [28, 28],
        "split": 1,
        "outputs": 138,
        "inputs": 2048,
    },
    "n12-2p-d4-138x14155": {
        "log_n": 12,
        "prime_bits": [52, 52],
        "split": 4,
        "outputs": 138,
        "inputs": 14155,
    },
}
combinations = pytest.mark.parametrize(
    "p", list(COMBINATIONS.values()), ids=list(COMBINATIONS)
)


def _combination(p: dict) -> tuple[RNSRing, list[RNSPolynomial], list[list[int]]]:
    ring = Ring(1 << p["log_n"], prime_size=p["prime_bits"], split_degree=p["split"])
    elements = [ring.random_element() for _ in range(p["inputs"])]
    rows = [
        [(31 * r + 17 * j) % 2049 - 1024 for j in range(p["inputs"])]
        for r in range(p["outputs"])
    ]
    return ring, elements, rows


@combinations
def integer_linear_combinations(benchmark: BenchmarkFixture, p: dict) -> None:
    """An integer matrix times ring elements in the NTT domain, the matrix prepared once."""
    ring, elements, rows = _combination(p)
    matrix = IntegerMatrix(rows)
    matrix.prepared(ring)
    result = benchmark(ring.integer_linear_combinations, elements, matrix)
    expected = elements[0] * rows[0][0]
    for w, e in zip(rows[0][1:], elements[1:], strict=True):
        expected = expected + e * w
    assert result[0] == expected


@combinations
def integer_matrix_prepare(benchmark: BenchmarkFixture, p: dict) -> None:
    """Preparing an integer matrix for a ring: entries reduced per prime, nonzero columns listed."""
    ring, _, rows = _combination(p)
    matrix = IntegerMatrix(rows)

    def op():
        matrix._prepared.clear()  # noqa: SLF001 - time the preparation, not the cache
        return matrix.prepared(ring)

    benchmark(op)
