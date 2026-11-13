# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial operations at the CKKS parameters.

Every ring is fully split (``split_degree=1``), as CKKS's are, so the NTT
domain is the evaluation domain and an automorphism there is a permutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import Polynomial, Ring, RNSRing

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
