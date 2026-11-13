# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BFV ciphertext multiplication.

The plaintext modulus is the chain's lowest prime, of 17 bits; the remaining primes
are 45 bits, with one 50-bit special prime for key switching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import Ring
from vfhe.fhe import BFV_Scheme

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

CHAINS = {
    "n12-4p": {"log_n": 12, "prime_bits": [17] + [45] * 3, "special_bits": [50]},
    "n13-6p": {"log_n": 13, "prime_bits": [17] + [45] * 5, "special_bits": [50]},
    "n14-8p": {"log_n": 14, "prime_bits": [17] + [45] * 7, "special_bits": [50]},
}


def _slot_values(scheme: BFV_Scheme, offset: int) -> list[int]:
    """One fixed value in ``[0, t)`` per slot."""
    return [(i * 7919 + offset) % scheme.t for i in range(scheme.n_slots)]


def _centered(scheme: BFV_Scheme, values: list[int]) -> list[int]:
    """``values`` modulo ``t``, as decoding gives them, in ``(-t/2, t/2]``."""
    t = scheme.t
    return [(v + t // 2) % t - t // 2 for v in values]


@pytest.mark.parametrize("p", list(CHAINS.values()), ids=list(CHAINS))
def bfv_multiply(benchmark: BenchmarkFixture, p: dict) -> None:
    """Two ciphertexts multiplied and relinearized, at the top level."""
    ring = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    scheme = BFV_Scheme(ring, special_primes=len(p["special_bits"]))
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    x, y = _slot_values(scheme, 0), _slot_values(scheme, 3)
    a = scheme.encrypt(scheme.encode(x), key)
    b = scheme.encrypt(scheme.encode(y), key)
    expected = _centered(scheme, [u * v for u, v in zip(x, y, strict=True)])
    assert (
        scheme.decode(scheme.decrypt(benchmark(scheme.multiply, a, b, rlk), key))
        == expected
    )
