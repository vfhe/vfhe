# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BFV ciphertext multiplication.

The plaintext modulus is the chain's lowest prime, of 17 bits; the remaining primes
are 45 bits, with one 50-bit special prime for key switching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness import Case, Shape, benchmark
from vfhe.arith import Ring
from vfhe.fhe import BFV_Scheme

if TYPE_CHECKING:
    from vfhe.mlwe import MLWE

SHAPES = [
    Shape("bfv-n12-4p", log_n=12, prime_bits=[17] + [45] * 3, special_bits=[50]),
    Shape("bfv-n13-6p", log_n=13, prime_bits=[17] + [45] * 5, special_bits=[50]),
    Shape("bfv-n14-8p", log_n=14, prime_bits=[17] + [45] * 7, special_bits=[50]),
]


def slot_values(scheme: BFV_Scheme, offset: int) -> list[int]:
    """Returns one fixed value in ``[0, t)`` per slot."""
    return [(i * 7919 + offset) % scheme.t for i in range(scheme.n_slots)]


def centered(scheme: BFV_Scheme, values: list[int]) -> list[int]:
    """Returns ``values`` modulo ``t``, as decoding gives them, in ``(-t/2, t/2]``."""
    t = scheme.t
    return [(v + t // 2) % t - t // 2 for v in values]


@benchmark(shapes=SHAPES)
def bfv_multiply(shape: Shape) -> Case:
    """Two ciphertexts multiplied and relinearized, at the top level."""
    ring = Ring(
        1 << shape["log_n"],
        prime_size=shape["prime_bits"] + shape["special_bits"],
        split_degree=1,
    )
    scheme = BFV_Scheme(ring, special_primes=len(shape["special_bits"]))
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    x, y = slot_values(scheme, 0), slot_values(scheme, 3)
    a = scheme.encrypt(scheme.encode(x), key)
    b = scheme.encrypt(scheme.encode(y), key)

    def check(result: MLWE) -> None:
        expected = centered(scheme, [u * v for u, v in zip(x, y, strict=True)])
        assert scheme.decode(scheme.decrypt(result, key)) == expected

    return Case(lambda: scheme.multiply(a, b, rlk), check=check)
