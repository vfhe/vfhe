# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two generators: the entropy stream and the seeded sampler."""

from __future__ import annotations

import math

from harness import Case, Shape, benchmark
from vfhe.crypto import entropy, seeded

#: Gaussian draws per call: `entropy.normal` returns one value, so a call loops.
NORMAL_DRAWS = 1024

#: A 49-bit prime, the kind of bound a protocol samples field elements below.
BOUND = 562949953421201


@benchmark(shapes=[Shape("b4096", amount=1 << 12), Shape("b1048576", amount=1 << 20)])
def entropy_bytes(shape: Shape) -> Case:
    """``entropy.bytes(amount)``: unpredictable bytes, expanded from a fresh seed."""
    amount = shape["amount"]

    def check(result: bytes) -> None:
        assert len(result) == amount
        assert len(set(result[:256])) > 1

    return Case(lambda: entropy.bytes(amount), check=check)


@benchmark()
def entropy_normal(_shape: Shape) -> Case:
    """``entropy.normal(3.2)``, one Gaussian draw per call into the native sampler."""

    def op() -> list[float]:
        return [entropy.normal(3.2) for _ in range(NORMAL_DRAWS)]

    def check(result: list[float]) -> None:
        assert all(math.isfinite(x) for x in result)
        assert len(set(result)) > 1

    return Case(op, check=check, items=NORMAL_DRAWS)


@benchmark(shapes=[Shape("count65536-q49", count=1 << 16)])
def seeded_below_many(shape: Shape) -> Case:
    """``seeded.below_many``: values below a 49-bit bound, a pure function of the seed."""
    count = shape["count"]
    context, seed = b"vfhe-bench", b"seeded-below-many-seed"
    expected = seeded.below_many(count, BOUND, context, seed)

    def check(result: list[int]) -> None:
        assert result == expected
        assert max(result) < BOUND

    return Case(
        lambda: seeded.below_many(count, BOUND, context, seed), check=check, items=count
    )
