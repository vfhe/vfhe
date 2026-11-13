# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two kinds of randomness: the entropy stream and the seeded sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.crypto.random import Random, seeded_below_many

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: A 49-bit prime, the kind of bound a protocol samples field elements below.
BOUND = 562949953421201


@pytest.mark.parametrize("amount", [1 << 12, 1 << 20], ids=["4096B", "1MiB"])
def random_bytes(benchmark: BenchmarkFixture, amount: int) -> None:
    """``Random().bytes(amount)``: unpredictable bytes, expanded from a fresh seed."""
    drawn = benchmark(Random().bytes, amount)
    assert len(drawn) == amount and len(set(drawn[:256])) > 1


def seeded_values(benchmark: BenchmarkFixture) -> None:
    """``seeded_below_many``: 2^16 values below a 49-bit bound, a pure function of the seed."""
    count, context, seed = 1 << 16, b"vfhe-bench", b"seeded-below-many-seed"
    drawn = benchmark(seeded_below_many, count, BOUND, context, seed)
    assert (
        max(drawn) < BOUND and seeded_below_many(count, BOUND, context, seed) == drawn
    )
