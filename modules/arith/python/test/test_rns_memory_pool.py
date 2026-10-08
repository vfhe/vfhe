# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""RNS polynomial rows come from and return to the memory pool."""

import gc
from pathlib import Path

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.engine import memory_pool


def _asan_runtime_loaded() -> bool:
    maps = Path("/proc/self/maps")
    return maps.exists() and "libasan" in maps.read_text()


# An ASan build bypasses the pool, so there is nothing here to observe.
pytestmark = pytest.mark.skipif(
    _asan_runtime_loaded(), reason="the memory pool is bypassed under ASan"
)

N = 4096
# Two 64-bit rows and one 32-bit row per polynomial.
PRIME_SIZES = [50, 50, 28]
ROW_BYTES = 2 * N * 8 + N * 4


@pytest.fixture(autouse=True)
def automatic_capacity():
    memory_pool.set_capacity("auto")
    memory_pool.release_all()
    yield
    memory_pool.set_capacity()


def test_a_dropped_polynomial_leaves_its_rows_in_the_pool():
    ring = Ring(N, prime_size=PRIME_SIZES)
    p = Polynomial(ring)
    memory_pool.release_all()
    del p
    gc.collect()
    assert memory_pool.statistics().retained_bytes >= ROW_BYTES

    hits = memory_pool.statistics().hits
    q = Polynomial(ring)
    assert memory_pool.statistics().hits >= hits + len(PRIME_SIZES)
    del q


def test_capacity_zero_keeps_no_rows():
    ring = Ring(N, prime_size=PRIME_SIZES)
    memory_pool.set_capacity(0)
    p = Polynomial(ring)
    del p
    gc.collect()
    assert memory_pool.statistics().retained_bytes == 0
