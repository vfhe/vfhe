# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`seeded_below_many`, the sampler whose contract callers depend on in detail:
`count` values uniform in [0, bound), a pure function of (context, seed) and
the index, so draws are independent of each other, the same arguments always
give the same values, and different domain-separation tags give unrelated
streams."""

from __future__ import annotations

import pytest
from vfhe.crypto.random import seeded_below_many

SEED = b"prng-test-seed"


def sample(
    count: int, bound: int, context: bytes = b"test", seed: bytes = SEED
) -> list[int]:
    return seeded_below_many(count, bound, context, seed)


def test_draws_are_independent_of_each_other():
    """Successive draws must differ: one value repeated `count` times is the
    failure mode of finalizing a hash state without advancing it."""
    bound = (1 << 61) - 1
    values = sample(64, bound)
    assert len(set(values)) == len(values)
    assert all(0 <= v < bound for v in values)


def test_is_a_pure_function_of_seed_and_context():
    assert sample(8, 1 << 40) == sample(8, 1 << 40)
    assert sample(8, 1 << 40, seed=b"other") != sample(8, 1 << 40)
    # domain separation: same seed, different tag -> independent stream
    assert sample(8, 1 << 40, context=b"other") != sample(8, 1 << 40)


def test_prefix_is_stable_as_count_grows():
    """Draw i must not depend on how many were asked for."""
    long = sample(32, 1 << 50)
    assert sample(8, 1 << 50) == long[:8]
    assert seeded_below_many(1, 1 << 50, b"test", SEED) == [long[0]]


def test_respects_tight_and_degenerate_bounds():
    assert sample(16, 1) == [0] * 16  # only 0 is below 1
    assert all(v < 3 for v in sample(64, 3))  # rejection-heavy: mask covers 0..3
    # a bound one above a power of two exercises the mask's widening
    assert all(v < (1 << 32) + 1 for v in sample(64, (1 << 32) + 1))


def test_covers_its_range():
    """A sanity check on spread, not a statistical test: 512 draws below 2^61
    should not collapse into a narrow band."""
    bound = 1 << 61
    values = sample(512, bound)
    assert len(set(values)) == 512
    assert min(values) < bound // 4
    assert max(values) > bound - bound // 4


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="bound must be in"):
        seeded_below_many(4, 0, b"test", SEED)
    with pytest.raises(ValueError, match="count must be non-negative"):
        seeded_below_many(-1, 8, b"test", SEED)
    assert seeded_below_many(0, 8, b"test", SEED) == []


def test_rejects_a_nul_in_the_context():
    """BLAKE3 reads the context as a C string, so a NUL would end it early and
    two tags that differ past it would share a stream."""
    with pytest.raises(ValueError, match="NUL"):
        seeded_below_many(1, 10, b"tag\0more", SEED)
    with pytest.raises(ValueError, match="NUL"):
        seeded_below_many(2, 10, b"\0", SEED)


def test_rejects_a_bound_past_64_bits():
    assert seeded_below_many(1, (1 << 64) - 1, b"test", SEED)[0] < (1 << 64) - 1
    with pytest.raises(ValueError, match=r"\[1, 2\*\*64\)"):
        seeded_below_many(1, 1 << 64, b"test", SEED)


def test_an_empty_seed_and_context_are_valid():
    assert sample(4, 1000, context=b"", seed=b"") == sample(
        4, 1000, context=b"", seed=b""
    )
    assert sample(4, 1000, context=b"", seed=b"") != sample(
        4, 1000, context=b"x", seed=b""
    )


def test_a_power_of_two_bound_uses_only_its_bits():
    """Rejection draws nothing for a bound of 2^k: every word already fits,
    so the sequence is simply the masked stream and never falls through."""
    values = sample(256, 1 << 16)
    assert all(0 <= v < 1 << 16 for v in values)
    assert len(set(values)) > 200
