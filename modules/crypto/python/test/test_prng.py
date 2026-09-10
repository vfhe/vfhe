# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for vfhe.crypto.prng: the library's two generators.

`seeded` is the one whose contract callers depend on in detail -- `count`
values uniform in [0, bound), a pure function of (context, seed) -- so most
of what is asserted here is about it: that draws are independent of each
other, that the same arguments always give the same values, and that
different domain-separation tags give unrelated streams.

`entropy` is unpredictable by construction, so what can be checked is its
shape, its unbiased rejection loop, and that the test-only deterministic
override does what it says.
"""

from __future__ import annotations

import pytest
from vfhe.crypto import SEED_WORDS, entropy, seeded

SEED = b"prng-test-seed"


def sample(
    count: int, bound: int, context: bytes = b"test", seed: bytes = SEED
) -> list[int]:
    return seeded.below_many(count, bound, context, seed)


# --- the seeded sampler ----------------------------------------------------


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
    assert seeded.below(1 << 50, b"test", SEED) == long[0]


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


def test_seeded_bytes_are_reproducible():
    assert seeded.bytes(0, b"test", SEED) == b""
    out = seeded.bytes(64, b"test", SEED)
    assert len(out) == 64
    assert out == seeded.bytes(64, b"test", SEED)
    assert out != seeded.bytes(64, b"other", SEED)


def test_seeded_rejects_bad_arguments():
    with pytest.raises(ValueError, match="bound must be at least 1"):
        seeded.below_many(4, 0, b"test", SEED)
    with pytest.raises(ValueError, match="count must be non-negative"):
        seeded.below_many(-1, 8, b"test", SEED)
    assert seeded.below_many(0, 8, b"test", SEED) == []


# --- the entropy-backed stream ---------------------------------------------


def test_entropy_bytes_have_the_right_shape_and_differ():
    assert entropy.bytes(0) == b""
    first, second = entropy.bytes(64), entropy.bytes(64)
    assert len(first) == len(second) == 64
    assert first != second
    # larger than the 1 KiB pool: served from a fresh seed
    assert len(entropy.bytes(4096)) == 4096
    with pytest.raises(ValueError, match="non-negative"):
        entropy.bytes(-1)


def test_entropy_below_stays_in_range_and_spreads():
    assert entropy.below(1) == 0
    assert all(0 <= entropy.below(3) < 3 for _ in range(64))
    values = [entropy.below((1 << 61) - 1) for _ in range(64)]
    assert len(set(values)) == len(values)
    with pytest.raises(ValueError, match="bound must be at least 1"):
        entropy.below(0)


def test_entropy_normal_is_centred_and_scaled():
    """Not a distribution test: 4096 draws at sigma=100 should have a mean
    near zero and a spread of the right order."""
    draws = [entropy.normal(100.0) for _ in range(4096)]
    mean = sum(draws) / len(draws)
    assert abs(mean) < 20.0
    assert 50.0 < max(abs(d) for d in draws) < 1000.0


def test_the_raw_entry_points_are_reachable():
    """The seed source and the two expanders, exposed for completeness."""
    words = entropy.seed_words()
    assert len(words) == SEED_WORDS
    assert all(0 <= w < (1 << 64) for w in words)
    assert len(entropy.bytes_from_fresh_seed(32)) == 32
    assert len(entropy.bytes_from_pool(32)) == 32
    assert entropy.bytes_from_fresh_seed(32) != entropy.bytes_from_fresh_seed(32)


def test_deterministic_override_pins_and_releases_the_stream():
    with entropy.deterministic(12345):
        pinned = entropy.bytes(32)
    with entropy.deterministic(12345):
        assert entropy.bytes(32) == pinned
    with entropy.deterministic(54321):
        assert entropy.bytes(32) != pinned
    assert entropy.bytes(32) != pinned  # hardware entropy is back


def test_deterministic_override_does_not_reach_the_seeded_sampler():
    """The test-only override pins the *entropy-backed* stream; a seeded draw
    is already a pure function of its arguments and must be unaffected."""
    baseline = sample(8, 1 << 40)
    with entropy.deterministic(12345):
        assert sample(8, 1 << 40) == baseline
