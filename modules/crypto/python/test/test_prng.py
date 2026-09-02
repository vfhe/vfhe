# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the seeded sampler in crypto's prng.c.

`prng_sample_below` is the library's one seeded generator: `count` values
uniform in [0, bound), a pure function of (context, seed). What is asserted
here is what its callers rely on -- that draws are independent of each other,
that the same arguments always give the same values, and that different
domain-separation tags give unrelated streams.

crypto has no cdef: its symbols are called from C. Like fhe's seeding
fixture, this binds the ones it needs in cffi's ABI mode from a declaration
owned here.
"""

from __future__ import annotations

import importlib.util
from functools import cache

from cffi import FFI

PRNG_CDEF = """
void prng_sample_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                       const uint8_t *seed, uint64_t seed_len);
void vfhe_prng_set_deterministic_seed(uint64_t seed);
void vfhe_prng_clear_deterministic_seed(void);
"""

ffi = FFI()
ffi.cdef(PRNG_CDEF)


@cache
def _lib():
    from vfhe.engine import active_engine

    spec = importlib.util.find_spec(f"_vfhe_native_{active_engine()}")
    if spec is None or spec.origin is None:
        raise RuntimeError(f"no extension found for engine {active_engine()!r}")
    return ffi.dlopen(spec.origin)


SEED = b"prng-test-seed"


def sample(
    count: int, bound: int, context: bytes = b"test", seed: bytes = SEED
) -> list:
    out = ffi.new("uint64_t[]", count)
    _lib().prng_sample_below(out, count, bound, context, seed, len(seed))
    return [out[i] for i in range(count)]


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


def test_deterministic_seed_override_does_not_reach_it():
    """The test-only override pins the *entropy-backed* stream; a seeded draw
    is already a pure function of its arguments and must be unaffected."""
    baseline = sample(8, 1 << 40)
    _lib().vfhe_prng_set_deterministic_seed(12345)
    try:
        assert sample(8, 1 << 40) == baseline
    finally:
        _lib().vfhe_prng_clear_deterministic_seed()
