# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The library's randomness, and the only randomness the library uses.

Two independent generators live in `c/src/prng.c` and are exposed here as two
objects, because the choice between them is a correctness question and not a
performance one:

- `entropy` (`EntropyPRNG`) — unpredictable bytes, seeded from RDRAND or
  `/dev/urandom` and expanded with the build's PRF. Use it for secrets:
  keys, noise, masks, anything an adversary must not guess.
- `seeded` (`SeededPRNG`) — values uniform in `[0, bound)` that are a pure
  function of `(context, seed)`. Use it where a value must be recomputable
  by another party from data it already has, such as a Fiat-Shamir
  transcript.

Both are module-level singletons over process-global C state; the classes
exist so the surface is typed and documented in one place, not so callers can
own an instance. Nothing here is thread-safe (the C pool is shared mutable
state).

`context` is a domain-separation tag: one seed under two different tags gives
two independent streams. Pass a fixed string literal per call site, never
anything caller-controlled.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from vfhe.engine import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterator

# What `generate_rnd_seed` fills, in 64-bit words.
SEED_WORDS = 4

# The AES keystream's step, in bytes: the smallest amount `get_rnd_from_hash`
# is defined for.
_KEYSTREAM_STEP = 256


class EntropyPRNG:
    """The entropy-backed stream: bytes no one can predict.

    Every method draws from the process-global generator in `prng.c`, so two
    calls never return the same thing (short of the test-only override in
    `deterministic`).
    """

    def bytes(self, amount: int) -> bytes:
        """`amount` unpredictable bytes — the default entry point.

        Serves requests under 512 bytes from an internal pool and expands a
        fresh seed for larger ones.
        """
        if amount < 0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        if amount == 0:
            return b""
        out = ffi.new("uint8_t[]", amount)
        lib.generate_random_bytes(amount, out)
        return bytes(out)

    def below(self, bound: int) -> int:
        """One value uniform in `[0, bound)`, from fresh entropy.

        Rejection sampling on whole bytes, so the result is unbiased. For a
        value that must be reproducible from a transcript, use `seeded`.
        """
        if bound < 1:
            raise ValueError(f"bound must be at least 1, got {bound}")
        if bound == 1:
            return 0
        width = (bound - 1).bit_length()
        nbytes = (width + 7) // 8
        mask = (1 << width) - 1
        while True:
            value = int.from_bytes(self.bytes(nbytes), "little") & mask
            if value < bound:
                return value

    def normal(self, sigma: float) -> float:
        """One sample from a zero-mean Gaussian with standard deviation
        `sigma` (Box-Muller over `bytes`).

        The support is unbounded: a caller needing a tail bound must clamp or
        resample.
        """
        return lib.generate_normal_random(sigma)

    def seed_words(self) -> list[int]:
        """One raw seed: `SEED_WORDS` words straight from the hardware source.

        The seed the expanders below draw from. Callers wanting random data
        want `bytes`, not this.
        """
        out = ffi.new("uint64_t[]", SEED_WORDS)
        lib.generate_rnd_seed(out)
        return [out[i] for i in range(SEED_WORDS)]

    def bytes_from_fresh_seed(self, amount: int) -> bytes:
        """`amount` bytes expanded from one freshly drawn seed, bypassing the
        pool. Every call pays for a seed draw, which dominates small requests.

        The AES keystream behind this on tuned x86-64 produces whole 256-byte
        steps, so a shorter request draws one step and keeps its prefix.
        """
        if amount < 1:
            raise ValueError(f"amount must be at least 1, got {amount}")
        width = max(amount, _KEYSTREAM_STEP)
        out = ffi.new("uint8_t[]", width)
        lib.get_rnd_from_hash(width, out)
        return bytes(out)[:amount]

    def bytes_from_pool(self, amount: int) -> bytes:
        """`amount` bytes from the 1 KiB pool, refilling it when what remains
        will not cover the request. A request larger than the pool is served
        from a fresh seed.
        """
        if amount < 1:
            raise ValueError(f"amount must be at least 1, got {amount}")
        out = ffi.new("uint8_t[]", amount)
        lib.get_rnd_from_buffer(amount, out)
        return bytes(out)

    @contextmanager
    def deterministic(self, seed: int) -> Iterator[None]:
        """Test-only: pin this stream to a splitmix64 sequence from `seed`.

        Makes every draw above reproducible for the duration of the block,
        within one build only — the expander is engine-dependent, so the same
        seed gives different bytes on different engines. Does not affect
        `seeded`, which is already a pure function of its arguments. Restores
        hardware entropy on exit.
        """
        lib.vfhe_prng_set_deterministic_seed(seed)
        try:
            yield
        finally:
            lib.vfhe_prng_clear_deterministic_seed()


class SeededPRNG:
    """The seeded sampler: values reproducible from `(context, seed)`.

    Byte-for-byte identical across runs and across engines, and untouched by
    `EntropyPRNG.deterministic`. This is what a protocol uses for values both
    parties must agree on.
    """

    def below(self, bound: int, context: bytes, seed: bytes) -> int:
        """One value uniform in `[0, bound)`."""
        return self.below_many(1, bound, context, seed)[0]

    def below_many(
        self, count: int, bound: int, context: bytes, seed: bytes, start: int = 0
    ) -> list[int]:
        """`count` values uniform in `[0, bound)`, from position `start` of
        the sequence.

        Entry `i` depends on `i` alone, so raising `count` extends the
        sequence instead of changing it, and a window costs what its own
        values cost: ``below_many(k, ..., start=j)`` is
        ``below_many(j + k, ...)[j:]`` without producing the first `j`.
        """
        if bound < 1:
            raise ValueError(f"bound must be at least 1, got {bound}")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if count == 0:
            return []
        out = ffi.new("uint64_t[]", count)
        lib.prng_sample_below_from(out, count, start, bound, context, seed, len(seed))
        return [out[i] for i in range(count)]

    def bytes(self, amount: int, context: bytes, seed: bytes) -> bytes:
        """`amount` bytes, a pure function of `(context, seed)`.

        Drawn as `amount` uniform bytes from `below_many`, so this is the
        seeded counterpart of `EntropyPRNG.bytes` and not a second expander.
        """
        if amount < 0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        if amount == 0:
            return b""
        return bytes(self.below_many(amount, 256, context, seed))


entropy = EntropyPRNG()
seeded = SeededPRNG()
