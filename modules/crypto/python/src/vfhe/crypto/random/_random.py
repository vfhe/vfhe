# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`Random`: the entropy stream, or a stream a seed reproduces."""

from __future__ import annotations

from vfhe.util.bindings import ffi, lib


class Random:
    """A source of randomness: the entropy stream, or a stream a seed reproduces.

    Without a seed, draws come from the entropy stream, seeded from the
    hardware and expanded by the engine's cipher in C, and nobody can predict
    them. Thread-safe: every thread draws from a pool of its own, and a forked
    child discards the pool it inherits. A failing seed source exits the
    process; a predictable value is never returned.

    With a seed, every draw is a function of the seed, the BLAKE3 XOF of it
    read in order, the same bytes on every engine: two sources from one seed
    give the same bytes in the same order, so a scheme built over one
    generates the same keys, masks and noise every time, and the seed is as
    secret as they are. A seeded source belongs to one thread at a time; its
    state is wiped when it is released.

    >>> Random(b"seed").bytes(4) == Random(b"seed").bytes(4)
    True
    """

    __slots__ = ("obj",)

    def __init__(self, seed: bytes | None = None) -> None:
        """
        Args:
            seed: Any length; None for the entropy stream.
        """
        #: The C source, or NULL for the entropy stream; what a sampler is handed.
        self.obj = (
            ffi.NULL
            if seed is None
            else ffi.gc(lib.random_seeded(seed, len(seed)), lib.random_free)
        )

    def bytes(self, amount: int) -> bytes:
        """Draws bytes.

        Args:
            amount: Bytes to draw; 0 and above.

        Returns:
            `amount` bytes.

        Raises:
            ValueError: `amount` is negative.
        """
        if amount < 0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        if amount == 0:
            return b""
        out = ffi.new("uint8_t[]", amount)
        lib.random_bytes(self.obj, out, amount)
        return ffi.buffer(out)[:]

    def below(self, bound: int) -> int:
        """Draws one value: `below_many` with a count of 1.

        Args:
            bound: The value is below it; 1 to 2**64 - 1.

        Returns:
            A value uniform in ``[0, bound)``.

        Raises:
            ValueError: `bound` is outside ``[1, 2**64)``.
        """
        return self.below_many(1, bound)[0]

    def below_many(self, count: int, bound: int) -> list[int]:
        """Draws values exactly uniform below a bound, by rejection.

        The draw branches on how many words were rejected and on nothing
        else, so a timing reveals only the rejection count.

        Args:
            count: Values to draw; 0 and above.
            bound: Every value is below it; 1 to 2**64 - 1.

        Returns:
            `count` values, each uniform in ``[0, bound)``.

        Raises:
            ValueError: `bound` is outside ``[1, 2**64)``, or `count` is negative.
        """
        if not 1 <= bound < 1 << 64:
            raise ValueError(f"bound must be in [1, 2**64), got {bound}")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return []
        out = ffi.new("uint64_t[]", count)
        lib.random_below(self.obj, out, count, bound)
        return [out[i] for i in range(count)]
