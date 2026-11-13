# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`seeded_below_many`: values uniform below a bound, each a pure function of its index."""

from __future__ import annotations

from vfhe.util.bindings import ffi, lib


def seeded_below_many(count: int, bound: int, context: bytes, seed: bytes) -> list[int]:
    """Computes values of the sequence ``(context, seed)`` names, uniform below a bound.

    The sequence is defined by index: value `i` is word `i` of a BLAKE3
    stream derived from `context` and `seed`, masked to the bits of
    ``bound - 1`` and redrawn from further streams while it is not below
    `bound`. So value `i` depends on `i` alone and raising `count` extends
    the sequence; in C, `seeded_below_from` starts at any index. The values
    are exactly uniform and byte for byte the same on every engine, which is
    what lets a verifier recompute a prover's draws. Thread-safe, no state.
    The running time depends on `bound`.

    Args:
        count: Values to compute; 0 and above.
        bound: Every value is below it; 1 to 2**64 - 1.
        context: The domain-separation tag: a fixed byte-string literal per
            call site, never caller data, so one seed under two tags gives
            two independent sequences. Holds no NUL byte.
        seed: Any length; public when the values are.

    Returns:
        Values ``[0, count)`` of the sequence, each in ``[0, bound)``.

    Raises:
        ValueError: `bound` is outside ``[1, 2**64)``, `count` is negative, or
            `context` holds a NUL byte, which would end it early in C.
    """
    if not 1 <= bound < 1 << 64:
        raise ValueError(f"bound must be in [1, 2**64), got {bound}")
    if b"\0" in context:
        raise ValueError("context must not contain a NUL byte")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return []
    out = ffi.new("uint64_t[]", count)
    lib.seeded_below(out, count, bound, context, seed, len(seed))
    return [out[i] for i in range(count)]
