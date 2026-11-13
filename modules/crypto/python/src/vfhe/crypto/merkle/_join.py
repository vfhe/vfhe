# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`join_digests`: the leaf digests of one tree over several packed digest arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.crypto.hash._digest import DIGEST_LEN, Bytes
from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Sequence


def join_digests(parts: Sequence[Bytes]) -> bytes:
    """Joins several packed digest arrays into one, index by index.

    Digest `i` of the result is the digest of digest `i` of every array in
    `parts`, concatenated in the order of `parts`. A tree over the result
    commits to leaf `i` of every array at once, so several codewords of one
    length sit under one tree and one opening. Thread-safe, no state.

    Args:
        parts: One or more arrays, each the same whole number of
            `DIGEST_LEN`-byte digests packed end to end.

    Returns:
        As many packed digests as each array holds.

    Raises:
        ValueError: `parts` is empty, an array's length is no multiple of
            `DIGEST_LEN`, or the arrays differ in length.
    """
    if not parts:
        raise ValueError("join_digests needs at least one array")
    length = len(parts[0])
    if length % DIGEST_LEN or any(len(part) != length for part in parts):
        raise ValueError(
            f"the arrays must hold the same number of {DIGEST_LEN}-byte digests"
        )
    buffers = [ffi.from_buffer(part) for part in parts]
    pointers = ffi.new("const uint8_t *[]", buffers)
    out = ffi.new("uint8_t[]", length)
    lib.merkle_join_digests(out, pointers, len(parts), length // DIGEST_LEN)
    return bytes(ffi.buffer(out))
