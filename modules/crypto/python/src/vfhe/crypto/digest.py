# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Canonical 32-byte digests of primitive values, and the chain that digests
a sequence of them.

Every value an object's `digest()` method is built from goes through one of
these, domain-separated by a tag, so that two values of different types
never digest the same. `chain_digest` is the recursive form,
h_i = H(h_{i-1} | part_i) over already-digested parts -- the BCS
sigma-chain [BCS16] -- rather than one concatenation: a growing sequence
can then cache its prefix states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .merkle import hash_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable


def chain_digest(parts: Iterable[bytes]) -> bytes:
    """The chained digest of already-digested `parts`, from a fixed seed."""
    state = hash_bytes(b"vfhe.chain")
    for part in parts:
        state = hash_bytes(state + part)
    return state


def bytes_digest(value: bytes | bytearray | memoryview) -> bytes:
    return hash_bytes(b"bytes" + bytes(value))


def int_digest(value: int) -> bytes:
    """Of an integer of any size, by its two's-complement little-endian
    bytes, in the fewest that hold it."""
    length = (value.bit_length() + 8) // 8 or 1
    return hash_bytes(b"int" + value.to_bytes(length, "little", signed=True))


def str_digest(value: str) -> bytes:
    return hash_bytes(b"str" + value.encode())
