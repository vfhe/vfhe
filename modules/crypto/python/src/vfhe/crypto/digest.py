# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Canonical 32-byte digests of primitive values, and the chain that digests
a sequence of them.

Every value an object's `digest()` method is built from goes through one of
these, domain-separated by a tag, so that two values of different types
never digest the same. `chain_digest` is the recursive form,
h_i = H(h_{i-1} | part_i) over already-digested parts -- the BCS
sigma-chain [BCS16] -- rather than one concatenation: a growing sequence
can then cache its prefix states. `DigestChain` is the same chain as an
accumulator, for a caller whose parts are already packed in a buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.util.bindings import ffi, lib

from .merkle import DIGEST_LEN, hash_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable

_CHAIN_SEED = hash_bytes(b"vfhe.chain")


class DigestChain:
    """The chain of `chain_digest` as an accumulator, its state in native
    memory.

    `absorb` takes one part, of any length; `absorb_packed` takes a run of
    parts of one length laid out end to end in one buffer, in one native
    call. `digest()` is the state after the parts absorbed so far.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state = ffi.new(f"uint8_t[{DIGEST_LEN}]", _CHAIN_SEED)

    def absorb(self, part: bytes | bytearray | memoryview) -> DigestChain:
        if not isinstance(part, bytes):
            part = ffi.from_buffer(memoryview(part).cast("B"))
        lib.hash_chain(self._state, part, 1, len(part))
        return self

    def absorb_packed(
        self, parts: bytes | bytearray | memoryview, length: int = DIGEST_LEN
    ) -> DigestChain:
        """Absorb `len(parts) // length` parts of `length` bytes each."""
        view = memoryview(parts).cast("B")
        if len(view) % length:
            raise ValueError(f"{len(view)} bytes are not whole parts of {length}")
        if view:
            lib.hash_chain(
                self._state, ffi.from_buffer(view), len(view) // length, length
            )
        return self

    def digest(self) -> bytes:
        return bytes(ffi.buffer(self._state))


def chain_digest(parts: Iterable[bytes]) -> bytes:
    """The chained digest of already-digested `parts`, from a fixed seed."""
    chain = DigestChain()
    for part in parts:
        chain.absorb(part)
    return chain.digest()


def bytes_digest(value: bytes | bytearray | memoryview) -> bytes:
    return hash_bytes(b"bytes" + bytes(value))


def int_digest(value: int) -> bytes:
    """Of an integer of any size, by its two's-complement little-endian
    bytes, in the fewest that hold it."""
    length = (value.bit_length() + 8) // 8 or 1
    return hash_bytes(b"int" + value.to_bytes(length, "little", signed=True))


def str_digest(value: str) -> bytes:
    return hash_bytes(b"str" + value.encode())
