# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`DigestChain`: the chain ``h_i = H(h_{i-1} | part_i)``, accumulated in native memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.util.bindings import ffi, lib

from ._digest import DIGEST_LEN, Bytes, digest

if TYPE_CHECKING:
    from collections.abc import Iterable

_CHAIN_SEED = digest(b"vfhe.chain")


class DigestChain:
    """Accumulates the chain ``h_i = H(h_{i-1} | part_i)`` from a fixed seed, in native memory.

    The BCS sigma-chain [BCS16] over already-digested parts: ``h_0`` is
    ``H("vfhe.chain")`` and each part replaces the state by the digest of the
    state followed by the part, so a sequence can cache the state after a
    prefix and extend it. `absorb` takes one part of any length;
    `absorb_packed` takes digests laid out end to end, in one native call.
    Both return the chain, so calls read in sequence;
    ``DigestChain(parts).digest()`` is the chain of `parts` in one
    expression. A chain belongs to one thread at a time.
    """

    __slots__ = ("_state",)

    def __init__(self, parts: Iterable[Bytes] = ()) -> None:
        """Starts the chain at the seed and absorbs `parts` in order.

        Args:
            parts: Byte strings of any length, absorbed one by one.
        """
        self._state = ffi.new(f"uint8_t[{DIGEST_LEN}]", _CHAIN_SEED)
        for part in parts:
            self.absorb(part)

    def absorb(self, part: Bytes) -> DigestChain:
        """Absorbs one part.

        Args:
            part: Bytes of any length.

        Returns:
            This chain, advanced.
        """
        view = memoryview(part).cast("B")
        lib.hash_blake3_chain(self._state, ffi.from_buffer(view), 1, len(view))
        return self

    def absorb_packed(self, digests: Bytes) -> DigestChain:
        """Absorbs digests packed end to end, each as one part, in one native call.

        Args:
            digests: A whole number of `DIGEST_LEN`-byte digests, in order.

        Returns:
            This chain, advanced once per digest.

        Raises:
            ValueError: The length of `digests` is no multiple of `DIGEST_LEN`.
        """
        view = memoryview(digests).cast("B")
        if len(view) % DIGEST_LEN:
            raise ValueError(f"{len(view)} bytes are not whole digests of {DIGEST_LEN}")
        if view:
            lib.hash_blake3_chain(
                self._state, ffi.from_buffer(view), len(view) // DIGEST_LEN, DIGEST_LEN
            )
        return self

    def digest(self) -> bytes:
        """Reads the state: the digest of every part absorbed so far.

        Returns:
            The `DIGEST_LEN`-byte state; the seed when nothing was absorbed.
        """
        return bytes(ffi.buffer(self._state))
