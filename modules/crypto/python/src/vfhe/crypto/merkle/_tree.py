# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`Merkle`, the tree."""

from __future__ import annotations

from vfhe.crypto.hash._digest import DIGEST_LEN, Bytes
from vfhe.util.bindings import ffi, lib


class Merkle:
    """A binary Merkle tree over BLAKE3 leaf digests: `root`, `open`, and the static `verify`.

    The tree is built on construction, in C, and holds every level, so
    openings cost no rebuild. An internal node is the digest of its left
    child followed by its right. A leaf count that is not a power of two is
    padded with zero digests up to the next one, which makes the root well
    defined for any size; the leaf count is public protocol data, so the
    padding needs no domain separation. An opening is the sibling digests
    along the leaf's path, from the leaf's level up to the root's children,
    packed in one `bytes`; the leaf index is no part of it, since a verifier
    checks the position it queried itself. A tree belongs to one thread at a
    time.
    """

    __slots__ = ("_height", "_obj", "_size", "root")

    def __init__(self, digests: Bytes) -> None:
        """Builds the tree.

        Args:
            digests: The leaf digests, one or more `DIGEST_LEN`-byte digests
                packed end to end, in leaf order. Read once.

        Raises:
            ValueError: `digests` is empty or its length is no multiple of
                `DIGEST_LEN`.
        """
        if len(digests) == 0 or len(digests) % DIGEST_LEN:
            raise ValueError(
                f"digests must be one or more whole digests of {DIGEST_LEN} bytes"
            )
        self._size = len(digests) // DIGEST_LEN
        self._height = (
            self._size - 1
        ).bit_length()  # ceil(log2(size)): siblings per opening
        self._obj = ffi.gc(lib.merkle_new(self._size), lib.merkle_free)
        lib.merkle_commit(self._obj, ffi.from_buffer("uint8_t[]", digests))
        out = ffi.new("uint8_t[]", DIGEST_LEN)
        lib.merkle_get_root(out, self._obj)
        #: The root: the `DIGEST_LEN`-byte commitment to every leaf.
        self.root = bytes(out)

    def __len__(self) -> int:
        """Returns the number of leaves committed to, before padding."""
        return self._size

    def open(self, index: int) -> bytes:
        """Opens one leaf.

        Args:
            index: The leaf, 0 to ``len(self) - 1``.

        Returns:
            The opening: ``ceil(log2(len(self)))`` sibling digests packed end
            to end, the leaf's sibling first and the root's child last.

        Raises:
            IndexError: `index` is negative or past the last leaf.
        """
        if not 0 <= index < self._size:
            raise IndexError(f"leaf index {index} out of range for {self._size} leaves")
        out = ffi.new("uint8_t[]", DIGEST_LEN * self._height)
        lib.merkle_open(out, self._obj, index)
        return bytes(ffi.buffer(out))

    @staticmethod
    def verify(root: Bytes, index: int, path: Bytes, digest: Bytes) -> bool:
        """Checks an opening against a root, with no tree at hand.

        Recomputes the root from the leaf digest and its opening, pairing
        each sibling on the side the bits of `index` say, and compares. A
        local check: no challenges, no rounds. Thread-safe, no state.

        Args:
            root: The `DIGEST_LEN`-byte root the leaf is claimed under.
            index: The leaf position the verifier queried.
            path: The opening, as `open` returns it: whole digests, as many
                as the tree is high.
            digest: The `DIGEST_LEN`-byte digest of the leaf.

        Returns:
            True when the leaf with `digest` sits at `index` under `root`.

        Raises:
            ValueError: `root` or `digest` is not `DIGEST_LEN` bytes, or the
                length of `path` is no multiple of `DIGEST_LEN`.
        """
        if (
            len(root) != DIGEST_LEN
            or len(digest) != DIGEST_LEN
            or len(path) % DIGEST_LEN
        ):
            raise ValueError(
                f"root and digest are {DIGEST_LEN} bytes, path whole digests"
            )
        return bool(
            lib.merkle_verify(
                ffi.from_buffer(root),
                index,
                ffi.from_buffer(path),
                len(path) // DIGEST_LEN,
                ffi.from_buffer(digest),
            )
        )
