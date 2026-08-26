# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Binary Merkle trees over BLAKE3 (piop.md §7).

A vector commitment: `Merkle(leaves)` commits to a list of arbitrary Python
objects, `open(index)` produces the sibling path of one leaf, and the static
`verify` replays that path against the root. The tree lives in C
(`c/src/merkle.c`) - this layer only turns leaves into digests and moves
32-byte buffers across the cffi boundary, so the per-leaf Python cost is one
`.hash()` call.

Leaf hashing is the one thing a leaf type must provide: a `.hash()` method
returning the leaf's digest (`vfhe.arith.FieldElement` has one), or a
`hash=` callable passed to `Merkle` / `Merkle.verify` for types that do not
(e.g. `hash=lambda p: p.get_hash()`). Nothing else about a leaf is assumed -
in particular the tree never keeps a copy of one, only a reference.

This module is a general-purpose primitive with no PIOP-specific content; it
lives here until the library grows a module for basic crypto primitives.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

from vfhe.engine import ffi, lib

# Node size, in bytes: BLAKE3's default output length.
DIGEST_LEN = 32


def hash_bytes(data: bytes) -> bytes:
    """The BLAKE3 digest of `data`, the hash function the tree is built from.

    Useful as the `hash=` argument for leaves that are already byte strings,
    and as the reference hash when checking a root by hand.
    """
    out = ffi.new("uint8_t[]", DIGEST_LEN)
    lib.merkle_hash(out, data, len(data))
    return bytes(out)


def _as_digest(value) -> bytes:
    """Normalize a hash value to `DIGEST_LEN` raw bytes.

    Bytes-like values pass through (what `FieldElement.hash()` returns); a
    sequence of four ints is packed little-endian, so the library's other
    hash producer, an element's `get_hash()` (a `uint64_t[4]` read back as
    ints), also works as a `hash=` callable.
    """
    if isinstance(value, bytes | bytearray | memoryview):
        digest = bytes(value)
    elif isinstance(value, Sequence) and len(value) == DIGEST_LEN // 8:
        digest = b"".join(int(word).to_bytes(8, "little") for word in value)
    else:
        raise TypeError(
            f"a hash must be {DIGEST_LEN} bytes or {DIGEST_LEN // 8} 64-bit "
            f"words, got {type(value).__name__}"
        )
    if len(digest) != DIGEST_LEN:
        raise ValueError(f"a hash must be {DIGEST_LEN} bytes, got {len(digest)}")
    return digest


def leaf_digest(leaf, hash: Callable | None = None) -> bytes:
    """The digest of one leaf: `hash(leaf)` if given, else `leaf.hash()`."""
    if hash is not None:
        return _as_digest(hash(leaf))
    try:
        leaf_hash = leaf.hash
    except AttributeError:
        raise TypeError(
            f"leaf of type {type(leaf).__name__} has no .hash() method; pass an "
            "explicit hash= callable"
        ) from None
    return _as_digest(leaf_hash())


class MerklePath:
    """A Merkle opening: the sibling digests along one leaf's path, bottom-up.

    `siblings[i]` is the sibling of the path node at height `i`, so the list
    is `log2(size)` long. The leaf index is deliberately *not* part of the
    path: a verifier checks the position it queried itself, never one the
    prover sent.
    """

    __slots__ = ("siblings",)

    def __init__(self, siblings: Sequence[bytes]):
        self.siblings = tuple(_as_digest(s) for s in siblings)

    def __len__(self) -> int:
        return len(self.siblings)

    def to_bytes(self) -> bytes:
        """The siblings concatenated, the layout the C verifier reads."""
        return b"".join(self.siblings)

    def __repr__(self) -> str:
        return f"MerklePath({[s.hex() for s in self.siblings]})"


class Merkle:
    """A binary Merkle tree committing to `leaves`, built on construction.

    The root is `self.root`; `open(index)` opens a leaf and the static
    `verify(root, index, path, leaf)` checks an opening against a root. Each
    leaf must expose `.hash()`, or `hash=` must supply the leaf hash for it
    (see the module docstring). A leaf count that is not a power of two is
    padded with zero digests.

    The same object may be recommitted: mutate `leaves` in place (keeping its
    length) and call `commit()` again - the C tree allocation is reused.
    """

    def __init__(self, leaves: Sequence, hash: Callable | None = None):
        self.leaves = list(leaves)
        if not self.leaves:
            raise ValueError("a Merkle tree needs at least one leaf")
        self.hash = hash
        self.log_size = (len(self.leaves) - 1).bit_length()
        self.obj = lib.merkle_new(len(self.leaves))
        self.root = self.commit()

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            lib.merkle_free(self.obj)

    def __len__(self) -> int:
        return len(self.leaves)

    def commit(self) -> bytes:
        """(Re)build the tree from the current leaves and return the root."""
        digests = b"".join(leaf_digest(leaf, self.hash) for leaf in self.leaves)
        lib.merkle_commit(self.obj, digests)
        out = ffi.new("uint8_t[]", DIGEST_LEN)
        lib.merkle_get_root(out, self.obj)
        self.root = bytes(out)
        return self.root

    def open(self, index: int) -> MerklePath:
        """The opening of the `index`-th leaf."""
        if not 0 <= index < len(self.leaves):
            raise IndexError(f"leaf index {index} out of range for {len(self)} leaves")
        if self.log_size == 0:  # a single leaf: the root is its digest
            return MerklePath([])
        out = ffi.new("uint8_t[]", DIGEST_LEN * self.log_size)
        lib.merkle_open(out, self.obj, index)
        path = bytes(out)
        return MerklePath(
            [path[i * DIGEST_LEN : (i + 1) * DIGEST_LEN] for i in range(self.log_size)]
        )

    @staticmethod
    def verify(
        root: bytes,
        index: int,
        path: MerklePath,
        leaf,
        hash: Callable | None = None,
    ) -> bool:
        """Whether `leaf` sits at position `index` of the tree with `root`."""
        return bool(
            lib.merkle_verify(
                _as_digest(root),
                index,
                path.to_bytes(),
                len(path),
                leaf_digest(leaf, hash),
            )
        )
