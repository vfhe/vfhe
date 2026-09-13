# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Binary Merkle trees over BLAKE3.

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

    The path is held as **one buffer**, not a digest object per sibling.
    That is the layout `merkle_open` writes and the layout `merkle_verify`
    reads, so both ends of a query answer touch it whole; cutting it into
    `log2(size)` objects and validating each, only to join them again on the
    way into the verifier, is work no one asked for. A caller that does want
    them apart reads `siblings`, which builds the tuple then.
    """

    __slots__ = ("_packed", "_siblings")

    def __init__(
        self,
        siblings: Sequence[bytes] | None = None,
        *,
        packed: bytes | None = None,
    ):
        """A path from the sibling digests, or from them already packed end
        to end. Exactly one of the two; `from_bytes` names the second."""
        if (siblings is None) == (packed is None):
            raise TypeError("a path takes either siblings or a packed buffer")
        if packed is not None:
            if len(packed) % DIGEST_LEN:
                raise ValueError(
                    f"a packed path must be a multiple of {DIGEST_LEN} bytes, "
                    f"got {len(packed)}"
                )
            self._packed = bytes(packed)
        else:
            self._packed = b"".join(_as_digest(s) for s in siblings or ())
        self._siblings: tuple[bytes, ...] | None = None

    @classmethod
    def from_bytes(cls, packed: bytes) -> MerklePath:
        """A path from the siblings already packed end to end -- what
        `merkle_open` produces, taken without being cut up."""
        return cls(packed=packed)

    @property
    def siblings(self) -> tuple[bytes, ...]:
        """The sibling digests, one object each -- cut from the buffer on
        first use, since most callers only ever hand the whole path on."""
        if self._siblings is None:
            self._siblings = tuple(
                self._packed[i : i + DIGEST_LEN]
                for i in range(0, len(self._packed), DIGEST_LEN)
            )
        return self._siblings

    def __len__(self) -> int:
        return len(self._packed) // DIGEST_LEN

    def to_bytes(self) -> bytes:
        """The siblings concatenated, the layout the C verifier reads."""
        return self._packed

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

    def __init__(
        self,
        leaves: Sequence | None = None,
        hash: Callable | None = None,
        *,
        digests: bytes | None = None,
    ):
        """A tree over `leaves`, or over `digests` already packed end to end.

        Exactly one of the two. `from_digests` is the name for the second,
        and says why it exists.
        """
        if (leaves is None) == (digests is None):
            raise TypeError("a Merkle tree takes either leaves or packed digests")
        if digests is not None:
            if len(digests) == 0 or len(digests) % DIGEST_LEN:
                raise ValueError(
                    f"packed digests must be a non-zero multiple of {DIGEST_LEN} "
                    f"bytes, got {len(digests)}"
                )
            self.leaves = None
            self.hash = None
            self._packed: bytes | None = bytes(digests)
            self._size = len(digests) // DIGEST_LEN
        else:
            self.leaves = list(leaves)  # type: ignore[arg-type]
            if not self.leaves:
                raise ValueError("a Merkle tree needs at least one leaf")
            self.hash = hash
            self._packed = None
            self._size = len(self.leaves)
        self.log_size = (self._size - 1).bit_length()
        self.obj = lib.merkle_new(self._size)
        self.root = self.commit()

    @classmethod
    def from_digests(cls, digests: bytes) -> Merkle:
        """A tree over leaf digests already packed end to end.

        `digests` is `DIGEST_LEN` bytes per leaf, contiguous -- the layout the
        C tree builder reads, so **nothing is materialised per leaf**. The
        ordinary constructor holds a Python object per leaf and calls a hash
        through it; past a few hundred thousand leaves that object churn, not
        the hashing and not the tree, is most of what committing costs.

        The tree keeps the buffer, so `commit()` still works. It has no
        `leaves` list, since not building one is the point -- `open` and
        `verify` need only the digests and the index.
        """
        return cls(digests=digests)

    def __del__(self) -> None:
        # interpreter shutdown may already have torn the lib down
        with contextlib.suppress(Exception):
            lib.merkle_free(self.obj)

    def __len__(self) -> int:
        return self._size

    def commit(self) -> bytes:
        """(Re)build the tree from the current leaves and return the root."""
        if self._packed is not None:
            digests = self._packed
        elif self.leaves is not None:
            digests = b"".join(leaf_digest(leaf, self.hash) for leaf in self.leaves)
        else:  # __init__ sets exactly one of the two, so this cannot happen
            raise RuntimeError("a Merkle tree with neither leaves nor digests")
        lib.merkle_commit(self.obj, digests)
        out = ffi.new("uint8_t[]", DIGEST_LEN)
        lib.merkle_get_root(out, self.obj)
        self.root = bytes(out)
        return self.root

    def open(self, index: int) -> MerklePath:
        """The opening of the `index`-th leaf."""
        if not 0 <= index < self._size:
            raise IndexError(f"leaf index {index} out of range for {len(self)} leaves")
        if self.log_size == 0:  # a single leaf: the root is its digest
            return MerklePath([])
        out = ffi.new("uint8_t[]", DIGEST_LEN * self.log_size)
        lib.merkle_open(out, self.obj, index)
        return MerklePath.from_bytes(ffi.buffer(out))

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
