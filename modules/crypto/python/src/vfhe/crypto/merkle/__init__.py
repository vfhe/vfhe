# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The vector commitment: `Merkle`, a binary tree over packed BLAKE3 leaf digests, and `join_digests`.

A tree commits to leaf digests packed end to end, `DIGEST_LEN` bytes each; an
opening is the sibling digests along the leaf's path, packed the same way,
and a verifier checks it against the root alone. `join_digests` hashes
several packed arrays together index by index, so one tree commits to all of
them.

>>> from vfhe.crypto.hash import digest
>>> from vfhe.crypto.merkle import Merkle
>>> digests = b"".join(digest(leaf) for leaf in (b"a", b"b", b"c"))
>>> tree = Merkle(digests)
>>> Merkle.verify(tree.root, 1, tree.open(1), digest(b"b"))
True
"""

from ._join import join_digests
from ._tree import Merkle

__all__ = ["Merkle", "join_digests"]
