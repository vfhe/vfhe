# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BLAKE3 hashing: `digest` of a byte string and the chain `DigestChain`.

Every digest is `DIGEST_LEN` bytes. `DigestChain` accumulates
``h_i = H(h_{i-1} | part_i)`` from a fixed seed. How a value of another type
becomes bytes, and how types are kept apart, is the caller's encoding.

>>> from vfhe.crypto.hash import DigestChain, digest
>>> digest(b"").hex()[:8]
'af1349b9'
>>> DigestChain([b"a", b"b"]).digest() == DigestChain().absorb(b"a").absorb(b"b").digest()
True
"""

from ._chain import DigestChain
from ._digest import DIGEST_LEN, digest

__all__ = ["DIGEST_LEN", "DigestChain", "digest"]
