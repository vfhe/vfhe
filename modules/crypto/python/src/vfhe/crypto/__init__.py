# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The basic cryptographic primitives every other module builds on.

- `random` is the randomness: `Random`, the source a scheme's samplers draw
  from, the entropy stream or a seeded stream, and `seeded_below_many`, values
  another party recomputes from a transcript. Nothing else in vfhe draws
  randomness from anywhere.
- `hash` is BLAKE3: `digest` of a byte string and the chain `DigestChain`;
  every digest is `DIGEST_LEN` bytes.
- `merkle` is the vector commitment: the tree `Merkle` and `join_digests`.

Importing `vfhe.crypto` imports none of them; import the one you use:

>>> from vfhe.crypto.random import Random
"""
