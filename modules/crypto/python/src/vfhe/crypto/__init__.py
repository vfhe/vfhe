# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
# vfhe.crypto public API re-exports: the basic cryptographic primitives —
# randomness, hashing and vector commitment — every other module builds on.
from vfhe.util.io import CHECKSUM_BLAKE3, register_checksum

from .hash import Blake3Stream
from .merkle import DIGEST_LEN, Merkle, MerklePath, hash_bytes, leaf_digest
from .prng import SEED_WORDS, EntropyPRNG, SeededPRNG, entropy, seeded

# vfhe.util.io checksums its records with BLAKE3 once this package is loaded.
register_checksum(CHECKSUM_BLAKE3, Blake3Stream)

__all__ = [
    "DIGEST_LEN",
    "SEED_WORDS",
    "Blake3Stream",
    "EntropyPRNG",
    "Merkle",
    "MerklePath",
    "SeededPRNG",
    "entropy",
    "hash_bytes",
    "leaf_digest",
    "seeded",
]
