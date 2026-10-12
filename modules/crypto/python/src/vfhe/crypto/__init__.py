# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
# vfhe.crypto public API re-exports: the basic cryptographic primitives —
# randomness, hashing and vector commitment — every other module builds on.
from .hash import Blake3Stream
from .merkle import DIGEST_LEN, Merkle, MerklePath, hash_bytes, leaf_digest
from .prng import SEED_WORDS, EntropyPRNG, SeededPRNG, entropy, seeded

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
