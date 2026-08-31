<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.crypto

Randomness, and nothing else. No Python surface: the kernels are its only
callers, so it ships a header and no cdef.

- `c/src/prng.c`: the byte source. Seeds from RDRAND or `/dev/urandom` and
  expands with BLAKE3, so this is the one module that depends on BLAKE3.
- `c/src/aes_rng.c`: an AES-NI CTR stream, used in place of BLAKE3 where the
  CPU offers AES and the build is not portable.
- `c/src/normal.c`: the Box-Muller draw, beside the bytes it consumes.

`prng.c` also carries the deterministic-seed override a probabilistic test needs.
Production never calls it, and no `.cdef` declares it, so it stays out of every
installed engine's public ABI.
