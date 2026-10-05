<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.fhe

FHE schemes built on `vfhe.mlwe`.

- `bfv.py`: `BFV_Scheme`: batched encoding into `N` integer slots,
  encrypt/decrypt, plaintext and ciphertext multiplication with
  relinearization, slot rotation and conjugation, and modulus switching. The
  plaintext modulus is the product of the ciphertext ring's lowest primes, so
  `Delta = q/t` is exact and a ciphertext needs no scheme-specific state: BFV
  ciphertexts are plain `mlwe.MLWE` samples.
- `ckks.py`: `CKKS_Scheme` / `CKKS_Ciphertext`: encode/decode complex vectors,
  encrypt/decrypt, slot rotation, rescale, and ciphertext-ciphertext /
  ciphertext-plaintext multiplication with relinearization.
- `cggi16.py`: `CGGI16`: LUT packing, blind rotation, and the functional
  bootstrap with LWE extraction, single or batched. The blind rotation can be
  unfolded over groups of `unfolding` key coefficients, one external product
  per group [ZYL+17], with the 2^u - 1 key of [BMMP18] (default) or the 2^u
  key of [ZYL+17]; it needs a binary input key.
- `gp25.py`: `GP25`: the sparse-amortized bootstrap (sparse-ternary key,
  blind rotate over the `gp25_*` kernels, packing / trace repacking).

`c/src/` holds `bfv.c` (an empty placeholder: the scheme needs no kernels of
its own), the CGGI16 blind rotation (prototyped in `c/include/fhe.h`:
`cggi16.c` runs the loop sequentially, pipelined across threads, or as a batch
over ciphertexts, `cggi16_team.c` data-parallel across threads, and
`zyl17.c` holds the unfolding's groups and key combinations) and the GP25
bootstrap kernels (`gp25.c`);
`python/cdef/fhe.cdef` declares the CGGI16 and GP25 ABI (BFV and CKKS reuse
the arith + mlwe surfaces).
