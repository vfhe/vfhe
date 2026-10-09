<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.mlwe

LWE / Module-LWE and MGSW over the `vfhe.arith` ring.

- `lwe.py`: `LWE_Key` / `LWE`: key generation (Gaussian, sparse-ternary),
  encryption, linear decryption, and coefficient extraction.
- `mlwe.py`: `MLWE_Scheme` / `MLWE` / `MLWE_Key` / `MLWE_Set`: module-LWE
  encryption and linear decryption, homomorphic add / sub / scalar and polynomial
  multiplication, BV and GHS key-switching, automorphisms, trace, packing
  key-switch, ring switching, and relinearized ciphertext multiplication.
- `mgsw.py`: `MGSW_Scheme` / `MGSW` with the external product and the
  `CMUX` / `NCMUX` gates that the bootstraps in `vfhe.fhe` build on.

Key generation -- key-switch keys, MGSW encryptions, and the bootstrapping
keys `vfhe.fhe` makes of them -- samples through one kernel,
`mlwe_RNS_sample_scaled_batch` (`mlwe.c`, behind `MLWE_Scheme.sample_scaled`),
on up to `n_threads` threads (0: the library limit). Each sample is drawn from
seeds of its own, taken in order on the calling thread, so the keys do not
depend on the thread count.

`c/src/` holds the kernels -- samples (`mlwe.c`), key switching
(`keyswitch.c`), the gadget (`gadget_rns.c`), MGSW (`mgsw.c`) and LWE
(`lwe.c`), with the parts that reach into RNS residues in the `_rns` files;
`python/cdef/mlwe.cdef` declares their ABI.
