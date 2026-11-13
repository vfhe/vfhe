<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.fhe

FHE schemes built on `vfhe.mlwe`.

## Layout

- `python/src/vfhe/fhe/bfv.py`: `BFV_Scheme`: batched encoding into `N` integer
  slots, encrypt/decrypt, plaintext and ciphertext multiplication with
  relinearization, slot rotation and conjugation, and modulus switching. The
  plaintext modulus is the product of the ciphertext ring's lowest primes, so
  `Delta = q/t` is exact and a ciphertext needs no scheme-specific state: BFV
  ciphertexts are plain `mlwe.MLWE` samples.
- `python/src/vfhe/fhe/ckks.py`: `CKKS_Scheme` / `CKKS_Ciphertext`:
  encode/decode complex vectors (one at a time or batched), encrypt/decrypt,
  slot rotation and conjugation, rescale, linear combinations with plaintexts,
  and ciphertext-ciphertext / ciphertext-plaintext multiplication with
  relinearization, batched and as a product tree.
- `python/src/vfhe/fhe/linear_transform.py`: `CKKS_LinearTransform`: a linear
  map on CKKS slots by baby-step giant-step [HS18], including SlotToCoeff and
  CoeffToSlot.
- `python/src/vfhe/fhe/cggi16.py`: `CGGI16`: LUT packing, blind rotation, and
  the functional bootstrap with LWE extraction, single (on the library's
  threads) or batched (one per thread). The blind rotation can be unfolded,
  each step consuming `unfolding` key coefficients [ZYL+17] with 2^u - 1 keys
  [BMMP18]; it needs a binary input key.
- `python/src/vfhe/fhe/gp25.py`: `GP25`: the sparse-amortized bootstrap
  [GP25]: every coefficient of an RLWE sample over `R_n` at once, for a sparse
  input key (ternary or binary, any rank), rotating over `R_N` for any `n` and
  `N`. Input keys with bounded gaps (`sample_input_key`), the blind rotation
  (`blind_rotate`, or `blind_rotate_exponents` on exponents already in
  `Z_2N`) with one or two key bits per layer [LWZ+26], its building blocks
  (`rotate`, `multiply_by_signed_monomials`), and repacking by the trace and a
  ring switch.
- `python/src/vfhe/fhe/ship.py`: `SHIP`: the SHIP bootstrap [CHKS25] of `N/2`
  real CKKS slots, for a sparse input key (`sample_input_key`).
- `python/src/vfhe/fhe/codec/`: the `vfhe.util.io` codecs for this package's
  schemes, ciphertexts, linear transforms and keys.
- `python/cdef/fhe.h`: what Python may call of the C below: the CGGI16, GP25
  and SHIP ABI (BFV and CKKS reuse the arith + mlwe surfaces).
- `c/include/fhe.h`: the prototypes.
- `c/src/`: `bfv.c` (an empty placeholder: the scheme needs no kernels of its
  own), the CGGI16 and GP25 blind rotations (`cggi16.c`, `gp25.c`) and the
  SHIP encoders (`ship.c`).

## Dependencies

`fhe` builds on `vfhe.mlwe`, and through it on `vfhe.arith` and
`vfhe.crypto`. Nothing in the library builds on it.
