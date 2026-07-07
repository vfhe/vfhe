# vfhe.fhe

FHE schemes built on `vfhe.mlwe`.

- `ckks.py` — `CKKS_Scheme` / `CKKS_Ciphertext`: encode/decode complex vectors,
  encrypt/decrypt, slot rotation, rescale, and ciphertext × ciphertext /
  ciphertext × plaintext multiplication with relinearization.
- `cggi16.py` — `CGGI16`: LUT packing, blind rotation (CMUX), and the
  functional bootstrap with LWE extraction.
- `gp25.py` — `GP25`: the sparse-amortized bootstrap (sparse-ternary key,
  blind rotate over the `gp25_*` kernels, packing / trace repacking).

`c/src/` holds `bfv.c` and the GP25 bootstrap kernels (`gp25.c`);
`python/cdef/fhe.cdef` declares the GP25 ABI (CKKS and CGGI16 reuse the arith +
mlwe surfaces).
