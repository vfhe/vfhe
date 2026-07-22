# vfhe.mlwe

LWE / Module-LWE and MGSW over the `vfhe.arith` ring.

- `lwe.py`: `LWE_Key` / `LWE`: key generation (Gaussian, sparse-ternary),
  encryption, phase, and coefficient extraction.
- `mlwe.py`: `MLWE_Scheme` / `MLWE` / `MLWE_Key` / `MLWE_Set`: module-LWE
  encryption and phase, homomorphic add / sub / scalar and polynomial
  multiplication, BV and GHS key-switching, automorphisms, trace, packing
  key-switch, and relinearized ciphertext multiplication.
- `mgsw.py`: `MGSW_Scheme` / `MGSW` with the external product and the
  `CMUX` / `NCMUX` gates that the bootstraps in `vfhe.fhe` build on.

`c/src/` holds the kernels (`lwe.c`, `mlwe.c`, `mgsw.c`); `python/cdef/mlwe.cdef`
declares their ABI.
