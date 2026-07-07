# vfhe.piop

Multilinear extensions and the interactive-oracle-proof scaffolding.

- `mle.py` — `MLE` and its forms: `ML_Polynomial` (monomial coefficients),
  `MLE_Sparse` (sparse evaluations), and `MLE_Dense` (dense vectors of
  `vfhe.arith.Polynomial`, backed by the `mle_dense_poly_*` C kernels). Supports
  add / sub / scale and variable-by-variable evaluation, including async
  evaluation against unresolved `IOPValue` futures.
- `piop.py` — the IOP primitives: `IOPValue` / `IOPVariable` (asyncio futures)
  and `IOPParty` / `IOPProver` / `IOPVerifier`.

`c/src/mle.c` holds the dense-MLE kernels; `python/cdef/piop.cdef` declares them.
