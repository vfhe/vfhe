<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.arith

RNS polynomial arithmetic over `Z_q[X]/(X^N+1)`: the compute engine every
other module builds on.

- `c/src/`, the kernels — incomplete (negacyclic) NTT with a configurable
  split degree, RNS/CRT limb arithmetic, base conversion / rescaling across the
  modulus tower, the CKKS complex FFT encoder, and multiprecision reconstruction.
- `c/test/`, C unit tests for the modular arithmetic, the NTT, and number
  theory.
- `python/src/vfhe/arith/`, the Python API over the cffi boundary:
  - `Ring` / `Polynomial` (`polynomial.py`): construct rings, sample, add /
    multiply, convert between coefficient and NTT domains, apply automorphisms,
    lift / rescale across quotient rings.
  - `ComplexRing` / `ComplexPolynomial` (`complex.py`): CKKS encode/decode FFT.
  - `Multiprecision` (`multiprecision.py`): big-integer/RNS bridge.
  - `number_theory.py` / `residue_selection.py`: pure-Python primality, CRT,
    and RNS prime selection.
  - `MLE` / `SparseMLE` (`mle.py`): multilinear extensions over any of the
    above. `MLE` is a dense table of `2^n` entries carrying two orthogonal
    properties — its `basis` (`MLE_Basis.eval` for hypercube evaluations,
    `MLE_Basis.coeff` for monomial coefficients) and its coefficient type
    (with a `ring` or `field`, entries are `Polynomial` / a `FieldVector` and
    the `mle_dense_poly_*` C kernels do the work; without one, plain Python
    values folded in Python). Supports add / sub / scale and
    variable-by-variable evaluation at concrete points; variables are plain
    identifiers (`MLE_Variable` or any hashable) and may be bound in any
    order, binding dispatching on the variable's position to the best pair
    layout (adjacent pairs for the LSB, table halves for the MSB, a strided
    generic fallback in between). `SparseMLE` is the unrelated sparse map of
    evaluations: add / sub / scale only, `evaluate` raises. The layer is
    asyncio-free; a consumer's unresolved protocol values are its own concern.

`python/cdef/arith.cdef` declares the C ABI Python calls (opaque handles plus a
few structs cdef'd for field access); `python/cdef/mle.cdef` the dense-MLE
kernels of `c/src/mle.c`.
