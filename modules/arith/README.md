<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.arith

RNS polynomial arithmetic over `Z_q[X]/(X^N+1)`, finite fields and their
vectors, and multilinear extensions over any of them: the compute engine every
other module builds on.

Sampling names follow one rule: on a ring or a field, `random_*` returns a
new element (`Ring.random_element`, `Ring.random_exceptional`,
`Field.random_element`, `Field.random_exceptional`); on an element or a
vector, `sample_*` overwrites it in place and returns it
(`Polynomial.sample_uniform`, `sample_gaussian`, `sample_exceptional`,
`FieldVector.sample_random`). Every `random_*` sampler takes a
`random=` source (`vfhe.crypto.random.Random`), the entropy stream by default. The exceptional samplers take an optional
`seed`, from which the element is derived; without one, a fresh seed is
drawn.

## Layout

- `python/src/vfhe/arith/base.py`: the contract every arithmetic parent
  satisfies, and what follows from it: `Ring` / `Polynomial` (construct rings,
  sample, add / multiply, convert between coefficient and NTT domains, apply
  automorphisms, lift / rescale across quotient rings, and re-express a value
  over a base that is not one of them, `convert_base`), `Field` /
  `FieldElement` / `FieldVector`, and `ArithParent` above them.
- `python/src/vfhe/arith/impl/`: one subpackage per implementation, each
  owning its parent and element classes and the native state they cache:
  `rns/` (`RNSRing` / `RNSPolynomial`, the RNS base and representations),
  `complex/` (`ComplexRing` / `ComplexPolynomial`, the CKKS encode/decode
  FFT), `mp/` (`Multiprecision`, the big-integer/RNS bridge), `field/`
  (`ExtensionField`, its elements and vectors, and their NTT) and `pmf/`
  (`PseudoMersenneField`, its elements, vectors and NTT). Reach them through
  `vfhe.arith`, which re-exports them. `rns/cuda.py` is the ``cuda`` backend
  of `rns`: `CUDARing` / `CUDAPolynomial`, elements held on a CUDA device,
  registered where `vfhe.util.gpu` finds one.
- `python/src/vfhe/arith/spec.py`, `registry.py`: an arithmetic object as
  data, along the implementation and backend axes (`Spec`, `Domain`,
  `Capability`), and the registry that resolves a caller's parameters to one
  and holds the conversions between them.
- `python/src/vfhe/arith/state.py`: the process-global caches implementations
  keep, and `reset`, which empties them.
- `python/src/vfhe/arith/number_theory.py`, `residue_selection.py`:
  pure-Python primality, CRT, and RNS prime selection.
- `python/src/vfhe/arith/mle.py`: `MLE` / `SparseMLE`, multilinear extensions
  over any of the above. `MLE` is a dense table of `2^n` entries carrying two
  orthogonal properties -- its `basis` (`MLE_Basis.eval` for hypercube
  evaluations, `MLE_Basis.coeff` for monomial coefficients) and its
  coefficient type (with a `ring` or `field`, entries are `Polynomial` / a
  `FieldVector` and the `mle_dense_poly_*` C kernels do the work; without one,
  plain Python values folded in Python). Supports add / sub / scale and
  variable-by-variable evaluation at concrete points; variables are plain
  identifiers (`MLE_Variable` or any hashable) and may be bound in any
  order, binding dispatching on the variable's position to the best pair
  layout (adjacent pairs for the LSB, table halves for the MSB, a strided
  generic fallback in between). `SparseMLE` is the unrelated sparse map of
  evaluations: add / sub / scale only, `evaluate` raises. The layer is
  asyncio-free; a consumer's unresolved protocol values are its own concern.
- `python/src/vfhe/arith/codec/`: the `vfhe.util.io` codecs for RNS rings and
  polynomials, `RingCodec` and `PolynomialCodec`, and their row encoding.
- `python/src/vfhe/arith/_alloc.py`: 64-byte-aligned cffi buffers from the
  memory pool, the alignment the tuned kernels require.
- `python/cdef/arith.h`, `python/cdef/mle.h`, `python/cdef/arith_gpu.h`:
  what Python may call of the C below: opaque handles plus a few structs
  cdef'd for field access, the dense-MLE kernels, and the CUDA kernels.
- `c/include/arith.h`, `arith_generic.h`, `mle.h`, `arith_gpu.h`: the
  prototypes.
- `c/src/kernels/`: the incomplete (negacyclic) NTT with a configurable split
  degree and the modular arithmetic, per word width and engine, and the
  complex FFT.
- `c/src/rns/`, `c/src/mp/`, `c/src/complex/`, `c/src/field/`, `c/src/pmf/`:
  one directory per implementation: RNS/CRT limb arithmetic, base conversion
  / rescaling across the modulus tower and the polynomial I/O; multiprecision
  reconstruction; the CKKS encoder; extension-field and pseudo-Mersenne
  vectors with their transforms and digests.
- `c/src/cuda/`: the device element and its kernels, which `nvcc` compiles
  into the CUDA extension.
- `c/src/mle.c`, `c/src/number_theory.c`: the dense-MLE kernels and the
  number theory the rings are built with.
- `c/test/unit/`: C unit tests for the modular arithmetic, the NTT, and
  number theory.

## Dependencies

`arith` builds on `vfhe.util` (allocation, threads, the memory pool) and
`vfhe.crypto` (samples, seeds and digests). Every module above builds on it.
