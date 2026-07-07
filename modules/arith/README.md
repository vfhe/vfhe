# vfhe.arith

RNS polynomial arithmetic over `Z_q[X]/(X^N+1)` — the compute engine every
other module builds on.

- `c/src/` — the kernels: incomplete (negacyclic) NTT with a configurable
  split degree, RNS/CRT limb arithmetic, base conversion / rescaling across the
  modulus tower, the CKKS complex FFT encoder, and multiprecision reconstruction.
- `c/test/` — C unit tests for the modular arithmetic, NTT round-trips, NTT
  multiplication, and number theory.
- `python/src/.../` — the Python API over the cffi boundary:
  - `Ring` / `Polynomial` (`polynomial.py`) — construct rings, sample, add /
    multiply, convert between coefficient and NTT domains, apply automorphisms,
    lift / rescale across quotient rings.
  - `ComplexRing` / `ComplexPolynomial` (`complex.py`) — CKKS encode/decode FFT.
  - `Multiprecision` (`multiprecision.py`) — big-integer ↔ RNS bridge.
  - `number_theory.py` / `residue_selection.py` — pure-Python primality, CRT,
    and RNS prime selection.

`python/cdef/arith.cdef` declares the C ABI Python calls (opaque handles plus a
few structs cdef'd for field access).
