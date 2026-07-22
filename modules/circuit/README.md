# vfhe.circuit

Layered GKR arithmetic-circuit representation and its polynomial export.

- `proto/vfhe/circuit/gkr/v1/gkr.proto`: the language-neutral wire format
  (layers of fan-in-2 ADD/MUL gates over a declared field modulus).
- `python/.../builder.py`: construct and (de)serialize circuits.
- `python/.../export.py`: the bridge to `vfhe.arith`: evaluates circuits,
  builds the dense MLE tables GKR reasons about (per-layer wire values `W_l`
  and wiring predicates `add_l` / `mul_l`), evaluates MLEs at arbitrary
  points (`mle_eval`, the sumcheck verifier's primitive), and packs tables
  into `vfhe.arith.Polynomial` coefficients for the polynomial-commitment
  layers.

The arith dependency is imported lazily inside the two packing functions, so
the circuit representation itself stays usable without the native extension.
The dense-MLE export is exponential in layer bit-widths by nature; sparse
representations for production-size circuits can be added behind the same API
(the multilinear-extension machinery lives in `vfhe.piop`).
