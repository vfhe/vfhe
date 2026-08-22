<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop

> **Under development.** The code is largely verified by the test suite, but
> the documentation text — this file and [piop.md](piop.md) — has not been
> reviewed yet and may be inaccurate or out of date.

Multilinear extensions and the interactive-oracle-proof scaffolding. The
architecture and its derivation from the PIOP literature are documented in
[piop.md](piop.md).

- `mle.py`: two unrelated types — `MLE` (a dense table of 2^n coefficients,
  what protocols use) and `SparseMLE` (a sparse map of evaluations;
  add / sub / scale only, `evaluate` raises).
  An MLE carries two orthogonal properties: its `basis` — `MLE_Basis.eval` (hypercube evaluations, constructor
  argument `evaluations=`) or `MLE_Basis.coeff` (monomial coefficients,
  `coefficients=`; `to_coefficients()` converts and returns an `MLE`)
  — and its coefficient type: with a `ring`, entries are
  `vfhe.arith.Polynomial` and the `mle_dense_poly_*` C kernels do the work;
  without one, entries are plain Python values (any type with `+`/`*`, e.g.
  ints) folded in Python.
  Supports add / sub / scale and variable-by-variable evaluation at concrete
  points; variables are plain identifiers (`MLE_Variable` or any hashable)
  and may be bound in any order — binding dispatches on the variable's
  position to the best pair layout (adjacent pairs for the LSB, table halves
  for the MSB, a strided generic fallback in between).
  `MLE.eq(ring, point)` builds the dense equality-polynomial table
  eq~(point, .). The layer is asyncio-free — unresolved protocol values are
  a Transcript / Statement concern (`Statement.resolved()`).
- `piop.py`: the IOP primitives: `IOP` (event loop, coefficient domain,
  transcript, parties, and the relation → protocol registry), `Transcript`
  (the ordered, labeled record of exchanged messages: `write(label, value)`
  / `await read(label)`; also its own Fiat-Shamir accumulator — `state()`
  is the chained prefix hash h_i = H(h_{i-1} | H(label_i) | digest(value_i)),
  seeded via `bind()` with the root statement's digest, with per-entry
  digests and chain values cached so interactive runs pay nothing and a new
  entry costs one hash; `element_digest` is the duck-typed walker digesting
  entry values and statements), `Value` / `Variable` (asyncio futures),
  `Party` / `Prover` / `Verifier` (worklist drivers; the IOP constructor
  instantiates both parties, taking optional subclasses;
  `iop.verifier.challenge(label)` is the single place challenges originate
  and either party may call it, with `challenge_bits(label, bits)` the
  second sampler — raw published coins that protocols expand into derived
  randomness, e.g. spot-check positions for a committed vector, which the
  prover must reproduce to answer with paths; a Fiat-Shamir verifier
  overrides both), `Protocol` (a reduction between products
  of relations, as paired prove/verify coroutines over statement bundles;
  `batching = True` protocols receive all frontier statements of their
  relation at once), and the relation / statement pair.
- `fs.py`: `FS_Verifier`, the Fiat-Shamir transform as a `Verifier` subtype
  overriding exactly the two draw hooks behind the samplers: each value is
  H(transcript.state() | tag | label), so it binds the root statement and
  every prior message. Enabled with `IOP(fiat_shamir=True)` (default
  `False`: interactive is the base model, FS the compiled artifact); an FS
  run is fully deterministic — same statement, same registry, byte-identical
  transcript. Ring challenges come from `ring_exceptional_from_seed`
  (deterministic constant-chunk exceptional elements) unless the domain
  provides its own `exceptional_from_seed`.
  With FS on, the two halves come apart: `iop.prove(statement)` runs the
  prover alone and returns a `Proof` (the prover messages only — challenges
  are recomputed, never carried), and a *separate* `iop.verify(statement,
  proof)` checks it with no prover, witnesses, or oracles present, reading
  the proof in a single forward pass and rejecting messages that arrive out
  of turn, run out, or are left over.
- `merkle.py`: `Merkle`, a binary Merkle tree over BLAKE3 — a vector
  commitment to a list of arbitrary Python objects (`root`, `open(index)`,
  static `verify(root, index, path, leaf)`; `MerklePath` is the sibling list
  of one opening, without the index, which the verifier supplies itself).
  The only requirement on a leaf type is a `.hash()` method returning its
  digest (`vfhe.arith.FieldElement` has one); a `hash=` callable supplies it
  for types that do not (`hash=Polynomial.get_hash`). Leaf hashing is the
  Python layer's only per-leaf work — the tree itself is built by
  `c/src/merkle.c`. A general-purpose primitive with nothing PIOP-specific
  about it: it lives here until the library grows a module for basic crypto
  primitives.
- `sumcheck.py`: `Sumcheck` (reduces `Relation_Sum` to `Relation_Eval` in
  `num_vars` rounds) and the Libra-style `SumcheckProd` (reduces
  `Relation_SumProd` to one `Relation_Eval` per factor). Round messages are
  the evaluations of the round polynomial at `0..deg` (Libra's format);
  both protocols work over any `MLE` oracle, in either basis and over
  either coefficient type.
  `iop.register(Relation_Sum, Sumcheck())` then `iop.run(statement)`.
  Both protocols share their round machinery (a template base class) and
  each body is written once: the native/pure-Python decision is made
  per call inside the round-message helpers — `Sumcheck.round_evals` /
  `SumcheckProd.prod_round_evals` delegate to the C kernels in
  `c/src/sumcheck.c` for native tables (for `SumcheckProd`: exactly two
  factors) and run pure Python otherwise, identical messages either way;
  the kernels come in pairs/halves/generic variants chosen by the round
  variable's location, like MLE binding. These helpers (plus
  `interpolate_evals`, the verifier's claim update) are also the building
  blocks for protocols that interleave sumcheck rounds with other messages
  (basefold in `vfhe.polycom`). Folding is MLE evaluation
  (`mle_dense_poly_evaluate`), not a sumcheck kernel; `supported_domains`
  (currently `(Ring,)`) is declarative metadata, not a dispatch switch.

A `Relation` is an indexed relation — a set of (index, instance, witness)
triples — declaring its instance shape (`fields`, in the canonical order a
Fiat-Shamir serialization will hash) and an ideal decider `check()`; the
toolbox currently has `Relation_Sum`, `Relation_SumProd`, `Relation_Zero`,
and `Relation_Eval` — whose instance may carry the polynomial as an
in-process oracle, as a commitment (optional field; instance data on the
statement, witness in `prover.witnesses[commitment]`), or both: left
terminal it is decided by one oracle query, and a registered PCS protocol
(basefold in `vfhe.polycom`) replaces that query with an opening argument. A `Statement` is the claim "instance ∈ L(relation)": a thin public
record whose fields come from its relation — the witness stays with
`Prover.witnesses`. Protocol steps reduce statement bundles to statement
bundles, linked through `parents` (the statements of a run form a DAG whose
`path`s namespace the transcript); field values may hold unresolved futures
and `resolved()` awaits them. Challenges are sampled from the exceptional
set of the coefficient domain (`Ring` / `Field`), never from a set carried
by the statement.

`c/src/mle.c` holds the dense-MLE kernels (`python/cdef/piop.cdef`);
`c/src/sumcheck.c` holds the sumcheck round-message kernels
(`python/cdef/sumcheck.cdef`); `c/src/merkle.c` holds the Merkle tree
(`python/cdef/merkle.cdef`), which depends on neither. The MLE layer is planned to move to
`vfhe.arith`, so the two stay strictly separated: sumcheck code may depend
on MLE kernels, never the reverse.
