<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop

Multilinear extensions and the interactive-oracle-proof scaffolding. The
architecture and its derivation from the PIOP literature are documented in
[piop.md](piop.md).

- `mle.py`: `MLE` and its forms: `ML_Polynomial` (monomial coefficients),
  `MLE_Sparse` (sparse evaluations), and `MLE_Dense` (dense vectors of
  `vfhe.arith.Polynomial`, backed by the `mle_dense_poly_*` C kernels).
  Supports add / sub / scale and variable-by-variable evaluation at concrete
  points; variables are plain identifiers (`MLE_Variable` or any hashable).
  The layer is asyncio-free — unresolved protocol values are a
  Transcript / Statement concern (`Statement.resolved()`).
- `piop.py`: the IOP primitives: `IOP` (event loop, coefficient domain,
  transcript, parties, and the relation → protocol registry), `Transcript`
  (the ordered, labeled record of exchanged messages: `write(label, value)`
  / `await read(label)`), `Value` / `Variable` (asyncio futures),
  `Party` / `Prover` / `Verifier` (worklist drivers; the IOP constructor
  instantiates both parties, taking optional subclasses;
  `iop.verifier.challenge(label)` is the single place challenges originate
  and either party may call it), `Protocol` (a reduction between products
  of relations, as paired prove/verify coroutines over statement bundles;
  `batching = True` protocols receive all frontier statements of their
  relation at once), and the relation / statement pair.
- `sumcheck.py`: `Sumcheck` (reduces `Relation_Sum` to `Relation_Eval` in
  `num_vars` rounds) and the Libra-style `SumcheckProd` (reduces
  `Relation_SumProd` to one `Relation_Eval` per factor). Round messages are
  the evaluations of the round polynomial at `0..deg` (Libra's format);
  both protocols work over `ML_Polynomial` and `MLE_Dense` oracles.
  `iop.register(Relation_Sum, Sumcheck())` then `iop.run(statement)`.
  Each protocol declares `supported_domains` (currently `(Ring,)`): the
  prover delegates to the C kernels in `c/src/sumcheck.c` when the domain
  is supported and the oracles are `MLE_Dense` (for `SumcheckProd`: exactly
  two factors), and falls back to pure Python otherwise — identical
  transcripts either way. Folding is MLE evaluation (`mle_dense_poly_evaluate`),
  not a sumcheck kernel.

A `Relation` is an indexed relation — a set of (index, instance, witness)
triples — declaring its instance shape (`fields`, in the canonical order a
Fiat-Shamir serialization will hash) and an ideal decider `check()`; the
toolbox currently has `Relation_Sum`, `Relation_SumProd`, `Relation_Zero`,
`Relation_Eval`, and `Relation_Open` (declared, pending a commitment
scheme). A `Statement` is the claim "instance ∈ L(relation)": a thin public
record whose fields come from its relation — the witness stays with
`Prover.witnesses`. Protocol steps reduce statement bundles to statement
bundles, linked through `parents` (the statements of a run form a DAG whose
`path`s namespace the transcript); field values may hold unresolved futures
and `resolved()` awaits them. Challenges are sampled from the exceptional
set of the coefficient domain (`Ring` / `Field`), never from a set carried
by the statement.

`c/src/mle.c` holds the dense-MLE kernels (`python/cdef/piop.cdef`);
`c/src/sumcheck.c` holds the sumcheck round-message kernels
(`python/cdef/sumcheck.cdef`). The MLE layer is planned to move to
`vfhe.arith`, so the two stay strictly separated: sumcheck code may depend
on MLE kernels, never the reverse.
