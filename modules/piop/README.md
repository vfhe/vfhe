<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop

> **Under development.** The code is largely verified by the test suite, but
> the documentation text — this file and [piop.md](piop.md) — has not been
> reviewed yet and may be inaccurate or out of date.

The interactive-oracle-proof scaffolding. The architecture and its
derivation from the PIOP literature are documented in [piop.md](piop.md).

The polynomials the protocols run on — `MLE`, `SparseMLE`, `MLE_Basis`,
`MLE_Variable` — are `vfhe.arith`'s (`arith/mle.py`), not this module's. A
statement never names a table directly: it names an *oracle*, an object of
this package that says what the verifier has of the polynomial (`oracle.py`,
below), and the prover works on the tables behind the oracles.

The package has three parts: the framework, exported by `vfhe.piop`; the
relation toolbox, `vfhe.piop.relations`; and the protocols,
`vfhe.piop.protocols`.

## The framework (`vfhe.piop`)

- `oracle.py`: `OracleKind` and the oracle classes — `Oracle`, the contract
  (`variables`, `kind`, `query(point)`, `digest()`); `TableOracle`, an
  oracle to an arith table the framework holds, `public` when the verifier
  holds it too (a wiring predicate, `eq̃`, a public output) and `plain`
  otherwise (the ideal model); `CommittedOracle`, a commitment and the
  variables, no table (the prover's table is in `Prover.witnesses[oracle]`);
  `ImplicitOracle`, a handle with variables and no content, the head of a
  statement of a reduction. `oracle_closure` and `map_oracles` walk the
  oracles a value mentions.
- `virtual.py`: `VirtualOracle`, a local definition over other oracles
  (`p*(x) = Σ_terms c·Π_j p_j(h_j(x))`, the constituents read through
  variable maps); `resolve_table`, the table behind an oracle on the
  prover's side; `expanded_tables`, the definition as a products form over
  equal-length dense tables.
- `relation.py`: `Relation` (an indexed relation: `fields` in canonical
  order, `index_fields`, the ideal decider `check`, heads through
  `head_index` / `compute_head`, and `subject` — the oracle a statement is
  a claim about, which the driver bundles, parks and routes by), `Statement`
  (the claim `instance ∈ L(relation)`, verifier-shaped; `reduce_to` for the
  DAG, `digest()` for Fiat-Shamir) and the hypercube helpers the deciders
  share.
- `evaluation.py`: `Relation_Eval`, the terminal claim `f(z) = v` on one
  oracle, the one relation the framework itself is written against, and the
  head machinery built on it: `with_start` derives the start relation
  `R.Start` of a headed relation, whose decider checks evaluation claims on
  the computed head, and `start_claims` rebuilds the claims of a started
  root.
- `protocol.py`: `Protocol`, a reduction between products of relations as
  paired `prove` / `verify` coroutines over statement bundles (`batching =
  True` protocols receive every frontier statement of their relation at
  once), and `Rejection`.
- `transcript.py`: `Transcript` (the ordered, labeled record of exchanged
  messages: `write(label, value)` / `await read(label)`; its Fiat-Shamir
  chain `state()`, seeded via `bind()` with the root statement's digest),
  `Proof` (the prover messages of a Fiat-Shamir run, in order) and the
  futures `Value` / `Variable`.
- `party.py`: `Party` / `Prover` / `Verifier`: the worklist driver both
  parties run (with the parking rule for evaluation claims), the prover's
  `witnesses`, and the verifier's two samplers — `challenge(label)`, an
  exceptional-set element, and `challenge_bits(label, bits)`, raw published
  coins; either party may call them, a Fiat-Shamir verifier overrides both.
- `fs.py`: `FS_Verifier`, the Fiat-Shamir transform as a `Verifier` subtype
  overriding exactly the two draw hooks: each value is
  `H(transcript.state() | tag | label)`, so it binds the root statement and
  every prior message. Ring challenges come from
  `ring_exceptional_from_seed` unless the domain provides its own
  `exceptional_from_seed`.
- `iop.py`: `IOP`, one protocol execution: the event loop, the coefficient
  domain, the transcript, the two parties and the relation → protocol
  registry (`register(relation, protocol, kind=None)`; `Relation_Eval` is
  registered per oracle kind and, for committed oracles, per commitment
  scheme, `Protocol.scheme`). `run(statement)` executes both parties;
  with `fiat_shamir=True`, `prove(statement) -> Proof` runs the prover
  alone and a separate `verify(statement, proof)` checks the proof with no
  prover, witnesses or oracles present.
- `composer.py`: `compose(layering)` / `Composed`, the protocol of a
  relation given as a layered reduction: headed statements introduce
  implicit oracles, the prover computes every head's table, and each
  layer's statements are started with the claims accumulated on their
  heads.
- `digest.py`: `element_digest`, the one entry point for digesting a
  transcript entry or statement field: primitives, containers, classes and
  protobuf messages itself, everything else through its own `digest()`
  (arith's tables, polynomials and field elements; `vfhe.crypto`'s Merkle
  paths; the oracles, statements and proofs here).

## The relation toolbox (`vfhe.piop.relations`)

- `hypercube.py`: `Relation_Sum`, `Relation_SumProd`, `Relation_Zero`.
- `sumdef.py`: `Relation_SumDef`, an oracle defined as the hypercube sum of
  a virtual oracle, the head of a statement in a layered reduction.
- `circuit.py`: `Relation_CircuitSparseIndex` (circuit satisfiability over
  the protobuf circuits of `vfhe.circuit`, the wiring predicates as public
  sparse index oracles, `wiring_predicates` the indexer), `Relation_Circuit`
  (the same with the predicates committed through one dense index oracle),
  `Relation_Layer` (a gate layer as a sum definition) and `Relation_Equal`.
- `sparse.py`: `FamilyShape`, `Relation_SpEval` and `dense_relation`: a
  family of sparse index tables packed into one master oracle, committed
  through a dense one.
- `lookup.py` / `range.py`: `Relation_Lookup` (every evaluation lies in
  `{0, …, β−1}`) and `Relation_Range` (every `Z_q` coefficient of every
  evaluation of an `R_q` oracle lies in `[0, B)`); see [range.md](range.md).

Every relation's `check` is the ideal (non-succinct) decider: what tests
run and what the verifier may do on the fully-reduced leaf statements. It
needs the tables, so a claim on a committed oracle has no decider.

## The protocols (`vfhe.piop.protocols`)

- `rounds.py`: the sumcheck round-message kernels as functions over tables
  (`round_evals`, `prod_round_evals`, their native, whole-vector and
  pure-Python paths, chosen per call), `interpolate_evals` (the verifier's
  claim update) and `exceptional_set_size`. The building blocks of the
  protocols below and of protocols that interleave sumcheck rounds with
  other messages (basefold in `vfhe.polycom`).
- `prover_view.py`: `prover_view(oracle, witnesses, strategy)`, the
  prover's working form of an oracle: the table behind a table, committed
  or implicit oracle; for a virtual oracle, the dense products form or the
  two-phase (Libra) form, which keeps sparse predicates sparse — both send
  identical messages.
- `sumcheck.py`: `Sumcheck` (reduces `Relation_Sum` to `Relation_Eval` in
  `num_vars` rounds) and the Libra-style `SumcheckProd` (reduces
  `Relation_SumProd` to one `Relation_Eval` per factor). Round messages are
  the evaluations of the round polynomial at `0..deg`; the `strategy`
  attribute picks the prover's form of a virtual oracle.
- `virtual.py`: `VirtualEval` (a claim on a virtual oracle becomes one
  claim per non-public constituent, no challenge) and `SumDefStart` (the
  composable protocol of a sum definition: the claims on the head combined
  by a random linear combination into one sum claim).
- `gkr.py`: `OutputTest`, `GKR` and `GKR_Start`: circuit satisfiability as
  a composition, the output test then one sum definition per gate layer.
- `sparse.py`: `dense_layering`, the two-layer composition of a dense-index
  relation.
- `lookup.py` / `range.py`: `Lookup` (offline memory checking over four
  grand products, two degree-3 zerochecks) and `RangeDecomposition` (the
  base-`β` digits of the coefficients as an oracle over a prime field, a
  recomposition check, and a lookup claim on the digits).

`c/src/sumcheck.c` holds the sumcheck round-message kernels
(`python/cdef/sumcheck.h`), the module's only C. The dense-MLE kernels they
mirror are `vfhe.arith`'s (`arith/c/src/mle.c`), and the vector commitment
the PCS layer uses is `vfhe.crypto`'s `Merkle`.
