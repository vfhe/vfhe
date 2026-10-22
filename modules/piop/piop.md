<!-- SPDX-FileCopyrightText: 2026 The vFHE Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop — design notes

> **Under development.** The code these notes describe is largely verified by
> the test suite, but the notes themselves have not been reviewed yet and may
> be inaccurate or out of date.

How the module's architecture is derived from the (polynomial) IOP
literature. Bracketed keys refer to the [bibliography](#bibliography).

## 1. Scope

`vfhe.piop` is a generic module for **polynomial interactive oracle proofs
(PIOPs) over multivariate polynomials** whose coefficients live either in a
quotient ring `R_q = Z_q[X]/(X^N + 1)` (`vfhe.arith.Ring` /
`vfhe.arith.Polynomial`) or in a finite field extension
(`vfhe.arith.Field` / `vfhe.arith.FieldElement`). The module provides:

- the polynomial objects the prover sends as oracles (`MLE`), and
- the protocol scaffolding: claims (`Statement`), the relations they refer
  to (`Relation`), the parties, and the asynchronous plumbing
  (`IOP`, `Value`, `Variable`).

Concrete protocols (sumcheck and its product variant, §5; the free
reduction of claims on defined oracles and the layer-by-layer proof of
circuit satisfiability, §4) are built on top of these primitives inside
this module; applications and compilations to succinct arguments belong to
other modules — `vfhe.polycom` compiles the evaluation claims into
commitment openings (§4).

## 2. Background: from IP to PIOP

An **interactive oracle proof (IOP)** [BCS16, RRR16] is an interactive proof
in which the verifier does not read the prover's messages in full: each
prover message is an *oracle* the verifier may query at a few positions. A
**polynomial IOP (PIOP)** [BFS20] restricts the oracles to (bounded-degree)
polynomials and the queries to polynomial evaluations; the same idea appears
as "polynomial protocols" in [GWC19] and, with a preprocessed index, as
"algebraic holographic proofs" in [CHMMVW20]. A PIOP is an
information-theoretic object: compiling each oracle into a polynomial
commitment and each query into an evaluation-opening turns it into a succinct
argument [BFS20, CHMMVW20], but that compilation is out of scope here.

This module targets the **multivariate / multilinear** flavor of PIOPs, in
the line of [Set20, CBBZ23]: oracles are multilinear polynomials over the
boolean hypercube `{0,1}^n`, claims are sums/evaluations over the hypercube,
and the workhorse reduction is the sumcheck protocol [LFKN92]. Compared with
univariate PIOPs, the multilinear setting needs no FFTs and admits
linear-time provers [CBBZ23], and every table of `2^n` values has a *unique*
multilinear extension (MLE) [Tha22, §3.5].

Because our coefficients may live in a ring `R_q` rather than a field, we
rely on the line of work extending these protocols to rings: sumcheck over
modules/rings [BCS21], GKR-style protocols over arbitrary rings [Sor22], and
soundness from *exceptional sets* via a generalized Schwartz–Zippel lemma
[GNS23, Lemma 2], including sumcheck PIOPs stated directly over `R_q`
[CCCFGS25, Thm 2.6]. See §6.

## 3. Relations, languages, statements

The literature fixes the following vocabulary [BCS16; CHMMVW20]:

- A **relation** `R` is a set of pairs `(x, w)`: *instance* (public) and
  *witness* (private to the prover). An **indexed relation** [CHMMVW20] is a
  set of triples `(i, x, w)` where the *index* `i` is a large, reusable part
  of the instance (e.g. a circuit description) that can be preprocessed.
- The **language** of `R` is `L(R) = { x : ∃w, (x, w) ∈ R }`.
- A **statement** is the claim "`x ∈ L(R)`" — what the prover asserts and the
  verifier must be convinced of. An IOP for `R` is complete if honest provers
  convince the verifier of true statements, and sound if for `x ∉ L(R)` every
  prover fails except with probability at most the soundness error.

The module mirrors this vocabulary directly:

| literature                  | `vfhe.piop`                                    |
| --------------------------- | ---------------------------------------------- |
| relation `R`                | a `Relation` instance                     |
| index `i`                   | `Relation.index`, over the names `Relation.index_fields` declares (or one opaque object) |
| instance `x`                | `Statement`; fields declared by `Relation.fields` |
| witness `w`                 | held by `Prover` (`witnesses`), never on the statement |
| membership test `(x,w) ∈ R` | `Relation.check(statement)`                |
| oracle (prover message)     | an `MLE` object (queried, not read)            |

Two consequences of this split, compared with putting everything on the
statement:

1. **Statements are verifier-shaped.** Everything on a `Statement` is
   public; handing one to the verifier can never leak a witness.
2. **Relations are the unit of protocol design.** A protocol is specified by
   which relation it starts from and which relation it reduces to (§5); the
   relation object is also the natural home for the index and for the
   "ideal" decider `check()` used by tests and by the end of a reduction
   chain.

**The index of a relation.** A relation declares its index fields beside
its instance fields (`index_fields`, in canonical order): its index is then
a mapping over exactly those names, validated by the constructor as
`Statement` validates `fields`, stored in the declared order and digested in
that order (`index_digest`, cached on the instance, since an index is built
once and reused across runs — the verifier key of [CHMMVW20]); a relation
declaring none keeps one opaque object. The oracles among the index values
— *index oracles*, found by `index_oracles()` through tuples, lists and
mappings, duck-typed on a `variables` attribute — are oracles of every
statement of the relation wherever the framework reads a statement's
oracles (§4, §5). The index is where public, preprocessable data goes: a
circuit description with its wiring predicates
(`Relation_CircuitSparseIndex`), the dense table that commits them
(`Relation_Circuit`), the table size of a lookup. A statement reads it as
`statement.index`.

In an in-process IOP the `oracles` list holds the actual `MLE` objects — the
same data an honest prover knows in full. The oracle *discipline* (the
verifier only evaluates them at points, it never inspects representations) is
enforced by convention; compiling the oracle into a polynomial commitment
enforces it cryptographically [BFS20] — registering
`vfhe.polycom.BasefoldEval` for `Relation_Eval` does exactly that (§4).

## 4. The relation toolbox

Following the multilinear PIOP toolboxes of [Set20, CBBZ23], the module
starts from a handful of relation kinds. In all of them `f` is (an MLE of) an
`n`-variate polynomial oracle and the coefficient domain is `R_q` or a field.

- **`Relation_Sum`** — instance `(f, v)`:
  `Σ_{b ∈ {0,1}^n} f(b) = v`.
  Discharged by the sumcheck protocol [LFKN92; Tha22, §4.1], which reduces it
  in `n` rounds to a `Relation_Eval` claim at a random point.
- **`Relation_SumProd`** — instance `(f_1..f_k, v)`:
  `Σ_{b ∈ {0,1}^n} Π_j f_j(b) = v`.
  Discharged by the Libra-style product sumcheck [XZZPS19, Alg. 3]
  (linear-time prover technique from [Tha13]; [Tha22, Lemma 4.5]), which
  reduces it to `k` `Relation_Eval` claims — one per factor, all at the same
  random point (§5).
- **`Relation_Zero`** — instance `(f)`:
  `f(b) = 0` for every `b ∈ {0,1}^n`.
  Discharged by zerocheck [Set20; CBBZ23, §3.2]: the verifier samples `r` and
  the claim reduces to the `Relation_Sum` claim
  `Σ_b eq̃(r, b)·f(b) = 0`, where `eq̃` is the multilinear equality polynomial
  (the `χ_w` Lagrange basis of [Tha22, §3.5, Lemma 3.6]).
- **`Relation_Eval`** (`evaluation.py`) — instance `(f, z, v)`: `f(z) = v`,
  the terminal claim of the sumcheck family, for one oracle `f`, which is
  the claim's `subject`: what the driver bundles, parks and routes it by. What the verifier has
  of `f` is the oracle's kind (below): a table it may query, or a
  commitment — "the commitment to f is simply the oracle π_f" [ZCF24, §4],
  so the compiled claim is *the same relation*: the standard PCS
  evaluation relation R_Eval = {[(C, z, y); f] : f(z) = y and C opens to f}
  [ZCF24, Def. 8; BFS20]. A `CommittedOracle` is instance data — created
  by a scheme's commit algorithm (possibly long before any IOP run, reused
  across runs), carried on statements, never on the per-execution
  transcript — and the prover's table lives in `prover.witnesses[oracle]`.
  Left terminal, the claim is decided by one oracle query
  (`Oracle.query`); registering a PCS evaluation protocol
  (`vfhe.polycom.BasefoldEval`) replaces that query with an opening
  argument — the "queries become opening claims" side of the [CHMMVW20]
  compiler, done in place. A claim on a committed oracle has no
  witness-free decider and must not be left terminal.

- **`Relation_CircuitSparseIndex`** — index: a layered arithmetic circuit
  (`vfhe.circuit`: fan-in-2 ADD and MUL gates and XMULT inner-product gates
  `Σ_i in[l_i]·in[r_i]`, the generalized gates of zkCNN [LXZ21, §4]) and its
  wiring predicates, sparse tables built once by the indexer
  (`indexed(circuit, domain)`); instance `oracles = (out, w_in)`: the input
  oracle evaluates under the circuit to the output oracle. `out` is the relation's *head* (§5): its
  table is computed from `w_in` (`compute_head`), so a circuit can define
  an implicit oracle of a larger reduction. Discharged by `GKR` [GKR15;
  Tha22, §4.6] (`circuit.py`), a *composition* (§5): the output test
  (`Relation_Equal`, below) between `out` and the output layer, then one
  sum definition per gate layer (`Relation_Layer`), from the output layer
  down to `w_in`; the rest of the proof is the framework's ordinary
  reduction of the claims the layers leave. It is a relation rather than a
  helper so that circuit satisfiability can be a factor of a larger product
  of relations, the target of another reduction, or the head of a
  statement of one.
- **`Relation_Circuit`** — the same relation with the predicates committed
  through one dense index oracle: index `("circuit", "dense", "shape",
  "speval")`, the dense table, the shape of the family of predicates and
  the realization decoding one from the other (§7, `sparse.py`). Discharged
  by `compose(dense_layering)`: `Relation_CircuitSparseIndex` over
  restrictions of a master oracle in the first layer, run by `GKR`, then
  the realization's statement headed at the master.
- **`Relation_SumDef`** — instance `oracles = (L, G)`:
  `L(z) = Σ_{b ∈ {0,1}^m} G(z, b)` on the hypercube, a *sum definition* of
  the head `L` by a virtual oracle `G` (below) over `L`'s variables `z` and
  the summed variables `b`: a layer of a layered circuit in terms of the
  layer below it [Tha22, §4.6], a grand-product level [Tha13, §5.3.1]. A
  statement headed at `L` is how an implicit oracle gets its definition;
  started with the claims on `L` (§5), it is discharged by `SumDefStart`,
  registered against `Relation_SumDef.Start`.
- **`Relation_Layer`** — a `Relation_SumDef` indexed by `(circuit, l)`:
  gate layer `l` as the sum definition of `W_{l+1}` by the generalized
  layer identity
  `W_{l+1}(z) = Σ_{x,y} lin_l(z,x,y) W_l(x) + xmult_l(z,x,y) W_l(x) W_l(y)`
  over public *sparse* wiring predicates — `lin` (the linear summands,
  supported on `y = 0`) and `xmult` (the multiplied pairs), one nonzero per
  wire read — and the layer below, read under two renamings. Same decider
  and same protocol as the generic sum definition; only `compute_head`
  differs: it evaluates the gates on the table of the layer below, linear
  in the wires, where the generic route is exponential.
- **`Relation_Equal`** — instance `oracles = (a, b)`: the two oracles agree
  on the hypercube; the headless statement that ties a circuit's output
  oracle to its output layer. Discharged by `OutputTest`, the
  random-evaluation identity test [Tha22, §4.6]: the verifier draws `r`,
  the value `v = a(r)` is computed by the verifier when `a` is public and
  sent by the prover otherwise, and the claim reduces to `b(r) = v` plus,
  unless `a` is public, `a(r) = v` — soundness `s/|A|` for `s` variables
  [GNS23, Lem. 2] (§6).

Further members of the toolbox (product-check, permutation-check, lookups
[CBBZ23]) can be added as new `Relation` subclasses without touching
`Statement`.

### Oracle kinds

`Relation_Eval` is one relation with several discharges, decided by *what
the verifier has* of the oracle (`OracleKind`, the oracle's `kind`; the
kinds are classes, `oracle.py`):

| kind | the verifier has | an evaluation claim on it |
|---|---|---|
| `public` (`TableOracle(table, public=True)`) | the table itself: a wiring predicate, `eq̃`, a public output | terminal — the verifier evaluates it |
| `committed` (`CommittedOracle`) | a commitment and the variables, no table | the PCS protocol (`vfhe.polycom.BasefoldEval`) |
| `virtual` (`VirtualOracle`) | a local definition over other oracles | `VirtualEval`: rewritten into claims on the constituents, no challenge, error 0 |
| `implicit` (`ImplicitOracle`) | nothing but the statement it heads | never discharged by a protocol: held by the Composer until its head's layer (§5) |
| `plain` (`TableOracle(table)`) | the oracle of the ideal model, which it may query | terminal, or the protocol registered without a kind |

The two *defined* kinds are one idea: an oracle given by a relation on
other oracles instead of by a table. A **virtual oracle**
[CBBZ23, §2.2; BCRSW19, Def. 4.6 ("rational constraints")] is
`p*(x) = Σ_terms c · Π_j p_j(h_j(x))` — a sum of products of constituents,
each read through a *variable map* `h_j` (a renaming of its variables,
some possibly fixed to constants: how one oracle appears as both `W(x)` and
`W(y)`, or as `W(p, 0)` and `W(p, 1)`). A query to it costs one query per
constituent, so a claim `p*(x) = v` reduces for free: the prover sends the
constituent values, the verifier evaluates the public constituents
itself, checks the combination against `v`, and one claim per distinct
(constituent, point) pair remains. An **implicit oracle** (`ImplicitOracle`) is
a handle with variables and no content of its own: it is the *head* of
exactly one statement of a reduction (`Statement.head`, §5), and that
statement is its definition — `L(z) = Σ_{b} G(z, b)` for a virtual oracle
`G` over the free variables `z` and the summed ones `b`
(`Relation_SumDef`), or the output of a circuit on an input oracle
(`Relation_CircuitSparseIndex`). The verifier has nothing of `L` but the statement,
and making `L(r)` explicit takes a protocol for it — for a sum
definition, a sumcheck. The multilinear extension of a layer of gate
values is of this second kind even when the gates are local, because
extension does not commute with products
(`W̃_l(r) ≠ W̃_{l+1}(r,0)·W̃_{l+1}(r,1)`): the `eq̃` factor is what makes
`W_l(z) = Σ_p eq̃(z,p) W_{l+1}(p,0) W_{l+1}(p,1)` an identity [Tha13,
§5.3.1]. (Jolt's code calls this second kind "virtual polynomials"; here
"virtual" is HyperPlonk's word for the first.)

A claim on an implicit oracle is therefore never discharged on its own: no
protocol is registered for the `implicit` kind, the Composer of the
reduction holds every claim on a head until the layer of the statement it
heads, and `IOP.protocol_for` refuses one that strays to the driver (the
`kind` exists for that guard only). At that layer the statement is
*started*: `R.start(statement, claims)` is the statement of `R.Start`
carrying the `k ≥ 1` claims (§5), and the protocol registered for
`R.Start` consumes them. For a sum definition that is `SumDefStart`: it
combines the claims by a random linear combination with one independent
challenge per claim (none for a single claim) — the ring precedents
[Sor22, Fig. 2, Lemma 11; CCCFGS25, Eqs. (19)–(20)] — into one Sum claim
over `Σ_i α_i G(r_i, ·)`, itself a virtual oracle; the registered sumcheck
reduces it to one evaluation claim on that virtual oracle, and
`VirtualEval` that claim to claims on `G`'s constituents. Zero
communication; soundness `(k−1)/|A|` for `k` claims. Binding a defined
oracle is *symbolic* (a bound variable becomes a constant in the variable
maps; no constituent is touched), so the claims that come out are about
the original constituents at explicit points — and so a *sparse*
constituent (`SparseMLE`) is only ever queried at a full point, which is
the one evaluation the sparse form supports (`O(nnz · vars)`, the honest
verifier cost of a public predicate). The prover's sumcheck works on
`prover_view`, the definition with every constituent resolved to a table
(`Prover.witnesses` for implicit constituents, where the Composer stores
every head's table, §5) and folded in place, under one of two
round-message *strategies* that send the same messages: the dense one
enumerates the remaining hypercube; the two-phase one of Libra
[XZZPS19, §3.3, Algs. 4–6], as generalized to arbitrary predicates by
zkCNN and Virgo++ [LXZ21; ZLWZS21], applies whenever the summed variables
split into two blocks `x` (bound first) and `y`, each dense factor lives
in one block, and each term carries at most one sparse predicate — the
shape of every layer identity `Σ_{x,y} pred(z,x,y)·A(x)·B(y)`. Per term it
builds, in time linear in the predicate's nonzeros, the bookkeeping table
`h(x) = Σ_y pred(z,x,y)·B(y)` for the `x` rounds (the term becomes
`A(x)·h(x)`) and `p(y) = pred(z,r_x,y)` for the `y` rounds (the term
becomes `A(r_x)·p(y)·B(y)`), so every round is a product of dense tables of
degree at most two — the `k = 2` native kernel's shape — and the prover's
work per layer is `O(#wires + 2^{s_x} + 2^{s_y})` instead of the dense
`O(2^{s_z + 2s})`. Round messages carry `deg_i + 1` nodes for the degree
*in the round variable* (two factors of a term reading `x_i`, not the
length of the longest product), so both strategies agree node for node.

Index oracles (§3) have the same kinds. A table in a root's index is
`public`, when the verifier evaluates it, or `committed`: `Basefold.commit`
returns the `CommittedOracle` — the commitment and the variables, no table
— that the index then names, so a claim on it is `committed` by its kind,
the root digest records it by its commitment and never by entries (a
verifier seeds Fiat–Shamir without reading a committed predicate table),
and the prover holds the table under the oracle in its witnesses. Public
and committed are two classes, not two marks on one table: they are two
discharges of a claim. The index a layering builds for a child statement may also hold an
`implicit` oracle (the head of a deeper statement, or the relation's own,
§5) and a `virtual` one when it is a *restriction*:
`VirtualOracle.restriction(P, variables, fixed)` reads one member of a
family of tables packed into one oracle `P` over slot variables `T` and
table variables `X`, member `i` being `P(bin(i), X)` — one constituent, one
term of coefficient 1, a map fixing some of `P`'s variables to 0/1
constants and renaming the rest. The slot is a variable map and nothing
more: a claim on the member is rewritten by `VirtualEval` into a claim on
`P` at `(bin(i), x)`, and on the prover's side `resolve_table` resolves the
restriction to a slice of `P`'s table (`SparseMLE.restrict` for a sparse
one, a partial evaluation for a dense one), which `prover_view` keeps
sparse for the two-phase strategy.

Each subclass implements `check(statement)`: the *ideal* (non-succinct)
membership test that simply enumerates the hypercube or queries the oracle.
This is the decider a test suite runs, and the check the verifier is entitled
to perform on the fully-reduced leaf statement; it is deliberately not the
succinct verifier.

## 5. Reductions between products of relations

Protocols in this module are **reductions between products of relations**:
an interactive step consumes a bundle of statements and emits the bundle it
reduces them to, such that (soundness) the new statements all being true
implies — up to the step's soundness error — that the old ones were. This is
the framing formalized as *reductions of knowledge* in [KP23]: Definition 10
defines the product ("relation pair") `R1 × R2` and the power `R^ℓ`,
Theorem 5 gives sequential composition, and Theorem 6 ("parallel
composition") composes reductions over products. Both directions occur:

- **one → many**: a product sumcheck on `Σ_b Π_j f_j(b) = v` ends with one
  evaluation claim per factor ([XZZPS19]; the same shape as [KP23]'s
  reduction from `R_VC(n)` to `R_VC(n/2) × R_VC(n/2)`), and the GKR layer
  reduction emits two claims `Ṽ(u), Ṽ(v)` per layer [XZZPS19, §2.3];
- **many → one (batching)**: [KP23, Def. 4] calls a reduction `R^ℓ → R` an
  ℓ-folding scheme; [CBBZ23, §3.8] batches k multilinear evaluation claims
  into one (`R^k_BATCH`, the BatchEval PIOP), and two evaluation claims of
  one polynomial reduce to one by line restriction [Tha22, §4.5.2] or by a
  random linear combination ([XZZPS19, §2.3], crediting Chiesa–Forbes–
  Spooner).

```text
Relation_Circuit ─(dense_layering = Composed: Relation_CircuitSparseIndex, then Relation_SpEval.Start)─▶ Relation_Eval^2 (w_in) [× Relation_Eval (out)] × Relation_Eval^* (P_dense)
Relation_CircuitSparseIndex ─(GKR = Composed: OutputTest, then per layer Relation_Layer.Start)─▶ Relation_Eval^2 (w_in) [× Relation_Eval (out)] [× Relation_Eval^* (the predicates' master)]
Relation_CircuitSparseIndex.Start ─(GKR_Start = Composed: per layer Relation_Layer.Start)─────▶ Relation_Eval^2 (w_in) [× Relation_Eval^* (the predicates' master)]
Relation_SpEval.Start ─(a realization's protocol)─▶ Relation_Eval^* (P_dense)
Relation_Equal ───(OutputTest, sample r)─▶ Relation_Eval (output layer) [× Relation_Eval (output)]
Relation_SumDef.Start ─(SumDefStart, k challenges)─▶ Relation_Sum
Relation_Zero ────(zerocheck, sample r)──▶ Relation_Sum
Relation_Sum ─────(sumcheck, n rounds)───▶ Relation_Eval
Relation_SumProd ─(sumcheckprod)─────────▶ Relation_Eval × … × Relation_Eval
Relation_Eval (virtual) ─(VirtualEval)──▶ Relation_Eval × … × Relation_Eval (constituents)
Relation_Eval^k (implicit) ─(held by the Composer, R.start)─▶ R.Start (the statement it heads)
Relation_Eval^k (committed) ─(PCS, per claim today)──────▶ accepted / rejected
Relation_Eval (public / plain) ─(oracle query)───────────▶ accepted / rejected
```

The statements of one run therefore form a **DAG**: each statement records
the bundle it came from in `parents`, and the verifier decides the leaves.
Organizing claims as a DAG has precedent — proof-carrying data defines
transcripts over a DAG [BCCT13], Virgo++ runs GKR over general DAG circuits
with claims flowing along the edges [ZLWZS21] — but the DAG here stays
*implicit* in the `parents` edges plus the worklist driver and the
Composer below; no dedicated graph object. (Implementations name what a
step emits a "claim" or "subclaim" — e.g. arkworks' `SubClaim` — which is
exactly our `Statement`.)

Each statement's instance fields are declared by its relation
(`Relation.fields`, in canonical order — the order Fiat-Shamir will hash)
and its `path` names its position in the DAG: the root is `""` and the j-th
statement reduced out of a bundle whose first parent has path `p` is
`f"{p}/{j}"`. Both parties derive identical DAGs, so paths agree and
namespace the transcript.

### Execution model

A reduction is realized as a `Protocol`: a pair of coroutines
(`prove`, `verify`) over the same bundle of statements
(`list[Statement] -> list[Statement]`) — the pair (P, V) of the IOP
definition [BCS16], specialized to one reduction step between products of
relations [KP23]. The only data that flows between the parties are oracle
messages and challenges; statements are never transmitted — each party
derives its own `parents`-linked DAG from the common input plus the
transcript.

- **Transcript.** The `IOP` holds a `Transcript`: the ordered, labeled
  record of everything exchanged — "transcript" is the standard term for
  this record in the IP/IOP literature [BCS16; Tha22]. The sender calls
  `transcript.write(label, value)`; the receiver `await transcript.read(label)`
  (entries are futures underneath, so a read can be scheduled before its
  write). Labels follow the style of Merlin transcripts (Henry de Valence's
  `merlin`, the transcript layer of Bulletproofs and Spartan): they are part
  of the record and give domain separation, and each label is single-use;
  protocols namespace them by the statement's `path`, which the identical
  DAGs on both sides keep in agreement. The *write order* is the
  canonical order a Fiat–Shamir transformation [FS86] hashes, and the
  transcript maintains that hash itself: `state(upto=None)` is the chained
  prefix digest `h_i = H(h_{i−1} ‖ H(label_i) ‖ digest(value_i))` — the
  recursive BCS form σ_i = ρ(rt_i ‖ σ_{i−1}) [BCS16], chosen over hashing
  one big concatenation because per-entry digests and chain values are
  cached (`digests` / `_states`): a new entry costs one compression, any
  prefix is a lookup, and interactive runs that never call `state()` pay
  nothing (the caches fill lazily). The chain is seeded by
  `bind(seed)` with the *root statement's* digest — σ_0 = ρ(x), whose
  omission is the "weak Fiat–Shamir" bug of [DMWG23] — and entry values
  are digested by `element_digest`: primitives, containers, classes and
  protobuf messages by itself, every other value by its own `digest()`
  (arith's tables, polynomials and field elements, `vfhe.crypto`'s Merkle
  paths, this module's oracles, statements and proofs).
- **Challenges.** `iop.verifier.challenge(label)` returns the challenge
  `label`: the first call samples it from the domain's exceptional set (§6)
  and writes it to the transcript; later calls return the recorded value.
  Challenge generation is a property of the *verifier*, not of a protocol —
  public coin means a challenge is fresh randomness and nothing else — so
  protocols call this method and never sample themselves. *Either party*
  may call it: under Fiat–Shamir a challenge is a deterministic function of
  the transcript so far, so prover and verifier compute the same value —
  the interactive machinery mirrors that symmetry. Public-coin honesty is
  the caller's responsibility: a challenge is drawn only after the round
  message it answers is written, keeping the transcript order canonical
  (the machinery does not enforce this; the FS hash binding does).
- **Fiat–Shamir.** `IOP(fiat_shamir=True)` (default `False`: interactive is
  the base model; FS is the compiled artifact you opt into) instantiates
  `fs.FS_Verifier`, a `Verifier` overriding exactly the two draw hooks
  behind the samplers: each value is `H(transcript.state() ‖ tag ‖ label)`
  — the tag separates the two samplers, the label keeps back-to-back
  challenges distinct — and `IOP.run` seeds the chain with
  `element_digest(root statement)`. Domain challenges need a deterministic
  bytes → exceptional-element map: a domain's own `exceptional_from_seed`
  takes precedence; over a `Ring`, `fs.ring_exceptional_from_seed` derives
  a constant chunk of `split_degree` coefficients below `min(pᵢ)` — an
  exceptional set of the same size `min(pᵢ)^split_degree` (nonzero
  small-coefficient chunks of degree < split_degree are coprime to every
  irreducible factor), replaceable by a seeded arith-side sampler later.
  Nothing else changes — same protocols, same transcript — and an FS run
  is fully deterministic: same statement, same registry ⇒ byte-identical
  transcript (the tests assert this end-to-end, basefold included).
- **Registry.** `iop.register(relation_type, protocol, kind=None)` chooses
  how statements of a relation are discharged; relations without a
  registered protocol are *terminal*. This keeps relations passive data and
  protocols swappable (e.g. a batched sumcheck can replace the plain one
  without touching relations or parties). For `Relation_Eval` the key also
  carries the oracle kind (§4): the PCS protocol is registered without a
  kind and receives committed and plain oracles, `VirtualEval` is
  registered for the virtual kind, and a kind without an entry falls back
  to the default; claims on public oracles never reach a protocol, and no
  protocol is registered for the implicit kind (a claim on an implicit
  oracle reaching `protocol_for` is an error, §4). Start relations are
  keyed like any other, by exact type, so a subclass's start relation has
  its own entry: `Relation_SumDef.Start` and `Relation_Layer.Start` get
  `SumDefStart`, `Relation_Equal` gets `OutputTest`,
  `Relation_CircuitSparseIndex` gets `GKR` and its `Start` `GKR_Start`,
  `Relation_Circuit` and its `Start` get `compose(dense_layering)`, and a
  realization's `Relation_SpEval.Start` gets that realization's protocol.
- **Drivers.** Both parties run the same worklist loop over the DAG's
  frontier: take the first statement that is not *parked*, hand the
  registered protocol its bundle, push the outputs; statements with no
  registered protocol are terminal. A composed protocol (below) is, to
  the driver, one protocol like any other. A protocol with
  `batching = True` is a *bundle* protocol — a folding reduction
  `R^ℓ → R` [KP23, Def. 4] — and receives, in one invocation, every
  frontier statement of its relation; for `Relation_Eval`, every frontier
  claim on one oracle — in practice the PCS bundle on a committed oracle,
  since claims on implicit oracles never reach the driver. Such a claim is
  **parked** while any other frontier statement, outside its bundle, has
  that oracle in its *oracle closure* (its oracles and, through the
  definitions of the defined ones, their constituents, recursively): that
  statement may still emit a claim on the same oracle, and the bundle
  should hold them all. Closures are public and static, so both parties
  park identically; definitions form a DAG and non-Eval statements never
  park, so some statement is always ready. This is the two-phase
  scheduling of staged provers (every evaluation claim accumulated, one
  batched discharge) [CBBZ23, §3.8], done per oracle. The verifier then
  decides every terminal leaf with its relation's own `check()` — for
  `Relation_Eval` that is exactly one oracle query per claim. A protocol's
  `verify` half raises `Rejection` on a failed round check, which the
  driver turns into a `False` verdict.
- **Run.** `iop.run(statement)` schedules both parties' coroutines on the
  IOP's event loop and returns the verifier's verdict. Each party drives
  its own *fork* of the root statement (same public content, fresh
  child-path counter): child paths are handed out by a per-statement
  counter, so a root shared between the parties would give the second
  party's children the next counter values and desynchronize every
  path-namespaced transcript label. Deadlock freedom
  follows from the round structure itself: the prover awaits challenge *i*
  before sending message *i+1*, the verifier awaits message *i* before
  sampling challenge *i* (a crashed prover is re-raised rather than left to
  deadlock the verifier). An `IOP` object is single-use — one run consumes
  its transcript.

### Layered reductions and the Composer

Some protocols are a schedule of reductions rather than one step. GKR is
the example: each gate layer defines the layer above it, the claims on a
layer come from the layer above, and the layer's sumcheck has to wait
until all of them are in. Such a protocol is written as a **layered
reduction** and run by the Composer (`composer.py`): `compose(layering)`
is the protocol for a relation `R` whose reduction is `layering`, a
function from a statement of `R` to its layers `L_0, …, L_D`, each a list
of child statements (built with `reduce_to`, so paths and labels agree on
both parties). A child is *headless*, an ordinary statement discharged by
its registered protocol, or *headed* (`Statement.head`): it introduces an
implicit oracle whose table its relation computes from the statement's
other oracles (`Relation.compute_head`), and it is discharged, once every
claim on the head has been made, by the protocol registered for its
relation's start relation (`Start`, §7): the relation's *sequentially
composable* protocol. Two rules make the layers a schedule, checked on
every run (`_check_rules`):

1. every implicit oracle heads exactly one statement;
2. a statement in `L_l` reads, besides its own head, only non-implicit
   oracles and heads of deeper layers — through virtual oracles
   (`oracle_closure`) and through its relation's index
   (`statement_oracles`) too.

An implicit oracle that no statement of the layering heads belongs to an
enclosing composition and counts as non-implicit for the rules. The run is
the same on both parties, except for the first step:

- **Head pass** (prover only): every head's table, once, deepest layer
  first, into `Prover.witnesses`, where the subprotocols read it —
  `compute_head` on `_statement_view`, the statement with every other
  oracle replaced by what the prover holds of it (rule 2 is what makes
  deepest-first possible). A head an enclosing composition has already
  stored is not recomputed.
- **Loop**: the layers in order, each statement in list order. A headless
  statement goes to its protocol; a headed one is started,
  `relation.start(statement, claims)`, with the claims accumulated on its
  head, and goes to the protocol of the start relation. A headed statement
  with no claim on its head is an error, and so is a claim left over after
  the last layer.
- **Routing** (`_discharge`) of what a subprotocol leaves: a claim on a
  head of this reduction waits for that head's layer; a statement whose
  protocol is not a bundle protocol (an evaluation claim on a virtual
  oracle, a sum a protocol handed back) is discharged inline, inside the
  layer, so a claim on a head hidden behind a virtual oracle is caught in
  time; everything else — a terminal statement, a claim for the PCS
  bundle, a claim on a head of an enclosing composition — is an output of
  the composed protocol, returned to the driver. A layer statement with no
  protocol that reads a head of this reduction is refused: nothing could
  decide it.

Everything above reads a statement's oracles through `statement_oracles`:
its instance oracles followed by the index oracles of its relation (§3). So
a head reached through an index counts for rule 2; the driver's parking
rule sees a claim a frontier statement can produce through its index; the
prover view (`_statement_view`) replaces the index oracles as it replaces
the instance ones — a virtual oracle rebuilt over resolved constituents, an
implicit one by its stored table — and hands `compute_head` a statement
whose relation is rebuilt over the viewed index (`type(r)(index=viewed)`,
the convention a relation with a head already follows), the relation thus
reading its index as it reads the oracles; and a terminal statement reading
a head through its index is refused. A relation may also name its *head* by
an index field: `head_index` is an `int`, a position in `oracles`, or a
`str`, the name of the index entry holding it, and a statement of such a
relation derives its head from that entry — an `ImplicitOracle` makes it headed,
a table an ordinary statement, and a `head=` passed must be the entry. The
Composer is unchanged by it: the head map, the rules and the routing are
keyed on the oracle object, wherever it sits. A root's index holds tables,
never an implicit oracle — except the head of a started root — and the
runners refuse one that does.

A composed protocol is an ordinary protocol to the driver: one statement
in, the leftover statements out. Hence **nesting**: a relation whose
protocol is composed can itself head a statement of another reduction
(`Relation_CircuitSparseIndex`, head `out`). Its `R.Start` protocol is then composed
too (`GKR_Start`): the claims the started statement carries seed the
claims on its head, and its statements may read the heads of the
enclosing reduction. A reduction that depends on the IOP it runs in (its
domain, say) subclasses `Composed` and overrides `layers(root, iop)`
instead of passing a layering, as `GKR` does.

### The non-interactive pair: producing and checking a proof

`run` keeps both parties in one process, which is the interactive model and
what protocol development wants. The point of Fiat–Shamir, though, is that
the two halves come apart: the prover derives its own challenges, so it can
finish alone and leave behind an object anyone can check later. Two runners
express that, both Fiat–Shamir-only (interactively the challenges are fresh
randomness a proof could not carry, so they refuse rather than emit an
uncheckable object):

- `iop.prove(statement) -> Proof` runs the prover alone. Nothing blocks:
  every `challenge` is computed on the spot from the chain, and prove halves
  only ever *write* to the transcript — a prove half that tried to read
  would have no counterparty, so the transcript raises instead of hanging.
- `iop.verify(statement, proof) -> bool` runs the verifier alone against
  that object, on a *separate* IOP: proving and checking cannot share one,
  since each consumes a transcript. The checking side needs the statement,
  the proof, and the registry — no prover, no witnesses, no oracles.

The object is the **argument string** [CY24, §4.1; CO25, §2.1], spelled
"NARG string" in implementations [CFRG-FS, §2]; the class keeps the
readable name `Proof`. It is deliberately *not* called a transcript: that
word is the interactive record, prover and verifier messages both
[CFRG-FS, §2], and it is the distinction spongefish drew when it deprecated
`transcript()` in favour of `narg_string()`.

A **`Proof` carries only the prover's messages**, in write order — "the
argument string π contains … all IP prover messages (and none of the IP
verifier messages)" [CO25, §4.3]. The challenges are a deterministic
function of the messages before them, so storing them would record what the
verifier recomputes anyway, and would invite a verifier to *trust* them —
precisely the hole Fiat–Shamir closes. (Every proof struct in the wild
agrees: plonky3's `FriProof` and Marlin's `Proof` carry commitments,
openings and prover messages, never the challenges or query indices.)
`Transcript.write(..., derived=True)` is how the samplers mark their
entries, and `Transcript.messages()` is everything not so marked.

Reading it back is a **single forward pass**: when the verifier reads a
label nobody has written, the transcript takes the proof's next message,
checks it is the expected one, and writes it — extending the chain in
exactly the order the prover built it, so the verifier *re-derives* the
challenges and reconstructs the interaction [CY24, §14.1]. Three
malformations are therefore rejected structurally: a message out of turn, a
proof that runs out, and one with entries left over. The last is the
**end-of-input check**, not a tidiness rule: unread entries make a proof
*malleable* — an adversary alters them for a second, distinct accepting
proof of the same statement, costing strong simulation-extractability
[CFRG-FS, §6.2] — and out-of-order or skipped messages are a real
implementation bug class, with CVEs behind the guidance. Everything else —
a tampered message, a substituted statement — is caught by the chain: any
change reshuffles every later challenge, and the run stops adding up.

**Note for the planned C port.** The non-interactive verifier is the part
worth making fast, and this shape is meant to survive the move. It also
happens to be what the implementation guidance prescribes [CFRG-FS, §8.6]:

> "A byte-level interface … is advisable in place of proof data structures
> whose fields are randomly addressable. A sequential interface, by
> contrast, enforces in-order processing. An end-of-input check is
> necessary to prevent malleability."

— guidance written against real CVEs for out-of-order or missing prover
messages, and the reason the same passage asks that absorbing a message and
(de)serializing it happen *in the same call*, which `Transcript.read` does.
So: reads are sequential with no random access, challenges are recomputed
rather than parsed, the chain is one 32-byte state updated per entry,
per-round verifier work is O(1) coefficient operations, and the verify path
provably never touches prover state.

What a C verifier additionally needs, and what Python does not yet have, is
a *canonical byte encoding* of each message type: `element_digest` hashes
values but does not serialize them, and `Proof` holds live Python objects,
so there is nothing to parse yet. Labels are the other Python-only
convenience — redundant in principle (the verifier knows what it expects
next), kept for clear errors, droppable in a byte-level format. That
encoding is the prerequisite for the port, and it is what would make this a
NARG *string* rather than a list of objects.

The first instantiation is the sumcheck protocol (`protocols/sumcheck.py`): round *i*
sends the univariate `g_i` (as the pair `g_i(0), g_i(1)` — the oracle is a
single multilinear polynomial, so `deg g_i = 1`), the verifier checks
`g_i(0) + g_i(1)` against the running claim, samples `r_i` from the domain's
exceptional set, and the claim becomes `g_i(r_i)`; after `n` rounds the
statement reduces to `Relation_Eval` at `(r_1, …, r_n)` [LFKN92; Tha22,
§4.1], with soundness error `d·n/|A|` (§6) reported by
`Sumcheck.soundness_error`.

The first many-output instantiation is the Libra-style product sumcheck
(`SumcheckProd`) for `Relation_SumProd` [XZZPS19, Alg. 3]: each factor is
linear in the round variable, so `g_i` has degree `k`; after the last round
the prover sends the per-factor values `v_j = f_j(r)`, the verifier checks
`Π_j v_j` against the final claim, and the claim reduces to `k`
`Relation_Eval` statements at the same point `r` — soundness `k·n/|A|`.

Round messages are the **evaluations of `g_i` at the integer nodes
`0..deg`**, Libra's format [XZZPS19, Alg. 3]: `MLE` tables are already
in the evaluation basis, so the linear-time prover kernels accumulate the
table halves directly (values above 1 extrapolated division-free as
`lo + t·(hi−lo)`, e.g. `2·hi − lo` at `t = 2`). The verifier updates its
claim by Lagrange interpolation at those nodes, which over a ring requires
`{0..deg}` to be an exceptional set — [Sor22, fn. 12]; trivially satisfied
in `R_q` with large primes, where the small integer denominators are
inverted per RNS prime. (Sending coefficients instead would avoid the
inversions entirely — the presentation of [Sor22, §5.4; BCS21, §2.1] — at
the cost of leaving the evaluation basis; we follow Libra. Evaluation-form
message optimizations, e.g. omitting `g(1)` [Gru24, §3.1; DT24], are
roadmap.)

### Native implementations

Each protocol body is written exactly once — there are no separate native
and Python provers. The native/pure-Python decision is made at the data
level, inside the round-message helpers: `Sumcheck.round_evals` and
`SumcheckProd.prod_round_evals` (static) compute one round message for a
round variable at any position and delegate to the C kernels when the
oracles are native tables (`mle.native_table`; for the product message,
exactly `k = 2` factors — the Libra `f·g` shape), running the naive
pure-Python path otherwise. Both paths produce identical messages, so
transcripts are interchangeable mid-stack — even mid-*protocol*: the
decision is per call, so e.g. a mixed native/coefficient-basis factor pair
simply takes the Python path. `supported_domains` (e.g. `(Ring,)`) remains
on the protocol as declarative metadata — which domains have kernels —
not as a dispatch switch. These helpers, with `interpolate_evals` (the
verifier's claim update), are also the building blocks for protocols that
interleave sumcheck rounds with other messages (basefold in
`vfhe.polycom`, which folds a committed codeword between rounds).

The two protocols share their round machinery (`_SumcheckRounds`, a
template both subclass): per round the prover writes the round message and
folds every table by the challenge (out of place on the first round to
keep the shared oracles intact, in place afterwards — folding is exactly
binding the round variable, i.e. `mle_dense_poly_evaluate*`, never a
sumcheck kernel), and the single verifier checks `g(0) + g(1)` against the
running claim and interpolates the next one in O(1) ring operations per
round. Subclasses supply only the transcript prefix, the round message,
and the closing step of each half (`Sumcheck`: the final `f(r)`;
`SumcheckProd`: the `/vals` message, its product check, and the per-factor
outputs).

The prover kernels (`piop/c/src/sumcheck.c`) only *accumulate round
messages* from the table pairs of the round variable — which, like MLE
binding (§7), may sit at any position: each message has a kernel per pair
layout (`sumcheck_round_pairs` / `_halves` and `sumcheck_prod2_round_pairs`
/ `_halves` for the LSB and MSB variables, plus the stride-computed
`sumcheck_round` / `sumcheck_prod2_round` generic fallbacks), chosen by the
round-eval helpers from the variable's position. Over a `Field` the table
is a single `vfhe.arith.FieldVector` and the same messages are a fixed
number of whole-vector operations (`round_evals_vector`,
`prod2_round_evals_vector`, gated on `mle.vector_table`) — arith's kernels
rather than piop's, for the first variable only; other positions take the
pure-Python path. Multithreaded kernels are roadmap.

## 6. Challenges and soundness

Verifier challenges drive every reduction, and their soundness comes from
Schwartz–Zippel-type arguments. Over a field `F`, a nonzero degree-`d`
polynomial vanishes on a random point of `F^n` with probability at most
`d/|F|`. Over a ring this fails in general; the fix is to sample challenges
from an **exceptional set** `A ⊆ R` — a set whose pairwise differences are
invertible [GNS23, Def. 5] — for which the generalized Schwartz–Zippel lemma
`Pr_{ā←A^n}[f(ā) = 0] ≤ deg(f)/|A|` holds [GNS23, Lemma 2]; the same notion
drives sumcheck over rings and modules [BCS21, Sor22] and sumcheck PIOPs
stated directly over `R_q` [CCCFGS25, Def. 2.1, Lemma 2.2]. The resulting
sumcheck soundness error is the familiar `d·n/|A|` [CCCFGS25, Thm 2.6;
Tha22, §4.1]. A field is simply the degenerate case `A = F`.

**Where this lives in vfhe:** the exceptional set is a property of the
*coefficient domain*, not of a statement or protocol, so the domain object is
responsible for sampling from it. `vfhe.arith.Ring` already provisions this:
its `exceptional_set_size` parameter picks the `split_degree` so the residue
factors are large enough, and `Ring.random_exceptional()` /
`Polynomial.sample_exceptional()` sample challenges. For `vfhe.arith.Field`
the whole field is exceptional and uniform sampling suffices, and
`vfhe.arith.Field` spells it under the same names a `Ring` has
(`random_exceptional`, `exceptional_from_seed`). The `IOP` object holds the domain and delegates; no
`sampling_set` is ever carried by statements. Sampling happens in exactly one
place — `Verifier.challenge`, which calls `domain.random_exceptional()`
(the name both `Ring` and `Field` have) — and per-protocol soundness accounting is a
protocol method —
`Sumcheck.soundness_error` reports `d·n/|A|`, with `|A| =
min(p_i)^split_degree` for a `Ring` and `p^d` for a `Field` (duck-typed until
the domain classes expose `|A|` themselves).

## 7. Class-by-class notes

### `IOP`

Owns one execution of a protocol: the asyncio event loop, the coefficient
`domain` (a `Ring` or `Field`, per §6), the `Transcript`, the two parties
(the constructor instantiates `Prover` / `Verifier`, or subclasses passed as
`prover=` / `verifier=` — e.g. the Fiat–Shamir verifier of §5), and the
relation → protocol registry (§5). The asynchronous model exists because
IOP messages have a *data-flow* structure: a prover message of round `i+1`
depends on the verifier challenge of round `i`, but everything that does not
depend on it can proceed. Unresolved messages are represented as futures
(below), and they live exclusively at the Transcript / Statement level:
`Statement.resolved()` awaits pending field values, and the MLE layer is
asyncio-free — it only ever evaluates at concrete points.

### `Value` / `Variable`

Asyncio futures standing for protocol messages that have not been sent yet.
`Variable` is a named future — a protocol variable such as a verifier
challenge `r_i`; `Value` is an anonymous one — typically a value derived
from unresolved inputs (e.g. the result of evaluating an MLE at a
yet-unsampled point). Statements may reference either in `point` / `value`;
`Statement.resolved()` awaits them.

### `Party` / `Prover` / `Verifier`

The two roles of the IOP model [BCS16]: the **prover** computes and sends
oracles, and holds the witnesses (`Prover.witnesses` maps statements to
their private data, and commitments and implicit oracles to what is behind
them — the witness never sits on the statement itself); the
**verifier** sends challenges and is restricted to oracle queries. Both keep
protocol-local `state` and run the reduce-until-terminal driver loops of §5:
`prove` follows the registered reductions and makes no decision; `verify`
does the same and decides the terminal leaf, translating a mid-protocol
`Rejection` into a `False` verdict. The verifier must never call
`check()` on a non-terminal statement — that decider enumerates the
hypercube.

The verifier draws two kinds of randomness, both compute-if-absent and both
published to the transcript, so the Fiat-Shamir subtype (`fs.FS_Verifier`,
§5) overrides exactly the two draw hooks behind them (`_draw_challenge`,
`_draw_bits`):

- `challenge(label)` — a coefficient-domain element from the exceptional
  set (§6), what reductions consume and what `soundness_error` accounts
  for;
- `challenge_bits(label, bits)` — raw uniform coins as a byte string,
  deliberately shapeless: how they become protocol randomness — spot-check
  positions (§2's "the verifier may query it at a few positions"), a
  permutation, a subset — is the *protocol's* business, expanded on its
  side (e.g. `BasefoldEval.query_positions` hashes the seed in counter
  mode and rejection-samples). The verifier stays generic; anything
  shaped, like index sampling with a distinctness constraint, would smuggle
  protocol knowledge into it. The coins are *published* for a reason worth
  stating: derived randomness must be reproducible by the prover — while
  an oracle is an in-process object the verifier just reads it, but once
  it is a commitment the prover must learn the query positions to answer
  them with authentication paths, which is the one structural change the
  Merkle layer makes to a message flow (`vfhe.polycom.BasefoldEval`). No
  exceptional-set structure, so `soundness_error` accounting never sees
  these coins.

### `MLE` and `SparseMLE` (`vfhe.arith`, `arith/mle.py`)

Every function `f : {0,1}^n → R` has a unique multilinear extension `f̃`,
`f̃(x) = Σ_{w ∈ {0,1}^n} f(w)·χ_w(x)` [Tha22, §3.5, Lemma 3.6]. The module
has two *independent* types — they share vocabulary (`variables`,
`num_vars`, `scale`, `constant`), not an inheritance chain, because a
sparse map supports none of the folding a dense table exists for, and a
common base could only promise an `evaluate` it cannot implement:

- `MLE` — a dense table of `2^n` coefficients; the type protocols work
  with. Two properties of the table, deliberately *not* subclasses, because
  they vary independently:
  - **`basis`** (`MLE_Basis.eval` / `MLE_Basis.coeff`) — the hypercube
    evaluations (constructor argument `evaluations=`) or the monomial
    coefficients (`coefficients=`, entry `b` multiplying
    `Π_{i ∈ bits(b)} x_i`, i.e. a multilinear *polynomial* rather than an
    extension table). Both bind a variable by folding the same (lo, hi)
    pairs; only the fold differs — interpolation `lo + r·(hi − lo)` in the
    evaluation basis, Horner `c_lo + r·c_hi` in the monomial one.
    `to_coefficients()` converts (the butterfly `c_hi = e_hi − e_lo`),
    returning another `MLE`; that LSB-first coefficient vector is
    what a code-based commitment encodes (`vfhe.polycom`).
  - **coefficient type** — with a `ring`, entries are
    `vfhe.arith.Polynomial` over that `Ring` and the `mle_dense_poly_*` C
    kernels do the work; with a `field`, the table is one
    `vfhe.arith.FieldVector` and the folds are whole-vector operations
    (the first variable's bind is a split plus an interpolation; other
    positions unpack to elements); without either, entries are plain Python
    values (any type with `+`/`*`, e.g. exact ints for reference-semantics
    tests) and the folds run in Python.

  The kernels are RNS routines *and* interpolate, so they need a ring
  **and** the evaluation basis: `native_table(f)` is that predicate, and it
  — not `isinstance` — is what protocols gate native delegation on (§5);
  `vector_table(f)` is its field counterpart.
  `MLE.eq(domain, point)` builds the dense table of the equality
  polynomial `eq̃(point, ·)` (the zerocheck reduction and basefold's
  virtual factor). A table carries no protocol mark: whether the verifier
  holds it is said by the oracle that names it (`TableOracle(table,
  public=True)`, §4's oracle kinds). `rename(mapping)`
  is a relabelled copy — the same polynomial read under other variable
  names, which is how the prover holds `W(x)` and `W(y)` as two tables.
- `SparseMLE` — a sparse map of hypercube evaluations
  (`evaluations[index] = value`, index packed LSB-first like a dense
  position), in a domain: the carrier of
  wiring predicates. It supports add / sub / scale, evaluation at a *full*
  point (`Σ_k value_k · Π_i eq̃(k_i, point_i)`, `O(nnz · vars)`; a partial
  point raises, since the form has no per-variable fold — defined oracles
  never need one, §4), and `materialize()` to the dense table. Digested by
  the FS walker in canonical index order.

Representation caveat, inherited from `vfhe.arith`: reading a table entry's
value (`== int`, iteration, `get_polynomial()`) converts *that entry* to
coefficient form in place, so a table can end up mixed — and the C kernels,
which read RNS form, would then silently fold the wrong data. Every native
path therefore normalizes first (`MLE.to_NTT()`, a per-entry flag check
when already normalized), and kernel outputs are stamped NTT-form
(`mark_ntt`) rather than mirroring a source flag that could itself be
stale. This is a **provisional workaround**: the fix belongs in arith,
which should stop mutating representation on read.

Evaluation binds one variable at a time by the standard fold
`f(…, x_i = r, …)` combining the two halves of the table as
`(1−r)·f|_{x_i=0} + r·f|_{x_i=1}` (equivalently `c_0 + r·c_1` in the monomial
basis), the linear-time technique used by memory-efficient sumcheck provers
[Tha22, §4.1–4.2]. Variable order is generic at the Python level — any
named variable may be bound, in any order — and each implementation
dispatches on the variable's *position* to the most efficient pair layout:
adjacent entries for the first (LSB) variable ("pairs"), the two table
halves for the last (MSB) one ("halves"), and a stride-computed generic
fallback for anything in between (each backed by its own C kernel on a
native table, by slicing on the Python path). The layer is deliberately
asyncio-free: variables are plain identifiers (`MLE_Variable`, or any
hashable — *not* protocol futures), and evaluation points are always
concrete values; anything unresolved is a Transcript / Statement concern.

### `Oracle`, `TableOracle`, `CommittedOracle`, `ImplicitOracle` (`oracle.py`)

What a statement or an index names is an oracle, an object of this
package, never an arith table: the table is arith's and carries no
protocol mark, the oracle says what the verifier has of it (§4's kinds).
`Oracle` is the contract — `variables`, `kind`, `query(point)` (the ideal
decider's one query, at a full point; it raises for an oracle the verifier
has no table of), `dependencies()`, `digest()` — and the four kinds are
classes: `TableOracle(table, public=...)` wraps a dense or sparse table,
`CommittedOracle(variables, commitment)` holds the commitment and the
variables and nothing else, `ImplicitOracle(variables)` is a handle,
`VirtualOracle` a definition. Oracles are compared by identity: claims are
bundled, parked and stored on the object. The prover works on tables:
`resolve_table(oracle, witnesses)` is the table behind an oracle on its
side — a table oracle's own, `witnesses[oracle]` for a committed or an
implicit one, a slice for a restriction.

### `VirtualOracle`, `Relation_SumDef` / `SumDefStart` (`virtual.py`, `relations/sumdef.py`, `protocols/virtual.py`, `protocols/prover_view.py`)

The virtual oracle of §4 is an object with the oracle surface the rest of
the module duck-types on (`variables`, `num_vars`, `evaluate`, `constant`,
`digest`), plus `dependencies()` for the closures of §5 and
`prover_view(witnesses)` for the prover's sumcheck. A `VirtualOracle` is
`(variables, constituents, terms, maps)`: `terms` is the products form
`[(coeff, (j, …)), …]` and `maps[j]` sends each variable of constituent
`j` to one of `variables` or to a constant; `VirtualOracle.product` and
`VirtualOracle.linear` build the common shapes. `evaluate` is symbolic
(maps rewritten, constituents untouched), and `constant()` on a fully
bound one is the ideal decider. The prover-side form (`prover_view`)
computes round messages over the products form: per hypercube point, each
factor at `t = 0..degree` from its two neighbours (`lo + t·(hi − lo)`),
summed per term — the degree is the longest product, and factors over a
subset of the variables are constant in the others.

`Relation_SumDef` is the sum definition of §4, head at index 0 of
`(L, G)`, the summed variables being `G`'s that are not `L`'s
(`summed_of`); its ideal decider enumerates both hypercubes. Its
`compute_head` is the generic, exponential route — `G`'s dense prover view
summed over the summed variables at every point of `L`'s hypercube; a
relation that knows the structure overrides it, as `Relation_Layer` does
(§4). `SumDefStart` is its composable protocol (§4), not a bundle
protocol: it takes one started statement, whose `points` and `values` are
the claims. `VirtualEval` and `SumDefStart` live in the same file, as do
the two round-message strategies (`_ProverVirtual`, `_LibraProver`; §4).
`VirtualOracle.strategy` (`"auto"` / `"dense"` / `"libra"`) pins one; the
tests assert message-for-message agreement between them.

### `Relation_CircuitSparseIndex` / `Relation_Circuit` / `GKR` (`relations/circuit.py`, `protocols/gkr.py`)

Circuit satisfiability over the protobuf circuits of `vfhe.circuit`
(imported lazily), with the wiring tables of `vfhe.circuit.export` as index
oracles: a table indexed `z‖x‖y` (y in the low bits) is the MLE over the
variables `[y…, x…, z…]`, so the framework's LSB-first tables need no
reordering. `Relation_CircuitSparseIndex` has `oracles = (out, w_in)` and
head `out`, and the index `("circuit", "predicates")`: the circuit and the
tuple `(lin_0, xmult_0, …, lin_{d-1}, xmult_{d-1})` the indexer
`indexed(circuit, domain)` builds once (`wiring_predicates`), public sparse
tables over variables of their own, the sparse index field
(`sparse_index_fields = ("predicates",)`). `layer_tables(w_in)` evaluates
the circuit gate by gate, which serves both the ideal decider and
`compute_head`.

`GKR` is a `Composed` subclass (§5) whose `layers(root, iop)` reads the
predicates from the root's index and renames each into its layer body
through the body's map (a predicate's own variables to the body's `z`, `x`
and `y`), so the index may hold public tables or restrictions of a master
(§4) and `GKR` runs the same. Layer `L_0` is the output test `Relation_Equal(out, W_d)`, `W_d` a
fresh `ImplicitOracle` handle for the output layer; then comes one
`Relation_Layer` statement per gate layer, from the output layer down,
headed at that layer's implicit oracle and reading the one below, the
bottom one reading `w_in` itself. The layering first checks that `out`
and `w_in` have the numbers of variables the circuit's layer sizes
require. `GKR_Start`, registered for `Relation_CircuitSparseIndex.Start`, is the same
layering without the output test: the top layer is headed at `out` itself
and started by the claims the statement carries. `GKR` refuses a started
statement and `GKR_Start` an unstarted one. Per layer the Composer starts
the definition with the claims the layer above left on it (one for the
top layer under `GKR`, the `k` the statement carries under `GKR_Start`,
two below it, one per renaming `x`, `y`), `SumDefStart`
combines them, and the sumcheck and `VirtualEval` reduce them to claims on
the layer below, down to the input oracle, whose claims leave the
composition and are decided by its kind. `GKR.soundness_error` reports the
output test's `s_out/|A|` only (`GKR_Start`'s is 0); the rest is the
layers' sumchecks and combinations.

The predicates come from `vfhe.circuit.export.sparse_wiring` (one nonzero
per wire read; the dense `wiring_tables` remain as their materialization),
the summed variables are listed `x` first then `y`, and the prover runs
the two-phase strategy, unchanged by the composition; the verifier's
end-of-layer cost is the sparse evaluation of the two predicates when they
are public.

`Relation_Circuit` is `dense_relation(Relation_CircuitSparseIndex)` (below),
decorated with `with_start`: the index `("circuit", "dense", "shape",
"speval")`, the same instance and head, `compute_head` the sparse-index
relation's. Its protocol is `compose(dense_layering)`, registered against
it and its `Start`; a small circuit keeps its predicates public and runs
`Relation_CircuitSparseIndex` as before, a large one commits them and runs
`Relation_Circuit`, and nothing in `GKR` changes between the two.

### `FamilyShape`, `Relation_SpEval`, `dense_relation` / `dense_layering` (`relations/sparse.py`, `protocols/sparse.py`)

The pattern for sparse index oracles committed through one dense one (§4),
with no encoding fixed by the framework. `FamilyShape({field: width or
(widths…)})` is the shape of a family of sparse tables — the slots in
order, their widths, which sparse index fields they fill — and nothing
else: data in the index, with a `digest()` by the widths and no `variables`
attribute. It builds the master's handle (`implicit()`, over `selectors`
`T`, `⌈log₂ a⌉` of them, and `xs` `X`, the largest width), each member as a
restriction of the master (`restriction(P, i)`: `T = bin(i)`, the padding
fixed to 0, the rest renamed) and their placement in the sparse fields
(`place(P)`); `FamilyShape.of(index, fields)` reads the widths off the
tables an index holds, and the master's index of entry `k` of slot `i` is
`i ‖ k` with `i` in the low bits (`index_of`).

`Relation_SpEval` is the abstract sparse-evaluation relation: index
`("sparse", "dense", "shape")`, the head named by `"sparse"` (§5), no
instance fields, `compute_head` the *decoding* of the master from the dense
table — abstract — and a generic decider comparing nonzeros. It is the
relation `R_speval` of [CCCFGS26] with its index turned inside out: the
dense oracle in the index beside the sparse one it defines, a claim naming
its polynomial in the `T` coordinate of its point. A *realization*
subclasses it, implements `compute_head`, holds its parameters as class
attributes (the layering builds it over the three index fields and nothing
else), is decorated with `with_start` itself (the registry keys on the
exact class) and registers against its `Start` a sequentially composable
protocol: any number of claims on the master, none of them a query to it,
every challenge drawn by the protocol, ending in evaluation claims on the
dense table carried in `oracles` — bundled by the driver for the
commitment scheme, or terminal when the table is plain. The tests hold a
basic one (`val` and `ind` in one dense table, one sumcheck of degree
`m + 1` per claim); a realization with sublinear verification is a module
built on this one (§8).

`dense_relation(inner)` derives from a relation with `sparse_index_fields`
the relation whose index is `inner`'s non-sparse fields followed by
`"dense"`, `"shape"` and `"speval"` (the realization, a class, digested by
its qualified name), with `inner`'s instance fields and head, a decider
that decodes once and decides `inner` with the slices in place, and
`inner`'s `compute_head`; the class comes back undecorated, `inner` kept as
its `inner` attribute (which `with_start` copies to the start relation).
`dense_layering(root)` is its protocol as a layered reduction (§5): `L_0`
the inner relation over the root's instance fields with its sparse fields
holding `place(P)` for a fresh master `P`, headed at the root's head when
the root is started — the Composer then starts it with the root's claims,
as it starts the top layer of `GKR_Start` — and `L_1` the realization over
`(P, P_dense, shape)`, headed at `P` through its index. Inside `L_0` the
claims the inner protocol leaves on the restrictions are rewritten by
`VirtualEval` into claims on `P`, which head no statement of the inner
layering and come back to this Composer, which holds them for `L_1`; no
message, no challenge, soundness error 0.

### `Statement.head`, `with_start`, `Composed` (`relation.py`, `evaluation.py`, `composer.py`)

`Statement.head` is structural, beside `parents` and `path`: not a field,
never digested, set through `reduce_to(…, head=…)` and kept by the
parties' forks of the root. A relation with a head declares `head_index`,
the head's position in `oracles`, and `compute_head(statement)`, the
head's table from the other oracles on the prover's view (concrete
tables); a statement checks that its head is the `ImplicitOracle` handle at that
position. The class decorator `with_start` derives `R.Start`, whose fields
are `R`'s plus `points` and
`values` and whose oracles and head are `R`'s: "`R` with its head bound to
the oracle it defines, plus the evaluation claims on it". Its decider is
generic — compute the head, decide `R` with that table in the head's
place, then one `Relation_Eval` check per claim — and `R.Start` rebuilds
`R` from its index, so a relation with a head must be constructible as
`type(r)(index=r.index)`. The decorator also installs
`R.start(statement, claims)`, which builds the started statement as a
child of the headed one, the claims as further parents. A sequentially
composable protocol for `R` is a protocol for `R.Start`.

`ImplicitOracle` is the handle itself: variables and nothing else. Its
`digest()` is by its variables, so a statement carrying it can seed a
Fiat–Shamir chain, and its `kind = implicit` exists only for the guard in
`protocol_for`. `Composed` is the protocol `compose` builds (§5); a
reduction that needs the IOP subclasses it and overrides the `layers`
hook.

### `Merkle` (`vfhe.crypto`, `crypto/merkle.py`)

A binary Merkle tree [Mer88] over BLAKE3: the vector commitment that turns a
long prover oracle into a short root plus per-query openings, which is how an
IOP is compiled into an argument [BCS16] and what a code-based
commitment (`vfhe.polycom`) needs for its codewords. Internal nodes are
`BLAKE3(left || right)`; a leaf count that is not a power of two is padded
with zero digests, so the root is defined for any size (the leaf count is
public protocol data, so the padding is not a domain separation concern).

The split follows the rest of the library: the tree — building the levels,
copying a path, replaying it — is C (one contiguous buffer per level), while
the Python layer only decides *what a leaf hashes to*. That
is the sole requirement on a leaf type: a `.hash()` method returning its
digest, or a `hash=` callable given to `Merkle` / `Merkle.verify` for types
without one. Leaves are never copied or interpreted, only referenced and
hashed, so the tree commits to `MLE` tables, codeword entries, or anything
else without knowing what they are.

An opening (`MerklePath`) carries the sibling digests bottom-up and
deliberately *not* the leaf index: the verifier checks the position it
queried, never one the prover chose. Nothing in it is PIOP-specific, which
is why it lives in `vfhe.crypto`; it is described here because the layering
argument below is what keeps it out of this module.

`Merkle` is a primitive, not a `Relation` or a `Protocol`, and that is a
deliberate layering decision rather than an omission. The root is the
random-oracle *instantiation* of an ideal oracle — the BCS compiler
[BCS16], as [ZCF24] §4 states for exactly this scheme — so it belongs to
the compiler layer that also owns Fiat-Shamir, one level below the IOP.
Structurally it could not be a `Relation` anyway: a relation here carries
statistical soundness over exceptional-set challenges (§6), while a Merkle
claim is computational (collision resistance) with no challenge, no round
and no reduction, so the framework's accounting has nothing to say about
it. And the relation a commitment participates in already exists —
`Relation_Eval` *is* [ZCF24]'s R_Eval, the oracle it names being a
`CommittedOracle` in the compiled form; swapping a table for a commitment
changes the oracle's kind, not the relation. Path checking
therefore lives inside the PCS's evaluation protocol
(`vfhe.polycom.BasefoldEval`), which is where the queried positions are.

## 8. Roadmap

1. **Prover kernel follow-ups**: the native Ring paths exist (§5); still
   open are multithreaded round kernels, a general-`k` product kernel
   (native is `k = 2`), the omit-`g(1)` message trick [Gru24, §3.1; DT24],
   and Field round kernels. The
   pure-Python fallbacks remain naive (per-round hypercube re-enumeration)
   by design — they are the reference semantics, not the fast path.
2. **Zerocheck** (`Relation_Zero → Relation_Sum` over the virtual oracle
   `eq̃(r, ·)·f`): a small protocol now that defined oracles exist (§4);
   a lazily evaluated `eq̃` factor (O(1) per query) would spare the dense
   table.
3. **Batching / folding**: done for claims on implicit oracles — the
   Composer holds them until their head's layer and `SumDefStart`
   combines them inside the composable protocol (§4–§5). Open: a
   `Composed.soundness_error` adding up its subprotocols' errors, the
   inline-discharged sumchecks and `VirtualEval` steps included — today
   `GKR.soundness_error` reports the output test alone. Open for
   committed oracles: the
   PCS bundle protocol still opens each claim on its own; a common-point
   reduction plus a batched opening (BatchEval [CBBZ23, §3.8]) is the
   `vfhe.polycom` roadmap. Sumcheck-side batching (one run for several
   Sum claims) is a bundle variant of `Sumcheck` still to write.
3a. **Defined-oracle follow-ups**: the two-phase strategy's `O(nnz)`
   bookkeeping loops and the sparse `evaluate` in C (every round is
   already the `k = 2` shape the native product kernel covers, so a
   general-`k` kernel is no longer on GKR's path); products-form round
   messages in C for the dense strategy; a sparse-evaluation realization
   with sublinear verification — the framework side is in place
   (`Relation_SpEval`, `dense_relation`, `dense_layering`, §7, with a basic
   realization in the tests), and the protocol of [CCCFGS26] for it, with
   its packing and its sumcheck over challenges from a small subset of
   `R_q`, belongs to a module built on this one, as a `Relation_SpEval`
   subclass registered against its `Start`; it takes one claim per sparse
   polynomial, so running GKR over it also needs a layer protocol that
   takes the common point at the layer oracle first, as the classic line
   reduction does; the line-restriction 2-to-1 [Tha22, §4.5.2] should a
   single-point opening ever pay.
4. **Lookup relation**: done (`relations/lookup.py`, `protocols/lookup.py`,
   `relations/range.py`, `protocols/range.py`; see
   [range.md](range.md)). `Relation_Lookup` reduces, by offline memory
   checking over four grand products, to two `Relation_Sum` claims over
   degree-3 virtual oracles plus four `Relation_Eval` claims;
   `Relation_Range` reduces to it. Open follow-ups: batching the four
   fingerprints into one sumcheck, and a products-form round kernel in C --
   the degree-3 rounds are pure Python today (item 3a).
5. **Field coefficient domains** are done (`MLE(field=...)`, the
   whole-vector round messages, `Field`'s samplers); binding a variable
   other than the first still unpacks the vector to elements.
6. **A canonical byte encoding for proof messages**: `element_digest`
   hashes transcript values but does not serialize them, so a `Proof`
   holds live Python objects rather than an argument *string*
   [CY24, §4.1] (§5). With it come `digest()` hooks for any new transcript
   value types, and a seeded arith-side `sample_exceptional` to replace
   `fs.ring_exceptional_from_seed` (§5, §6).
7. **A native non-interactive verifier**: port the FS verifier to C once
   the encoding above exists — the hot path is hashing and per-round
   coefficient arithmetic, and the Python design already constrains it to a
   single forward pass over the proof (§5).

## Bibliography

- **[LXZ21]** Tianyi Liu, Xiang Xie, Yupeng Zhang. *zkCNN: Zero Knowledge
  Proofs for Convolutional Neural Network Predictions and Accuracy*. ACM
  CCS 2021, pp. 2968–2985. <https://doi.org/10.1145/3460120.3485379>
  (Generalized GKR gates: §4.)
- **[BCRSW19]** Eli Ben-Sasson, Alessandro Chiesa, Michael Riabzev, Nicholas
  Spooner, Madars Virza, Nicholas P. Ward. *Aurora: Transparent Succinct
  Arguments for R1CS*. EUROCRYPT 2019, LNCS 11476, pp. 103–128, Springer,
  2019. ePrint 2018/828. <https://eprint.iacr.org/2018/828>
- **[BCS16]** Eli Ben-Sasson, Alessandro Chiesa, Nicholas Spooner.
  *Interactive Oracle Proofs*. TCC 2016-B, LNCS 9986, pp. 31–60, Springer,
  2016. ePrint 2016/116. <https://eprint.iacr.org/2016/116>
- **[RRR16]** Omer Reingold, Guy N. Rothblum, Ron D. Rothblum.
  *Constant-Round Interactive Proofs for Delegating Computation*. STOC 2016,
  pp. 49–62, ACM, 2016. <https://dl.acm.org/doi/10.1145/2897518.2897652>
- **[Mer88]** Ralph C. Merkle. *A Digital Signature Based on a Conventional
  Encryption Function*. CRYPTO '87, LNCS 293, pp. 369-378, Springer, 1988.
  <https://doi.org/10.1007/3-540-48184-2_32>
- **[LFKN92]** Carsten Lund, Lance Fortnow, Howard Karloff, Noam Nisan.
  *Algebraic Methods for Interactive Proof Systems*. Journal of the ACM
  39(4), pp. 859–868, 1992. <https://dl.acm.org/doi/10.1145/146585.146605>
- **[GKR15]** Shafi Goldwasser, Yael Tauman Kalai, Guy N. Rothblum.
  *Delegating Computation: Interactive Proofs for Muggles*. STOC 2008;
  Journal of the ACM 62(4), Article 27, 2015.
  <https://dl.acm.org/doi/10.1145/2699436>
- **[CHMMVW20]** Alessandro Chiesa, Yuncong Hu, Mary Maller, Pratyush Mishra,
  Psi Vesely, Nicholas Ward. *Marlin: Preprocessing zkSNARKs with Universal
  and Updatable SRS*. EUROCRYPT 2020, LNCS 12105, pp. 738–768, Springer,
  2020. ePrint 2019/1047. <https://eprint.iacr.org/2019/1047>
- **[BFS20]** Benedikt Bünz, Ben Fisch, Alan Szepieniec. *Transparent SNARKs
  from DARK Compilers*. EUROCRYPT 2020, LNCS 12105, pp. 677–706, Springer,
  2020. ePrint 2019/1229. <https://eprint.iacr.org/2019/1229>
- **[FS86]** Amos Fiat, Adi Shamir. *How to Prove Yourself: Practical
  Solutions to Identification and Signature Problems*. CRYPTO '86, LNCS 263,
  pp. 186–194, Springer, 1987.
  <https://doi.org/10.1007/3-540-47721-7_12>
- **[CY24]** Alessandro Chiesa, Eylon Yogev. *Building Cryptographic Proofs
  from Hash Functions*. 2024. <https://snargsbook.org/> (§4.1 defines NARG
  = non-interactive argument and the argument string; §14.1 the
  emulate/re-derive framing; ch. 25 casts BCS as a NARG).
- **[CO25]** Alessandro Chiesa, Michele Orrù. *A Fiat–Shamir Transformation
  from Duplex Sponges*. TCC 2025. ePrint 2025/536.
  <https://eprint.iacr.org/2025/536> (§4.3: the argument string carries all
  IP prover messages and none of the verifier's; §8.1–8.2 the byte-level
  interface).
- **[CFRG-FS]** Michele Orrù et al. *Fiat-Shamir Transformation*. IRTF/CFRG
  Internet-Draft, draft-irtf-cfrg-fiat-shamir.
  <https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-fiat-shamir> (§2
  transcript vs NARG string; §6.2 the end-of-input MUST and malleability;
  §8.6 sequential interface guidance).
- **[DMWG23]** Quang Dao, Jim Miller, Opal Wright, Paul Grubbs. *Weak
  Fiat-Shamir Attacks on Modern Proof Systems*. IEEE Symposium on Security
  and Privacy 2023. ePrint 2023/691. <https://eprint.iacr.org/2023/691>
- **[XZZPS19]** Tiancheng Xie, Jiaheng Zhang, Yupeng Zhang, Charalampos
  Papamanthou, Dawn Song. *Libra: Succinct Zero-Knowledge Proofs with
  Optimal Prover Computation*. CRYPTO 2019, LNCS 11694, pp. 733–764,
  Springer, 2019. ePrint 2019/317. <https://eprint.iacr.org/2019/317>
- **[Tha13]** Justin Thaler. *Time-Optimal Interactive Proofs for Circuit
  Evaluation*. CRYPTO 2013, Springer, 2013. ePrint 2013/351.
  <https://eprint.iacr.org/2013/351>
- **[BCCT13]** Nir Bitansky, Ran Canetti, Alessandro Chiesa, Eran Tromer.
  *Recursive Composition and Bootstrapping for SNARKs and Proof-Carrying
  Data*. STOC 2013, ACM, 2013. ePrint 2012/095.
  <https://eprint.iacr.org/2012/095>
- **[ZLWZS21]** Jiaheng Zhang, Tianyi Liu, Weijie Wang, Yinuo Zhang, Dawn
  Song, Xiang Xie, Yupeng Zhang. *Doubly Efficient Interactive Proofs for
  General Arithmetic Circuits with Linear Prover Time*. ACM CCS 2021.
  ePrint 2020/1247. <https://eprint.iacr.org/2020/1247>
- **[Gru24]** Angus Gruen. *Some Improvements for the PIOP for ZeroCheck*.
  Cryptology ePrint Archive, Paper 2024/108, 2024.
  <https://eprint.iacr.org/2024/108>
- **[DT24]** Quang Dao, Justin Thaler. *More Optimizations to Sum-Check
  Proving*. Cryptology ePrint Archive, Paper 2024/1210, 2024.
  <https://eprint.iacr.org/2024/1210>
- **[GWC19]** Ariel Gabizon, Zachary J. Williamson, Oana Ciobotaru. *PLONK:
  Permutations over Lagrange-bases for Oecumenical Noninteractive arguments
  of Knowledge*. Cryptology ePrint Archive, Paper 2019/953, 2019.
  <https://eprint.iacr.org/2019/953>
- **[Set20]** Srinath Setty. *Spartan: Efficient and General-Purpose
  zkSNARKs Without Trusted Setup*. CRYPTO 2020, LNCS 12172, pp. 704–737,
  Springer, 2020. ePrint 2019/550. <https://eprint.iacr.org/2019/550>
- **[CBBZ23]** Binyi Chen, Benedikt Bünz, Dan Boneh, Zhenfei Zhang.
  *HyperPlonk: Plonk with Linear-Time Prover and High-Degree Custom Gates*.
  EUROCRYPT 2023, LNCS 14005, pp. 499–530, Springer, 2023. ePrint 2022/1355.
  <https://eprint.iacr.org/2022/1355>
- **[BCS21]** Jonathan Bootle, Alessandro Chiesa, Katerina Sotiraki.
  *Sumcheck Arguments and Their Applications*. CRYPTO 2021, LNCS 12825,
  pp. 742–773, Springer, 2021. ePrint 2021/333.
  <https://eprint.iacr.org/2021/333>
- **[GNS23]** Chaya Ganesh, Anca Nitulescu, Eduardo Soria-Vazquez.
  *Rinocchio: SNARKs for Ring Arithmetic*. Journal of Cryptology 36,
  Article 41, 2023. ePrint 2021/322. <https://eprint.iacr.org/2021/322>
- **[Sor22]** Eduardo Soria-Vazquez. *Doubly Efficient Interactive Proofs
  over Infinite and Non-Commutative Rings*. TCC 2022, LNCS 13747,
  pp. 497–525, Springer, 2022. ePrint 2022/587.
  <https://eprint.iacr.org/2022/587>
- **[CCCFGS26]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Batch, Pack, and Prove:
  More Efficient Verifiable Computation for CKKS*. ASIACRYPT 2026, to appear.
  (Cited for its sparse-evaluation relation and the protocol proving it
  through one committed dense polynomial, the realization §8 names.)
- **[CCCFGS25]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Verifiable Computation
  for Approximate Homomorphic Encryption Schemes*. CRYPTO 2025, LNCS 16006,
  pp. 643–677, Springer, 2025. ePrint 2025/286.
  <https://eprint.iacr.org/2025/286> (cited for its exceptional-set
  formulation and sumcheck PIOP over `R_q`).
- **[KP23]** Abhiram Kothapalli, Bryan Parno. *Algebraic Reductions of
  Knowledge*. CRYPTO 2023, LNCS 14084, pp. 669–701, Springer, 2023. ePrint
  2022/009. <https://eprint.iacr.org/2022/009>
- **[Tha22]** Justin Thaler. *Proofs, Arguments, and Zero-Knowledge*.
  Foundations and Trends in Privacy and Security 4(2–4), pp. 117–660, now
  publishers, 2022.
  <https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf>
