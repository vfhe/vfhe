<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop — design notes

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

Concrete protocols (sumcheck, zerocheck, …) will be built on top of these
primitives inside this module; applications and compilations to succinct
arguments belong to other modules.

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
| index `i`                   | `Relation.index` (optional, preprocessed)  |
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

In an in-process IOP the `oracles` list holds the actual `MLE` objects — the
same data an honest prover knows in full. The oracle *discipline* (the
verifier only evaluates them at points, it never inspects representations) is
enforced by convention for now; a commitment compiler can enforce it
cryptographically later [BFS20].

## 4. The relation toolbox

Following the multilinear PIOP toolboxes of [Set20, CBBZ23], the module
starts from four relation kinds. In all of them `f` is (an MLE of) an
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
- **`Relation_Eval`** — instance `(f and/or its commitment, z, v)`:
  `f(z) = v`, the terminal claim of the sumcheck family. The polynomial may
  appear as the in-process oracle (`oracles`), as its commitment
  (`commitment`, optional), or both — they are one object at two levels of
  instantiation, "the commitment to f is simply the oracle π_f"
  [ZCF24, §4], so the compiled claim is *the same relation*: the standard
  PCS evaluation relation R_Eval = {[(C, z, y); f] : f(z) = y and C opens
  to f} [ZCF24, Def. 8; BFS20]. A commitment is instance data — created by
  a scheme's commit algorithm (possibly long before any IOP run, reused
  across runs), carried on statements, never on the per-execution
  transcript — and its witness lives in `prover.witnesses[commitment]`.
  Left terminal, the claim is decided by one oracle query (`MLE.evaluate`);
  registering a PCS evaluation protocol (`vfhe.polycom.BasefoldEval`)
  replaces that query with an opening argument — the "queries become
  opening claims" side of the [CHMMVW20] compiler, done in place, with the
  commitment taken from the field or resolved from the scheme's records of
  the oracle. Commitment-only claims have no witness-free decider and must
  not be left terminal.

Further members of the toolbox (product-check, permutation-check, lookups
[CBBZ23]) can be added as new `Relation` subclasses without touching
`Statement`.

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

```
Relation_Zero ────(zerocheck, sample r)──▶ Relation_Sum
Relation_Sum ─────(sumcheck, n rounds)───▶ Relation_Eval
Relation_SumProd ─(sumcheckprod)─────────▶ Relation_Eval × … × Relation_Eval
Relation_Eval^k ──(batching, future)─────▶ Relation_Eval
Relation_Eval ────(oracle query/opening)─▶ accepted / rejected
```

The statements of one run therefore form a **DAG**: each statement records
the bundle it came from in `parents`, and the verifier decides the leaves.
Organizing claims as a DAG has precedent — proof-carrying data defines
transcripts over a DAG [BCCT13], Virgo++ runs GKR over general DAG circuits
with claims flowing along the edges [ZLWZS21] — but the DAG here stays
*implicit* in the `parents` edges plus the worklist driver below; no
dedicated graph object. (Implementations name what a step emits a "claim" or
"subclaim" — e.g. arkworks' `SubClaim` — which is exactly our `Statement`.)

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
  are digested by the duck-typed `element_digest` walker (bytes, ints,
  Polynomials via `get_hash`, MerklePaths, statements, dense MLEs, or any
  object providing `digest()`).
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
- **Registry.** `iop.register(relation_type, protocol)` chooses how
  statements of a relation are discharged; relations without a registered
  protocol are *terminal*. This keeps relations passive data and protocols
  swappable (e.g. a batched sumcheck can replace the plain one without
  touching relations or parties).
- **Drivers.** Both parties run the same worklist loop over the DAG's
  frontier: pop a statement, hand the registered protocol its bundle, push
  the outputs; statements with no registered protocol are terminal. A
  protocol with `batching = True` receives *all* frontier statements of its
  relation in one invocation — a folding reduction `R^ℓ → R` [KP23, Def. 4]
  needs no other machinery. The verifier then decides every terminal leaf
  with its relation's own `check()` — for `Relation_Eval` that is exactly
  one oracle query per claim. A protocol's `verify` half raises `Rejection`
  on a failed round check, which the driver turns into a `False` verdict.
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
  only ever *write* to the transcript. (That invariant is now enforced —
  reading with no counterparty raises instead of hanging, the failure mode
  described in `workflow.md`.)
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

The first instantiation is the sumcheck protocol (`sumcheck.py`): round *i*
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
round-eval helpers from the variable's position. Fields join
`supported_domains` after the MLE layer moves to `vfhe.arith`;
multithreaded kernels are roadmap.

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
the whole field is exceptional and uniform sampling suffices
(`FieldElement.sample_random`; a `sample_exceptional` alias on both `Ring`
and `Field` is planned so the piop module can stay domain-agnostic — future
arith work). The `IOP` object holds the domain and delegates; no
`sampling_set` is ever carried by statements. Sampling happens in exactly one
place — `Verifier.challenge`, which calls `domain.random_exceptional()`
(the name `Ring` already has) — and per-protocol soundness accounting is a
protocol method —
`Sumcheck.soundness_error` reports `d·n/|A|`, with `|A| =
min(p_i)^split_degree` for a `Ring` and `p^d` for a `Field` (duck-typed until
the domain classes expose `|A|` themselves).

## 7. Class-by-class notes

### `IOP`

Owns one execution of a protocol: the asyncio event loop, the coefficient
`domain` (a `Ring` or `Field`, per §6), the `Transcript`, the two parties
(the constructor instantiates `Prover` / `Verifier`, or subclasses passed as
`prover=` / `verifier=` — e.g. a future Fiat–Shamir verifier), and the
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
their private data — the witness never sits on the statement itself); the
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

### `MLE` and `SparseMLE` (`mle.py`)

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
    kernels do the work; without one, entries are plain Python values (any
    type with `+`/`*`, e.g. exact ints for reference-semantics tests) and
    the folds run in Python.

  The kernels are RNS routines *and* interpolate, so they need a ring
  **and** the evaluation basis: `native_table(f)` is that predicate, and it
  — not `isinstance` — is what protocols gate native delegation on (§5).
  `MLE.eq(ring, point)` builds the dense table of the equality
  polynomial `eq̃(point, ·)` (the zerocheck reduction and basefold's
  virtual factor).
- `SparseMLE` — a sparse map of hypercube evaluations (bookkeeping form:
  add / sub / scale only; `evaluate` raises).

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

### `Merkle` (`merkle.py`)

A binary Merkle tree [Mer88] over BLAKE3: the vector commitment that turns a
long prover oracle into a short root plus per-query openings, which is how an
IOP is compiled into an argument [BCS16] and what a code-based
commitment (`vfhe.polycom`) needs for its codewords. Internal nodes are
`BLAKE3(left || right)`; a leaf count that is not a power of two is padded
with zero digests, so the root is defined for any size (the leaf count is
public protocol data, so the padding is not a domain separation concern).

The split follows the rest of the module: the tree — building the levels,
copying a path, replaying it — is C (`c/src/merkle.c`, one contiguous buffer
per level), while the Python layer only decides *what a leaf hashes to*. That
is the sole requirement on a leaf type: a `.hash()` method returning its
digest, or a `hash=` callable given to `Merkle` / `Merkle.verify` for types
without one. Leaves are never copied or interpreted, only referenced and
hashed, so the tree commits to `MLE` tables, codeword entries, or anything
else without knowing what they are.

An opening (`MerklePath`) carries the sibling digests bottom-up and
deliberately *not* the leaf index: the verifier checks the position it
queried, never one the prover chose. Nothing in the file is PIOP-specific —
it is here for want of a crypto-primitives module, and should move when one
exists (like `MLE` moving to `vfhe.arith`, §1).

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
`Relation_Eval` *is* [ZCF24]'s R_Eval, its optional `commitment` field
carrying the compiled form of the oracle; swapping a codeword for a root
changes the type of an instance field, not the relation. Path checking
therefore lives inside the PCS's evaluation protocol
(`vfhe.polycom.BasefoldEval`), which is where the queried positions are.

## 8. Roadmap

1. **Prover kernel follow-ups**: the native Ring paths exist (§5); still
   open are the multithreaded round kernels (the reference `sumCheck_mt`),
   a general-`k` product kernel (native is `k = 2`), the omit-`g(1)`
   message trick [Gru24, §3.1; DT24], and Field domains once the MLE layer
   moves to `vfhe.arith`. The pure-Python fallbacks remain naive
   (per-round hypercube re-enumeration) by design — they are the reference
   semantics, not the fast path.
2. **Zerocheck** (`Relation_Zero → Relation_Sum(Prod)` via `eq̃`): the
   dense `eq̃` table exists (`MLE.eq`); a lazy/virtual-polynomial
   form of `MLE` is still open.
3. **Batching / folding** (`Relation_Eval^k → Relation_Eval`): a
   `batching = True` protocol per [KP23, Def. 4], via random linear
   combination over the exceptional set or the BatchEval PIOP
   [CBBZ23, §3.8]; the driver already supports it.
4. **Lookup relation**, reducing to a mix of Eval and Sum claims.
5. **Field coefficient domains**: `MLE` already has one code path for
   ring-backed and plain-Python coefficients (§7), so a `Field` domain needs
   `vfhe.arith.Field` elements to satisfy the same `+`/`*` protocol plus a
   `sample_exceptional` alias (§6); native kernels for them are a separate
   step.
6. ~~**Evaluation claims backed by a polynomial commitment scheme**~~ —
   done: registering `vfhe.polycom.BasefoldEval` for `Relation_Eval`
   replaces the terminal oracle query with a basefold opening argument
   (commit separately, evaluate many times; the commitment rides the
   statement's optional `commitment` field or is resolved from the
   scheme's records), with codewords committed by Merkle root (§7) and
   spot-checked at positions expanded from `challenge_bits`. A separate
   `Relation_Open` and an `EvalToOpen` bridge existed briefly and were
   folded back in: the bridge was a pure relabeling between two spellings
   of the same relation.
7. ~~**Fiat-Shamir**~~ — done (`fs.FS_Verifier`, `IOP(fiat_shamir=True)`,
   §5): both samplers derived from the transcript's chained `state()`,
   seeded with the root statement digest [DMWG23], plus the non-interactive
   pair `IOP.prove -> Proof` / `IOP.verify(statement, proof)`. Still open:
   a **canonical byte encoding** for proof messages (the prerequisite for
   both a real proof *string* and the planned C verifier, §5), `digest()`
   hooks for any new transcript value types, and a seeded arith-side
   `sample_exceptional` to replace `fs.ring_exceptional_from_seed`.
8. **A native non-interactive verifier**: port the FS verifier to C once
   the encoding above exists — the hot path is hashing and per-round
   coefficient arithmetic, and the Python design already constrains it to a
   single forward pass over the proof (§5).

## Bibliography

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
