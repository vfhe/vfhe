<!-- SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop.range — the range proof, design and implementation plan

> **Under development.** This is the design note *and* the implementation
> plan for `lookup.py` / `range.py`: the two protocols of Appendix I of
> [CCCFGS26] expressed in this module's vocabulary. Sections marked as
> deliverables or verification describe work in progress.

- [Context](#context)
- [Deliverables](#deliverables)
- [Background: the tools](#background-the-tools-being-used)
- [Conventions](#conventions-domains-and-index-order)
- [The lookup protocol (Figure 11)](#the-lookup-protocol--figure-11)
- [The decomposition protocol (Figure 13)](#the-decomposition-protocol--figure-13)
- [Soundness](#soundness-bookkeeping-both-protocols)
- [Files](#files)
- [Verification](#verification)
- [Risks](#risks-in-the-order-they-will-bite)
- [Appendix: what was checked](#appendix-what-was-checked-and-how)

## Context

[CCCFGS26], Appendix I, specifies the range proof this module implements, in
two layers:

- **Figure 13** (`Π̂*range`, §I.5) — the *optimized decomposition* protocol. It
  reduces "every coefficient of every `ṽ(x)` lies in `[0, B)`" to a lookup
  claim over a big prime field `F_p`, plus a handful of evaluation claims. It
  replaces the older Figure 2 / Galois-ring construction.
- **Figure 11** (`Π̂^F_range`, §I.4) — the *lookup argument* over `F_p`
  (Spartan-style memory checking, Lemma I.3, plus Quarks grand products),
  which reduces to two sumchecks and evaluation claims. **Figure 12** defines
  the function `g` that builds the virtual oracle those sumchecks run on.

A standalone reference implementation exists outside the library (the
authors' `vcckks` scratch tree): its `decomposition_prove` /
`decomposition_verify` follow the *older* decomposition protocol, and its
`lookup_prove` / `lookup_verify` follow Figure 11 procedurally, with
hand-rolled sumcheck loops. It is the semantics to agree with, not the
structure to copy.

The goal is to re-express both inside `vfhe` in the language of `vfhe.piop` —
relations, statements, registered protocols, async prove/verify coroutines —
so the range proof becomes a reduction in the framework's statement DAG rather
than a standalone script, plus tests that accept honest decompositions and
reject dishonest ones.

**Where this lands.** The working tree is on `feature/range-proof` at
`9762a2c`, which is exactly `main`'s tip — clean, tracking
`origin/feature/range-proof`, nothing to rebase. That tree already has the
three pieces this protocol needs, and they are what make the construction
below short:

| piece | why it matters here |
| --- | --- |
| `arith.PseudoMersenneField` | the big prime field `F_p`: uniform and seeded sampling, inversion, hashing — enough to serve directly as `IOP(domain=…)` |
| `arith.MLE(field=…)` | field-backed tables, plus a domain-generic `MLE.eq(domain, point)` that comes back marked `public` |
| `piop/virtual.py` | `VirtualOracle`: `p*(x) = Σ c·Π p_j(h_j(x))`, constituents read through variable maps that may pin variables to constants — which is *literally* Figure 12's function `g` |

`piop.md`'s own roadmap, item 4, is "**Lookup relation**, reducing to a mix of
Eval and Sum claims". This work is that item.

**Status.** Both protocols are implemented: `Relation_Range` and
`RangeDecomposition` (Figure 13) in `range.py`, `Relation_Lookup` and `Lookup`
(Figure 11) in `lookup.py`, covered by `test_range.py` and `test_lookup.py`.
A range claim now reduces all the way down — through the lookup, its two
degree-3 zerochecks and their virtual-oracle openings — to evaluation claims
on `ṽ` over `R_q` and on `µ̃'`, `read_ts`, `final_cts` and the `f^{(1)}_α`
over `F_p`, which are left terminal. Compiling those with a PCS is the next
step; see [Sending an oracle](#sending-an-oracle-both-protocols).

## Deliverables

- two new modules, `vfhe/piop/lookup.py` and `vfhe/piop/range.py`;
- two new relations, `Relation_Lookup` and `Relation_Range`, and the two
  protocols that discharge them, `Lookup` and `RangeDecomposition`;
- a design note `modules/piop/range.md` (the background below, expanded);
- two test files, `test_lookup.py` and `test_range.py`.

Everything else is reuse. The whole construction is this much wiring:

```python
iop = IOP(domain=field)                       # field = F_p; or fiat_shamir=True
iop.register(Relation_Range,  RangeDecomposition(field, m, alphabet, kappa))
#                                     m = split point, alphabet = |Sigma| = sigma,
#                                     kappa = repetitions           (all Fig. 13)
iop.register(Relation_Lookup, Lookup(field))
iop.register(Relation_Sum,    Sumcheck())                       # existing
iop.register(Relation_Eval,   VirtualEval(), kind=OracleKind.virtual)  # existing
# Relation_Eval for plain oracles stays unregistered -> terminal oracle query
assert iop.run(Statement(Relation_Range(B, c), oracles=[v_tilde]))
```

In the notation of `piop.md` §5's reduction table, the two new lines are

```text
Relation_Range  ─(RangeDecomposition, κ)─▶ Relation_Lookup × Relation_Eval^{2κ}
Relation_Lookup ─(Lookup, σ_L τ χ ρ ξ)───▶ Relation_Sum^2 × Relation_Eval^4
```

and the run produces this statement DAG (`→` = registered protocol; leaves are
decided by the relation's own `check()`):

```text
Relation_Range(B, c) on ṽ
├── κ × Relation_Eval on ṽ   at x_η              [terminal: one oracle query]
├── κ × Relation_Eval on µ̃'  at (r'_η, s')       [terminal]
└── Relation_Lookup(β) on µ̃'
    ├── Relation_Sum, ℓ* vars, degree 3   → Sumcheck → Eval on the Fig.-12 VirtualOracle
    │     → VirtualEval → Eval on f^(1)_WSµ, f^(1)_RS at r  +  Eval on the virtual f_WSµ, f_RS
    │       → VirtualEval again → Eval on µ̃', read_ts, f^(1)_α at the shifted points [terminal]
    ├── Relation_Sum, b* vars,  degree 3  → … same, ending on t̃β (public), final_cts, f^(1)_α
    └── 4 × Relation_Eval on f^(1)_α at (0,1,…,1) [terminal]
```

## Background: the tools being used

*(This section, expanded, becomes `modules/piop/range.md`.)*

### Asynchronous programming, and why a PIOP library wants it

A *coroutine* is a function that can pause in the middle and hand control back
to a scheduler — the *event loop* — which resumes it later. `await x` is the
pause point: it means "suspend me until `x` has a value". A *future* is that
not-yet-available value: an empty box someone else will fill. None of this is
parallel or threaded. Exactly one coroutine runs at a time and the loop only
switches at `await` points, so there are no races and no locks.

What it buys a proof system is this: an interactive proof is a *data-flow*
problem — the prover's round-`i+1` message depends on the verifier's round-`i`
challenge, but anything not downstream of that challenge can proceed. With
coroutines, prover and verifier are each written as one straight-line script in
the natural order of the protocol, and the interleaving is *inferred* rather
than hand-scheduled. `vfhe.piop` does exactly that (the class is called `IOP`;
"PIOP" is the model it implements):

```python
async def prove(self, prover, statements):            # the prover's script
    iop.transcript.write("msg1", table)               # fills a future
    r = iop.verifier.challenge("chal1")               # derived, not awaited
    iop.transcript.write("msg2", f(r))
    return [ ... ]                                    # the statements this reduces to

async def verify(self, verifier, statements):         # the verifier's script
    t = await iop.transcript.read("msg1")             # suspends until prove writes it
    r = verifier.challenge("chal1")                   # same value, same label
    m2 = await iop.transcript.read("msg2")
    if not check(t, r, m2):
        raise Rejection("...")
    return [ ... ]                                    # the identical list
```

Neither side declares a round number; the ordering is forced by who awaits
what. `iop.run(statement)` schedules both coroutines on one loop and returns
the verifier's verdict. Under `IOP(fiat_shamir=True)` the halves come apart:
`iop.prove(stmt) -> Proof`, and on a *separate* `IOP`,
`iop.verify(stmt, proof) -> bool`. (Two implementation details worth knowing:
under Fiat–Shamir every challenge is a hash of the transcript so far, so the
prover never actually waits; and a prover coroutine that tries to *read*
raises rather than hanging, since it has no counterparty.)

**The one rule this imposes on protocol code:** both halves must emit the
*identical* sequence of child statements. Child statement `path`s come from a
counter on the parent, and `path` namespaces every transcript label, so a
verify half cannot look at a witness to decide how many children to make.

### The PIOP scaffolding

- **`Relation`** — an indexed relation `R ⊆ (index, instance, witness)`.
  `fields` declares the instance shape in the canonical Fiat–Shamir order;
  `check(statement)` is the *ideal* (hypercube-enumerating) decider.
- **`Statement`** — the claim `instance ∈ L(relation)`. Public only; the
  witness lives in `Prover.witnesses`. `parents` links the run's DAG.
- **`Protocol`** — a reduction between *products* of relations, as the pair of
  coroutines above, each `list[Statement] -> list[Statement]`.
  `iop.register(RelationType, protocol)` picks the discharge. A statement whose
  relation has no registered protocol is **terminal**: it is not reduced
  further, and the verifier decides it directly with `check()`.
- The two relations that carry this whole construction:
  - **`Relation_Sum`** — `Σ_{x ∈ {0,1}^n} f(x) = value`, discharged by
    `Sumcheck` in `n` rounds, reducing to
  - **`Relation_Eval`** — `f(point) = value`, the terminal claim. The
    instance may carry `f` as an in-process oracle, as a `commitment`, or both.
- **Oracle kinds** (`OracleKind`) — `Relation_Eval` is *one* relation with
  several discharges, keyed on what the verifier *has* of the oracle:
  `public` (terminal, the verifier evaluates the table itself), `committed`
  (a PCS protocol), `virtual` (`VirtualEval`), `implicit` (`ImplicitEval`),
  `plain` (terminal oracle query). `register` takes a `kind=` argument.
- **`VirtualOracle(variables, constituents, terms, maps)`** —
  `p*(x) = Σ_terms coeff · Π_j p_j(h_j(x))`. `terms` is `[(coeff, (j, …)), …]`
  over constituent indices; `maps[j]` sends each variable of constituent `j`
  either to one of `variables` (a renaming) or **to a constant**.
  `degree_in(var)` is the largest number of factors of one term reading `var`.
- **`Sumcheck` already accepts a `VirtualOracle`** as the oracle of a
  `Relation_Sum`: the round message carries `degree_in(var) + 1` evaluation
  nodes. `VirtualEval` then rewrites the final `Relation_Eval` on the virtual
  oracle: for each distinct (constituent, point) pair it emits one
  `Relation_Eval`. Constituents marked `public` are dropped, because the
  verifier evaluates them itself. The rewrite draws no challenges and is a
  deterministic identity, not a probabilistic reduction, so it costs no
  soundness.

That last pair of bullets is why this maps onto `main` so cleanly: **Figure
12's `g` is a `VirtualOracle`**, and Figure 11's two sumchecks are two
`Relation_Sum` statements over one.

### Notation

Paper notation, all of it used below:

| symbol | meaning |
| --- | --- |
| `R_q = Z_q[Y]/(Y^N+1)`, `q = Π p_l` | the ring `ṽ` lives in; `N = 2^ν` |
| `ṽ` | the oracle under test: `ℓ`-variate multilinear over `R_q` |
| `B = 2^b` | the coefficient bound; `T_B` is the paper's table of in-range ring elements |
| `c = 2^γ`, `β = B^{1/c} = 2^{b*}` | number of digits per coefficient, and the digit alphabet |
| `t_β = {0,…,β−1}`, `t̃β` | the lookup table and its MLE (a *public* oracle) |
| `ℓ* = ℓ + ν + γ` | variables of the digit polynomial |
| `ũ = unpack(ṽ)` | `ũ(x, k) = ṽ(x)[k]`: the `Z_q` coefficients spread over `ν` extra variables (§3.1). Never committed — queried through Lemma 3.1 |
| `µ̃` | the integer digit polynomial, `µ̃(i,k,j) = dig_j^{(β)}(ṽ(i)[k])`, the `j`-th base-`β` digit of the `k`-th coefficient of `ṽ(i)` |
| `µ̃' = π_p(µ̃)` | its reduction into `F_p` — the prime marks "over `F_p`", not a derivative |
| `ι` | the balanced lift `F_p → I_p = {−⌊p/2⌋,…,⌊p/2⌋} ⊂ Z` |
| `Σ = {0,…,σ−1} ⊂ Z`, `m ≤ ℓ`, `κ` | the small-integer challenge alphabet, the split point, the repetition count (Fig. 13). `σ ≤ p_min` is required, so that `π_q(Σ)` lands in the exceptional set `Ŝ_0 ⊂ Z_q` and the query to `ṽ` is sound |
| `r_η ∈ Σ^{ℓ+ν}` | repetition `η`'s challenge. Split as `x_η = (r_{η,1..ℓ})` (the `X` block, for `ṽ`) and `z_η = (r_{η,ℓ+1..ℓ+ν})` (the `Z` block); and separately as `r'_η = (r_{η,1..m})` (bound into `ψ̃'`) and `r_η^{>m} = (r_{η,m+1..ℓ+ν})` (used as `eq̃` weights) |
| `ψ̃'_η = µ̃'(r'_η, ·)` | the partial evaluation the prover sends, over `ℓ*−m` variables; `δ ∈ {0,1}^{ℓ+ν−m}` and `k ∈ {0,1}^γ` index it |
| `s' ∈ F_p^{ℓ*−m}` | the spot-check point for `ψ̃'_η` |
| `σ_L`, `τ`, `χ`, `ρ`, `ξ` | Figure 11's challenges: two fingerprints, one batching scalar, and two `eq̃` points (`ρ ∈ F_p^{ℓ*}`, `ξ ∈ F_p^{b*}`) |
| `α ∈ {WSµ, RS, WSt, S}` | the four multiset fingerprints of Lemma I.3: write-set and read-set on the `µ̃` side, write-set and final-counts on the table side |
| `f^{(0)}_α`, `f^{(1)}_α`, `f_α` | the grand-product layers: leaves, internal nodes, and the `(n+1)`-variate polynomial combining them, with `n = ℓ*` for `α ∈ {WSµ, RS}` and `n = b*` for `α ∈ {WSt, S}` |

> **Name clash, inherited from the paper.** §I uses `σ` for `|Σ|` and Figure 11
> uses `σ` for the lookup fingerprint challenge. Here the fingerprint is
> written `σ_L` (code: `zeta`), and `σ` keeps its §I meaning (code:
> `alphabet`).

## Conventions: domains and index order

These are the things that are easiest to get wrong; fix them before writing
any code.

### The challenge domain

**`iop.domain = F_p`**, a `PseudoMersenneField`. Every `challenge()` returns an
`F_p` element, and `exceptional_set_size` is `p` itself.

`ṽ` is a table of `R_q` elements, not `F_p` elements. The two coexist because
`iop.domain` is consulted in exactly one place — when the verifier samples a
challenge — and the only claim ever made about `ṽ` is a terminal evaluation at
a point with plain-integer coordinates, which a ring-backed table can bind.
(This combination is untested on `main`; see [Risk 3](#risks-in-the-order-they-will-bite).)

The small-integer challenges `r_η ∈ Σ^{ℓ+ν}` are **not** domain challenges.
They come from `verifier.challenge_bits(label, bits)` — raw published coins
that the protocol interprets however it needs — reduced mod `σ`, with `σ` a
power of two so there is no bias.

### LSB-first table order

`MLE` tables are LSB-first: `variables[i]` is bit `i` of the table index.

**`µ̃'`** has variables `[X_1..X_ℓ, Z_1..Z_ν, Z'_1..Z'_γ]`, so
`index = i + 2^ℓ·k + 2^{ℓ+ν}·j` and `µ̃'(i,k,j) = dig_j^{(β)}(ṽ(i)[k])`
(paper §C.1). The `X` block varies fastest, the digit index `j` slowest. At
toy parameters `ℓ = ν = γ = 1`:

| index | `X_1` | `Z_1` | `Z'_1` | `(i,k,j)` | value |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | (0,0,0) | `dig_0(ṽ(0)[0])` |
| 1 | 1 | 0 | 0 | (1,0,0) | `dig_0(ṽ(1)[0])` |
| 2 | 0 | 1 | 0 | (0,1,0) | `dig_0(ṽ(0)[1])` |
| 3 | 1 | 1 | 0 | (1,1,0) | `dig_0(ṽ(1)[1])` |
| 4 | 0 | 0 | 1 | (0,0,1) | `dig_1(ṽ(0)[0])` |
| … | | | | | |

**The grand-product array** is `f[2^n + k] = f[2k]·f[2k+1]` for
`k = 0 … 2^n−2`, with the last entry **pinned to zero**, `f[2^{n+1}−1] = 0`,
and `P = f[2^{n+1}−2]`. The pinning is not cosmetic: applying the recurrence at
`k = 2^n−1` would be self-referential (`f[2^{n+1}−1] = P·f[2^{n+1}−1]`), and
the zerocheck below is only identically zero on the *whole* hypercube because
both sides vanish there. The old `scripts/grand_product.py` does exactly this
(`f[n-1] = 0`, and the fill loop stops one short); the paper's Lemma I.4
carries the same convention implicitly. Reading the array LSB-first, with
`f_α`'s variables `W = [W_0 … W_n]` and `index = W_0 + 2W_1 + … + 2^n W_n`, at
`n = 2`:

| index | `W_0 W_1 W_2` | half | contents |
| --- | --- | --- | --- |
| 0 | 0 0 0 | `f^(0)` | leaf 0 |
| 1 | 1 0 0 | `f^(0)` | leaf 1 |
| 2 | 0 1 0 | `f^(0)` | leaf 2 |
| 3 | 1 1 0 | `f^(0)` | leaf 3 |
| 4 | 0 0 1 | `f^(1)` | `f[0]·f[1]` |
| 5 | 1 0 1 | `f^(1)` | `f[2]·f[3]` |
| 6 | 0 1 1 | `f^(1)` | `f[4]·f[5] = P` |
| 7 | 1 1 1 | `f^(1)` | `0` (pinned, not computed) |

With arguments listed in variable order `W_0 … W_n`, `f_α(X, 1)` is index
`int(X) + 2^n`, while `f_α(0, X)` and `f_α(1, X)` are indices `2·int(X)` and
`2·int(X) + 1`. Check one row: for `X = (1,0)`, `f_α(X,1) = f[5]` and
`f_α(0,X)·f_α(1,X) = f[2]·f[3]` — which is exactly what index 5 holds. So

- the **bottom half is `f^{(0)}_α`, index-aligned with `µ̃'` entry for entry** —
  this alignment is the whole reason to read LSB-first, and it is what removes
  every bit reversal from the implementation;
- the **top half is `f^{(1)}_α`**, also index-aligned;
- the identity is `f_α(X, 1) = f_α(0, X) · f_α(1, X)` for **all** `X ∈ {0,1}^n`
  — including `X = (1,…,1)`, where the pinned `f[2^{n+1}−1] = 0` makes both
  sides zero (`0 = P · 0`). This is what makes the zerocheck claim exactly `0`;
- `P_α = f_α(0, 1, …, 1)`, index `2 + 4 + … + 2^n = 2^{n+1} − 2`, i.e. entry
  `2^n − 2` of the top half. (The paper writes this query point as `(1,0)`,
  meaning `(1,…,1,0)` in its own argument order.)

*Checked numerically for `n = 2, 3, 4` before writing this plan.*

> **Comparing with the paper side by side.** Figure 11 writes the layer
> selector as the *first* argument (`f(1,i) = f(i,0)·f(i,1)`, with
> `f^{(1)}_α(i) = f^{(i_1)}_α(i', 0)·f^{(i_1)}_α(i', 1)`); above it is the
> *last* variable `W_n`, and the new bit is inserted at `W_0`. These are the
> same binary tree read with "first" and "last" exchanged — the paper's
> reading is the MSB-first reading of the very same flat array. Nothing in the
> protocol depends on which naming is used (`ρ` and `ξ` are uniform), and
> putting the selector last is what buys the index alignment with `µ̃'`.

## The lookup protocol — Figure 11

Taken first, because Figure 13 reduces to it.
New file **`modules/piop/python/src/vfhe/piop/lookup.py`**.

```python
class Relation_Lookup(Relation):
    """Every hypercube evaluation of the oracle lies in t_beta = {0..beta-1}."""
    name = "lookup"
    fields = ("oracles",)
    def __init__(self, beta: int) -> None: super().__init__(index=beta)
```

`check()` walks the hypercube and tests `0 <= int(value) < beta`.

Everything below is done **twice**: once on the `µ̃'` side, where the
fingerprint pair is `α ∈ {WSµ, RS}` and `n = ℓ*`, and once on the table side,
where it is `α ∈ {WSt, S}` and `n = b*`. The two are independent apart from
sharing the challenges `σ_L, τ, χ`.

`Lookup(Protocol)`: `reduces_from = Relation_Lookup`,
`reduces_to = (Relation_Sum, Relation_Eval)`. Steps 1–2 below are Figure 11's
steps 1–2; steps 3–6 are all of its step 3.

1. Build `read_ts` (`ℓ*` vars) and `final_cts` (`b*` vars) by the counter
   algorithm of Lemma I.3 — the reference tree's `create_counters_optimized`,
   over `F_p`. One pass:
   `read_ts[i]` is how many times the value `µ̃'[i]` has been seen *before*
   position `i`, and `final_cts[v]` is the total count of `v`. For
   `β = 4`, `µ̃' = [2, 0, 2, 2]`: `read_ts = [0, 0, 1, 2]` and
   `final_cts = [1, 0, 3, 0]`. Send both (see
   [Sending an oracle](#sending-an-oracle-both-protocols)). Draw `σ_L`, `τ`.
2. Build the four leaf tables `f^{(0)}_α` and their grand products; send the
   four top halves `f^{(1)}_α`. Draw `χ`, `ρ ∈ F_p^{ℓ*}`, `ξ ∈ F_p^{b*}`.

   The leaves are the paper's fingerprints, eqs. (41)–(43):

   | `α` | `f^{(0)}_α[i]` | over |
   | --- | --- | --- |
   | `WSµ` | `µ̃'[i] + σ_L·(read_ts[i] + 1) − τ` | `2^{ℓ*}` entries |
   | `RS` | `µ̃'[i] + σ_L·read_ts[i] − τ` | `2^{ℓ*}` |
   | `WSt` | `t̃β[i] − τ` | `2^{b*}` |
   | `S` | `t̃β[i] + σ_L·final_cts[i] − τ` | `2^{b*}` |

   Only `f^{(1)}_α` is sent. `f^{(0)}_α` is *not* an oracle: the verifier can
   reconstruct it from oracles it already has (`µ̃'`, `read_ts`, `final_cts`)
   and the public `t̃β`, which is the whole point of steps 3–4.
3. Express that reconstruction as a `VirtualOracle`. The combined layered
   polynomial is `f_α = (1 − W_n)·f^{(0)}_α + W_n·f^{(1)}_α` over
   `W = [W_0 … W_n]`; the `(1 − W_n)` / `W_n` selectors are just two
   one-variable public tables, and each summand of `f^{(0)}_α` becomes one
   term. For `α = WSµ`:

   ```python
   notlast = MLE.eq(field, [0], variables=[w])   # public, table [1, 0] = 1 - w
   last    = MLE.eq(field, [1], variables=[w])   # public, table [0, 1] = w
   onto_W  = lambda t: {v: W[i] for i, v in enumerate(t.variables)}
   f_WSmu = VirtualOracle(
       variables=W,
       constituents=[notlast, last, mu_prime, read_ts, f1_WSmu],
       terms=[(one,        (0, 2)),     # (1-W_n) * mu'
              (zeta,       (0, 3)),     # (1-W_n) * sigma_L * read_ts
              (zeta - tau, (0,)),       # (1-W_n) * (sigma_L - tau)
              (one,        (1, 4))],    # W_n * f^(1)
       maps=[{w: W[n]}, {w: W[n]},
             onto_W(mu_prime), onto_W(read_ts), onto_W(f1_WSmu)],
   )
   ```

   The third term is why the selectors are needed at all: a `VirtualOracle`
   term must index at least one constituent, so a bare constant has nowhere to
   live — multiplying it by `notlast` is both the fix and the correct algebra.
   `RS` differs only in that constant (`−τ` instead of `σ_L − τ`); `WSt` uses
   the public `t̃β` with `−τ`; `S` uses `t̃β`, `σ_L·final_cts`, `−τ`.
4. Build the two child views of the shift, over the sumcheck variables
   `X = [X_0 … X_{n−1}]`:

   ```python
   child0 = {W[0]: 0} | {W[i+1]: X[i] for i in range(n)}   # f_a(0, X)
   child1 = {W[0]: 1} | {W[i+1]: X[i] for i in range(n)}   # f_a(1, X)
   ```

   The parent slot needs no view of `f_α` at all: `f_α(X, 1) = f^{(1)}_α(X)`
   by construction, so the real oracle `f^{(1)}_α` goes in directly, read under
   the plain renaming `{f1.variables[i]: X[i]}`. (Putting `f_α` there instead
   would work arithmetically but would make `claims()` — which is structural
   and cannot see that the `(1 − W_n)` factor is zero at `W_n = 1` — emit
   provably-zero claims on `µ̃'` and `read_ts`: two wasted PCS openings per
   fingerprint under the compiled variant.)
5. Emit **two** `Relation_Sum` statements with `value = field.zero`: one over
   `ℓ*` variables with `eq̃(ρ,·)` combining `WSµ` and `RS`, one over `b*`
   variables with `eq̃(ξ,·)` combining `WSt` and `S`. Build each as a single
   `VirtualOracle` with `eq̃` as one shared constituent and four terms:

   ```python
   constituents = [eq_rho, f1_WSmu, f_WSmu, f_WSmu, f1_RS, f_RS, f_RS]
   terms = [( one, (0, 1)), (-one, (0, 2, 3)),
            ( chi, (0, 4)), (-chi, (0, 5, 6))]
   maps  = [identity, rename1, child0, child1, rename2, child0, child1]
   ```

   This is Figure 12 transcribed: one `eq̃(ρ, X)` factor, `f_1(1,X) −
   f_1(X,0)·f_1(X,1) + χ·(…)`. `degree_in` is 3 in every variable (`eq̃ · f · f`
   in the second and fourth terms), matching the "degree 3 in each variable"
   of the proof of Thm. I.5.

   **Invariant:** the `WSµ` and `RS` halves must share the *same* `X` objects
   (and likewise `WSt`/`S`). `MLE_Variable` compares by identity, so two
   separately built `X` lists would give a `2n`-variable oracle — twice the
   rounds, wrong claim — silently. This is also why the combined oracle is
   built directly rather than with `VirtualOracle.linear([one, chi], […])`,
   which would additionally list `eq̃` twice.
6. Emit the four `Relation_Eval` claims on `f^{(1)}_α` at `(0, 1, …, 1)`. The
   prover writes the four values; the verifier reads them, checks
   `P_WSt · P_WSµ = P_RS · P_S` — raising `Rejection` otherwise — and puts the
   *read* values on the four claims. (In the ideal model the verifier holds the
   tables and could read `P_α` off them directly, making the message
   redundant; sending it is what keeps the step intact once the oracles become
   commitments. Note also that the paper's proof of Thm. I.5 calls this "step
   6" while Figure 11 numbers it 3 — the figure is the right reading.)

The prover records `prover.witnesses[f_α] = <the (n+1)-variable table>` — the
same handoff `GKR` uses for its layer tables (`circuit.py`), and the only way
`resolve_table` can find concrete data for a virtual constituent. It is
mandatory, not an optimization; see [Risk 1](#risks-in-the-order-they-will-bite).

> **Not the `ImplicitOracle` grand product.** `test_virtual.py` already has a
> grand product, `L_i(z) = Σ_p eq̃(z,p)·L_{i+1}(p,0)·L_{i+1}(p,1)` — the
> Thaler/GKR form, which must be *implicit* because multilinear extension does
> not commute with products. Figure 11 uses Quarks' form instead: the internal
> nodes `f^{(1)}_α` are sent as an oracle, and the tree identity becomes a
> zerocheck on `f_α(X,1) − f_α(0,X)·f_α(1,X)` — a claim about a polynomial the
> verifier can already name. So there is no `ImplicitOracle` here, no
> `ImplicitEval`, and no bundling: just a `VirtualOracle` and a
> `Relation_Sum`.

### What the framework does after step 5

The paper's verifier recombines the sumcheck's final evaluations by hand
(`f10/f11/f20/f21` in the old `lookup_verify`). Here that recombination is the
framework's standard chain, and it is worth spelling out, because it is this
plan's central bet:

1. `Sumcheck` runs `n` rounds on the `Relation_Sum` statement. Because the
   oracle is a `VirtualOracle`, each round message carries `degree_in(var)+1 = 4`
   evaluation nodes, and the prover computes them from `prover_view` — the
   definition with every constituent resolved to a table and folded in place.
2. The sumcheck ends in `Relation_Eval` on the *combined* virtual oracle at the
   sumcheck point `r`. Its kind is `virtual`, so `VirtualEval` discharges it.
3. `VirtualEval` asks the oracle for `claims(r)`: one entry per distinct
   (constituent, point) pair, deduplicated by constituent *identity*. `eq̃` is
   `public` and is dropped — the verifier evaluates `eq̃(ρ, r)` itself. That
   leaves six: `f^{(1)}_WSµ` at `r`, `f_WSµ` at the child0 and child1 points,
   and the same three for `RS`. The prover sends the six values; the verifier
   checks that they recombine, through the four terms of step 5, to the
   sumcheck's final claim.
4. The four claims on `f_α` are `Relation_Eval` on a `VirtualOracle` again, so
   `VirtualEval` fires a second time. `notlast`, `last` and `t̃β` are public and
   are evaluated by the verifier; what remains are claims on the real oracles
   `µ̃'`, `read_ts`, `final_cts` and `f^{(1)}_α` at the shifted points — exactly
   the `evals[2..9]` of the old implementation, with the linear recombination
   it spells out as `f10/f11/f20/f21` performed for us.
5. Those claims are `plain`, hence terminal: one oracle query each.

One deliberate deviation from the paper, to be noted in `range.md`:

- The counter construction of Lemma I.3 is only defined for in-range values, so
  `Lookup.prove` must reject an out-of-range entry with a clear error rather
  than index out of bounds — and the rejection tests must therefore go through
  a *cheating prover* that fabricates counters, so the `False` verdict comes
  from the verifier and not from a prover crash.

## The decomposition protocol — Figure 13

New file **`modules/piop/python/src/vfhe/piop/range.py`**.

```python
class Relation_Range(Relation):
    """Every coefficient of every hypercube evaluation of the R_q oracle
    lies in [0, B): the table T_B of the paper."""
    name = "range"
    fields = ("oracles",)
    def __init__(self, bound: int, chunks: int) -> None:   # B, c
        super().__init__(index=(bound, chunks))
```

`check()` walks the hypercube and tests `max(poly.get_polynomial()) < B`.

`RangeDecomposition(Protocol)`: `reduces_from = Relation_Range`,
`reduces_to = (Relation_Lookup, Relation_Eval)`. The constructor takes
`(field, m, alphabet, kappa)`; `B` and `c` come from the relation's index,
`N`/`ν` from `ṽ.ring`, `ℓ` from `ṽ.num_vars`.

1. The prover builds the integer digit table `µ̃` by reading each
   `Polynomial.get_polynomial()` and slicing out `c` base-`β` digits, lifts it
   to `µ̃' = MLE(field=F_p, …)`, and sends it. (Not via
   `Polynomial.decompose`, which produces `ring.bit_size // base` limbs by
   floor division, is unsigned, has zero callers and no test — it is the wrong
   shape here.)
2. `r_0 … r_{κ−1} ← Σ^{ℓ+ν}` via `challenge_bits`.
3. The prover sends `ψ̃'_η = µ̃'(r'_η, ·)` for each `η` — bind the first `m`
   variables, `2^{ℓ+ν−m+γ}` field elements each — and the `κ` ring values
   `w_η = ṽ(x_η)`. Note the two liftings of the same integer challenge: `ṽ` is
   bound at the plain integers `x_η` (a ring table takes an `int` directly),
   while `µ̃'` must be bound at `π_p(r'_η)`, i.e. `field(r)` per coordinate — a
   field-backed table folds through `FieldVector.fold`, which wants a field
   element, not an `int`.
4. `s' ← F_p^{ℓ*−m}` via `challenge`.
5. The verifier checks, for each `η`: compute both sides as exact Python
   integers, using the balanced lift `ι` to leave `F_p`, then reduce and
   compare mod `q`:

   ```text
   Σ_{δ,j} ι(ψ̃'_η[δ,j]) · eq̃(δ; r_η^{>m}) · β^{to-int(j)}   ≡   uEval(w_η, z_η)   (mod q)
   ```

   where `uEval(a, z) = Σ_{k∈{0,1}^ν} a[k] · eq̃(k, z)` (Lemma 3.1). This is how
   `ũ = unpack(ṽ)` is queried without anyone ever committing to `ũ`: a single
   query to `ṽ` plus `O(N)` local work. Checking mod `q` is checking mod every
   `p_l`, by CRT.

   *Index warning:* Figure 13 writes the digit index as `k` in this formula,
   having used `j` for it in eq. (4). This plan uses `j` for the digit and `k`
   for the coefficient **everywhere**; transposing the two blocks is the
   easiest way to get this wrong.
6. Emit `κ` `Relation_Eval` on `µ̃'` at `(r'_η, s')` with value `ψ̃'_η(s')`
   (both parties compute it from the `ψ` message), `κ` `Relation_Eval` on `ṽ`
   at `x_η` with value `w_η`, and one `Relation_Lookup(β)` on `µ̃'`.

## Soundness bookkeeping (both protocols)

Each protocol gets a `soundness_error` method, matching the convention the
existing protocols follow (`Sumcheck.soundness_error`, `piop.md` §6), reporting
the paper's own bounds:

- `RangeDecomposition` (Thm. I.6): `max{ ((ℓ+ν)/σ)^κ , (ℓ+ν+γ−m)/p , δ'_range }`,
  where the third term is what the child `Relation_Lookup` accounts for;
- `Lookup` (Thm. I.5): `δ'_range = (2^{ℓ*} + 2^{b*}) / p`.

The correctness side conditions — `p > 2β(2σ)^m` (eq. 38) and `p > 2^{ℓ*}` —
are asserted in the protocol constructors, so a bad parameter set fails loudly
at construction rather than producing a proof that is quietly unsound.

## Sending an oracle (both protocols)

In the ideal model a prover message *is* an oracle, so `µ̃'`, `read_ts`,
`final_cts` and the four `f^{(1)}_α` are written to the transcript and read
back by the verifier.

They are written as plain **tuples of field elements**, not as `MLE` objects.
`element_digest` would handle either (it has a dense-MLE branch), so the reason
is variable identity: `MLE_Variable`s compare by identity, and each party
should build its own table, with its own variable objects, from a message that
is plain data. It also keeps the message a value rather than a live object,
which is what the planned canonical byte encoding (`piop.md` §8, item 6) will
need.

The compiled variant is a contained change, not a rewrite: the prover commits
with `vfhe.polycom` and puts the commitment on the statements instead of the
table; `Relation_Eval` is registered with `kind=OracleKind.committed`; the PCS
protocol discharges those claims. The relations, the protocols and the DAG are
unchanged. `range.md` will record this.

## Files

| file | what |
| --- | --- |
| `modules/piop/python/src/vfhe/piop/lookup.py` | **new** — `Relation_Lookup`, `Lookup`, counter builder, grand-product builder |
| `modules/piop/python/src/vfhe/piop/range.py` | **new** — `Relation_Range`, `RangeDecomposition`, digit decomposition, `uEval` |
| `modules/piop/python/src/vfhe/piop/__init__.py` | add the four names to the imports and `__all__` |
| `modules/piop/range.md` | **new** — the background above, expanded: the async/PIOP tour, the two figures, the index conventions, the soundness parameters, the compiled-PCS variant |
| `modules/piop/piop.md` | one-line update: roadmap item 4 done, pointer to `range.md` |
| `modules/piop/python/test/test_lookup.py` | **new** |
| `modules/piop/python/test/test_range.py` | **new** |

Reused, not rewritten: `VirtualOracle` / `VirtualEval` / `OracleKind`
(`piop/virtual.py`), `Sumcheck` (`piop/sumcheck.py`), `MLE` / `MLE.eq` /
`MLE.rename` (`arith/mle.py`), `PseudoMersenneField` (`arith/impl/pmf/`),
`Ring` / `Polynomial.get_polynomial` (`arith/impl/rns/polynomial.py`), and the
`Prover.witnesses` handoff pattern from `piop/circuit.py::GKR.prove`.

## Verification

**Done when:** (i) an honest `ṽ` with every coefficient in `[0,B)` is accepted
end to end at the test parameters below; (ii) each cheating strategy listed is
rejected, and rejected *for the reason named*; (iii) the Fiat–Shamir round trip
across two separate `IOP`s accepts and is byte-reproducible.

Test parameters, small enough for the pure-Python degree-3 rounds:
`N = 8` (`ν = 3`), `ℓ = 2`, `B = 2^6`, `c = 2` (so `γ = 1`, `β = 8`, `b* = 3`),
giving `ℓ* = 6` and 64-entry tables; `m = 2`, `σ = 2^8`, `κ = 2`;
`p = PseudoMersenneField.generate(260, two_adicity=8)` — the existing fixture in
`test_field_domain.py`, reused verbatim — so `p ≈ 2^260`, against the correctness
bound `2β(2σ)^m = 2^22` and the lookup requirement `2^{ℓ*} = 2^6`. (`κ = 2` is
a test-size choice, not a security one; `range.md` records the real bound
`((ℓ+ν)/σ)^κ ≈ 2^{−λ}`, eq. 44.) `σ = 2^8 ≤ p_min` holds for any realistic
prime size.

**Pre-flight, not an assumption:** nothing on `main` builds a ring this small —
every `Ring` in `modules/piop` and `modules/polycom` is `N = 512` or `N = 1024`,
and `modules/arith`'s own tests bottom out at `N = 2^9`. Bare `Ring(8)` raises
(`prime_size` defaults to the *int* `49`, which is not one of the four accepted
modulus specifications). The call to try first is

```python
ring = Ring(8, prime_size=[30, 30], split_degree=1)   # rou_order = 2N/split_degree = 16
```

and step zero of the implementation is to check that it constructs and that
`MLE(ring=ring, …).evaluate({var: 3})` returns the right thing. If a small `N`
does not survive the native NTT, fall back to `N = 16` (`ν = 4`, `ℓ* = 7`), and
beyond that to `N = 512` with `ℓ = 1`, `c = 2` (`ℓ* = 11`), which is slower but
certainly supported.

`test_lookup.py`

- `Relation_Lookup().check()` accepts an in-range table and rejects one
  out-of-range entry.
- An honest run accepts. A cheating prover that fabricates counters for an
  out-of-range entry is rejected — by the step-6 product check or by the
  sumcheck, and the test asserts which.
- The transcript labels are the expected sequence, and both `Relation_Sum`
  children report `degree == 3` (round messages of 4 nodes).
- Fiat–Shamir: `IOP(fiat_shamir=True).prove` then a *separate* `IOP(…).verify`
  accepts; two runs are byte-identical.
- Replay an honest transcript into a fresh IOP with one label rewritten and
  assert rejection (the `_replay_with_tamper` harness of
  `modules/polycom/python/test/test_basefold.py`), keeping the identity-tamper
  positive control.

`test_range.py`

- A direct unit test of the Figure-13 step-2 identity on honest data — "the
  decomposition was done correctly": the digits recompose to the coefficients,
  and `Σ_{δ,k} ι(ψ̃')·eq̃·β^k ≡ uEval(ṽ(x_η), z_η) (mod q)`.
- End to end: `Relation_Range` over a `ṽ` with coefficients in `[0,B)` accepts.
- Rejection, as three separate cases, each naming the branch that catches it:
  1. a coefficient `≥ B` — the digits cannot recompose, so the **mod-`q` check**
     in step 5 fails;
  2. an honest `ṽ` with a prover that perturbs one digit of `µ̃` (subclass the
     protocol to corrupt the table) — again the **mod-`q` check**;
  3. an honest `ṽ` with a `µ̃` whose digits recompose correctly but where one
     digit exceeds `β` (borrow a unit from a higher digit) — this passes step 5
     and is caught by the **lookup branch**.
- The `κ` claims on `ṽ` and the `κ` on `µ̃'` reach the terminal frontier, and
  each `check()`s true on an honest run.

(The core arithmetic was already checked numerically before writing this plan;
see the [appendix](#appendix-what-was-checked-and-how). These tests are what
pins it inside the framework.)

Commands:

```bash
python3 -m pytest modules/piop modules/polycom -q
```

```bash
ruff check modules/piop/python
```

No C is added, so no rebuild is required.

## Risks, in the order they will bite

1. **Nested `VirtualOracle` as a constituent.** `f_α` is virtual and appears as
   a constituent of the Figure-12 oracle. Virtual-inside-virtual has no test on
   `main`, but both call paths were traced and they hold: `resolve_table`
   returns `witnesses[f_α]` (a `VirtualOracle` has no `__eq__`, so it is a
   valid identity-keyed dict key); `_dense_constituent` binds the map's fixed
   coordinate *out of place*, so the one shared witness table survives being
   read under two child maps; and `claims()` dedups by constituent identity, so
   the two shifted views stay two claims.
   **The one hard requirement is that `prover.witnesses[f_α]` be set.** This is
   the same handoff `GKR` uses for its layer tables, but *not* with the same
   safety net: `ImplicitOracle` defines `materialize()`, so a missing witness
   there merely falls back to a slow path, whereas `VirtualOracle` has neither
   `.table` nor `materialize`, so a missing witness is a hard
   `TypeError: no table for oracle` — raised from `VirtualOracle.prover_view`
   during `Sumcheck.prove`, and again from `VirtualEval.prove`.
   *Fallback if the nesting misbehaves anyway:* flatten `f_α` into the terms.
   Under the child maps the `W_n` indicator becomes `X_{n−1}` rather than a
   constant, so each child view is 4 summands and each product is 16 terms;
   with the parent slot already collapsed to `f^{(1)}_α`, that is
   2 fingerprints × (1 + 16) = 34 terms per sumcheck instead of 4. Mechanical,
   slower, no framework dependency.
2. **`VirtualOracle` over a `Field` domain, end to end.** `test_libra.py`
   exercises `prover_view` over a field and `test_virtual.py` exercises the
   full IOP over integers and `Ring`, but no test crosses the two. Write a
   small degree-2 virtual sumcheck over `PseudoMersenneField` first, before the
   real protocol.
3. **Mixed domains in one run.** `ṽ` over `R_q`, everything else over `F_p`,
   `iop.domain = F_p`. Verify early that a ring-backed `MLE` binds plain
   integer points and that a terminal `Relation_Eval` on it decides correctly.
4. **`element_digest` coverage.** Field-element tuples take the
   `list|tuple` → `hash()` path and `Polynomial` values take `get_hash()`. Any
   new message type outside those branches should be given a `digest()` method
   rather than widening the walker.
5. **Cost.** Degree-3 products-form round messages run in pure Python
   (`piop.md` §8, item 3a). Keep `ℓ*` at 6–8 in tests.

## Appendix: what was checked, and how

Every API claim above was read off the working tree, and the three pieces of
arithmetic that are easy to get wrong were checked numerically in standalone
Python before the plan was written:

- the grand-product identity `f_α(X,1) = f_α(0,X)·f_α(1,X)` under the LSB-first
  reading, for `n = 2,3,4` — including at `X = (1,…,1)`, where it holds only
  because the last entry is pinned to zero; and `P = f[2^{n+1}−2]` landing at
  entry `2^n−2` of the top half;
- that the paper's reading (selector first, MSB-first) and this plan's
  (selector last, LSB-first) are two readings of the *same* flat array, both
  satisfied by it;
- the Figure-13 step-2 identity end to end — digits → `ψ`-fold →
  `Σ ι(ψ)·eq̃·β^j`. Against `ṽ`'s *integer* coefficients this holds **exactly
  over `Z`**; against `uEval` of the ring element the verifier actually
  queries it is an identity mod `q` and nothing stronger, because reading
  `ṽ(x_η)` out of `R_q` has already reduced it. `test_recomposition_identity`
  pins both, and the `iota` lift's precondition (`|·| < 2^m β σ^m`) besides;
- the Figure-11 product check `WSt·WSµ = RS·S` on honest counters.

The plan's three risks were then settled by pre-flight, before any protocol
code was written:

- a nested `VirtualOracle` resolves correctly when `prover.witnesses[f_α]` is
  set, and raises exactly the predicted `TypeError: no table for oracle`
  when it is not;
- a `VirtualOracle` runs end to end over a `PseudoMersenneField` domain,
  including the degree-3 `eq̃ · f · f` shape Figure 12 needs;
- `Ring(8, prime_size=[30, 30], split_degree=1)` constructs, multiplies
  correctly, and binds plain integer points — so the `N = 16` / `N = 512`
  fallbacks above are not needed.

One correction the tests forced on this document is recorded above: the
recomposition identity is exact over `Z` only against `ṽ`'s integer
coefficients, not against the value read out of `R_q`.

## Bibliography

- **[CCCFGS26]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Batch, Pack, and Prove:
  More Efficient Verifiable Computation for CKKS*. ASIACRYPT 2026, to appear.
  (Appendix I: Figure 11 the lookup argument over a prime field, Figure 12 its
  virtual oracle, Figure 13 the optimized decomposition. Section 3.1:
  `unpack` / `uEval`. Section 4: Lemma 4.1 and the table `T_B`.)
- **[CCCFGS25]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Verifiable Computation
  for Approximate Homomorphic Encryption Schemes*. CRYPTO 2025. ePrint
  2025/286. <https://eprint.iacr.org/2025/286> (Lemma I.3, the memory-checking
  reduction of a lookup in `t_beta`.)
- **[SL20]** Srinath Setty, Jonathan Lee. *Quarks: Quadruple-efficient
  transparent zkSNARKs*. ePrint 2020/1275. <https://eprint.iacr.org/2020/1275>
  (the grand-product layering of Lemma I.4.)
- **[Set20]** Srinath Setty. *Spartan: Efficient and General-Purpose zkSNARKs
  Without Trusted Setup*. CRYPTO 2020. ePrint 2019/550.
  <https://eprint.iacr.org/2019/550> (the offline memory-checking technique.)
- **[Tha22]** Justin Thaler. *Proofs, Arguments, and Zero-Knowledge*.
  Foundations and Trends in Privacy and Security 4(2-4), 2022.
