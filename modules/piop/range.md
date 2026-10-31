<!-- SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.piop.range — the range proof

> The design note for the range proof: the two protocols of Appendix I of
> [CCCFGS26] expressed in this module's vocabulary, as implemented in
> `relations/lookup.py`, `protocols/lookup.py`, `relations/range.py` and
> `protocols/range.py`.

- [Context](#context)
- [Notation](#notation)
- [Conventions](#conventions-domains-and-index-order)
- [The lookup protocol (Figure 11)](#the-lookup-protocol--figure-11)
- [The decomposition protocol (Figure 13)](#the-decomposition-protocol--figure-13)
- [Soundness](#soundness-bookkeeping-both-protocols)
- [Sending an oracle](#sending-an-oracle-both-protocols)

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

Both are expressed in the language of `vfhe.piop` — relations, statements,
registered protocols, async prove/verify coroutines — so the range proof is a
reduction in the framework's statement DAG rather than a standalone script.

`Relation_Range` and `RangeDecomposition` (Figure 13) are
`relations/range.py` and `protocols/range.py`, `Relation_Lookup` and
`Lookup` (Figure 11) `relations/lookup.py` and `protocols/lookup.py`,
covered by `test_range.py` and `test_lookup.py`. A range claim reduces all
the way down — through the lookup, its two degree-3 zerochecks and their
virtual-oracle openings — to evaluation claims on `ṽ` over `R_q` and on
`µ̃'`, `read_ts`, `final_cts` and the `f^{(1)}_α` over `F_p`, which are left
terminal; see [Sending an oracle](#sending-an-oracle-both-protocols) for the
compiled variant.

Everything else is the framework's. The whole construction is this much
wiring:

```python
iop = IOP(domain=field)                       # field = F_p; or fiat_shamir=True
iop.register(Relation_Range,  RangeDecomposition(field, m, alphabet, kappa, ring))
#                                     m = split point, alphabet = |Sigma| = sigma,
#                                     kappa = repetitions (all Fig. 13); ring = R_q
iop.register(Relation_Lookup, Lookup(field))
iop.register(Relation_Sum,    Sumcheck())
iop.register(Relation_Eval,   VirtualEval(), kind=OracleKind.virtual)
# Relation_Eval for plain oracles stays unregistered -> terminal oracle query
assert iop.run(Statement(Relation_Range(B, c), oracles=[TableOracle(v_tilde)]))
```

In the notation of `piop.md` §5's reduction table, the two lines are

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

(When `c` does not divide `b`, a second `Relation_Lookup(β_top)` follows,
on the top digits of `µ̃'`, with the same subtree; see "A shorter top
digit".)

The framework's vocabulary used below — relations, statements, protocols,
oracle kinds, `VirtualOracle`, `VirtualEval` — is that of `piop.md` §3–§5.

## Notation

Paper notation, all of it used below:

| symbol | meaning |
| --- | --- |
| `R_q = Z_q[Y]/(Y^N+1)`, `q = Π p_l` | the ring `ṽ` lives in; `N = 2^ν` |
| `ṽ` | the oracle under test: `ℓ`-variate multilinear over `R_q` |
| `B = 2^b` | the coefficient bound; `T_B` is the paper's table of in-range ring elements |
| `c = 2^γ`, `β = 2^{⌈b/c⌉} = 2^{b*}` | number of digits per coefficient, and the digit alphabet (`B^{1/c}` when `c` divides `b`) |
| `β_top = B / β^{c−1}` | the alphabet of the most significant digit: `β` when `c` divides `b`, smaller otherwise (`top_beta`) |
| `t_β = {0,…,β−1}`, `t̃β` | the lookup table and its MLE (a *public* oracle) |
| `ℓ* = ℓ + ν + γ` | variables of the digit polynomial |
| `ũ = unpack(ṽ)` | `ũ(x, k) = ṽ(x)[k]`: the `Z_q` coefficients spread over `ν` extra variables (§3.1). Never committed — queried through Lemma 3.1 |
| `µ̃` | the integer digit polynomial, `µ̃(i,k,j) = dig_j^{(β)}(ṽ(i)[k])`, the `j`-th base-`β` digit of the `k`-th coefficient of `ṽ(i)` |
| `µ̃' = π_p(µ̃)` | its reduction into `F_p` — the prime marks "over `F_p`", not a derivative |
| `ι` | the balanced lift `F_p → I_p = {−⌊p/2⌋,…,⌊p/2⌋} ⊂ Z` |
| `Σ = {0,…,σ−1} ⊂ Z`, `m ≤ ℓ+ν`, `κ` | the small-integer challenge alphabet, the split point, the repetition count (Fig. 13). With `m > ℓ` the bound prefix reaches into the `Z` block and `ψ̃'_η` has fewer than `c·N` entries. `σ ≤ p_min` is required, so that `π_q(Σ)` lands in the exceptional set `Ŝ_0 ⊂ Z_q` and the query to `ṽ` is sound |
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

These are the things that are easiest to get wrong.

### The challenge domain

Both protocols draw every challenge of their own in their `field`
(`challenge(label, self.field)`), an `F_p` element whose exceptional set is
`p` itself, whatever the IOP's domain is; the sumchecks of the statements
they reduce to draw in their oracles' domain (`Oracle.domain`), which is
`F_p` too. So the range proof runs in an IOP over any domain, `R_q`
included, beside reductions over that domain on the same committed `ṽ`:
the claims on `ṽ` from both share one bundle of its scheme.

`ṽ` is a table of `R_q` elements, not `F_p` elements. The two coexist because
the only claim ever made about `ṽ` is a terminal evaluation at a point with
plain-integer coordinates, which a ring-backed table can bind.

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
both sides vanish there; the paper's Lemma I.4 carries the same convention
implicitly. Reading the array LSB-first, with
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

> **Comparing with the paper side by side.** Figure 11 writes the layer
> selector as the *first* argument (`f(1,i) = f(i,0)·f(i,1)`, with
> `f^{(1)}_α(i) = f^{(i_1)}_α(i', 0)·f^{(i_1)}_α(i', 1)`); above it is the
> *last* variable `W_n`, and the new bit is inserted at `W_0`. These are the
> same binary tree read with "first" and "last" exchanged — the paper's
> reading is the MSB-first reading of the very same flat array. Nothing in the
> protocol depends on which naming is used (`ρ` and `ξ` are uniform), and
> putting the selector last is what buys the index alignment with `µ̃'`.

## The lookup protocol — Figure 11

Taken first, because Figure 13 reduces to it. The relation is
`relations/lookup.py`, the protocol `protocols/lookup.py`.

```python
class Relation_Lookup(Relation):
    """Every hypercube evaluation of the oracle lies in t_beta = {0..beta-1}."""
    name = "lookup"
    fields = ("oracles",)
    index_fields = ("beta",)
    def __init__(self, beta: int) -> None: super().__init__(index={"beta": beta})
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
   algorithm of Lemma I.3, over `F_p`. One pass:
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
   notlast = TableOracle(MLE.eq(field, [0], variables=[w]), public=True)  # [1, 0] = 1 - w
   last    = TableOracle(MLE.eq(field, [1], variables=[w]), public=True)  # [0, 1] = w
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
   live — multiplying it by `notlast` is the correct algebra.
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
same handoff the Composer uses for the heads of a reduction, and the only way
`resolve_table` can find concrete data for a virtual constituent. It is
mandatory, not an optimization: a virtual oracle has no table of its own, so
a missing witness is a hard `TypeError` from the sumcheck prover.

> **Not the implicit-oracle grand product.** The sum definitions of
> `test_composition.py` have a grand product,
> `L_i(z) = Σ_p eq̃(z,p)·L_{i+1}(p,0)·L_{i+1}(p,1)` — the Thaler/GKR form,
> which must be *implicit* because multilinear extension does not commute
> with products. Figure 11 uses Quarks' form instead: the internal nodes
> `f^{(1)}_α` are sent as an oracle, and the tree identity becomes a
> zerocheck on `f_α(X,1) − f_α(0,X)·f_α(1,X)` — a claim about a polynomial
> the verifier can already name. So there is no implicit oracle here, no
> sum definition, and no Composer: just a `VirtualOracle` and a
> `Relation_Sum`.

### What the framework does after step 5

The paper's verifier recombines the sumcheck's final evaluations by hand.
Here that recombination is the framework's standard chain, worth spelling
out:

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
   `µ̃'`, `read_ts`, `final_cts` and `f^{(1)}_α` at the shifted points.
5. Those claims are `plain`, hence terminal: one oracle query each.

## The decomposition protocol — Figure 13

The relation is `relations/range.py`, the protocol `protocols/range.py`.

```python
class Relation_Range(Relation):
    """Every coefficient of every hypercube evaluation of the R_q oracle
    lies in [0, B): the table T_B of the paper."""
    name = "range"
    fields = ("oracles",)
    index_fields = ("bound", "chunks")
    def __init__(self, bound: int, chunks: int) -> None:   # B, c
        super().__init__(index={"bound": bound, "chunks": chunks})
```

`check()` walks the hypercube and tests `max(poly.get_polynomial()) < B`.

`RangeDecomposition(Protocol)`: `reduces_from = Relation_Range`,
`reduces_to = (Relation_Lookup, Relation_Eval)`. The constructor takes
`(field, m, alphabet, kappa, ring)`; `B` and `c` come from the relation's
index, `N`/`ν` from the ring (an oracle does not expose its domain: a
committed one has no table on the verifier's side), `ℓ` from `ṽ.num_vars`.

1. The prover builds the integer digit table `µ̃` by reading each
   `Polynomial.get_polynomial()` and slicing out `c` base-`β` digits, lifts it
   to `µ̃' = MLE(field=F_p, …)`, and sends it.
2. `r_0 … r_{κ−1} ← Σ^{ℓ+ν}` via `challenge_bits`.
3. The prover sends `ψ̃'_η = µ̃'(r'_η, ·)` for each `η` — bind the first `m`
   variables, `2^{ℓ+ν−m+γ}` field elements each — and the `κ` ring values
   `w_η = ṽ(x_η)`. Note the two liftings of the same integer challenge: `ṽ` is
   bound at the plain integers `x_η` (a ring table takes an `int` directly),
   while `µ̃'` must be bound at `π_p(r'_η)`, i.e. `field(r)` per coordinate.
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
   having used `j` for it in eq. (4). This note uses `j` for the digit and `k`
   for the coefficient **everywhere**; transposing the two blocks is the
   easiest way to get this wrong.
6. Emit `κ` `Relation_Eval` on `µ̃'` at `(r'_η, s')` with value `ψ̃'_η(s')`
   (both parties compute it from the `ψ` message), `κ` `Relation_Eval` on `ṽ`
   at `x_η` with value `w_η`, and one `Relation_Lookup(β)` on `µ̃'`; when
   `β_top < β`, then one `Relation_Lookup(β_top)` on the restriction of `µ̃'`
   to the top digit (below).

### A shorter top digit

When `c` does not divide `b`, the digits are `c − 1` of `b* = ⌈b/c⌉` bits
and a top one of `b − (c−1)b*` bits, at least one (`Relation_Range`
refuses a `(B, c)` that leaves none). The prover's digit table, the
recomposition check and its eq. (38) bound are those of a `β^c` bound: an
honest coefficient below `B` simply has a top digit below `β_top`. What the
first lookup and the recomposition prove together is a coefficient in
`[0, β^c)`, so a last claim bounds the top digits:
`Relation_Lookup(β_top)` on `µ̃'(X, Z, 1…1)`, the
`VirtualOracle.restriction` of `µ̃'` to `Z' = (1, …, 1)`, whose hypercube
is exactly the top digit of every coefficient. Its claims end, through
`VirtualEval`, in claims on `µ̃'` itself.

Soundness: with every digit in `[0, β)` and every top digit in `[0, β_top)`,
the integer the digits rebuild is at most
`(β^{c−1} − 1) + (β_top − 1)β^{c−1} = B − 1`, and the recomposition makes it
congruent to the coefficient mod `q`; for `B ≤ q` both are representatives
in `[0, q)`, so the coefficient is that integer, below `B`. The extra error
is the second lookup's, `(2^{ℓ+ν} + β_top)/p`. Looking up `2^s · d` for the
top digit `d` in `t_β` instead (`s = c·b* − b`) would not do on its own: the
digit is a field element, and `2^s · d` in `[0, β)` holds as well for
`d = e / 2^s` with `e` not a multiple of `2^s`, a field element far outside
any small range.

## Soundness bookkeeping (both protocols)

Each protocol's `soundness_error` (`piop.md` §6) reports the paper's own
bound:

- `RangeDecomposition` (Thm. I.6): `max{ ((ℓ+ν)/σ)^κ , (ℓ+ν+γ−m)/p , δ'_range }`,
  where the third term is what the child `Relation_Lookup` accounts for;
- `Lookup` (Thm. I.5): `δ'_range = (2^{ℓ*} + 2^{b*}) / p`.

The first term dominates: `κ` is chosen so that `((ℓ+ν)/σ)^κ ≈ 2^{−λ}`
(eq. 44).

The correctness side conditions — `p > 2β(2σ)^m` (eq. 38) and `p > 2^{ℓ*}` —
mix constructor parameters with claim dimensions, so they are checked at the
protocol's first use of a claim (`_shape`), and a bad parameter set fails
loudly there rather than producing a proof that is quietly unsound.

## Sending an oracle (both protocols)

Both protocols send their oracles through an `OracleMessages` object
(`vfhe.piop.messages`), passed as `messages=` and the ideal model when
omitted: `send(prover, label, table)` returns the prover's oracle, and
`receive(verifier, label, variables, field)` the verifier's.

In the ideal model a prover message *is* an oracle, so `µ̃'`, `read_ts`,
`final_cts` and the four `f^{(1)}_α` are written to the transcript and read
back by the verifier.

They are written as plain **tuples of field elements**, not as `MLE` objects.
A table digests itself, so either would do for Fiat–Shamir; the reason is
variable identity: `MLE_Variable`s compare by identity, and each party should
build its own table, with its own variable objects, from a message that is
plain data. Each party then names its table through a plain `TableOracle`.

The compiled variant is `vfhe.polycom.BasefoldMessages`, built from one
field `Basefold` scheme per table size (`2^{ℓ*}` for `µ̃'`, `read_ts`,
`f^{(1)}_WSµ`, `f^{(1)}_RS`; `2^{b*}` for `final_cts`, `f^{(1)}_WSt`,
`f^{(1)}_S`; with a shorter top digit also `2^{ℓ+ν}` and `β_top` for the
second lookup's): the prover commits and the message is the Merkle root;
both parties name a `CommittedOracle`. Each scheme's `BasefoldEval`, registered
for `Relation_Eval`, opens every claim on its oracles in one batched
opening. Once a scheme is registered, claims on plain oracles are refused,
so `ṽ` must be committed too (ring Basefold, say, with its own
`BasefoldEval`):

```python
messages = BasefoldMessages([main_scheme, table_scheme])
iop.register(Relation_Range, RangeDecomposition(field, m, alphabet, kappa, ring,
                                                messages=messages))
iop.register(Relation_Lookup, Lookup(field, messages=messages))
for scheme in (main_scheme, table_scheme, ring_scheme):
    iop.register(Relation_Eval, BasefoldEval(scheme))
```

The relations, the reductions and the DAG are the same in both models.

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
