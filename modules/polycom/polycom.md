<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.polycom — design notes

> **Under development.** The code these notes describe is largely verified by
> the test suite, but the notes themselves have not been reviewed yet and may
> be inaccurate or out of date.

How the module is derived from the polynomial-commitment literature.
Bracketed keys refer to the [bibliography](#bibliography); the PIOP
machinery this module builds on is documented in `modules/piop/piop.md`.

## 1. Scope

`vfhe.polycom` holds polynomial commitment schemes for the multilinear
oracles of `vfhe.piop`. The first scheme is **basefold** [ZCF24] over
`R_q = Z_q[X]/(X^N + 1)` or over a finite field: a code-based commitment
whose evaluation proof is a sumcheck interleaved with codeword folds. The
scheme and protocol are written once against the code's interface; the two
codes (`FoldableRS` over `R_q`, `FieldFoldableRS` over a field, §2) carry
everything representation-specific.

The module follows the standard four-algorithm PCS syntax
[ZCF24, Def. 8; BFS20]: `Basefold` is the scheme, `Basefold.commit` and
`Basefold.open` its algorithms, and `BasefoldEval` the Eval protocol — an
argument for the relation
`R_Eval = {[(C, z, y); f] : f(z) = y and C opens to f}`, which is exactly
piop's `Relation_Eval` with its optional `commitment` field set (the
commitment is just the oracle's compiled form [ZCF24, §4], so there is no
separate "open" relation; witness `f`). **Committing and evaluating are separate moments**: `commit(f)` runs
whenever the polynomial exists — possibly long before any IOP, and once
per polynomial however many evaluation claims follow — and the commitment
is *instance* data: it rides on statements (Fiat-Shamir binds it through
`sigma_0 = rho(x)`), never on the per-execution transcript. The prover's
opening — the polynomial and what commit precomputed for it — is stored
under the commitment in `prover.witnesses` (piop's witness map, keyed by
the commitment). This is
the structure of [ZCF24, Protocol 4] ("public input: oracle
`pi_f := Enc_d(f)`"), of the Marlin compiler [CHMMVW20] ("commit to
oracles, then open query answers" — with index commitments produced in an
offline phase and reused across proofs), and of the `commit -> (Commitment,
ProverData)` shape of PCS implementations (arkworks' poly-commit, plonky3's
`Pcs`).

Codewords are committed by **Merkle root** (`vfhe.crypto.Merkle`, BLAKE3):
the RO-model instantiation of [ZCF24] §4's ideal oracle via the BCS
compiler [BCS16]. So `commit` returns `(BasefoldCommitment, BasefoldOpening)`
— a root, and the prover data (the codeword and its tree) that every later
evaluation proof reuses. Two consequences worth stating:

- The commitment is **succinct**: one digest, whatever `n_d` is — no
  oracle-discipline convention needed to keep the verifier from reading the
  codeword it cannot have.
- Binding is **computational** (collision resistance) layered on the
  code-distance argument, where the oracle-model version is
  information-theoretic; [ZCF24, Thm. 4] reduces one to the other. The
  `soundness_error` methods still report only the information-theoretic
  part, and say so.

Fiat-Shamir needs nothing from this module: `fs.FS_Verifier`
(piop.md §5) derives `challenge` and `challenge_bits` from the transcript
chain and seeds it with σ₀ = ρ(x), which binds the commitment because the
commitment is a statement field. No message below changes — the same
`BasefoldEval` runs interactively under `iop.run` and non-interactively
under `iop.prove` / `iop.verify`: the prover alone emits a `Proof`, and a
separate IOP holding no witnesses, codewords or trees checks it — the
BCS-compiled argument [BCS16] (piop.md §5).

## 2. The foldable code (`code.py`, `field_code.py`, `c/src/rscode_*.c`)

Basefold commits with a family of **foldable linear codes** [ZCF24,
Def. 5]: from a base `[n0, k0]` code `C_0` and, per level, two diagonal
twist vectors `T_l`, `T'_l` with `T_l[j] != T'_l[j]`, the level-`l` code
`C_l` (dimension `k0·2^l`, length `n0·2^l`) encodes `m` from the codewords
of its two halves. As in the paper's random foldable code, **`T'_l = -T_l`**
with `T_l` nonzero: one table serves both positions, the fold's divisor is
`2·T_l[j]`, and the general `T'` buys nothing the distance argument uses.
In vfhe's indexing — the message split into its even and odd entries `m_e`,
`m_o`, and the two twisted copies *adjacent* rather than half a codeword
apart as in the paper (see the conventions below):

```text
Enc_l(m)[2j]   = Enc_{l-1}(m_e)[j] + T_l[j] · Enc_{l-1}(m_o)[j]
Enc_l(m)[2j+1] = Enc_{l-1}(m_e)[j] - T_l[j] · Enc_{l-1}(m_o)[j]
```

so a codeword of `m` folds, pair by pair and with only the table known,
into a codeword of `m_e + r·m_o` for any challenge `r`:

```text
b[j]   = (pi[2j] - pi[2j+1]) / (2·T_l[j])
pi'[j] = pi[2j+1] + b[j]·T_l[j] + b[j]·r
```

Both codes hold `T` (`twists`; `twists_odd` is `-T`, derived on request)
and the fold's `1/(2T)`, the latter inverted in **one batch per level by
Montgomery's trick** (`FieldVector.inverse`, a C kernel on the extension
family; `code.batch_inverse` per prime over a ring) rather than an
exponentiation per entry; `fold_pair` is the two lines above and `fold` the
whole-codeword form (`FieldVector.fold_twisted` over a field, a loop over a
ring).

The family is instantiated in one of two ways, chosen by the constructor's
`instantiation=`:

- **`"general"`, the default — [ZCF24]'s random foldable code.** `T_l` is
  drawn, per level, from the code's `seed` (a public parameter,
  `DEFAULT_SEED` unless named; two parties building the code from the same
  arguments hold the same code), through the library's seeded samplers
  (`FieldVector.sample_random`, `crypto.seeded`), and resampled under a
  fresh derived seed until no entry is zero (`diag(T) ∈ (F^×)^n`, the
  paper's condition). The base code is a
  Reed-Solomon code on `n0` points, which is what the distance argument
  needs (below) — and one code with **two encoders**: arith's negacyclic
  NTT where the field has a root of unity of order `2·n0` (`2·n0 | p - 1`;
  over a ring `n0 | N/split_degree`), a Vandermonde product on `n0` seeded
  distinct points otherwise. Above the base there is no transform at all,
  so **any codeword length is served on any field** — Basefold's
  field-agnosticism. Over a field the encoder is `O(n log n)` in
  whole-vector passes: the `2^l` base messages of a level-`l` message are
  encoded together by Horner's scheme over the whole result, and every
  level above is one `FieldVector.fma_interleave` over all of that level's
  codewords at once (laid end to end, even half-messages' codewords first,
  the tables tiled over them). Over a ring it is the recursion, on lists.
- **`"rs"` — every level a Reed-Solomon code on roots of unity**, the FRI
  folding structure [BBHR18], with one transform per level from the same
  providers. With `psi` the `2n`-th root of unity `ntt_new_plan` picks,
  position `p` of a length-`n` codeword holds `P(psi^(2·brv(p)+1))` (the
  transform is CT_NR, natural in, bit-reversed out), so the adjacent pairs
  are `(P(x_i), P(-x_i))` for `x_i = psi^(2·brv(i)+1)` and the table is
  `T = x`: the same fold, on a structured rather than a random `T`. The
  levels share a
  root tower, `psi_{n/2} = psi_n^2`: `ntt_new_plan` finds its root as
  `g^((q-1)/2n)` for the smallest quadratic non-residue `g`, a condition
  independent of `n`, so the squared fold points are exactly the
  half-length code's points (`test_rs_root_orders_and_level_consistency`
  pins it). Deterministic, faster (a transform per level in C), and with
  the exact distance `1 - 1/c + 1/n_d`; it needs `2·n_d | p - 1` (over a
  ring `n_d | N/split_degree`), which is the restriction the default
  removes. [ZCF24, App. E] shows Reed-Solomon codes are themselves foldable
  codes, so this is a legitimate member of the family — what differs is the
  distance argument, not the fold.

**Distance** (`relative_distance(security_bits, bound)`, the `delta` of
`BasefoldEval.soundness_error`): for the option, the exact Reed-Solomon
value at the longest level. For the general code, a probabilistic lower
bound over the choice of twists (`foldable_relative_distance`), of the
shape [ZCF24, Thm. 2] gives it — `t_0 = k_0`, `t_i = 2·t_{i-1} + l_i`,
`distance >= 1 - t_d/n_d` with probability at least `1 - d·2^-lambda` —
where `l_i` bounds the zeros level `i` may add. **The default is
[CCCFGS26]'s tightening** of that theorem (its appendix on BaseFold),
proven for exactly this code (`T' = -T`, `diag(T)` uniform in `F^×`, an MDS
base):

```text
l_i = (lambda + 2·t_{i-1}·(1 + log2(|F|/(|F|-1))) + 0.585·n_i) / (log2(|F|-1) - 1)
```

against [ZCF24]'s `l_i = (2(i-1)·log2 n_0 + lambda + 2.002·t_{i-1} + 0.6·n_i) /
(log2|F| - 1.001)`, reachable as `bound="zcf24"` and checked against that
paper's Table 1. The tightening drops the `2(i-1)·log2 n_0` term and
sharpens `0.6` to `0.585` (`log2(9/4)/2`), and it is stated for any field
with more than three elements where [ZCF24]'s form assumes `|F| >= 2^10`;
it is never smaller, and the gain is where the field is small relative to
the depth (about two percentage points of distance, and eight fewer
queries at 128 bits, for `log2|F| = 61` and `d = 20`; nothing measurable
over a 180-bit residue field). It is the default because it is the better
bound for the code the library actually builds, and because the library's
own soundness parameters should not quote a looser number than its authors
have proven.

`t_0 = k_0` is the base code's distance as an MDS code, which is why the
base stays Reed-Solomon whichever encoder produces it. `|F|` is the field
the twists are drawn from — the whole extension field over
`FieldFoldableRS`, the smallest residue field over `FoldableRS` (the bound
is per RNS-prime component). The value is 0 where the theorem gives
nothing (a rate-one code, or too few field bits for the depth).

Three conventions to keep in mind:

- **Message split is even/odd (LSB), not halves.** The message is the
  monomial-basis coefficient vector of the MLE in LSB-first index order
  (`MLE.to_coefficients`), and `P_even/P_odd` are the even/odd
  coefficient subsequences — so folding the codeword binds the *first*
  variable of the MLE, exactly the round variable of the piop sumcheck
  kernels (which pair table entries `(2i, 2i+1)`). [ZCF24] splits the
  message into contiguous halves (MSB) and puts the twisted copies half a
  codeword apart; the two are relabelings of the same family ([ZCF24,
  Def. 5] is stated up to row permutation), and this choice is what lets
  basefold share round machinery with `vfhe.piop.sumcheck`.
- **Everything is per RNS prime** over a ring. The code acts on vectors of
  ring elements coefficient-slot-wise and per RNS prime (an interleaved
  code), so each (prime, coefficient slot) pair carries an independent
  codeword over `Z_p`: the kernels read `coeffs[i][j]` directly (hence
  every entry must be in the same — RNS/NTT — representation, which the
  kernel paths normalize first), roots and twists are per-prime integers
  applied through `Polynomial * list`, and the fold's `1/(2T)` are
  per-prime modular inverses — no ring inversions.
- **The Merkle leaf is the adjacent pair** `(2i, 2i+1)`, the unit the fold
  reads (§3). Only the `"rs"` option makes it a `(P(x), P(-x))` pair; in
  general the two entries are two twisted copies with nothing to do with
  negation.

**The field code** (`field_code.py`) is the same construction over a
finite field, with the same interface: a codeword is one
`vfhe.arith.FieldVector`, the tables are held as vectors (and as lists of
elements on request), `fold` is one kernel call and the encoder the
whole-vector passes above. Merkle leaves come from the vector's own
windowed digest (`hash_elements(group=2, stride=2)`), and the verifier
recomputes a leaf as the digest of the two-element vector.

The base transform (and, for the option, every level's) sits behind a
provider the field selects; both providers share arith's basis and output
order, so everything above them is written once.

- `_ExtensionTransforms` serves an `ExtensionField` `F_p[x]/(x^d - w)`
  through the `rs_field_*` kernels (`c/src/rscode_field.c`), which run
  arith's NTT over the field's prime once per coefficient plane. The
  evaluation points lie in `F_p`, so encoding is `F_p`-linear and the
  extension structure never enters: the base is an RS code over `F_(p^d)`
  with points in `F_p`.
- `_PseudoMersenneTransforms` serves a `PseudoMersenneField` through
  arith's own `PseudoMersenneNTT` (`field.ntt_plan(n)`, memoized there),
  which transforms the vector whole. Encoding is the forward transform of
  the message zero-padded to the codeword length, decoding the inverse plus
  a zero check on what the truncation cuts off.

`decode` is the encoder's other half: the recursion inverted down to the
base, then the base's inverse transform plus the degree check — or, on a
Vandermonde base, interpolation from the first `k0` positions plus a
re-encoding check — deciding whether a vector is in the code. The
evaluation protocol does not need it — the verifier *computes* the level-0
codeword rather than decoding one — but it is the natural round-trip test
for the encoder and a ready-made proximity check.

Soundness over the ring reduces to the per-prime components: the sumcheck
side samples challenges from the exceptional set as in the piop module
([GNS23]; [CCCFGS25]), and the code side is a foldable code over each
residue field.

## 3. The scheme and its evaluation protocol (`basefold.py`)

`Basefold.commit(f)` encodes `f`'s coefficient vector once, builds its
Merkle tree, and returns the root as the `BasefoldCommitment` (with the
committed variables, whose order is the canonical order of evaluation
points) plus a `BasefoldOpening` — the polynomial, the codeword and the
tree — for `prover.witnesses[commitment]`. It also records the public
oracle -> commitment association in `scheme.commitments` (and the opening
in the prover-side `scheme.openings`, from which `BasefoldEval` backfills
`prover.witnesses` so pipelines need no manual installation).
`Basefold.open` is the Open algorithm of [ZCF24, Def. 8]: re-encode,
rebuild the tree (`Basefold.merkle_commit`), compare roots — as binding as
the hash is collision-resistant. A commitment-only `Relation_Eval` has no
witness-free decider (`check` raises), so such claims are always
discharged by the registered Eval protocol, never left terminal.

**Leaves are adjacent pairs, not single positions.** The fold reads a
pair (§2), so committing one leaf per pair makes a single path
authenticate both operands of a fold check — [ZCF24, Remark 9]'s packed
leaves, standard in FRI deployments — which halves both the number of paths
and the tree height.

`BasefoldEval` is that protocol: register
`iop.register(Relation_Eval, BasefoldEval(scheme, rep))` and every
evaluation claim is proved against a commitment — the statement's own
`commitment` field when set, else the scheme's record for the statement's
oracle (a never-committed oracle is a `LookupError`). It is a *bundle*
protocol (`batching = True`, piop.md §5): the driver parks the claims on
one committed oracle until no other frontier statement can produce
another, then hands them over together, in path order — today each claim
in the bundle is opened on its own (the `M = 1` case; the batched opening
is roadmap), so the transcript is the concatenation of the runs below. For
a claim on `n` variables with a depth-`d` code (`k_d = 2^n`, `kappa = n - d`
base variables):

1. **d interleaved rounds** (round `s`): the prover sends the degree-2
   round message of the product sumcheck `sum_b f(b)·eq~(z, b) == v` —
   the same wire format, round kernel (`SumcheckProd.prod_round_evals`,
   which decides native vs pure Python itself) and Lagrange interpolation
   (`interpolate_evals`) as `SumcheckProd` — the challenge `r_s` is drawn,
   both sumcheck tables are folded by MLE evaluation, and the codeword
   (starting from the *committed* one — no re-encoding at eval time) is
   folded with the *same* `r_s`. The folded codeword is published as a
   **root** (`pi_{s+1}`), for every level down to 1: `log n` trees per
   opening, as [ZCF24] notes for BaseFold/FRI.
2. **Base case**: the prover sends the remaining `kappa`-variable table
   `h0` in the clear. The verifier checks the final sumcheck claim as a
   `Relation_SumProd` decider over `h0` and its own eq~ tail table (scaled
   by the bound-variable eq~ factor), and computes the level-0 codeword
   `Enc_0(coefficients(h0))` itself — a codeword by construction, held in
   full, and therefore needing no tree.
3. **Query phase** — the one place committing the codewords changes the
   message flow: the verifier cannot read a committed vector, so the
   positions come from a *published* bit-string challenge
   (`Verifier.challenge_bits`, drawn after `h0`), which both parties expand
   identically with `BasefoldEval.query_positions` — the raw coins are the
   verifier's, their shape is the protocol's; under Fiat-Shamir the coins
   are derived from the transcript chain (piop.md §5), changing nothing in
   the expansion. The expansion hashes the seed
   in counter mode (BLAKE3, the tree's own hash; the ranges are powers of
   two, so masking is unbiased) and **rejection-samples the positions so
   their projections to the level-0 codeword are pairwise distinct**: two
   queries meeting at the bottom would rerun the same final fold check, so
   sampling with replacement buys less soundness than its `rep` claims.
   That in turn bounds the parameters — the distinct projections live in
   the level-0 codeword the verifier holds in full, so `rep < n_0`
   (checked at construction). The prover answers each position with the
   pair and its path at every level from d down to 1; the verifier checks
   each path against that level's root, folds the authenticated pair, and
   requires the result to reappear one level down — at offset `j & 1` of
   the pair the walk moves to, or, at the bottom, in the level-0 codeword
   it built itself. The walk `j -> j // 2` is forced rather than chosen:
   with only authenticated pairs the verifier must reuse the value it just
   derived, which is what chains the levels into a proximity test instead
   of d independent checks.

The verifier's round work is O(1) ring operations plus the O(2^kappa +
rep·d) base and query work; it never touches the opening `f`. The
soundness error is `2d/(gamma^3·|A|) + (1 - delta + gamma·d)^rep` for
admissible `(gamma, delta)` [ZCF24], with `|A|` the per-prime residue field
size (`BasefoldEval.soundness_error`).

**The PIOP world composes directly**: sumcheck-style reductions end in
`Relation_Eval` claims carrying oracles, and `BasefoldEval` *is* the
compiler step of [CHMMVW20] ("queries become opening claims") for them —
it resolves the commitment from `scheme.commitments` and proves against
it, no bridge relation or extra DAG level: a reduction from
`Relation_Eval` to a separate "open" relation would be a pure relabeling —
no messages, no soundness cost.
The full compiled chain is `Relation_Sum -> Sumcheck -> Relation_Eval ->
BasefoldEval`, with `scheme.commit(f)` run *before* the IOP, so under
Fiat-Shamir the commitment precedes every challenge that fixes the
evaluation point.

## 4. Roadmap

1. **Batched openings**: `BasefoldEval` already receives every pending
   claim on a commitment as one bundle (piop.md §5) and opens them one by
   one; the step left is the protocol itself — different-point claims
   reduced to a common point by sumcheck, then one basefold run on their
   random linear combination (an `M -> 1` folding reduction).
2. **C kernels for the ring code above the base** (`fold_at` is
   per-position Python over `Polynomial`, one `Polynomial * list` scaling
   per entry, and the general code's encoder is the same recursion in
   Python; the field code is whole-vector throughout, and only the base
   transform is native over the ring). Multithreading the per-prime,
   per-coefficient-slot columns of `rs_encode` is the other easy win —
   they are independent transforms.

## Bibliography

- **[ZCF24]** Hadas Zeilberger, Binyi Chen, Ben Fisch. *BaseFold:
  Efficient Field-Agnostic Polynomial Commitment Schemes from Foldable
  Codes*. CRYPTO 2024. ePrint 2023/1705.
  <https://eprint.iacr.org/2023/1705>
- **[BBHR18]** Eli Ben-Sasson, Iddo Bentov, Yinon Horesh, Michael Riabzev.
  *Fast Reed-Solomon Interactive Oracle Proofs of Proximity*. ICALP 2018.
  <https://drops.dagstuhl.de/opus/volltexte/2018/9018/>
- **[BCS16]** Eli Ben-Sasson, Alessandro Chiesa, Nicholas Spooner.
  *Interactive Oracle Proofs*. TCC 2016-B. ePrint 2016/116.
  <https://eprint.iacr.org/2016/116> (the compiler that turns oracles into
  Merkle roots).
- **[BFS20]** Benedikt Bünz, Ben Fisch, Alan Szepieniec. *Transparent
  SNARKs from DARK Compilers*. EUROCRYPT 2020. ePrint 2019/1229.
  <https://eprint.iacr.org/2019/1229>
- **[CHMMVW20]** Alessandro Chiesa, Yuncong Hu, Mary Maller, Pratyush
  Mishra, Psi Vesely, Nicholas Ward. *Marlin: Preprocessing zkSNARKs with
  Universal and Updatable SRS*. EUROCRYPT 2020. ePrint 2019/1047.
  <https://eprint.iacr.org/2019/1047>
- **[GNS23]** Chaya Ganesh, Anca Nitulescu, Eduardo Soria-Vazquez.
  *Rinocchio: SNARKs for Ring Arithmetic*. Journal of Cryptology 36, 2023.
  ePrint 2021/322. <https://eprint.iacr.org/2021/322>
- **[CCCFGS25]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Verifiable Computation
  for Approximate Homomorphic Encryption Schemes*. CRYPTO 2025. ePrint
  2025/286. <https://eprint.iacr.org/2025/286>
- **[CCCFGS26]** Ignacio Cascudo, Anamaria Costache, Daniele Cozzo, Dario
  Fiore, Antonio Guimarães, Eduardo Soria-Vazquez. *Batch, Pack, and Prove:
  More Efficient Verifiable Computation for CKKS*. ASIACRYPT 2026, to
  appear. (The appendix on BaseFold has the minimum-distance bound
  `foldable_relative_distance` reports by default.)
