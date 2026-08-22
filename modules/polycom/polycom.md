<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.polycom — design notes

How the module is derived from the polynomial-commitment literature.
Bracketed keys refer to the [bibliography](#bibliography); the PIOP
machinery this module builds on is documented in `modules/piop/piop.md`.

## 1. Scope

`vfhe.polycom` holds polynomial commitment schemes for the multilinear
oracles of `vfhe.piop`. The first scheme is **basefold** [ZCF24] over
`R_q = Z_q[X]/(X^N + 1)`: a code-based commitment whose evaluation proof is
a sumcheck interleaved with codeword folds.

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

Codewords are committed by **Merkle root** (`vfhe.piop.Merkle`, BLAKE3):
the RO-model instantiation of [ZCF24] §4's ideal oracle via the BCS
compiler [BCS16]. So `commit` returns `(BasefoldCommitment, BasefoldOpening)`
— a root, and the prover data (the codeword and its tree) that every later
evaluation proof reuses. Two consequences worth stating:

- The commitment is **succinct**: one digest, whatever `n_d` is. It was
  previously the whole codeword, readable only under an oracle-discipline
  convention.
- Binding becomes **computational** (collision resistance) layered on the
  code-distance argument, where the oracle-model version was
  information-theoretic; [ZCF24, Thm. 4] reduces one to the other. The
  `soundness_error` methods still report only the information-theoretic
  part, and say so.

Fiat-Shamir is done, and it needed nothing from this module: `fs.FS_Verifier`
(piop.md §5) derives `challenge` and `challenge_bits` from the transcript
chain and seeds it with σ₀ = ρ(x), which binds the commitment because the
commitment is a statement field. No message below changes — the same
`BasefoldEval` runs interactively under `iop.run` and non-interactively
under `iop.prove` / `iop.verify`.

## 2. The foldable code (`code.py`, `c/src/rscode.c`)

Basefold commits with a family of **foldable linear codes** [ZCF24,
Def. 3.2]: from a base `[n0, k0]` code `C_0` and diagonal twist vectors
`t_l`, the level-`l` code `C_l` (dimension `k0·2^l`, length `n0·2^l`)
encodes `m = (m_e, m_o)` as

```
Enc_l(m) = ( Enc_{l-1}(m_e) + t_l ∘ Enc_{l-1}(m_o),
             Enc_{l-1}(m_e) - t_l ∘ Enc_{l-1}(m_o) )
```

so a codeword of `m` folds, position-wise and with only `t_l` known, into a
codeword of `m_e + r·m_o` for any challenge `r` — in the paper's indexing,
where the `±` partners sit half a codeword apart:

```
pi'[j] = pi[j + n] + (t_l[j] + r) · (pi[j] - pi[j + n]) / (2·t_l[j]).
```

Our instantiation pairs them *adjacently* instead (see below); the fold is
the same map, re-indexed.

`FoldableRS` instantiates the family so that **every level is a
Reed-Solomon code** — the FRI folding structure [BBHR18], rather than
[ZCF24]'s random twists — and gets the transform from **arith's negacyclic
NTT**, via the `rs_*` C kernels (`c/src/rscode.c`), following the structure
of the reference implementation: one `NTT_proc` per (level, RNS prime),
and per (prime, coefficient slot) column a gather, zero-pad, `ntt_forward`,
scatter. This keeps the code deterministic and its distance exact
(`1 - k/n + 1/n` per level) instead of probabilistic, and reuses the
library's tuned kernels rather than a hand-rolled FFT.

The layout follows from what `ntt_forward` computes. With `psi` the
`2n`-th root of unity `ntt_new_proc` picks, position `p` of a length-`n`
codeword holds `P(psi^(2·brv(p)+1))`, where `brv` reverses the `log2(n)`
index bits — the transform is CT_NR (natural in, bit-reversed out). Two
structural consequences, both load-bearing:

- **The `±x` pairs are adjacent**: positions `2i` and `2i+1` hold `P(x_i)`
  and `P(-x_i)` for `x_i = psi^(2·brv(i)+1)` (because `psi^n = -1` and
  `brv_n(2i) = brv_{n/2}(i)`), so the fold reads adjacent entries and
  `t_l[i] = x_i` is the twist:
  `pi'[i] = pi[2i+1] + (t[i] + r)·(pi[2i] - pi[2i+1]) / (2·t[i])`.
- **The levels share a root tower**, `psi_{n/2} = psi_n^2`. `ntt_new_proc`
  finds its root as `g^((q-1)/2n)` for the first `g` with `g^((q-1)/2) =
  -1` — i.e. the smallest quadratic *non-residue*, a condition independent
  of `n`. So the same `g` is chosen at every length, and the squared fold
  points `x_i^2` are exactly the half-length code's evaluation points.
  Without this the folded codeword would not be a level-`(l-1)` codeword
  at all; `test_root_orders_and_level_consistency` pins it.

Three conventions to keep in mind:

- **Message split is even/odd (LSB), not halves.** The message is the
  monomial-basis coefficient vector of the MLE in LSB-first index order
  (`MLE.to_coefficients`), and `P_even/P_odd` are the even/odd
  coefficient subsequences — so folding the codeword binds the *first*
  variable of the MLE, exactly the round variable of the piop sumcheck
  kernels (which pair table entries `(2i, 2i+1)`). [ZCF24] splits the
  message into contiguous halves (MSB) instead; the two are relabelings of
  the same family, and this choice is what lets basefold share round
  machinery with `vfhe.piop.sumcheck`.
- **Everything is per RNS prime.** The code acts on vectors of ring
  elements coefficient-slot-wise and per RNS prime (an interleaved RS
  code), so each (prime, coefficient slot) pair carries an independent
  codeword over `Z_p`: the kernels read `coeffs[i][j]` directly (hence
  every entry must be in the same — RNS/NTT — representation, which
  `encode`/`decode` normalize first), roots are per-prime integers, twists
  are applied through `Polynomial * list` (per-RNS-residue scaling), and
  the twist inverses `(2t)^{-1}` are per-prime modular inverses — no ring
  inversions.
- **Two bounds on the codeword length.** Upward: the negacyclic transform
  of length `n_d` needs `2·n_d | p - 1`, and ring primes only guarantee
  `p = 1 mod 2N/split_degree`, so `n_d` must divide `N/split_degree`.
  Downward: arith's vectorized NTT kernels are guarded on `sub_n >= 16`
  (one AVX512 lane group per butterfly stage) and read past the buffer
  below that, so the *base* length `n_0 = c·k_0` must be at least 16 —
  every level gets encoded, `n_0` included. Both are checked in the
  constructor.

`decode` is the reference's other half: the inverse transform plus the
degree check (coefficients above the dimension must vanish) that decides
whether a vector is in the code. The evaluation protocol does not need it
— the verifier *computes* the level-0 codeword rather than decoding one —
but it is the natural round-trip test for the encoder and a ready-made
proximity check.

Soundness over the ring reduces to the per-prime components: the sumcheck
side samples challenges from the exceptional set as in the piop module
([GNS23]; [CCCFGS25]), and the code side is an RS code over each residue
field.

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

**Leaves are `±x` pairs, not single positions.** Our fold reads adjacent
entries (§2), so committing one leaf per pair makes a single path
authenticate both operands of a fold check — [ZCF24, Remark 9]'s packed
leaves, standard in FRI deployments — which halves both the number of paths
and the tree height.

`BasefoldEval` is that protocol: register
`iop.register(Relation_Eval, BasefoldEval(scheme, rep))` and every
evaluation claim is proved against a commitment — the statement's own
`commitment` field when set, else the scheme's record for the statement's
oracle (a never-committed oracle is a `LookupError`). For a claim on `n` variables
with a depth-`d` code (`k_d = 2^n`, `kappa = n - d` base variables):

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
3. **Query phase** (new with the Merkle layer, and the one real change to
   the message flow): the verifier can no longer read a committed vector,
   so the positions come from a *published* bit-string challenge
   (`Verifier.challenge_bits`, drawn after `h0`), which both parties expand
   identically with `BasefoldEval.query_positions` — the raw coins are the
   verifier's, their shape is the protocol's. The expansion hashes the seed
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
it, no bridge relation or extra DAG level. (An `EvalToOpen` protocol
reducing `Relation_Eval -> Relation_Open` existed briefly; it was a pure
relabeling — no messages, no soundness cost — and was folded back in.)
The full compiled chain is `Relation_Sum -> Sumcheck -> Relation_Eval ->
BasefoldEval`, with `scheme.commit(f)` run *before* the IOP — so the
commitment precedes every challenge that fixes the evaluation point,
resolving the ordering caveat the earlier commit-inside-prove design had
under Fiat-Shamir.

## 4. Roadmap

1. ~~**Merkle vector commitments**~~ — done: roots commit the codewords,
   the codeword and tree live in the `BasefoldOpening` under
   `prover.witnesses[commitment]`, and spot checks carry authentication
   paths over pair leaves. The **Fiat-Shamir** layer above it is done too
   (piop.md §5): with `IOP(fiat_shamir=True)` the run collapses to a single
   non-interactive `Proof`, produced by `iop.prove(statement)` and checked
   by a separate `iop.verify(statement, proof)` that holds no witnesses.
2. **Batched openings**: M evaluation claims at a common point batch into
   one basefold run on a random linear combination (an `M -> 1` folding
   reduction; `batching = True` on `BasefoldEval` itself, whose driver
   support already exists), and different-point claims reduce to
   common-point ones by sumcheck.
3. **C kernel for the fold** (`fold_at` is still per-position Python over
   `Polynomial`, one `Polynomial * list` scaling per entry); the encoder
   and decoder are native already. Multithreading the per-prime,
   per-coefficient-slot columns of `rs_encode` is the other easy win —
   they are independent transforms.
4. ~~**Query derandomization under Fiat-Shamir**~~ — done: `challenge_bits`
   comes from the transcript chain under `FS_Verifier`, and
   `query_positions` expands it exactly as before, so both parties still
   agree on the positions with no protocol change.

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
