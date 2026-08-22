<!-- SPDX-FileCopyrightText: 2026 Antonio Guimarães -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# vfhe.polycom

Polynomial commitment schemes for the multilinear oracles of `vfhe.piop`.
The architecture and its derivation from the literature are documented in
[polycom.md](polycom.md).

- `code.py` + `c/src/rscode.c`: `FoldableRS`, a depth-d foldable
  Reed-Solomon code over `R_q` applied per RNS prime — every level is
  itself an RS code (FRI-style), messages are LSB-first coefficient
  vectors, and folding a codeword with a challenge binds the MLE's first
  variable. The transform is arith's negacyclic NTT through the `rs_*`
  kernels (`rs_encode` / `rs_decode`, one `NTT_proc` per level and prime):
  `ntt_forward` is bit-reversed, so a codeword's `±x` pairs sit at adjacent
  positions `(2i, 2i+1)` and successive levels share a root tower
  (`psi_{n/2} = psi_n^2`). `encode` infers the level from the message
  length, `decode` adds the degree check, and `fold` / `fold_at` are the
  verifier-checkable fold. Codeword lengths run from 16 (arith's shortest
  vectorized transform) up to `N/split_degree`.
- `basefold.py`: the basefold PCS in the standard four-algorithm shape
  (Setup, Commit, Open, Eval), interactive or non-interactive — the
  protocol is the same either way, since Fiat-Shamir lives in the piop
  verifier (`IOP(fiat_shamir=True)`, then `prove` / `verify`).
  `Basefold` is the scheme: `commit(f)` encodes and builds the Merkle tree
  once, returning `(BasefoldCommitment, BasefoldOpening)` — a succinct
  root (instance data: it rides on statements, never on the transcript, and
  is reused by any number of evaluation claims across IOP runs) and the
  prover data (codeword + tree) each proof reuses; `open` is the opening
  decider (roots via `merkle_commit`, one leaf per `±x` pair, so one path
  authenticates both operands of a fold check). `BasefoldEval` is the Eval
  protocol, registered for `Relation_Eval` in place of the terminal oracle
  query: the claim's commitment comes from its optional `commitment` field
  or from the scheme's record for its oracle (witness = the
  `BasefoldOpening` in `prover.witnesses[commitment]`, backfilled from the
  scheme). It runs a
  product sumcheck for `sum_b f(b)·eq~(z, b) == v` interleaved with folds of
  the committed codeword (sharing challenges, round kernels, and wire format
  with `vfhe.piop.sumcheck`), each fold published as a root, a clear base
  table the verifier re-encodes, and then a query phase — a published
  bit-string challenge (`Verifier.challenge_bits`) that both parties expand
  into positions with `query_positions` (counter-mode BLAKE3, rejection-
  sampled to be distinct in the level-0 codeword, hence `rep < n0`), the
  prover answering with pairs and paths that are checked against each
  level's root and against the fold above. A chain like sumcheck composes
  end-to-end with no bridge: `scheme.commit(f)`, then
  `Relation_Sum -> Sumcheck -> Relation_Eval -> BasefoldEval`.

Fiat-Shamir needs nothing from this module — `IOP(fiat_shamir=True)` derives
the fold challenges and the query coins from the transcript and binds the
commitment-carrying statement, so `iop.prove(statement)` yields a `Proof`
another IOP checks with no prover present. Batched openings and a C kernel
for the fold are roadmap (polycom.md §4).
