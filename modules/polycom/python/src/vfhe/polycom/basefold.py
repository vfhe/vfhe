# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The basefold polynomial commitment scheme [ZCF24] over R_q or a field.

Structured as the standard four-algorithm PCS [ZCF24, Def. 8; BFS20]:
`Basefold` is the scheme (Setup = choosing the code; `commit` and `open`
are its algorithms), and `BasefoldEval` is the Eval protocol — a piop
`Protocol` discharging `Relation_Eval` in place of the terminal oracle
query, proving R_Eval = {[(C, z, y); f] : f(z) = y and C opens to f}
against the claim's commitment (its own field, or the scheme's record for
its oracle — the commitment is just the oracle's other form [ZCF24, §4],
so no separate relation or bridge protocol is needed). Committing and
evaluating are separate moments: `commit(f)` may run long before any IOP
exists (and its cost — the encoding and one Merkle tree — is paid once per
polynomial, however many evaluation claims follow); the resulting
commitment is *instance* data, riding on `Relation_Eval` statements, never
on the transcript. The prover's opening data is stored under the commitment
in `prover.witnesses`.

The commitment is a **Merkle root** over the level-d codeword
(`vfhe.piop.Merkle`, BLAKE3): the RO-model instantiation of [ZCF24] §4's
ideal oracle, via the BCS compiler [BCS16]. It commits to the codeword's
`±x` *pairs*, one leaf each — our fold reads adjacent entries, so a single
path authenticates both operands of a fold check ([ZCF24, Remark 9]'s
packed leaves), halving both the paths and the tree height. Note the
consequence for the security statement: binding is *computational*
(collision resistance) layered on the code-distance argument, where the
oracle-model version is information-theoretic — see `soundness_error`.

The evaluation claim f(z) == v is proved as the sumcheck claim
sum_b f(b) * eq~(z, b) == v, run for d rounds interleaved with folds of the
committed codeword — the same challenge r_s binds the round variable of the
sumcheck tables and folds the codeword, so the code's message stays the
coefficient vector of the partially-bound polynomial (see polycom.md for
the derivation and bibliography). Each round publishes the folded codeword
as a root; after the d folds the prover sends the remaining kappa-variable
table in the clear. The verifier checks the base sumcheck claim with its
own eq~ table, re-encodes that table into the level-0 codeword (which it
therefore holds in full, and which needs no tree), and spot-checks the
folds: a published bit-string challenge (`Verifier.challenge_bits`) is
expanded into `rep` query positions, distinct in the level-0 codeword
(`query_positions`), and each answer — a pair plus its path, at every level
from d down to 1 — is checked against that level's root and against the
value the previous level's fold produced.

Merkle itself is a primitive here, not a `Relation` or a `Protocol`: it
carries computational binding with no challenges, rounds or reduction, so
it belongs to the compiler layer rather than the IOP layer (piop.md §7).
Fiat-Shamir sits above it and needs nothing from this module: it binds the
commitment by hashing the statement (sigma_0 = rho(x)) and derives both
samplers (`challenge`, `challenge_bits`) from the transcript, without
changing any message here.
"""

from __future__ import annotations

from vfhe.arith import Field, Polynomial, Ring
from vfhe.piop import (
    MLE,
    Merkle,
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Relation_SumProd,
    Statement,
    SumcheckProd,
    Verifier,
    element_digest,
)
from vfhe.piop.merkle import DIGEST_LEN, hash_bytes
from vfhe.piop.mle import native_table, vector_table
from vfhe.piop.sumcheck import _exceptional_set_size, interpolate_evals

from .code import FoldableRS, pair_digest  # noqa: F401  (pair_digest: public here)
from .field_code import FieldFoldableRS


def _committed_table(f) -> bool:
    """Whether `f` is a table basefold can commit to: a dense evaluation-basis
    MLE over a ring (Polynomial entries) or over a field (a FieldVector)."""
    return native_table(f) or vector_table(f)


def _domain_of(f):
    """The Ring or Field of a committed table's coefficients."""
    return f.ring if f.ring is not None else f.field


def _one(domain):
    """The multiplicative identity of the domain, as one of its elements."""
    if isinstance(domain, Field):
        return domain.one
    return Polynomial(domain).from_array([1])


def _digest_leaf(leaf: bytes) -> bytes:
    """The Merkle `hash=` for trees whose leaves are already digests."""
    return leaf


class BasefoldCommitment:
    """A commitment to a multilinear polynomial: public instance data,
    created once by `Basefold.commit` and referenced by any number of
    `Relation_Eval` statements (across IOP runs — it never touches a
    transcript).

    `root` is the Merkle root of the level-d codeword's pair leaves, and
    `variables` the committed variables, whose order is the canonical order
    of evaluation points. Nothing else: the codeword itself is prover data
    (`BasefoldOpening`).
    """

    def __init__(self, variables: list, root: bytes):
        if len(root) != DIGEST_LEN:
            raise ValueError(f"a commitment root is {DIGEST_LEN} bytes")
        self.variables = list(variables)
        self.root = bytes(root)

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    def digest(self) -> bytes:
        """The `piop.element_digest` hook: what a Fiat-Shamir chain absorbs
        when this commitment sits in a statement (the root already binds the
        codeword; the variables fix the point order)."""
        return hash_bytes(
            b"basefold-commitment" + self.root + element_digest(self.variables)
        )

    def __repr__(self) -> str:
        return f"BasefoldCommitment(vars={self.num_vars}, root={self.root.hex()[:16]}…)"


class BasefoldOpening:
    """The prover's data for one commitment — the witness of the evaluation
    claim, stored as `prover.witnesses[commitment]`.

    Holds the polynomial plus what commit already paid for and every
    evaluation proof reuses: the level-d codeword and its Merkle tree. The
    shape mirrors the `commit -> (Commitment, ProverData)` split of PCS
    implementations (arkworks' poly-commit, plonky3's `Pcs`).
    """

    __slots__ = ("polynomial", "tree", "word")

    def __init__(self, polynomial: MLE, word: list, tree: Merkle):
        self.polynomial = polynomial
        self.word = word
        self.tree = tree

    def __repr__(self) -> str:
        return f"BasefoldOpening(vars={self.polynomial.num_vars}, n={len(self.word)})"


class Basefold:
    """The basefold PCS [ZCF24, Def. 8] over the domain of its code — R_q
    (`FoldableRS`) or a field (`FieldFoldableRS`): this object is the scheme
    (its `code` is the Setup output); `commit` / `open` below, and Eval is
    the `BasefoldEval` protocol, registered for `Relation_Eval`.

    `commitments` is the scheme's public oracle -> commitment association:
    which polynomial a commitment binds is instance data both parties know,
    and it is how `BasefoldEval` finds the commitment behind an evaluation
    claim that carries only the oracle (the compiled-pipeline shape). `openings` is the matching *prover-side*
    map (commitment -> `BasefoldOpening`); a verifier's copy of a scheme
    simply never holds one, and the in-process sharing of this object is the
    same convention that lets a verifier hold MLE oracles it may only query.
    """

    def __init__(self, code: FoldableRS | FieldFoldableRS):
        self.code = code
        self.commitments: dict[MLE, BasefoldCommitment] = {}
        self.openings: dict[BasefoldCommitment, BasefoldOpening] = {}

    def _check_polynomial(self, f) -> None:
        if not _committed_table(f):
            raise TypeError(
                "basefold commits to ring- or field-backed evaluation-basis MLEs"
            )
        if (1 << f.num_vars) != self.code.k_d:
            raise ValueError(
                f"polynomial has 2^{f.num_vars} coefficients but the code "
                f"encodes {self.code.k_d}"
            )

    def merkle_commit(self, word: list) -> Merkle:
        """The Merkle tree commitment to a codeword — the vector-commitment
        step of the scheme, one leaf per `±x` pair (digested by the code,
        `leaf_digests`); its `root` is what travels (as the polynomial
        commitment or as a round message)."""
        return Merkle(self.code.leaf_digests(word), hash=_digest_leaf)

    def commit(self, f: MLE) -> tuple[BasefoldCommitment, BasefoldOpening]:
        """Commit to f: encode, build the Merkle tree, return the root as the
        commitment and the codeword plus tree as the prover's opening data.

        Both costs are paid here, once; store the opening under the
        commitment on whichever provers will run evaluation proofs
        (`prover.witnesses[commitment] = opening`).
        """
        self._check_polynomial(f)
        word = self.code.encode(f.to_coefficients().table)
        tree = self.merkle_commit(word)
        commitment = BasefoldCommitment(f.variables, tree.root)
        self.commitments[f] = commitment
        self.openings[commitment] = BasefoldOpening(f, word, tree)
        return commitment, self.openings[commitment]

    def open(self, commitment: BasefoldCommitment, f) -> bool:
        """The Open algorithm [ZCF24, Def. 8]: does `commitment` open to f?
        Re-encode, rebuild the tree, compare roots — so it is exactly as
        binding as the hash is collision-resistant."""
        if not isinstance(f, MLE) or f.variables != commitment.variables:
            return False
        try:
            self._check_polynomial(f)
        except (TypeError, ValueError):
            return False
        return self.merkle_commit(self.code.encode(f.to_coefficients().table)).root == (
            commitment.root
        )


class BasefoldEval(Protocol):
    """Basefold's Eval protocol: discharges `Relation_Eval` statements
    against the scheme's code, replacing the terminal oracle query.

    Register with `iop.register(Relation_Eval, BasefoldEval(scheme))` —
    every evaluation claim reaching the driver is then proved against a
    commitment: the statement's own `commitment` field when set (the
    standalone commit-then-eval shape), else the one the scheme recorded
    for the statement's oracle at commit time (the compiled-pipeline shape,
    where sumcheck emits oracle-carrying claims; an oracle the scheme never
    committed is a `LookupError`). This is [CHMMVW20]'s "queries become
    opening claims" done in place — no separate relation, since the
    commitment is just the oracle's other form [ZCF24, §4].

    The prover's witness is the `BasefoldOpening`, from
    `prover.witnesses[commitment]` (falling back to the scheme's own
    records, which commit() filled); the verifier works from the root
    alone. The code's dimension k_d must equal 2^num_vars of the
    commitment, and its depth d must not exceed num_vars (kappa =
    num_vars - d variables remain in the clear base table).
    """

    reduces_from: type[Relation] = Relation_Eval
    reduces_to: tuple[type[Relation], ...] = ()
    supported_domains: tuple[type, ...] = (Ring, Field)

    def __init__(self, scheme: Basefold, rep: int = 32):
        if rep >= scheme.code.n0:
            raise ValueError(
                f"rep = {rep} query positions but the base codeword has only "
                f"{scheme.code.n0}: the queries must land on distinct "
                "positions of the level-0 codeword (the one the verifier "
                "holds in full), so it must be larger than their number"
            )
        self.scheme = scheme
        self.code = scheme.code
        self.rep = rep

    def soundness_error(
        self, statement: Statement, domain, gamma: float, delta: float
    ) -> float | None:
        """The information-theoretic part,
        2d / (gamma^3 |A|) + (1 - delta + gamma * d)^rep [ZCF24], per
        RNS-prime component (|A| = the residue field size); the caller must
        pick gamma, delta satisfying the theorem's distance conditions for
        the code (polycom.md). None if |A| is unknown.

        With Merkle commitments this is no longer the whole story: binding
        is computational, so the argument's error also carries the hash's
        collision term ([ZCF24, Thm. 4] reduces one to the other). That term
        is a property of BLAKE3 and the security parameter, not of these
        protocol parameters, so it is deliberately not folded in here.
        """
        size = _exceptional_set_size(domain)
        if size is None:
            return None
        d = self.code.d
        return 2 * d / (gamma**3 * size) + (1 - delta + gamma * d) ** self.rep

    def _check_statement(self, statement: Statement) -> BasefoldCommitment:
        """The commitment this claim is proved against: the statement's own
        field when set, else the scheme's record for the statement's oracle."""
        commitment = statement.commitment
        if commitment is None and statement.oracles:
            (f,) = statement.oracles
            commitment = self.scheme.commitments.get(f)
            if commitment is None:
                raise LookupError(
                    "evaluation claim on an oracle the scheme never "
                    "committed; call scheme.commit(f) before running the IOP"
                )
        if not isinstance(commitment, BasefoldCommitment):
            raise TypeError(
                "the evaluation claim carries no BasefoldCommitment (neither "
                "a commitment field nor a committed oracle)"
            )
        if (1 << commitment.num_vars) != self.code.k_d:
            raise ValueError(
                f"commitment has 2^{commitment.num_vars} coefficients but "
                f"the code encodes {self.code.k_d}"
            )
        return commitment

    def query_positions(self, seed: bytes) -> tuple[int, ...]:
        """The `rep` spot-check positions derived from a bit-string
        challenge: pair indices at level d, in [0, n_d / 2).

        The seed is expanded with the tree's own hash in counter mode
        (`BLAKE3(seed || counter)`, eight candidates per digest); a candidate
        is a masked 64-bit word — the range is a power of two, so masking is
        unbiased — and is **rejection-sampled on its projection to the
        level-0 codeword** (`q >> (d - 1)`): two queries meeting at the
        bottom would run the same final fold check twice, so with-
        replacement sampling silently buys less soundness than its `rep`
        claims. Distinct bottom projections imply distinct positions at
        every level (the higher projections extend the bottom one), and the
        constructor guarantees termination (`rep < n0`).

        Deterministic in the seed, so both parties derive the same positions
        from the published challenge — the prover to answer, the verifier to
        check.
        """
        top = self.code.n_d // 2
        shift = self.code.d - 1
        positions: list[int] = []
        seen: set[int] = set()
        counter = 0
        while len(positions) < self.rep:
            digest = hash_bytes(seed + counter.to_bytes(8, "little"))
            counter += 1
            for off in range(0, DIGEST_LEN, 8):
                candidate = int.from_bytes(digest[off : off + 8], "little") & (top - 1)
                base = candidate >> shift
                if base in seen:
                    continue
                seen.add(base)
                positions.append(candidate)
                if len(positions) == self.rep:
                    break
        return tuple(positions)

    def _walk(self, query: int, d: int):
        """The (level, pair index) chain one query position visits, top level
        first.

        Folding pair `j` of level `l` yields the single value `W_{l-1}[j]`,
        which sits in pair `j // 2` of level `l - 1` — at offset `j & 1`
        inside it. So the walk is forced: with only authenticated pairs the
        verifier must reuse the value it just derived, which is what chains
        the levels together (and what makes the queries a proximity test
        rather than d independent ones).
        """
        j = query
        for level in range(d, 0, -1):
            yield level, j
            j //= 2

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        commitment = self._check_statement(statement)
        opening = prover.witnesses.get(commitment)
        if opening is None:
            # In-process convenience: commit() already recorded the opening
            # on the scheme, so a pipeline needs no manual installation.
            opening = self.scheme.openings.get(commitment)
            if opening is not None:
                prover.witnesses[commitment] = opening
        if opening is None:
            raise LookupError(
                "no opening for this commitment: store the BasefoldOpening "
                "commit() returned under prover.witnesses[commitment]"
            )
        if not isinstance(opening, BasefoldOpening):
            raise TypeError(
                "the witness for a basefold commitment must be the "
                "BasefoldOpening commit() returned"
            )
        f = opening.polynomial
        if not _committed_table(f) or f.variables != commitment.variables:
            raise TypeError("the opening does not match the commitment")
        iop = prover.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        label = f"basefold{statement.path}"
        d = self.code.d
        zs = [statement.point[v] for v in commitment.variables]

        # Codewords and their trees by level: level d was built at commit
        # time and only folded here; each fold's result is committed by root.
        words = {d: opening.word}
        trees = {d: opening.tree}
        cur_f = f
        cur_eq = MLE.eq(_domain_of(f), zs, variables=f.variables)
        for s in range(d):
            var = cur_f.variables[0]
            evals = SumcheckProd.prod_round_evals([cur_f, cur_eq])
            iop.transcript.write(f"{label}/g{s}", evals)
            r = iop.verifier.challenge(f"{label}/r{s}")
            cur_f = cur_f.evaluate({var: r}, in_place=cur_f is not f)
            cur_eq = cur_eq.evaluate({var: r}, in_place=True)
            level = d - s
            folded = self.code.fold(words[level], r, level=level)
            if level > 1:  # level 0 is the one the verifier re-encodes itself
                words[level - 1] = folded
                trees[level - 1] = self.scheme.merkle_commit(folded)
                iop.transcript.write(f"{label}/pi{s + 1}", trees[level - 1].root)
        # The base case: the remaining kappa-variable table, in the clear
        # (the verifier re-encodes its coefficients itself).
        iop.transcript.write(f"{label}/h0", cur_f)

        # The query phase, which only exists once the codewords are roots:
        # the verifier cannot read a committed vector, so query positions are
        # derived from a published bit-string challenge (either party may
        # draw it; honesty = only after h0 is written) and the prover answers
        # each with the pair and path at every level.
        queries = self.query_positions(iop.verifier.challenge_bits(f"{label}/queries"))
        iop.transcript.write(
            f"{label}/answers",
            tuple(
                tuple(
                    (self.code.pair_at(words[level], j), trees[level].open(j))
                    for level, j in self._walk(q, d)
                )
                for q in queries
            ),
        )
        return []

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        commitment = self._check_statement(statement)
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        domain = iop.domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        label = f"basefold{statement.path}"
        d = self.code.d
        zs = [statement.point[v] for v in commitment.variables]

        # Roots by level: the commitment is level d, each round message the
        # level below it (label pi_k carries level d - k).
        roots = {d: commitment.root}
        claim = statement.value
        rs = []
        for s in range(d):
            evals = await iop.transcript.read(f"{label}/g{s}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {s}: g(0) + g(1) != claim")
            r = verifier.challenge(f"{label}/r{s}")
            rs.append(r)
            claim = interpolate_evals(evals, r)
            if d - s > 1:
                roots[d - s - 1] = await iop.transcript.read(f"{label}/pi{s + 1}")

        h0 = await iop.transcript.read(f"{label}/h0")
        if not _committed_table(h0) or (1 << h0.num_vars) != self.code.k0:
            raise Rejection(f"{label}: malformed base table")

        # Base sumcheck check: sum_b h0(b) * eq~(z, (r, b)) == claim, with
        # eq~ factored into the bound-variable scalar and the tail table
        # (built over h0's own variables — the pairing with z is positional).
        one = _one(domain)
        eq_factor = one
        for z, r in zip(zs[:d], rs, strict=True):
            eq_factor = eq_factor * (z * r + (one - z) * (one - r))
        eq_tail = MLE.eq(domain, zs[d:], variables=h0.variables)
        base = Statement(
            Relation_SumProd(),
            oracles=[h0, eq_tail.scale(eq_factor)],
            value=claim,
        )
        if not base.check():
            raise Rejection(f"{label}: base table does not match the claim")

        # The level-0 codeword is computed, not received, so it is a codeword
        # by construction and needs no tree — it terminates every query walk.
        word0 = self.code.encode(h0.to_coefficients().table)
        queries = self.query_positions(verifier.challenge_bits(f"{label}/queries"))
        answers = await iop.transcript.read(f"{label}/answers")
        if len(answers) != len(queries):
            raise Rejection(f"{label}: wrong number of query answers")

        for q, answer in zip(queries, answers, strict=True):
            steps = list(self._walk(q, d))
            if len(answer) != len(steps):
                raise Rejection(f"{label}: query {q} answered for {len(answer)} levels")
            for i, ((level, j), (pair, path)) in enumerate(
                zip(steps, answer, strict=True)
            ):
                if not Merkle.verify(
                    roots[level], j, path, pair, hash=self.code.leaf_digest
                ):
                    raise Rejection(
                        f"{label}: Merkle path rejected at level {level}, pair {j}"
                    )
                folded = self.code.fold_pair(pair[0], pair[1], rs[d - level], level, j)
                # The folded value must reappear in the next level down: at
                # offset j & 1 of the pair the walk moves to, or — at the
                # bottom — in the level-0 codeword the verifier built itself.
                below = word0[j] if level == 1 else answer[i + 1][0][j & 1]
                if not (below == folded):
                    raise Rejection(
                        f"{label}: fold check failed at level {level - 1}, position {j}"
                    )
        return []
