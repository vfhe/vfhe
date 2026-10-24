# SPDX-FileCopyrightText: 2026 The vFHE Authors
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
(`vfhe.crypto.Merkle`, BLAKE3): the RO-model instantiation of [ZCF24] §4's
ideal oracle, via the BCS compiler [BCS16]. It commits to the codeword's
adjacent *pairs*, one leaf each — the fold reads a pair, so a single path
authenticates both operands of a fold check ([ZCF24, Remark 9]'s packed
leaves), halving both the paths and the tree height. A layer schedule
(`Basefold`) widens a leaf to the `2^s` adjacent symbols that `s` folds
reduce to one. Note the
consequence for the security statement: binding is *computational*
(collision resistance) layered on the code-distance argument, where the
oracle-model version is information-theoretic — see `soundness_error`.

The evaluation claim f(z) == v is proved as the sumcheck claim
sum_b f(b) * eq~(z, b) == v, run for d rounds interleaved with folds of the
committed codeword — the same challenge r_s binds the round variable of the
sumcheck tables and folds the codeword, so the code's message stays the
coefficient vector of the partially-bound polynomial (see polycom.md for
the derivation and bibliography). The folds are grouped into layers by the
scheme's schedule, and the round that ends a layer publishes the folded
codeword as a root -- every round, under the default schedule of one fold
per layer; after the d folds the prover sends the remaining kappa-variable
table in the clear. The verifier checks the base sumcheck claim with its
own eq~ table, re-encodes that table into the level-0 codeword (which it
therefore holds in full, and which needs no tree), and spot-checks the
folds: a published bit-string challenge (`Verifier.challenge_bits`) is
expanded into `rep` query positions, distinct in the level-0 codeword
(`query_positions`), and each answer — a leaf plus its path, for every
layer from the top down — is checked against that layer's root and against
the value the previous layer's folds produced, and refolded with the
layer's challenges.

Merkle itself is a primitive here, not a `Relation` or a `Protocol`: it
carries computational binding with no challenges, rounds or reduction, so
it belongs to the compiler layer rather than the IOP layer (piop.md §7).
Fiat-Shamir sits above it and needs nothing from this module: it binds the
commitment by hashing the statement (sigma_0 = rho(x)) and derives both
samplers (`challenge`, `challenge_bits`) from the transcript, without
changing any message here.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from vfhe.arith import MLE, Field, FieldVector, Polynomial, Ring
from vfhe.arith.mle import native_table, vector_table
from vfhe.crypto import DIGEST_LEN, Merkle, hash_bytes
from vfhe.piop import (
    CommittedOracle,
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Statement,
    TableOracle,
    Verifier,
    element_digest,
    exceptional_set_size,
)
from vfhe.piop.protocols.sumcheck.rounds import interpolate_evals, prod_round_evals
from vfhe.piop.relations import Relation_SumProd

# `pair_digest` is re-exported here: the aliased form is what marks it
# as a public name of this module rather than an unused import.
from .code import pair_digest as pair_digest
from .queries import query_positions

if TYPE_CHECKING:
    from collections.abc import Callable

    from .code import FoldableRS
    from .field_code import FieldFoldableRS

#: A coefficient table or a codeword, in whichever form the domain's code
#: takes: a list of ring elements, or one `FieldVector`. A `Basefold` is built
#: with the code matching its tables, so which of the two travels through the
#: protocol is fixed by that pairing rather than by any one signature here.
Word = Any


def _gather(word: Word, positions: list[int]) -> Word:
    """The symbols of `word` at `positions`, in the word's own form."""
    if isinstance(word, FieldVector):
        return word.query(array("Q", positions))
    return [word[i] for i in positions]


def _identity(digest):
    """`Merkle.verify`'s leaf hash when the digest is already in hand."""
    return digest


def _coefficients(f: MLE) -> Word:
    """`f`'s coefficient table, in the form its own domain's code encodes."""
    return f.to_coefficients().table


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

    def __init__(self, polynomial: MLE, word: Word, tree: Merkle) -> None:
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

    `commit` returns the committed oracle (`piop.CommittedOracle`: the
    commitment, the variables and this scheme, what a statement names) and
    the opening.
    `openings` is the *prover-side* map (commitment -> `BasefoldOpening`)
    `BasefoldEval` falls back to; a verifier's copy of a scheme simply never
    holds one, and the in-process sharing of this object is the same
    convention that lets a verifier hold table oracles it may only query.

    `schedule` groups the code's `d` folds into layers: `schedule[i]` folds
    make layer `i`, and only the codeword each layer starts from is committed,
    by a Merkle tree whose leaves are its fibers of `2^schedule[i]` adjacent
    symbols (the symbols those folds reduce to one). The counts are positive
    and sum to `d`; the default ``[1] * d`` commits every folded codeword over
    pair leaves. The commitment is the first layer's tree, so it depends on
    `schedule[0]`.
    """

    def __init__(
        self, code: FoldableRS | FieldFoldableRS, schedule: list[int] | None = None
    ):
        schedule = [1] * code.d if schedule is None else list(schedule)
        if not schedule or any(
            not isinstance(folds, int) or folds < 1 for folds in schedule
        ):
            raise ValueError(f"schedule must be positive fold counts, got {schedule}")
        if sum(schedule) != code.d:
            raise ValueError(
                f"schedule sums to {sum(schedule)} but the code has depth {code.d}"
            )
        self.code = code
        self.schedule = schedule
        #: The code level each layer starts from, top first, then the level
        #: the last layer ends at (0).
        self.levels = [code.d]
        for folds in schedule:
            self.levels.append(self.levels[-1] - folds)
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

    def merkle_commit(self, word: Word, folds: int | None = None) -> Merkle:
        """The Merkle tree commitment to a codeword — the vector-commitment
        step of the scheme, one leaf per fiber of `2^folds` adjacent symbols
        (digested by the code, `leaf_digests`), `folds` the first layer's
        count by default; its `root` is what travels (as the polynomial
        commitment or as a round message).

        The digests reach the tree **packed**, which is why nothing here
        builds a Python object per leaf. A codeword has a leaf per pair, so an
        object per leaf would be one on the way in and another on the way back
        out when the builder rejoined them -- at a large codeword more than
        either the hashing or the tree costs, and paid once per fold level as
        well as once per commit.
        """
        folds = self.schedule[0] if folds is None else folds
        return Merkle.from_digests(self.code.leaf_digests(word, 1 << folds))

    def commit(
        self, f: MLE, prover: Prover | None = None
    ) -> tuple[CommittedOracle, BasefoldOpening]:
        """Commit to f: encode, build the Merkle tree, return the committed
        oracle -- the root as the commitment, with the variables and this
        scheme -- and the codeword plus tree as the prover's opening data.

        Both costs are paid here, once. The opening is recorded on the scheme
        and, with `prover` given, installed on it: `witnesses[commitment] =
        opening` is what the evaluation protocol proves against, and
        `witnesses[oracle] = f` is the table a sumcheck prover folds when
        the oracle is one of its factors (`piop.resolve_table`). Without a
        prover, store them yourself on whichever provers will run. Mutating
        the table after the commit breaks the binding.
        """
        self._check_polynomial(f)
        word = self.code.encode(_coefficients(f))
        tree = self.merkle_commit(word)
        commitment = BasefoldCommitment(f.variables, tree.root)
        opening = BasefoldOpening(f, word, tree)
        self.openings[commitment] = opening
        oracle = CommittedOracle(f.variables, commitment, self)
        if prover is not None:
            prover.witnesses[oracle] = f
            prover.witnesses[commitment] = opening
        return oracle, opening

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
        return self.merkle_commit(self.code.encode(_coefficients(f))).root == (
            commitment.root
        )


class BasefoldEval(Protocol):
    """Basefold's Eval protocol: discharges `Relation_Eval` statements
    against the scheme's code, replacing the terminal oracle query.

    Register with `iop.register(Relation_Eval, BasefoldEval(scheme))`;
    the registry keys it by `scheme`, so it receives the evaluation claims
    on the oracles committed under that scheme (`CommittedOracle`s, as
    `commit` returns them) and proves each against the commitment its
    oracle carries. A claim on any other oracle is a `TypeError`. This is
    [CHMMVW20]'s "queries become opening claims" done in place — no
    separate relation, since the commitment is just the oracle's other
    form [ZCF24, §4].

    It is a bundle protocol, and its bundle is every pending claim on the
    oracles committed under its scheme (`bundle_key`): the driver hands
    them over once no other statement can produce another. A single claim
    is opened by one basefold run; several, at any points and on any of the
    scheme's commitments, are opened together by one sumcheck to a common
    point and one basefold run on a random linear combination
    (`_prove_batch`, [CCCFGS26, §7, Figs. 3-4]).

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
    batching = True  # receives every pending claim under its scheme at once

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
        self,
        statement: Statement,
        domain,
        gamma: float,
        delta: float,
        claims: int = 1,
        polynomials: int = 1,
    ) -> float | None:
        """The information-theoretic part,
        2d / (gamma^3 |A|) + (1 - delta + gamma * d)^rep [ZCF24], per
        RNS-prime component (|A| = the residue field size). `delta` is the
        code's relative distance -- `self.code.relative_distance()` reports
        it -- and `gamma` is the caller's, subject to the theorem's
        conditions (polycom.md). None if |A| is unknown.

        For a bundle of `claims` evaluation claims on `polynomials` distinct
        polynomials, opened together, the batching adds
        (2l + claims - 1) / |A| for the common-point sumcheck when
        claims > 1 [CCCFGS26, §7.1] (l = the number of variables; the bound
        holds whether or not the points coincide) and
        (polynomials - 1) / |A| for the random linear combination
        [CCCFGS26, §7.2, Lemma 5].

        With Merkle commitments this is no longer the whole story: binding
        is computational, so the argument's error also carries the hash's
        collision term ([ZCF24, Thm. 4] reduces one to the other). That term
        is a property of BLAKE3 and the security parameter, not of these
        protocol parameters, so it is deliberately not folded in here.

        The bound is stated for [ZCF24]'s protocol, which commits every
        folded codeword: the default schedule. Under a schedule that groups
        several folds into a layer, the codewords inside a layer are not
        committed; the same expression is returned, and that case is not
        covered by the cited theorem.
        """
        size = exceptional_set_size(domain)
        if size is None:
            return None
        d = self.code.d
        error = 2 * d / (gamma**3 * size) + (1 - delta + gamma * d) ** self.rep
        if claims > 1:
            num_vars = self.code.k_d.bit_length() - 1
            error += (2 * num_vars + claims - 1) / size
        return error + (polynomials - 1) / size

    def _check_statement(self, statement: Statement) -> BasefoldCommitment:
        """The commitment this claim is proved against: the one its oracle
        carries."""
        (oracle,) = statement.oracles
        commitment = getattr(oracle, "commitment", None)
        if not isinstance(oracle, CommittedOracle) or not isinstance(
            commitment, BasefoldCommitment
        ):
            raise TypeError(
                "the evaluation claim is not on a basefold-committed oracle; "
                "name the oracle scheme.commit(f) returned"
            )
        if oracle.scheme is not self.scheme:
            raise TypeError(
                "the evaluation claim is on an oracle committed under another "
                "scheme; register that scheme's evaluation protocol"
            )
        if (1 << commitment.num_vars) != self.code.k_d:
            raise ValueError(
                f"commitment has 2^{commitment.num_vars} coefficients but "
                f"the code encodes {self.code.k_d}"
            )
        return commitment

    def query_positions(self, seed: bytes) -> tuple[int, ...]:
        """The `rep` spot-check positions derived from a bit-string
        challenge: leaf indices of the commitment's tree, in
        ``[0, n_d / 2^schedule[0])`` -- pair indices at level d under the
        default schedule.

        This scheme's geometry handed to `polycom.query_positions`, which is
        where the sampler and its rejection rule are: a leaf index is a
        position of the codeword the first layer ends at, so a walk from it
        consumes that level's `levels[1]` index bits to reach the level-0
        codeword, and the positions are rejection-sampled to land there
        pairwise distinct. The constructor's `rep < n0` guarantees
        termination.

        Deterministic in the seed, so both parties derive the same positions
        from the published challenge — the prover to answer, the verifier to
        check.
        """
        schedule = self.scheme.schedule
        return query_positions(
            seed, self.code.n_d >> schedule[0], self.scheme.levels[1], self.rep
        )

    def _walk(self, query: int) -> list[tuple[int, int]]:
        """The (leaf index, offset) chain one query position visits, one entry
        per layer, top first.

        Refolding leaf `j` of a layer yields the single value `W[j]` of the
        codeword the layer ends at, which is the next layer's starting
        codeword: there it sits in leaf ``j >> s`` at offset ``j & (2^s - 1)``,
        `s` that layer's fold count. So the walk is forced: with only
        authenticated leaves the verifier must reuse the value it just
        derived, which is what chains the layers together (and what makes the
        queries a proximity test rather than independent ones). The first
        layer has no value coming in, and its offset is 0.
        """
        steps, j = [(query, 0)], query
        for folds in self.scheme.schedule[1:]:
            steps.append((j >> folds, j & ((1 << folds) - 1)))
            j >>= folds
        return steps

    def bundle_key(self, oracle) -> Any:
        """Claims on every oracle committed under this protocol's scheme form
        one bundle, which `prove` opens together; any other oracle keeps a
        bundle of its own."""
        if isinstance(oracle, CommittedOracle) and oracle.scheme is self.scheme:
            return ("scheme", id(self.scheme))
        return id(oracle)

    def _bundle(
        self, statements: list[Statement]
    ) -> tuple[list[BasefoldCommitment], list[int], list[list]]:
        """The bundle's distinct commitments, in order of first appearance;
        for each claim, the index of its commitment in that list; and for
        each claim, its point read in its commitment's variable order (the
        order the tables are laid out in, so claims on different polynomials
        pair positionally)."""
        commitments: list[BasefoldCommitment] = []
        owners, points = [], []
        for statement in statements:
            commitment = self._check_statement(statement)
            index = next(
                (i for i, c in enumerate(commitments) if c is commitment), None
            )
            if index is None:
                index = len(commitments)
                commitments.append(commitment)
            owners.append(index)
            points.append([statement.point[v] for v in commitment.variables])
        return commitments, owners, points

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        # The driver hands over every pending claim on the oracles committed
        # under this scheme at once, in path order (identical on both sides).
        if len(statements) == 1:
            await self._prove_one(prover, statements[0])
        else:
            await self._prove_batch(prover, statements)
        return []

    def _opening(
        self, prover: Prover, commitment: BasefoldCommitment
    ) -> BasefoldOpening:
        """The prover's opening of `commitment`: `prover.witnesses`, or the
        scheme's own record, which commit() filled (and which is then
        installed on the prover, so a pipeline needs no manual step)."""
        opening = prover.witnesses.get(commitment)
        if opening is None:
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
        return opening

    async def _prove_one(self, prover: Prover, statement: Statement) -> None:
        await statement.resolved()
        commitment = self._check_statement(statement)
        opening = self._opening(prover, commitment)
        f = opening.polynomial
        if not _committed_table(f) or f.variables != commitment.variables:
            raise TypeError("the opening does not match the commitment")
        iop = prover.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        label = f"basefold{statement.path}"
        zs = [statement.point[v] for v in commitment.variables]
        size = 1 << self.scheme.schedule[0]
        run = _ProverRun(
            self,
            label,
            f,
            zs,
            opening.word,
            lambda j: _layer_answer(self.code, opening.word, opening.tree, j, size),
        )
        run.fold(iop)
        # The query phase, which only exists once the codewords are roots:
        # the verifier cannot read a committed vector, so query positions are
        # derived from a published bit-string challenge (either party may
        # draw it; honesty = only after h0 is written) and the prover answers
        # each with the leaf and path of every layer's tree.
        queries = self.query_positions(iop.verifier.challenge_bits(f"{label}/queries"))
        iop.transcript.write(f"{label}/answers", run.answers(queries))

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        if len(statements) == 1:
            await self._verify_one(verifier, statements[0])
        else:
            await self._verify_batch(verifier, statements)
        return []

    async def _verify_one(self, verifier: Verifier, statement: Statement) -> None:
        await statement.resolved()
        commitment = self._check_statement(statement)
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        domain = iop.domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        label = f"basefold{statement.path}"
        zs = [statement.point[v] for v in commitment.variables]
        size = 1 << self.scheme.schedule[0]
        run = _VerifierRun(
            self,
            label,
            zs,
            statement.value,
            lambda answers, positions: _check_layer(
                self.code, label, self.code.d, commitment.root, answers, positions, size
            ),
        )
        await run.fold(iop, verifier, domain)
        queries = self.query_positions(verifier.challenge_bits(f"{label}/queries"))
        run.check(queries, await iop.transcript.read(f"{label}/answers"))

    async def _prove_batch(self, prover: Prover, statements: list[Statement]) -> None:
        """The batched opening [CCCFGS26, §7, Figs. 3-4] of M > 1 claims
        f_{c(m)}(z_m) = y_m on C distinct committed polynomials.

        1. Common point (Fig. 3; skipped when every z_m is the same point):
           the verifier draws s, and a sumcheck over the l variables proves
           ``sum_b h(b) = sum_m s^m y_m`` for
           ``h(b) = sum_m s^m eq~(z_m, b) f_{c(m)}(b)``. It ends at a point r;
           the prover sends q_c = f_c(r) for every polynomial, and the
           verifier checks the last round against
           ``h(r) = sum_m s^m eq~(z_m, r) q_{c(m)}``. With one point z, r = z
           and q_c is the claimed value.
        2. Common-point opening (Fig. 4): the verifier draws alpha (when
           C > 1), and one basefold run proves u(r) = sum_c alpha^c q_c for
           ``u = sum_c alpha^c f_c``, on the codeword sum_c alpha^c pi_c of
           the committed ones -- the code is linear. A query on the first
           layer is answered by each commitment's leaf and path, and the
           verifier combines the authenticated leaves with the same alphas.
        """
        for statement in statements:
            await statement.resolved()
        commitments, owners, points = self._bundle(statements)
        openings = [self._opening(prover, c) for c in commitments]
        fs = [opening.polynomial for opening in openings]
        for f, commitment in zip(fs, commitments, strict=True):
            if not _committed_table(f) or f.variables != commitment.variables:
                raise TypeError("the opening does not match the commitment")
        iop = prover.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        label = f"basefold{statements[0].path}"
        domain = _domain_of(fs[0])
        one = _one(domain)

        if _same_point(points):
            point = points[0]
            values = [statements[owners.index(c)].value for c in range(len(fs))]
        else:
            s = iop.verifier.challenge(f"{label}/point/s")
            powers = _powers(s, len(statements), one)
            # h = sum_c f_c * E_c, E_c = sum of s^m eq~(z_m, .) over the
            # claims on f_c: one product per polynomial, not per claim.
            weights: list[MLE] = []
            for c, f in enumerate(fs):
                terms = [
                    MLE.eq(domain, points[m], variables=f.variables).scale(powers[m])
                    for m in range(len(statements))
                    if owners[m] == c
                ]
                weight = terms[0]
                for term in terms[1:]:
                    weight = weight + term
                weights.append(weight)
            tables = list(fs)
            point = []
            for t in range(len(points[0])):
                evals = None
                for f, e in zip(tables, weights, strict=True):
                    term = prod_round_evals([f, e])
                    evals = (
                        term
                        if evals is None
                        else tuple(a + b for a, b in zip(evals, term, strict=True))
                    )
                iop.transcript.write(f"{label}/point/g{t}", evals)
                r = iop.verifier.challenge(f"{label}/point/r{t}")
                point.append(r)
                tables = [
                    f.evaluate({f.variables[0]: r}, in_place=f is not fs[c])
                    for c, f in enumerate(tables)
                ]
                weights = [
                    e.evaluate({e.variables[0]: r}, in_place=True) for e in weights
                ]
            values = [f.constant() for f in tables]
            iop.transcript.write(f"{label}/point/q", tuple(values))

        size = 1 << self.scheme.schedule[0]
        if len(fs) == 1:
            u, word = fs[0], openings[0].word

            def top_answer(j: int) -> tuple:
                return _layer_answer(
                    self.code, openings[0].word, openings[0].tree, j, size
                )
        else:
            alphas = _powers(iop.verifier.challenge(f"{label}/alpha"), len(fs), one)
            u = fs[0].scale(alphas[0])
            for f, alpha in zip(fs[1:], alphas[1:], strict=True):
                term = f.scale(alpha)
                term.variables = u.variables  # the tables pair positionally
                u = u + term
            word = _combination([opening.word for opening in openings], alphas)

            def top_answer(j: int) -> tuple:
                return tuple(
                    _layer_answer(self.code, opening.word, opening.tree, j, size)
                    for opening in openings
                )

        run = _ProverRun(self, label, u, point, word, top_answer)
        run.fold(iop)
        queries = self.query_positions(iop.verifier.challenge_bits(f"{label}/queries"))
        iop.transcript.write(f"{label}/answers", run.answers(queries))

    async def _verify_batch(
        self, verifier: Verifier, statements: list[Statement]
    ) -> None:
        """The verifier's half of `_prove_batch`."""
        for statement in statements:
            await statement.resolved()
        commitments, owners, points = self._bundle(statements)
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        domain = iop.domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        label = f"basefold{statements[0].path}"
        one = _one(domain)

        if _same_point(points):
            point = points[0]
            values = [
                statements[owners.index(c)].value for c in range(len(commitments))
            ]
            for statement, c in zip(statements, owners, strict=True):
                if not (statement.value == values[c]):
                    raise Rejection(
                        f"{label}: two values claimed for one polynomial at one point"
                    )
        else:
            s = verifier.challenge(f"{label}/point/s")
            powers = _powers(s, len(statements), one)
            claim = None
            for power, statement in zip(powers, statements, strict=True):
                term = power * statement.value
                claim = term if claim is None else claim + term
            point = []
            for t in range(len(points[0])):
                evals = await iop.transcript.read(f"{label}/point/g{t}")
                if not (evals[0] + evals[1] == claim):
                    raise Rejection(f"{label} point round {t}: g(0) + g(1) != claim")
                r = verifier.challenge(f"{label}/point/r{t}")
                point.append(r)
                claim = interpolate_evals(evals, r)
            values = list(await iop.transcript.read(f"{label}/point/q"))
            if len(values) != len(commitments):
                raise Rejection(f"{label}: wrong number of values at the common point")
            expected = None
            for power, c, z in zip(powers, owners, points, strict=True):
                term = power * _eq_at(z, point, one) * values[c]
                expected = term if expected is None else expected + term
            if not (expected == claim):
                raise Rejection(f"{label}: the common-point values do not match")

        size = 1 << self.scheme.schedule[0]
        if len(commitments) == 1:
            value = values[0]

            def top_check(answers, positions) -> Word:
                return _check_layer(
                    self.code,
                    label,
                    self.code.d,
                    commitments[0].root,
                    answers,
                    positions,
                    size,
                )
        else:
            alphas = _powers(
                verifier.challenge(f"{label}/alpha"), len(commitments), one
            )
            value = None
            for alpha, v in zip(alphas, values, strict=True):
                value = alpha * v if value is None else value + alpha * v

            def top_check(answers, positions) -> Word:
                if any(
                    not isinstance(a, tuple) or len(a) != len(commitments)
                    for a in answers
                ):
                    raise Rejection(f"{label}: malformed first-layer answer")
                blocks = [
                    _check_layer(
                        self.code,
                        label,
                        self.code.d,
                        commitment.root,
                        [a[c] for a in answers],
                        positions,
                        size,
                    )
                    for c, commitment in enumerate(commitments)
                ]
                return _combination(blocks, alphas)

        run = _VerifierRun(self, label, point, value, top_check)
        await run.fold(iop, verifier, domain)
        queries = self.query_positions(verifier.challenge_bits(f"{label}/queries"))
        run.check(queries, await iop.transcript.read(f"{label}/answers"))


def _layer_answer(code, word: Word, tree: Merkle, j: int, size: int) -> tuple:
    """The answer to a query on leaf `j` of a committed codeword: the leaf --
    the fiber of `size` adjacent symbols -- and its authentication path."""
    return (code.fiber_at(word, j, size), tree.open(j))


def _same_point(points: list[list]) -> bool:
    """Whether every claim of a bundle is at the same point."""
    return all(
        all(a == b for a, b in zip(z, points[0], strict=True)) for z in points[1:]
    )


def _powers(x, n: int, one) -> list:
    """1, x, ..., x^(n-1)."""
    out = [one]
    for _ in range(1, n):
        out.append(out[-1] * x)
    return out


def _eq_at(a: list, b: list, one):
    """eq~(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i))."""
    out = one
    for x, y in zip(a, b, strict=True):
        out = out * (x * y + (one - x) * (one - y))
    return out


def _combination(words: list[Word], coefficients: list) -> Word:
    """``sum_c coefficients[c] * words[c]``, position by position, in the
    words' own form (a list of ring elements, or one vector)."""
    if isinstance(words[0], FieldVector):
        out = words[0].scale(coefficients[0])
        for word, coefficient in zip(words[1:], coefficients[1:], strict=True):
            out = out.fma(word, coefficient)
        return out
    out = [coefficients[0] * x for x in words[0]]
    for word, coefficient in zip(words[1:], coefficients[1:], strict=True):
        out = [y + coefficient * x for y, x in zip(out, word, strict=True)]
    return out


def _check_layer(
    code, label: str, level: int, root: bytes, answers, positions, size: int
) -> Word:
    """The leaves `answers` give for `positions` of the codeword at `level`
    committed under `root`, checked against it: each answer is a
    `_layer_answer`, and the result is the leaves joined into one word, in
    query order (`code.join_fibers`). Raises `Rejection` on a malformed leaf
    or a path that does not authenticate.

    The leaves are digested in one call; the paths are independent and are
    checked one at a time.
    """
    try:
        block: Word = code.join_fibers([answer[0] for answer in answers], size)
    except ValueError as error:
        raise Rejection(f"{label}: malformed leaf at level {level}: {error}") from error
    digests = code.leaf_digests(block, size)
    for k, (j, answer) in enumerate(zip(positions, answers, strict=True)):
        if not Merkle.verify(
            root, j, answer[1], digests[32 * k : 32 * (k + 1)], hash=_identity
        ):
            raise Rejection(f"{label}: Merkle path rejected at level {level}, leaf {j}")
    return block


class _ProverRun:
    """The prover's half of one basefold evaluation run: the claim
    ``sum_b f(b) * eq~(z, b) == v`` proved against `word`, the level-d
    codeword of f's coefficients under the protocol's code.

    The run starts from `word` and does not hold a tree for it: `top_answer`
    answers a query on leaf `j` of the first layer, in whatever form the
    caller's commitment to `word` takes. Every later layer's codeword is
    committed here and published by its root. `fold` writes the rounds and
    the base table; `answers` builds the reply to the query positions, once
    the caller has drawn them.
    """

    def __init__(
        self,
        protocol: BasefoldEval,
        label: str,
        f: MLE,
        zs: list,
        word: Word,
        top_answer: Callable[[int], Any],
    ) -> None:
        self.protocol = protocol
        self.label = label
        self.f = f
        self.zs = zs
        self.top_answer = top_answer
        #: The codeword each layer starts from.
        self.words: list[Word] = [word]
        #: The trees of the layers after the first, in order (the first
        #: layer's answers come from `top_answer`).
        self.trees: list[Merkle] = []

    def fold(self, iop) -> None:
        """The d interleaved rounds -- round message, challenge, the bind of
        both sumcheck tables and the fold of the codeword with that challenge,
        and the root of every codeword a layer starts from -- then the
        remaining kappa-variable table in the clear."""
        protocol, label = self.protocol, self.label
        scheme, code = protocol.scheme, protocol.code
        d = code.d
        schedule, levels = scheme.schedule, scheme.levels
        f = self.f
        word: Word = self.words[0]
        cur_f = f
        cur_eq = MLE.eq(_domain_of(f), self.zs, variables=f.variables)
        for s in range(d):
            var = cur_f.variables[0]
            evals = prod_round_evals([cur_f, cur_eq])
            iop.transcript.write(f"{label}/g{s}", evals)
            r = iop.verifier.challenge(f"{label}/r{s}")
            cur_f = cur_f.evaluate({var: r}, in_place=cur_f is not f)
            cur_eq = cur_eq.evaluate({var: r}, in_place=True)
            word = code.fold(word, r, level=d - s)
            # Level 0 is the one the verifier re-encodes itself.
            if d - s - 1 in levels[1:-1]:
                self.words.append(word)
                layer = len(self.words) - 1
                self.trees.append(scheme.merkle_commit(word, schedule[layer]))
                iop.transcript.write(f"{label}/pi{s + 1}", self.trees[-1].root)
        # The base case: the remaining kappa-variable table, in the clear
        # (the verifier re-encodes its coefficients itself).
        iop.transcript.write(f"{label}/h0", cur_f)

    def answers(self, queries) -> tuple:
        """For each query position, the answer of every layer along its walk
        (`BasefoldEval._walk`), first layer first."""
        protocol = self.protocol
        schedule, code = protocol.scheme.schedule, protocol.code
        return tuple(
            tuple(
                self.top_answer(j)
                if i == 0
                else _layer_answer(
                    code, self.words[i], self.trees[i - 1], j, 1 << schedule[i]
                )
                for i, (j, _) in enumerate(protocol._walk(q))
            )
            for q in queries
        )


class _VerifierRun:
    """The verifier's half of one basefold evaluation run: the claim
    ``sum_b f(b) * eq~(z, b) == value`` against a level-d codeword it cannot
    read.

    `top_check(answers, positions)` checks the first layer's answers to the
    queries, whose leaf indices are `positions`, against the caller's
    commitment to that codeword, and returns their leaves joined into one
    word (as `_check_layer` does for a single tree); it raises `Rejection`
    when they do not check. `fold` reads the rounds and the base table, and
    `check` the answers to the query positions the caller drew.
    """

    def __init__(
        self,
        protocol: BasefoldEval,
        label: str,
        zs: list,
        value,
        top_check: Callable[[list, list[int]], Word],
    ) -> None:
        self.protocol = protocol
        self.label = label
        self.zs = zs
        self.value = value
        self.top_check = top_check
        #: The roots of the layers after the first, in order.
        self.roots: list[bytes] = []
        #: The round challenges.
        self.rs: list = []
        #: The level-0 codeword, which the verifier computes itself.
        self.word0: Word = None

    async def fold(self, iop, verifier: Verifier, domain) -> None:
        """Reads the d rounds -- each round message checked against the
        running claim -- and the roots, then the base table, which must
        satisfy the final claim; computes the level-0 codeword from it."""
        protocol, label = self.protocol, self.label
        code = protocol.code
        d = code.d
        levels = protocol.scheme.levels
        # The root written after the fold that ends a layer is the next
        # layer's (label pi_k carries the codeword at level d - k).
        claim = self.value
        for s in range(d):
            evals = await iop.transcript.read(f"{label}/g{s}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {s}: g(0) + g(1) != claim")
            r = verifier.challenge(f"{label}/r{s}")
            self.rs.append(r)
            claim = interpolate_evals(evals, r)
            if d - s - 1 in levels[1:-1]:
                self.roots.append(await iop.transcript.read(f"{label}/pi{s + 1}"))

        h0 = await iop.transcript.read(f"{label}/h0")
        if not _committed_table(h0) or (1 << h0.num_vars) != code.k0:
            raise Rejection(f"{label}: malformed base table")

        # Base sumcheck check: sum_b h0(b) * eq~(z, (r, b)) == claim, with
        # eq~ factored into the bound-variable scalar and the tail table
        # (built over h0's own variables — the pairing with z is positional).
        one = _one(domain)
        eq_factor = one
        for z, r in zip(self.zs[:d], self.rs, strict=True):
            eq_factor = eq_factor * (z * r + (one - z) * (one - r))
        eq_tail = MLE.eq(domain, self.zs[d:], variables=h0.variables)
        base = Statement(
            Relation_SumProd(),
            oracles=[TableOracle(h0), TableOracle(eq_tail.scale(eq_factor))],
            value=claim,
        )
        if not base.check():
            raise Rejection(f"{label}: base table does not match the claim")

        # The level-0 codeword is computed, not received, so it is a codeword
        # by construction and needs no tree — it terminates every query walk.
        self.word0 = code.encode(_coefficients(h0))

    def check(self, queries, answers) -> None:
        """Checks the answers to `queries`: every layer's leaves against its
        commitment, and each query's walk -- the value refolded from one
        layer's leaf must reappear in the next layer's leaf, and at the end in
        the level-0 codeword."""
        protocol, label = self.protocol, self.label
        code = protocol.code
        d = code.d
        schedule, levels = protocol.scheme.schedule, protocol.scheme.levels
        if len(answers) != len(queries):
            raise Rejection(f"{label}: wrong number of query answers")
        walks = []
        for q, answer in zip(queries, answers, strict=True):
            steps = protocol._walk(q)
            if len(answer) != len(steps):
                raise Rejection(f"{label}: query {q} answered for {len(answer)} layers")
            walks.append(steps)

        # Every query visits the same layers in the same order, so step `i` is
        # one layer for all of them, and a layer's leaves are checked together:
        # one word, refolded by whole-word operations.
        carried = None  # per query, the value refolded from the layer above
        for i, folds in enumerate(schedule):
            level, size = levels[i], 1 << folds
            positions = [steps[i][0] for steps in walks]
            layer = [answer[i] for answer in answers]
            if i == 0:
                block = self.top_check(layer, positions)
            else:
                block = _check_layer(
                    code, label, level, self.roots[i - 1], layer, positions, size
                )
            # The value refolded from the layer above must reappear in this
            # layer's leaf, at the offset the walk gives.
            if carried is not None:
                inside = _gather(
                    block, [k * size + steps[i][1] for k, steps in enumerate(walks)]
                )
                if not (inside == carried):
                    raise Rejection(f"{label}: fold check failed at level {level}")
            first = d - level  # the index of this layer's first fold
            carried = code.fold_fibers(
                block, self.rs[first : first + folds], level, positions
            )
        # The last layer lands in the level-0 codeword the verifier built.
        if not (_gather(self.word0, [steps[-1][0] for steps in walks]) == carried):
            raise Rejection(f"{label}: fold check failed at level 0")
