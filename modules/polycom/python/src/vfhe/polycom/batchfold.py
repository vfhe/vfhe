# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BatchFold [CCCFGS26, §7]: evaluation claims on multilinear polynomials
over R_q, proved by basefold runs over the residue fields of R_q.

Let R_q = Z_q[X]/(X^N + 1), with q the product of L primes p_i, each of
which splits X^N + 1 into k = N/d irreducible factors of degree d
(d = `split_degree`). At the prime p_i, the CRT components
(`RNSRing.crt_components`) map an element of R_q to k elements of the field
F_i = `RNSRing.residue_field(i)`, which has p_i^d elements, and the
components at every prime together determine the element. The components
are ring homomorphisms, so for a multilinear polynomial f over R_q the claim
f(z) = y holds exactly when f_{i,s}(z_{i,s}) = y_{i,s} for every prime i and
every slot s < k: here f_{i,s} is the multilinear polynomial over F_i whose
table is component s of f's table, and z_{i,s}, y_{i,s} are component s of
the point's coordinates and of the value.

`BatchFoldEval` proves a bundle of M claims f_{c(m)}(z_m) = y_m on C
polynomials committed under one `Basefold` scheme over R_q. At each prime,
the k*M component claims on the k*C component polynomials are reduced, all
in F_i:

1. to claims at one common point r_i, by one sumcheck [CCCFGS26, Fig. 3],
   which is skipped when every component of every point is the same point
   (for example, points with integer coordinates);
2. to one claim u_i(r_i) = v_i on a random linear combination u_i of the
   k*C component polynomials [CCCFGS26, Fig. 4];
3. which one basefold run over the residue code at p_i
   (`FieldFoldableRS.residue_code`) proves.

The primes run in lockstep, as the sub-protocols of one AND-composition
(`IOP.and_composition`): in each round every prime sends its message, then
one challenge is drawn for all of them. It is an exceptional element of
R_q (`Ring.random_exceptional`), whose CRT components at every prime are all
one element with integer coefficients below the smallest prime, and every
prime reads it as that element of F_i. The sumcheck is skipped at every
prime or at none, so the primes' rounds stay aligned.

Points made of such elements -- what a sumcheck over R_q produces -- are one
point in every slot. Then the sumcheck is skipped when the claims share
their point, and otherwise its prover runs on one combined table per
polynomial, ``F_c = sum_s sigma^s f_(c,s)``, instead of k: with
``z_(m,s) = z_m``, ``h(b) = sum_m sigma^(m k) eq~(z_m, b) F_(c(m))(b)``,
so the round messages are the same.

The residue code's codewords are the CRT components of the ring code's
codewords. So u_i's codeword is the same linear combination of the
components of the committed ring codewords, and the ring commitments serve
as the commitment of every u_i. The L runs answer one set of query
positions, drawn after every run has folded: a query's first layer is
answered once, by the leaves of the committed ring codewords, which the
verifier authenticates against the ring commitments and maps to the leaves
of every u_i. Each later layer is committed once for all primes: one Merkle
tree whose leaf j joins every prime's leaf j (`vfhe.crypto.join_digests`),
built when every prime has folded to that layer, so a query answers it with
every prime's fiber and a single path.

[CCCFGS26, §7] instead commits to each of the k*L component polynomials
separately and runs each prime with its own query positions; see
`BatchFoldEval.soundness_error` for why sharing both costs no soundness.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from vfhe.arith import MLE, FieldVector, Polynomial, Ring, RNSPolynomial
from vfhe.crypto import Merkle, join_digests
from vfhe.piop import (
    Prover,
    Rejection,
    Statement,
    Verifier,
    exceptional_set_size,
)
from vfhe.piop.protocols.sumcheck.rounds import interpolate_evals

from .basefold import (
    Basefold,
    BasefoldEval,
    BasefoldOpening,
    _check_layer,
    _committed_table,
    _identity,
    _layer_answer,
    _ProverRun,
    _VerifierRun,
)
from .code import FoldableRS
from .field_code import FieldFoldableRS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vfhe.arith import Field


class BatchFoldEval(BasefoldEval):
    """BatchFold's Eval protocol: discharges `Relation_Eval` statements on
    oracles committed under a `Basefold` scheme over R_q (a `FoldableRS`
    code, with its twists in either field), by basefold runs over the
    residue fields of R_q (module docstring).

    Register with `iop.register(Relation_Eval, BatchFoldEval(scheme))` in
    place of `BasefoldEval`: the commitments are the same, so a polynomial
    committed once may be opened by either. As with `BasefoldEval`, the
    bundle is every pending claim on the oracles committed under the scheme,
    and a single claim is a bundle of one.

    `rep` is the number of query positions, shared by the runs at every
    prime.
    """

    supported_domains: tuple[type, ...] = (Ring,)

    def __init__(self, scheme: Basefold, rep: int = 32):
        super().__init__(scheme, rep)
        code = scheme.code
        if not isinstance(code, FoldableRS):
            raise TypeError("BatchFold runs on a basefold scheme over R_q")
        ring = code.ring
        self.ring = ring
        #: k = N / split_degree, the number of CRT components at each prime.
        self.slots = ring.N // ring.split_degree
        #: Per prime p_i, the field F_i the components at p_i lie in.
        self.fields: list[Field] = [ring.residue_field(i) for i in range(ring.ell)]
        #: Per prime p_i, basefold over the residue code at p_i with this
        #: scheme's schedule. These run the per-prime folds and are never
        #: registered with an IOP.
        self.residue_protocols = [
            BasefoldEval(
                Basefold(FieldFoldableRS.residue_code(code, i), scheme.schedule),
                rep,
            )
            for i in range(ring.ell)
        ]

    def soundness_error(
        self,
        statement: Statement,
        domain,
        gamma: float,
        delta: float,
        claims: int = 1,
        polynomials: int = 1,
    ) -> float | None:
        """The information-theoretic soundness error for a bundle of `claims`
        evaluation claims on `polynomials` distinct polynomials:
        ``2d / (gamma^3 |F|) + (2l + k claims - 1) / |F|
        + (k polynomials - 1) / |F| + (1 - delta + gamma d)^rep``,
        one prime's error: the common-point sumcheck [CCCFGS26, §7.1], the
        random linear combination [CCCFGS26, §7.2, Lemma 5] and basefold over
        a residue code [ZCF24]. Here |F| = min_i p_i^d is the size of the
        smallest residue field, which is also the size of the set the shared
        challenges come from, l the number of variables, k the number of
        components at a prime, and d the code's depth (not the split degree).
        `delta` is the smallest of the residue codes' relative distances
        (``min(p.code.relative_distance() for p in residue_protocols)``),
        and `gamma` is the caller's, as in `BasefoldEval.soundness_error`.
        None if a field's size is unknown.

        Why one prime's error, not L times it. The ring commitment fixes the
        committed codeword before the protocol, and the statement fixes the
        claims, so whether the components at prime i are codewords whose
        polynomials satisfy the component claims is decided, for every i,
        before any challenge or query is drawn. When the bundle is false,
        some prime i fails in this way, and the verifier accepts only if
        that prime's checks pass, which happens with probability at most
        that prime's error. Each prime's error is bounded by the expression
        above: its challenges come after the messages they answer and are
        uniform over a set of size |F| (they are one ring element's
        components, so the primes' challenges are correlated, which no
        prime's argument looks at), and its query positions are uniform and
        drawn after every oracle they check is committed. The same holds
        round by round, which Fiat-Shamir needs: the composition is doomed
        when that prime's run is ([BCFW25] proves it for parallel copies of
        an interactive oracle reduction). The ring sumcheck over R_q, which
        through the CRT is the per-prime sumchecks on one shared challenge,
        likewise has no factor for the number of primes [CCCFGS25, Thm. 2.6].
        [CCCFGS26, §7] states its bound with a factor L + 1, for its L + 1
        primes run independently.

        The paper commits to the k*L component polynomials separately and
        opens each prime with its own query positions and challenges; here
        the ring commitments serve every prime and the primes share both, as
        well as one tree per later layer. A ring leaf holds every component
        at its position, and a later layer's leaf every prime's fiber at its
        position, which includes what the paper's verifier reads there; and
        every layer's tree is committed before the queries are drawn.

        Two terms rest on assumptions. The fold term 2d / (gamma^3 |F|) is
        [ZCF24]'s for fold challenges uniform over the whole field; here
        they are uniform over a subset of size |F|, which keeps the bound if
        it comes from counting the challenges that fold badly (the ring
        `BasefoldEval` makes the same assumption). And as in
        `BasefoldEval.soundness_error`, the bound is stated for the default
        schedule, and the hash's collision term is not included.
        """
        sizes = [exceptional_set_size(field) for field in self.fields]
        if any(size is None for size in sizes):
            return None
        size = min(s for s in sizes if s is not None)
        d, k = self.code.d, self.slots
        num_vars = self.code.k_d.bit_length() - 1
        return (
            2 * d / (gamma**3 * size)
            + (2 * num_vars + k * claims - 1) / size
            + (k * polynomials - 1) / size
            + (1 - delta + gamma * d) ** self.rep
        )

    def _components(self, values: list, index: int) -> FieldVector:
        """The CRT components at prime `index` of `values` (ring elements or
        non-negative integers), entry ``b * k + s`` being component s of
        ``values[b]``."""
        elements = [
            x if isinstance(x, RNSPolynomial) else Polynomial(self.ring).from_array([x])
            for x in values
        ]
        return self.ring.crt_components(elements, index, element_major=True)

    def _points(self, points: list[list]) -> tuple[list[FieldVector], bool]:
        """Per prime, the components of the claims' points, entry
        ``(m * l + t) * k + s`` being component s of coordinate t of z_m;
        and whether every component of every point is one point at every
        prime, in which case no prime runs the common-point sumcheck."""
        flat = [x for point in points for x in point]
        zs = [self._components(flat, i) for i in range(self.ring.ell)]
        num_vars = len(points[0])
        return zs, all(_one_point(z, num_vars, self.slots) for z in zs)

    def _challenge_at_prime(self, seed: bytes, field) -> Any:
        """One round's challenge in `field`, the residue field of a prime:
        the element of `field` whose coefficients are the integers below the
        smallest prime of R_q that `seed` determines
        (`Field.random_exceptional` with that bound). Every prime derives the
        same integers from the seed, so the primes' challenges are the CRT
        components, the same in every slot, of one exceptional element of
        R_q."""
        return field.random_exceptional(seed=seed, bound=min(self.ring.primes))

    def _layer_committer(
        self, iop, shared: str, index: int, trees: dict[int, Merkle]
    ) -> Callable[[int, int, Any], Awaitable[None]]:
        """How prime `index`'s run commits a later layer: by one tree for that
        layer across every prime, whose leaf j joins the primes' leaves j
        (`_joint_tree`). The tree is built once every prime has folded to the
        layer (`IOP.batched_call`) and kept in ``trees[layer]``; its root is
        written once, as ``{shared}/pi{step}``, by prime 0, which the
        composition's turn order runs first at every step."""

        async def commit_layer(layer: int, step: int, word) -> None:
            name = f"{shared}/pi{step}"
            size = 1 << self.scheme.schedule[layer]
            tree = await iop.batched_call(
                name, word, lambda words: self._joint_tree(words, size)
            )
            trees[layer] = tree
            if index == 0:
                iop.transcript.write(name, tree.root)

        return commit_layer

    def _joint_tree(self, words: list, size: int) -> list[Merkle]:
        """The tree over the primes' codewords `words` (one per prime, of one
        length) whose leaf j joins every prime's fiber j of `size` symbols:
        its digest hashes the primes' leaf digests j together. Returned once
        per prime, as `IOP.batched_call` hands out results."""
        digests = [
            protocol.code.leaf_digests(word, size)
            for protocol, word in zip(self.residue_protocols, words, strict=True)
        ]
        tree = Merkle.from_digests(join_digests(digests))
        return [tree] * len(words)

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        for statement in statements:
            await statement.resolved()
        commitments, owners, points = self._bundle(statements)
        openings = [self._opening(prover, c) for c in commitments]
        for opening, commitment in zip(openings, commitments, strict=True):
            f = opening.polynomial
            if not _committed_table(f) or f.variables != commitment.variables:
                raise TypeError("the opening does not match the commitment")
        iop = prover.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        label = f"batchfold{statements[0].path}"
        zs, skip = self._points(points)
        # The tree of each layer after the first, shared by every prime.
        trees: dict[int, Merkle] = {}
        runs = await iop.and_composition(
            label,
            [
                self._prove_prime(
                    iop,
                    f"{label}/p{i}",
                    i,
                    openings,
                    owners,
                    zs[i],
                    skip,
                    self._layer_committer(iop, label, i, trees),
                )
                for i in range(self.ring.ell)
            ],
            challenge_sampler=self._challenge_at_prime,
        )
        queries = self.query_positions(
            await iop.verifier.challenge_bits(f"{label}/queries")
        )
        # Per query: the ring leaves, which answer the first layer for every
        # prime; then, for each later layer, every prime's fiber and the one
        # path of the shared tree.
        schedule = self.scheme.schedule
        size = 1 << schedule[0]
        answers = tuple(
            (
                tuple(
                    _layer_answer(self.code, opening.word, opening.tree, q, size)
                    for opening in openings
                ),
                tuple(
                    (
                        tuple(
                            protocol.code.fiber_at(run.words[i], j, 1 << schedule[i])
                            for protocol, run in zip(
                                self.residue_protocols, runs, strict=True
                            )
                        ),
                        trees[i].open(j),
                    )
                    for i, (j, _) in enumerate(self._walk(q))
                    if i > 0
                ),
            )
            for q in queries
        )
        iop.transcript.write(f"{label}/answers", answers)
        return []

    async def _prove_prime(
        self,
        iop,
        label: str,
        index: int,
        openings: list[BasefoldOpening],
        owners: list[int],
        z: FieldVector,
        skip: bool,
        commit_layer: Callable[[int, int, Any], Awaitable[None]],
    ) -> _ProverRun:
        """The prover's sub-protocol at prime `index`: the reduction (steps 1-2 of
        the module docstring), skipping the sumcheck when `skip`, then the
        folds of the basefold run on u_i, whose query answers are left to
        the caller. `z` holds the points' components at this prime
        (`_points`), and `commit_layer` commits the run's later layers
        (`_layer_committer`)."""
        field, k = self.fields[index], self.slots
        claims = len(owners)
        num_vars = len(z) // (claims * k)
        ring = self.ring

        def component_tables() -> FieldVector:
            # Every polynomial's element-major component table, side by side:
            # entry (c * 2^l + b) * k + s is f_{c,s}(b).
            return FieldVector.concat(
                [
                    ring.crt_components(o.polynomial.table, index, element_major=True)
                    for o in openings
                ]
            )

        if skip:
            r = [z[t * k] for t in range(num_vars)]
        else:
            # Step 1: with sigma the verifier's challenge, the sumcheck proves
            # sum_b h(b) = sum_{m,s} sigma^(m k + s) y_{m,s} for
            # h(b) = sum_{m,s} sigma^(m k + s) eq~(z_{m,s}, b) f_{c(m),s}(b).
            sigma = await iop.verifier.challenge(f"{label}/point/s", field)
            powers = _power_vector(field, sigma, claims * k)
            if _constant_in_slots(z, k):
                # Every point is one point in every slot, z_{m,s} = z_m, so
                # h(b) = sum_m sigma^(m k) eq~(z_m, b) F_{c(m)}(b) with
                # F_c = sum_s sigma^s f_{c,s}: the same round messages, from
                # one table per polynomial instead of k.
                table = FieldVector.concat(
                    [
                        ring.crt_linear_combination(
                            [o.polynomial.table], index, _window(powers, 0, k)
                        )
                        for o in openings
                    ]
                )
                weight = FieldVector.concat(
                    _slot_free_weights(z, owners, powers, len(openings), num_vars, k)
                )
                block = 1
            else:
                table = component_tables()
                weight = FieldVector.concat(
                    _slot_weights(z, owners, powers, len(openings), num_vars, k)
                )
                block = k
            # Every polynomial's table and weights side by side: a round binds
            # bit 0 of b in all of them, and in this layout the entries that
            # bit pairs are `block` apart.
            r = []
            for t in range(num_vars):
                t_lo, t_hi = table.split_even_odd(block)
                w_lo, w_hi = weight.split_even_odd(block)
                evals = (
                    (t_lo * w_lo).sum(),
                    (t_hi * w_hi).sum(),
                    ((t_hi * 2 - t_lo) * (w_hi * 2 - w_lo)).sum(),
                )
                iop.transcript.write(f"{label}/point/g{t}", evals)
                x = await iop.verifier.challenge(f"{label}/point/r{t}", field)
                r.append(x)
                table = t_lo.fma(t_hi - t_lo, x)
                weight = w_lo.fma(w_hi - w_lo, x)
            if block == 1:
                # The rounds bound the combinations F_c; the values at r of
                # the component polynomials themselves come from their tables,
                # bound to r the same way.
                table = component_tables()
                for x in r:
                    t_lo, t_hi = table.split_even_odd(k)
                    table = t_lo.fma(t_hi - t_lo, x)
            # Entry c * k + s is f_{c,s}(r).
            iop.transcript.write(f"{label}/point/q", table)
        # Step 2: u = sum_{c,s} alpha^(c k + s) f_{c,s}, on the same
        # combination of the components of the committed codewords, each
        # formed in one call without the components themselves.
        alphas = _power_vector(
            field,
            await iop.verifier.challenge(f"{label}/alpha", field),
            len(openings) * k,
        )
        u = MLE(
            field=field,
            variables=list(openings[0].polynomial.variables),
            evaluations=ring.crt_linear_combination(
                [o.polynomial.table for o in openings], index, alphas
            ),
        )
        word = ring.crt_linear_combination([o.word for o in openings], index, alphas)
        run = _ProverRun(
            self.residue_protocols[index],
            label,
            u,
            r,
            word,
            _no_answer,
            commit_layer=commit_layer,
        )
        await run.fold(iop)
        return run

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        for statement in statements:
            await statement.resolved()
        commitments, owners, points = self._bundle(statements)
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        label = f"batchfold{statements[0].path}"
        values = [statement.value for statement in statements]
        count = len(commitments)
        zs, skip = self._points(points)
        # The first-layer leaves of u_i at the query positions, computed from
        # the ring leaves once those are authenticated.
        tops: list[Any] = [None] * self.ring.ell
        # (layer, prime) -> the prime's leaves of that later layer at the
        # query positions, once the shared trees authenticate them.
        later: dict[tuple[int, int], Any] = {}
        results = await iop.and_composition(
            label,
            [
                self._verify_prime(
                    iop,
                    verifier,
                    f"{label}/p{i}",
                    i,
                    count,
                    owners,
                    zs[i],
                    skip,
                    values,
                    tops,
                    later,
                    label,
                )
                for i in range(self.ring.ell)
            ],
            challenge_sampler=self._challenge_at_prime,
        )
        runs = [run for run, _ in results]
        alphas = [alpha for _, alpha in results]
        queries = self.query_positions(
            await verifier.challenge_bits(f"{label}/queries")
        )
        answers = await iop.transcript.read(f"{label}/answers")
        schedule = self.scheme.schedule
        if (
            not isinstance(answers, tuple)
            or len(answers) != len(queries)
            or any(
                not _answer_shape(a, count, self.ring.ell, len(schedule) - 1)
                for a in answers
            )
        ):
            raise Rejection(f"{label}: malformed answers")
        size = 1 << schedule[0]
        blocks = [
            _check_layer(
                self.code,
                label,
                self.code.d,
                commitment.root,
                [a[0][c] for a in answers],
                queries,
                size,
            )
            for c, commitment in enumerate(commitments)
        ]
        walks = [self._walk(q) for q in queries]
        codes = [protocol.code for protocol in self.residue_protocols]
        for layer in range(1, len(schedule)):
            leaves = _check_joint_layer(
                codes,
                label,
                self.scheme.levels[layer],
                runs[0].roots[layer - 1],
                [a[1][layer - 1] for a in answers],
                [walk[layer][0] for walk in walks],
                1 << schedule[layer],
            )
            for prime, block in enumerate(leaves):
                later[layer, prime] = block
        # Every layer is checked through `tops` and `later`, so the runs see
        # no answers of their own.
        placeholders = tuple((None,) * len(schedule) for _ in answers)
        for i, run in enumerate(runs):
            tops[i] = self.ring.crt_linear_combination(blocks, i, alphas[i])
            run.check(queries, placeholders)
        return []

    async def _verify_prime(
        self,
        iop,
        verifier: Verifier,
        label: str,
        index: int,
        count: int,
        owners: list[int],
        z: FieldVector,
        skip: bool,
        values: list,
        tops: list,
        later: dict,
        shared: str,
    ) -> tuple[_VerifierRun, FieldVector]:
        """The verifier's sub-protocol at prime `index`, for claims on `count`
        polynomials: the reduction (steps 1-2 of the module docstring),
        skipping the sumcheck when `skip`, then the folds of the basefold
        run on u_i. Returns the run, whose query checks are left to the
        caller and read u_i's first-layer leaves from ``tops[index]`` and its
        later layers' from ``later[layer, index]``, and the combination's
        coefficients. The later layers' roots are the shared trees', under
        `shared`."""
        field, k = self.fields[index], self.slots
        claims = len(owners)
        num_vars = len(z) // (claims * k)
        one = field.one
        # Entry m * k + s is component s of y_m.
        y = self._components(values, index)
        if skip:
            r = [z[t * k] for t in range(num_vars)]
            blocks: list[Any] = [None] * count
            for m, c in enumerate(owners):
                block = _window(y, m * k, k)
                if blocks[c] is None:
                    blocks[c] = block
                elif block != blocks[c]:
                    raise Rejection(
                        f"{label}: two values claimed for one polynomial at one point"
                    )
            q = FieldVector.concat(blocks)
        else:
            sigma = await verifier.challenge(f"{label}/point/s", field)
            powers = _power_vector(field, sigma, claims * k)
            claim = (powers * y).sum()
            r = []
            for t in range(num_vars):
                evals = await iop.transcript.read(f"{label}/point/g{t}")
                if not isinstance(evals, tuple) or len(evals) != 3:
                    raise Rejection(f"{label} point round {t}: malformed message")
                if not (evals[0] + evals[1] == claim):
                    raise Rejection(f"{label} point round {t}: g(0) + g(1) != claim")
                x = await verifier.challenge(f"{label}/point/r{t}", field)
                r.append(x)
                claim = interpolate_evals(evals, x)
            q = await iop.transcript.read(f"{label}/point/q")
            if (
                not isinstance(q, FieldVector)
                or q.field is not field
                or len(q) != count * k
            ):
                raise Rejection(f"{label}: malformed values at the common point")
            # h(r) = sum_{m,s} sigma^(m k + s) eq~(z_{m,s}, r) f_{c(m),s}(r),
            # each eq~ factor z r + (1 - z)(1 - r) taken over the k slots.
            expected = None
            for m, c in enumerate(owners):
                eq = None
                for t, x in enumerate(r):
                    zt = _window(z, (m * num_vars + t) * k, k)
                    factor = zt * (x + x - one) + (one - x)
                    eq = factor if eq is None else eq * factor
                term = (_window(powers, m * k, k) * eq * _window(q, c * k, k)).sum()
                expected = term if expected is None else expected + term
            if not (expected == claim):
                raise Rejection(f"{label}: the common-point values do not match")
        alphas = _power_vector(
            field, await verifier.challenge(f"{label}/alpha", field), count * k
        )
        run = _VerifierRun(
            self.residue_protocols[index],
            label,
            r,
            (alphas * q).sum(),
            lambda _answers, _positions: tops[index],
            read_root=lambda step: iop.transcript.read(f"{shared}/pi{step}"),
            check_layer=lambda layer, _answers, _positions: later[layer, index],
        )
        await run.fold(iop, verifier, field)
        return run, alphas


def _no_answer(_leaf: int) -> None:
    """A per-prime run's first-layer answer: none, since the ring leaves
    answer that layer for every prime."""
    return None


def _answer_shape(answer, commitments: int, primes: int, layers: int) -> bool:
    """Whether one query's answer is ``(ring leaves, later layers)``: a tuple
    of one entry per commitment, and a tuple of one ``(fibers, path)`` per
    later layer, with one fiber per prime."""
    return (
        isinstance(answer, tuple)
        and len(answer) == 2
        and isinstance(answer[0], tuple)
        and len(answer[0]) == commitments
        and isinstance(answer[1], tuple)
        and len(answer[1]) == layers
        and all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], tuple)
            and len(entry[0]) == primes
            for entry in answer[1]
        )
    )


def _check_joint_layer(
    codes: list, label: str, level: int, root: bytes, answers, positions, size: int
) -> list:
    """Every prime's leaves of a later layer at the query positions, checked
    against the layer's shared tree: each answer is ``(fibers, path)``, one
    fiber per prime, and the leaf the path authenticates joins the primes'
    leaf digests (`vfhe.crypto.join_digests`). Returns, per prime, its
    fibers joined into one word in query order, as `_check_layer` does for a
    tree of one codeword. Raises `Rejection` on a malformed fiber or a path
    that does not authenticate."""
    blocks = []
    prime = 0
    try:
        for prime, code in enumerate(codes):
            fibers = [answer[0][prime] for answer in answers]
            blocks.append(code.join_fibers(fibers, size))
    except ValueError as error:
        raise Rejection(
            f"{label}: malformed leaf at level {level}, prime {prime}: {error}"
        ) from error
    digests = join_digests(
        [
            code.leaf_digests(block, size)
            for code, block in zip(codes, blocks, strict=True)
        ]
    )
    for k, (j, answer) in enumerate(zip(positions, answers, strict=True)):
        if not Merkle.verify(
            root, j, answer[1], digests[32 * k : 32 * (k + 1)], hash=_identity
        ):
            raise Rejection(f"{label}: Merkle path rejected at level {level}, leaf {j}")
    return blocks


def _window(vector: FieldVector, start: int, length: int) -> FieldVector:
    """Entries ``start`` to ``start + length - 1`` of `vector`, copied."""
    return vector.query(array("Q", range(start, start + length)))


def _tile(vector: FieldVector, n: int) -> FieldVector:
    """`vector` repeated to length `n`, a multiple of its length."""
    out = vector
    while 2 * len(out) <= n:
        out = FieldVector.concat([out, out])
    if len(out) < n:
        out = FieldVector.concat([out, _window(out, 0, n - len(out))])
    return out


def _power_vector(field: Field, x, n: int) -> FieldVector:
    """The vector 1, x, ..., x^(n-1) over `field`."""
    out = FieldVector(field, [field.one])
    step = x
    while len(out) < n:
        out = FieldVector.concat([out, out.scale(step)])
        step = step * step
    return out if len(out) == n else _window(out, 0, n)


def _slot_weights(
    z: FieldVector,
    owners: list[int],
    powers: FieldVector,
    count: int,
    num_vars: int,
    k: int,
) -> list[FieldVector]:
    """Per polynomial c, laid out as its element-major component table: entry
    ``b * k + s`` is the sum over the claims m on f_c of
    ``sigma^(m k + s) eq~(z_{m,s}, b)``, `powers` holding the sigma powers and
    `z` the points' components (`_points`)."""
    weights: list[Any] = [None] * count
    for m, c in enumerate(owners):
        # eq~(z, b) is the product over t of z_t if bit t of b is 1 and of
        # 1 - z_t if it is 0, so each coordinate doubles the table, bit t
        # becoming the highest bit of b so far.
        weight = _window(powers, m * k, k)
        for t in range(num_vars):
            zt = _tile(_window(z, (m * num_vars + t) * k, k), len(weight))
            high = weight * zt
            weight = FieldVector.concat([weight - high, high])
        weights[c] = weight if weights[c] is None else weights[c] + weight
    return weights


def _slot_free_weights(
    z: FieldVector,
    owners: list[int],
    powers: FieldVector,
    count: int,
    num_vars: int,
    k: int,
) -> list[FieldVector]:
    """`_slot_weights` for points that are one point in every slot
    (`_constant_in_slots`): per polynomial c, entry b is the sum over the
    claims m on f_c of ``sigma^(m k) eq~(z_m, b)``."""
    weights: list[Any] = [None] * count
    for m, c in enumerate(owners):
        weight = FieldVector(z.field, [powers[m * k]])
        for t in range(num_vars):
            high = weight.scale(z[(m * num_vars + t) * k])
            weight = FieldVector.concat([weight - high, high])
        weights[c] = weight if weights[c] is None else weights[c] + weight
    return weights


def _constant_in_slots(z: FieldVector, k: int) -> bool:
    """Whether every coordinate whose components `z` holds (entry
    ``i * k + s`` = component s of coordinate i) has one component in every
    slot."""
    heads = z.query(array("Q", range(0, len(z), k)))
    # Each head repeated k times: interleaving a vector with itself doubles
    # every entry in place, and k is a power of two.
    repeated = k
    while repeated > 1:
        heads = FieldVector.interleave(heads, heads)
        repeated //= 2
    return heads == z


def _one_point(z: FieldVector, num_vars: int, k: int) -> bool:
    """Whether the points whose components are `z` (entry
    ``(m * num_vars + t) * k + s`` = component s of coordinate t of point
    m) are all one point in every component."""
    first = [z[t * k] for t in range(num_vars) for _ in range(k)]
    return _tile(FieldVector(z.field, first), len(z)) == z
