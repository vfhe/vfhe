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
one after the other, the k*M component claims on the k*C component
polynomials are reduced, all in F_i:

1. to claims at one common point r_i, by one sumcheck [CCCFGS26, Fig. 3],
   which is skipped when every component of every point is the same point
   (for example, points with integer coordinates);
2. to one claim u_i(r_i) = v_i on a random linear combination u_i of the
   k*C component polynomials [CCCFGS26, Fig. 4];
3. which one basefold run over the residue code at p_i
   (`FieldFoldableRS.residue_code`) proves.

The residue code's codewords are the CRT components of the ring code's
codewords. So u_i's codeword is the same linear combination of the
components of the committed ring codewords, and the ring commitments serve
as the commitment of every u_i. The L runs answer one set of query
positions, drawn after every run has folded: a query's first layer is
answered once, by the leaves of the committed ring codewords, which the
verifier authenticates against the ring commitments and maps to the leaves
of every u_i. The later layers are committed by each run, over F_i.

[CCCFGS26, §7] instead commits to each of the k*L component polynomials
separately; see `BatchFoldEval.soundness_error`.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from vfhe.arith import MLE, FieldVector, Polynomial, Ring, RNSPolynomial
from vfhe.piop import Prover, Rejection, Statement, Verifier, exceptional_set_size
from vfhe.piop.protocols.sumcheck.rounds import interpolate_evals

from .basefold import (
    Basefold,
    BasefoldEval,
    BasefoldOpening,
    _check_layer,
    _committed_table,
    _layer_answer,
    _ProverRun,
    _VerifierRun,
)
from .code import FoldableRS
from .field_code import FieldFoldableRS

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        """[CCCFGS26, §7]'s bound for a bundle of `claims` evaluation claims
        on `polynomials` distinct polynomials:
        ``L * (2d / (gamma^3 |F|) + (2l + k claims - 1) / |F|
        + (k polynomials - 1) / |F| + (1 - delta + gamma d)^rep)``,
        a union bound over the L primes of the common-point sumcheck, the
        random linear combination and basefold over a residue code. Here
        |F| is the size of the smallest residue field F_i, l the number of
        variables, k the number of components at a prime, and d the code's
        depth (not the split degree). `delta` is the residue codes' relative distance (each
        `residue_protocols[i].code.relative_distance()` reports its own),
        and `gamma` is the caller's, as in `BasefoldEval.soundness_error`.
        None if a field's size is unknown.

        The cited bound is for k*L separately committed component
        polynomials, each opened with its own query positions. Here the ring
        commitments serve every prime and the L runs share their query
        positions; the same expression is returned, and that case is not
        covered by the cited argument. As in `BasefoldEval.soundness_error`,
        the hash's collision term is not included.
        """
        sizes = [exceptional_set_size(field) for field in self.fields]
        if any(size is None for size in sizes):
            return None
        size = min(s for s in sizes if s is not None)
        d, k = self.code.d, self.slots
        num_vars = self.code.k_d.bit_length() - 1
        per_prime = (
            2 * d / (gamma**3 * size)
            + (2 * num_vars + k * claims - 1) / size
            + (k * polynomials - 1) / size
            + (1 - delta + gamma * d) ** self.rep
        )
        return len(self.fields) * per_prime

    def _components(self, values: list, index: int) -> FieldVector:
        """The CRT components at prime `index` of `values` (ring elements or
        non-negative integers), entry ``b * k + s`` being component s of
        ``values[b]``."""
        elements = [
            x if isinstance(x, RNSPolynomial) else Polynomial(self.ring).from_array([x])
            for x in values
        ]
        return self.ring.crt_components(elements, index, element_major=True)

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
        runs = [
            self._prove_prime(iop, f"{label}/p{i}", i, openings, owners, points)
            for i in range(self.ring.ell)
        ]
        queries = self.query_positions(iop.verifier.challenge_bits(f"{label}/queries"))
        # Each run answers its first layer with None: the ring leaves below
        # answer it for every prime.
        layers = [run.answers(queries) for run in runs]
        size = 1 << self.scheme.schedule[0]
        answers = tuple(
            (
                tuple(
                    _layer_answer(self.code, opening.word, opening.tree, q, size)
                    for opening in openings
                ),
                tuple(prime[n][1:] for prime in layers),
            )
            for n, q in enumerate(queries)
        )
        iop.transcript.write(f"{label}/answers", answers)
        return []

    def _prove_prime(
        self,
        iop,
        label: str,
        index: int,
        openings: list[BasefoldOpening],
        owners: list[int],
        points: list[list],
    ) -> _ProverRun:
        """The prover's reduction at prime `index` (steps 1-2 of the module
        docstring), then the folds of the basefold run on u_i, whose query
        answers are left to the caller."""
        field, k = self.fields[index], self.slots
        claims, num_vars = len(points), len(points[0])
        ring = self.ring
        # Entry b * k + s of tables[c] is component s of entry b of f_c's
        # table; entry (m * l + t) * k + s of z is component s of coordinate
        # t of z_m.
        tables = [
            ring.crt_components(o.polynomial.table, index, element_major=True)
            for o in openings
        ]
        z = self._components([x for point in points for x in point], index)
        if _one_point(z, num_vars, k):
            r = [z[t * k] for t in range(num_vars)]
        else:
            # Step 1: with sigma the verifier's challenge, the sumcheck proves
            # sum_b h(b) = sum_{m,s} sigma^(m k + s) y_{m,s} for
            # h(b) = sum_{m,s} sigma^(m k + s) eq~(z_{m,s}, b) f_{c(m),s}(b).
            # weights[c] is laid out as tables[c], with entry b * k + s the sum
            # over the claims m on f_c of sigma^(m k + s) eq~(z_{m,s}, b).
            sigma = iop.verifier.challenge(f"{label}/point/s", field)
            powers = _power_vector(field, sigma, claims * k)
            weights: list[Any] = [None] * len(tables)
            for m, c in enumerate(owners):
                # eq~(z, b) is the product over t of z_t if bit t of b is 1
                # and of 1 - z_t if it is 0, so each coordinate doubles the
                # table, bit t becoming the highest bit of b so far.
                weight = _window(powers, m * k, k)
                for t in range(num_vars):
                    zt = _tile(_window(z, (m * num_vars + t) * k, k), len(weight))
                    high = weight * zt
                    weight = FieldVector.concat([weight - high, high])
                weights[c] = weight if weights[c] is None else weights[c] + weight
            # Every polynomial's table and weights side by side: a round binds
            # bit 0 of b in all of them, and in this layout the entries that
            # bit pairs are k apart.
            table = FieldVector.concat(tables)
            weight = FieldVector.concat(weights)
            r = []
            for t in range(num_vars):
                t_lo, t_hi = table.split_even_odd(k)
                w_lo, w_hi = weight.split_even_odd(k)
                evals = (
                    (t_lo * w_lo).sum(),
                    (t_hi * w_hi).sum(),
                    ((t_hi * 2 - t_lo) * (w_hi * 2 - w_lo)).sum(),
                )
                iop.transcript.write(f"{label}/point/g{t}", evals)
                x = iop.verifier.challenge(f"{label}/point/r{t}", field)
                r.append(x)
                table = t_lo.fma(t_hi - t_lo, x)
                weight = w_lo.fma(w_hi - w_lo, x)
            # Entry c * k + s is f_{c,s}(r).
            iop.transcript.write(f"{label}/point/q", table)
        # Step 2: u = sum_{c,s} alpha^(c k + s) f_{c,s}, on the same
        # combination of the components of the committed codewords.
        alphas = _power_vector(
            field, iop.verifier.challenge(f"{label}/alpha", field), len(tables) * k
        )
        u = MLE(
            field=field,
            variables=list(openings[0].polynomial.variables),
            evaluations=_combine(tables, alphas, k),
        )
        word = _combine(
            [ring.crt_components(o.word, index, element_major=True) for o in openings],
            alphas,
            k,
        )
        run = _ProverRun(self.residue_protocols[index], label, u, r, word, _no_answer)
        run.fold(iop)
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
        count, k = len(commitments), self.slots
        runs, alphas = [], []
        # The first-layer leaves of u_i at the query positions, computed from
        # the ring leaves once those are authenticated.
        tops: list[Any] = [None] * self.ring.ell
        for i in range(self.ring.ell):
            r, alpha, value = await self._verify_prime(
                iop, verifier, f"{label}/p{i}", i, count, owners, points, values
            )
            run = _VerifierRun(
                self.residue_protocols[i],
                f"{label}/p{i}",
                r,
                value,
                lambda _answers, _positions, i=i: tops[i],
            )
            await run.fold(iop, verifier, self.fields[i])
            runs.append(run)
            alphas.append(alpha)
        queries = self.query_positions(verifier.challenge_bits(f"{label}/queries"))
        answers = await iop.transcript.read(f"{label}/answers")
        if (
            not isinstance(answers, tuple)
            or len(answers) != len(queries)
            or any(not _answer_shape(a, count, self.ring.ell) for a in answers)
        ):
            raise Rejection(f"{label}: malformed answers")
        size = 1 << self.scheme.schedule[0]
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
        for i, run in enumerate(runs):
            tops[i] = _combine(
                [self.ring.crt_components(b, i, element_major=True) for b in blocks],
                alphas[i],
                k,
            )
            run.check(queries, tuple((None, *a[1][i]) for a in answers))
        return []

    async def _verify_prime(
        self,
        iop,
        verifier: Verifier,
        label: str,
        index: int,
        count: int,
        owners: list[int],
        points: list[list],
        values: list,
    ) -> tuple[list, FieldVector, Any]:
        """The verifier's reduction at prime `index` (steps 1-2 of the module
        docstring) for claims on `count` polynomials: returns the common
        point r_i, the combination's coefficients and the value v_i that the
        basefold run on u_i must prove."""
        field, k = self.fields[index], self.slots
        claims, num_vars = len(points), len(points[0])
        one = field.one
        z = self._components([x for point in points for x in point], index)
        # Entry m * k + s is component s of y_m.
        y = self._components(values, index)
        if _one_point(z, num_vars, k):
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
            sigma = verifier.challenge(f"{label}/point/s", field)
            powers = _power_vector(field, sigma, claims * k)
            claim = (powers * y).sum()
            r = []
            for t in range(num_vars):
                evals = await iop.transcript.read(f"{label}/point/g{t}")
                if not isinstance(evals, tuple) or len(evals) != 3:
                    raise Rejection(f"{label} point round {t}: malformed message")
                if not (evals[0] + evals[1] == claim):
                    raise Rejection(f"{label} point round {t}: g(0) + g(1) != claim")
                x = verifier.challenge(f"{label}/point/r{t}", field)
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
            field, verifier.challenge(f"{label}/alpha", field), count * k
        )
        return r, alphas, (alphas * q).sum()


def _no_answer(_leaf: int) -> None:
    """A per-prime run's first-layer answer: none, since the ring leaves
    answer that layer for every prime."""
    return None


def _answer_shape(answer, commitments: int, primes: int) -> bool:
    """Whether one query's answer is ``(ring leaves, per-prime layers)``:
    a tuple of one entry per commitment, and a tuple of one tuple per
    prime."""
    return (
        isinstance(answer, tuple)
        and len(answer) == 2
        and isinstance(answer[0], tuple)
        and len(answer[0]) == commitments
        and isinstance(answer[1], tuple)
        and len(answer[1]) == primes
        and all(isinstance(layers, tuple) for layers in answer[1])
    )


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


def _one_point(z: FieldVector, num_vars: int, k: int) -> bool:
    """Whether the points whose components are `z` (entry
    ``(m * num_vars + t) * k + s`` = component s of coordinate t of point
    m) are all one point in every component."""
    first = [z[t * k] for t in range(num_vars) for _ in range(k)]
    return _tile(FieldVector(z.field, first), len(z)) == z


def _combine(tables: Sequence[FieldVector], alphas: FieldVector, k: int) -> FieldVector:
    """``sum_{c,s} alphas[c k + s] * (component s of table c)``, entry by
    entry, for tables of components laid out as `crt_components` with
    ``element_major`` lays them out (entry ``b * k + s`` = component s of
    element b); entry b of the result is the sum for element b."""
    out = tables[0] * _tile(_window(alphas, 0, k), len(tables[0]))
    for c in range(1, len(tables)):
        coefficients = _tile(_window(alphas, c * k, k), len(tables[c]))
        out = out.fma(tables[c], coefficients)
    # Sum each run of k adjacent entries, halving the runs: k is a power of
    # two.
    while k > 1:
        lo, hi = out.split_even_odd()
        out = lo + hi
        k //= 2
    return out
