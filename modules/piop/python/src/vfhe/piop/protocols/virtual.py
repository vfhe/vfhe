# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The protocols over defined oracles (piop.md section 4): the evaluation
reduction on a virtual oracle, and the composable protocol of a sum
definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.protocol import Protocol, Rejection, exceptional_set_size
from vfhe.piop.relations import Relation_Sum, Relation_SumDef
from vfhe.piop.virtual import VirtualOracle, resolve_table, scaled

if TYPE_CHECKING:
    from vfhe.piop.party import Prover, Verifier
    from vfhe.piop.relation import Relation, Statement


class VirtualEval(Protocol):
    """`Eval(p*, x, v)` on a virtual oracle -> one `Eval(p_j, h_j(x), y_j)`
    per distinct non-public constituent query (piop.md §4).

    The prover sends the constituent values y_j; the verifier evaluates
    the public constituents itself, checks `ev(y, x) == v`, and emits the
    claims. No challenge: each y_j is bound by its own claim and the check
    pins v, so the step has soundness error 0.
    """

    name = "virtual"
    reduces_from: type[Relation] = Relation_Eval
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)

    def _outputs(self, statement: Statement, claims: list, values: list):
        return [
            self.reduce([statement], oracles=[c], point=pt, value=y)
            for (c, pt, _), y in zip(claims, values, strict=True)
        ]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        iop = prover.iop
        (g,) = statement.oracles
        point = dict(statement.point)
        claims = g.claims(point)
        values = [
            resolve_table(c, prover.witnesses).evaluate(pt, in_place=False).constant()
            for c, pt, _ in claims
        ]
        iop.transcript.write(f"{self.name}{statement.path}/vals", tuple(values))
        return self._outputs(statement, claims, values)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        await statement.resolved()
        iop = verifier.iop
        (g,) = statement.oracles
        point = dict(statement.point)
        claims = g.claims(point)
        label = f"{self.name}{statement.path}"
        values = await iop.transcript.read(f"{label}/vals")
        if len(values) != len(claims):
            raise Rejection(f"{label}: wrong number of constituent values")
        by_index: dict = {}
        for (_, _, idxs), y in zip(claims, values, strict=True):
            for j in idxs:
                by_index[j] = y
        for j, c in enumerate(g.constituents):
            if j not in by_index:  # public: evaluate it here
                by_index[j] = c.query(g.query_point(j, point))
        combined = g.combine([by_index[j] for j in range(len(g.constituents))])
        if not (combined == statement.value):
            raise Rejection(f"{label}: constituent values do not combine to the claim")
        return self._outputs(statement, claims, list(values))


class SumDefStart(Protocol):
    """The composable protocol of a sum definition: the started statement's
    claims `L(r_i) = y_i` become one `Sum(sum_i alpha_i G(r_i, .), sum_i
    alpha_i y_i)` over the summed variables, the alphas independent challenges
    (none for a single claim); the registered sumcheck then reduces it to one
    evaluation claim on that virtual oracle, and `VirtualEval` that claim to
    claims on G's constituents, both run by the Composer inside the layer (or
    by the driver, for a started statement run on its own). Zero
    communication; soundness error (k-1)/|A| for k claims [Sor22, Lem. 11;
    CCCFGS25, Eqs. (19)-(20)].
    """

    name = "sumdef"
    reduces_from: type[Relation] = Relation_SumDef.Start
    reduces_to: tuple[type[Relation], ...] = (Relation_Sum,)

    def soundness_error(self, statement: Statement, domain=None) -> float | None:
        """(k - 1) / |A|; the domain defaults to the definition's
        (`Oracle.domain` of its body)."""
        if domain is None:
            domain = Relation_SumDef.definition(statement)[1].domain
        size = exceptional_set_size(domain)
        if size is None:
            return None
        return (len(statement.points) - 1) / size

    async def _combine(self, party, statements: list[Statement]) -> list[Statement]:
        iop = party.iop
        (s,) = statements
        await s.resolved()
        L, G = Relation_SumDef.definition(s)
        points, values = list(s.points), list(s.values)
        if not points or len(points) != len(values):
            raise ValueError("a started sum definition carries k >= 1 paired claims")
        label = f"{self.name}{s.path}"
        if len(points) == 1:
            alphas = [1]
        else:
            alphas = [
                await iop.verifier.challenge(f"{label}/a{i}", G.domain)
                for i in range(len(points))
            ]
        bodies = [
            G.evaluate({v: p[v] for v in L.variables}, in_place=False) for p in points
        ]
        combined = VirtualOracle.linear(alphas, bodies)
        value = None
        for alpha, y in zip(alphas, values, strict=True):
            t = scaled(alpha, y)
            value = t if value is None else value + t
        return [self.reduce([s], Relation_Sum, oracles=[combined], value=value)]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(prover, statements)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        return await self._combine(verifier, statements)
