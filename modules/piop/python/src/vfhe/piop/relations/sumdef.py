# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sum definitions (piop.md section 4): an oracle defined as a hypercube sum
of a virtual oracle, the head of a statement in a layered reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.piop.evaluation import with_start
from vfhe.piop.relation import Relation, Statement, hypercube
from vfhe.piop.virtual import (
    VirtualOracle,
    expanded_tables,
    hypercube_index,
    table_like,
)

if TYPE_CHECKING:
    from vfhe.piop.oracle import Oracle


@with_start
class Relation_SumDef(Relation):
    """`L(z) = sum_{b in {0,1}^m} G(z, b)` on the hypercube: a *sum definition*
    of the oracle L, the head, by the virtual oracle G over L's variables z
    and the summed variables b, the others of G, in G's order: a layer of a
    layered circuit in terms of the layer below it [Tha22, §4.6], a
    grand-product level [Tha13, §5.3.1]. A statement headed at L is how an
    oracle that is defined and never committed enters a layered reduction:
    it sits in the layer of L, and `SumDefStart`, registered against
    `Relation_SumDef.Start`, is its composable protocol.
    """

    name = "sumdef"
    fields = ("oracles",)
    head_index = 0

    @staticmethod
    def summed_of(L, G) -> list:
        """The summed variables: G's that are not L's."""
        return [v for v in G.variables if not any(v is u for u in L.variables)]

    @classmethod
    def summed(cls, statement: Statement) -> list:
        L, G = cls.definition(statement)
        return cls.summed_of(L, G)

    @staticmethod
    def definition(statement: Statement) -> tuple[Oracle, VirtualOracle]:
        """The head L and the defining virtual oracle G of a sum-definition
        statement, G's type checked."""
        L, G = statement.oracles
        if not isinstance(G, VirtualOracle):
            raise TypeError(
                "the defining oracle of a sum definition is a virtual oracle"
            )
        return L, G

    def check(self, statement: Statement) -> bool:
        L, G = self.definition(statement)
        summed = self.summed(statement)
        for z in hypercube(L.variables):
            total = None
            for b in hypercube(summed):
                e = G.query({**z, **b})
                total = e if total is None else total + e
            if L.query(z) != total:
                return False
        return True

    def compute_head(self, statement: Statement):
        """L's table, the hypercube sums of G over the summed variables, from
        G's constituent tables (the prover view): the generic, exponential
        route. A relation that knows G's structure (a circuit layer) computes
        the table directly instead."""
        L, G = self.definition(statement)
        summed = self.summed(statement)
        tables = expanded_tables(G)
        evaluations = []
        for z in hypercube(L.variables):
            total = None
            for b in hypercube(summed):
                i = hypercube_index(G.variables, {**z, **b})
                e = G.combine([t.table[i] for t in tables])
                total = e if total is None else total + e
            evaluations.append(total)
        return table_like(tables, L.variables, evaluations)
