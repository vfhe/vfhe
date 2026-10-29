# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The hypercube relations of the toolbox (piop.md section 4): a sum, a sum
of products, and a zero test over {0,1}^n."""

from __future__ import annotations

from vfhe.piop.relation import (
    Relation,
    Statement,
    aligned_tables,
    hypercube,
    hypercube_sum,
    ring_sum_of_products,
    value_of,
)


class Relation_Sum(Relation):
    """sum_{b in {0,1}^n} f(b) == value, for a single oracle f."""

    name = "sum"
    fields = ("oracles", "value")

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        return hypercube_sum(f) == value_of(statement.value)


class Relation_SumProd(Relation):
    """sum_{b in {0,1}^n} prod_j f_j(b) == value, for oracles f_1..f_k over
    the same variables (an instance whose oracles disagree on them is not
    in the relation)."""

    name = "sumprod"
    fields = ("oracles", "value")

    def check(self, statement: Statement) -> bool:
        first = statement.oracles[0]
        if any(set(f.variables) != set(first.variables) for f in statement.oracles):
            return False
        tables = aligned_tables(statement.oracles)
        if tables is not None and isinstance(tables[0], list):
            return ring_sum_of_products(tables) == value_of(statement.value)
        if tables is not None:
            # One product vector, then folded in place: the first multiply
            # allocates it, the rest write through it. Never into `tables[0]`,
            # which belongs to the oracle.
            product = tables[0]
            if len(tables) > 1:
                product = product * tables[1]
                for table in tables[2:]:
                    product.mul(table, out=product)
            return product.sum() == value_of(statement.value)

        total = None
        for b in hypercube(statement.oracles[0].variables):
            prod = None
            for f in statement.oracles:
                e = f.query(b)
                prod = e if prod is None else prod * e
            total = prod if total is None else total + prod
        return total == value_of(statement.value)


class Relation_Zero(Relation):
    """f(b) == 0 for every b in {0,1}^n, for a single oracle f."""

    name = "zero"
    fields = ("oracles",)

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        return all(f.query(b) == 0 for b in hypercube(f.variables))
