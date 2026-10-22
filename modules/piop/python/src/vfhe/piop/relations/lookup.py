# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The lookup relation: every evaluation of an oracle lies in a small table.

`Relation_Lookup` is the target of the range proof's decomposition step
(`range.py`): once the digits of a
bounded ring element are on the table, all that is left to prove is that each
digit is a member of `t_beta = {0, ..., beta - 1}`.

Left terminal, the claim is decided by the ideal decider below, which reads
every hypercube evaluation; `protocols.Lookup` is the succinct discharge.
See `modules/piop/range.md`.
"""

from __future__ import annotations

from typing import cast

from vfhe.arith.mle import native_table, vector_table
from vfhe.piop.oracle import TableOracle
from vfhe.piop.relation import Relation, Statement, hypercube


class Relation_Lookup(Relation):
    """`f(b) in t_beta` for every `b` in `{0,1}^n`, with
    `t_beta = {0, ..., beta - 1}` the relation's index.

    The table is an index rather than an instance field because it is the
    large, reusable, preprocessable part of the claim: one `beta` serves every
    lookup of a given digit width, and it is public data both parties hold
    (`piop.md` §3).
    """

    name = "lookup"
    fields = ("oracles",)
    index_fields = ("beta",)

    def __init__(self, beta: int) -> None:
        if beta <= 0 or beta & (beta - 1):
            raise ValueError(f"beta must be a positive power of two, got {beta}")
        super().__init__(index={"beta": beta})

    @property
    def beta(self) -> int:
        """The table size; `t_beta` is `{0, ..., beta - 1}`."""
        return cast("int", self.index["beta"])

    def check(self, statement: Statement) -> bool:
        """The ideal decider: every hypercube evaluation is in range.

        A dense table in the evaluation basis already *is* the list of those
        evaluations, so it is read directly; anything else (a coefficient
        basis, a defined oracle) is enumerated the slow way.
        """
        (f,) = statement.oracles
        if isinstance(f, TableOracle):
            table = f.polynomial
            if native_table(table) or vector_table(table):
                return all(self._in_range(e) for e in table.table)
        return all(self._in_range(f.query(b)) for b in hypercube(f.variables))

    def _in_range(self, value) -> bool:
        """Whether one coefficient-domain value is a member of `t_beta`.

        `int()` on a field element is its canonical representative in
        `[0, p)`, so a digit that overflowed the table is a large integer
        here, not a small negative one.
        """
        return 0 <= int(value) < self.beta
