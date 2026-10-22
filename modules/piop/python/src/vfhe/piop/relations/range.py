# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The range relation: every `Z_q` coefficient of every hypercube evaluation
of an `R_q` oracle lies in `[0, B)` -- membership in the table `T_B` of
[CCCFGS26] §4. `protocols.RangeDecomposition` discharges it; the
conventions -- LSB-first table order, which variable block is which -- are
set out in `modules/piop/range.md`.
"""

from __future__ import annotations

from typing import cast

from vfhe.arith.mle import native_table
from vfhe.piop.oracle import TableOracle
from vfhe.piop.relation import Relation, Statement, hypercube


class Relation_Range(Relation):
    """Every `Z_q` coefficient of every hypercube evaluation of the oracle
    lies in `[0, B)`: the table `T_B` of [CCCFGS26] §4.

    The index is `(B, c)`: the bound, and the number of base-`beta` digits the
    protocol splits each coefficient into (`beta = B ** (1/c)`). Both are
    public, reusable, preprocessable data, which is what an index is for.
    `c` is a pure efficiency knob -- it trades the digit alphabet against the
    size of the digit oracle -- and does not change the language.
    """

    name = "range"
    fields = ("oracles",)
    index_fields = ("bound", "chunks")

    def __init__(self, bound: int, chunks: int) -> None:
        if bound <= 1 or bound & (bound - 1):
            raise ValueError(f"bound B must be a power of two > 1, got {bound}")
        if chunks <= 0 or chunks & (chunks - 1):
            raise ValueError(f"chunks c must be a power of two, got {chunks}")
        if (bound.bit_length() - 1) % chunks:
            raise ValueError(
                f"c must divide log2(B): log2({bound}) = {bound.bit_length() - 1} "
                f"is not a multiple of {chunks}"
            )
        super().__init__(index={"bound": bound, "chunks": chunks})

    @property
    def bound(self) -> int:
        """The coefficient bound `B`."""
        return cast("int", self.index["bound"])

    @property
    def chunks(self) -> int:
        """The number of base-`beta` digits per coefficient, `c`."""
        return cast("int", self.index["chunks"])

    @property
    def beta(self) -> int:
        """The digit alphabet `B ** (1/c)`."""
        return 1 << ((self.bound.bit_length() - 1) // self.chunks)

    def check(self, statement: Statement) -> bool:
        (v,) = statement.oracles
        if isinstance(v, TableOracle) and native_table(v.polynomial):
            entries = v.polynomial.table
        else:
            entries = [v.query(b) for b in hypercube(v.variables)]
        # get_polynomial() is the unsigned representative in [0, q), so a
        # negative coefficient shows up here as a large one and is rejected.
        return all(max(e.get_polynomial()) < self.bound for e in entries)
