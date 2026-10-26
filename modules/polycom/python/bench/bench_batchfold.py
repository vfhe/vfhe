# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BatchFold [CCCFGS26, §7]: one proof of several evaluation claims over ``R_q``.

Four multilinear polynomials of 64 ring elements (6 variables) are committed
with basefold, each with one evaluation claim at its own exceptional point.
`BatchFoldEval` proves the four claims together, by one basefold run per
prime over the residue code. Proofs are non-interactive (Fiat-Shamir).
"""

from __future__ import annotations

from typing import Any

from harness import Case, Shape, benchmark
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Protocol, Relation, Relation_Eval, Statement
from vfhe.polycom import Basefold, BatchFoldEval, FoldableRS

SHAPES = [
    Shape("rq-n8-1p-v6", log_n=8, prime_bits=[49], num_vars=6),
    Shape("rq-n10-3p-v6", log_n=10, prime_bits=[60, 40, 40], num_vars=6),
]
CLAIMS = 4
K0_LOG = 4
C = 4


class Relation_Claims(Relation):
    """A bundle of evaluation claims, ``claims = ((oracle, point, value), ...)``."""

    name = "claims"
    fields = ("claims",)

    def check(self, statement: Any) -> bool:
        return False


class SplitClaims(Protocol):
    """Reduces a bundle to one `Relation_Eval` statement per claim, with no messages."""

    reduces_from = Relation_Claims
    reduces_to = (Relation_Eval,)

    def _claims(self, statements: list) -> list:
        return [
            self.reduce(statements, Relation_Eval, oracles=(o,), point=p, value=v)
            for s in statements
            for o, p, v in s.claims
        ]

    async def prove(self, prover: Any, statements: list) -> list:
        return self._claims(statements)

    async def verify(self, verifier: Any, statements: list) -> list:
        return self._claims(statements)


@benchmark(shapes=SHAPES)
def batchfold_prove(shape: Shape) -> Case:
    """Prove four evaluation claims on four committed polynomials with BatchFold.

    Each call builds a fresh IOP, since a run consumes its transcript. The
    openings are the ones the scheme recorded at commit time.
    """
    ring = Ring(1 << shape["log_n"], prime_size=shape["prime_bits"], split_degree=1)
    num_vars = shape["num_vars"]
    scheme = Basefold(FoldableRS(ring, k0=1 << K0_LOG, c=C, d=num_vars - K0_LOG))
    claims = []
    for j in range(CLAIMS):
        variables = [MLE_Variable(f"f{j}_{i}") for i in range(num_vars)]
        entries = [ring.random_element() for _ in range(1 << num_vars)]
        f = MLE(ring=ring, variables=variables, evaluations=entries)
        oracle, _ = scheme.commit(f)
        point = {v: ring.random_exceptional() for v in variables}
        claims.append((oracle, point, f.evaluate(point, in_place=False).constant()))

    def iop() -> IOP:
        iop = IOP(domain=ring, fiat_shamir=True)
        iop.register(Relation_Claims, SplitClaims())
        iop.register(Relation_Eval, BatchFoldEval(scheme))
        return iop

    def statement() -> Statement:
        return Statement(Relation_Claims(), claims=tuple(claims))

    def check(proof: Proof) -> None:
        assert iop().verify(statement(), proof)

    return Case(lambda: iop().prove(statement()), check=check)
