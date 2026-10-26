# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The sumcheck protocol for the sum of a product of two multilinear tables.

The claim is ``sum_b f(b) g(b) = v`` over the hypercube, proved by
`SumcheckProd` under Fiat-Shamir and ending in one evaluation claim per
factor. Over ``R_q`` the ring has N = 64 and one 49-bit prime, split into 16
residue fields; over ``F_(p^2)``, p is the 50-bit prime of polycom's tests.
"""

from __future__ import annotations

from typing import Any

from harness import Case, Shape, benchmark
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Statement, TableOracle
from vfhe.piop.protocols import SumcheckProd
from vfhe.piop.relations import Relation_SumProd

RING_SHAPES = [
    Shape("rq-n6-1p-v16", log_n=6, prime_bits=[49], split_degree=4, num_vars=16),
    Shape("rq-n6-1p-v20", log_n=6, prime_bits=[49], split_degree=4, num_vars=20),
]
FIELD_SHAPES = [
    Shape("fp2-q50-v16", prime=562949948178433, w=5, num_vars=16),
    Shape("fp2-q50-v20", prime=562949948178433, w=5, num_vars=20),
]
FACTORS = 2

#: A ring table repeats this many random elements, each entry its own copy:
#: the memory traffic of a table of distinct elements, at a fraction of the
#: setup cost, and a claimed sum computed from one period.
PERIOD = 256


class Setting:
    """Two factor tables, their oracles and the true sum of their product."""

    def __init__(self, shape: Shape) -> None:
        num_vars = shape["num_vars"]
        variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
        n = 1 << num_vars
        mles: list[MLE] = []
        if "log_n" in shape:
            ring = Ring(
                1 << shape["log_n"],
                prime_size=shape["prime_bits"],
                split_degree=shape["split_degree"],
            )
            self.domain: Any = ring
            periods = [
                [ring.random_element() for _ in range(PERIOD)] for _ in range(FACTORS)
            ]
            for period in periods:
                entries = [period[i % PERIOD].copy() for i in range(n)]
                mles.append(MLE(ring=ring, variables=variables, evaluations=entries))
            value = periods[0][0] * periods[1][0]
            for j in range(1, PERIOD):
                value += periods[0][j] * periods[1][j]
            value *= n // PERIOD
        else:
            field = Field(shape["prime"], 2, shape["w"])
            self.domain = field
            tables = []
            for j in range(FACTORS):
                table = FieldVector(field, n)
                table.sample_random(f"bench-sumcheck-{j}".encode())
                tables.append(table)
                mles.append(MLE(field=field, variables=variables, evaluations=table))
            value = (tables[0] * tables[1]).sum()
        self.oracles = [TableOracle(f) for f in mles]
        self.value = value

    def statement(self) -> Statement:
        return Statement(Relation_SumProd(), oracles=self.oracles, value=self.value)

    def iop(self) -> IOP:
        iop = IOP(domain=self.domain, fiat_shamir=True)
        iop.register(Relation_SumProd, SumcheckProd())
        return iop

    def prove(self) -> Proof:
        return self.iop().prove(self.statement())

    def verify(self, proof: Proof) -> bool:
        return self.iop().verify(self.statement(), proof)


def prove_case(shape: Shape) -> Case:
    """Times the prover: the round messages and the folds of both tables.

    Each call builds a fresh IOP, since a run consumes its transcript; the
    first fold is out of place, so the tables are unchanged between calls.
    """
    setting = Setting(shape)

    def check(proof: Proof) -> None:
        assert setting.verify(proof)

    return Case(setting.prove, check=check)


def verify_case(shape: Shape) -> Case:
    """Times the verifier, including its query to each factor's oracle.

    The factors are plain table oracles, so the verifier settles the final
    evaluation claims by evaluating both tables at the last point; at these
    sizes that evaluation is most of the time. Behind a commitment scheme an
    evaluation proof answers those claims instead.
    """
    setting = Setting(shape)
    proof = setting.prove()

    def check(accepted: bool) -> None:
        assert accepted

    return Case(lambda: setting.verify(proof), check=check)


@benchmark(shapes=RING_SHAPES, reps=3)
def sumcheck_prove_ring(shape: Shape) -> Case:
    """Prove the sum of a product of two tables over ``R_q``."""
    return prove_case(shape)


@benchmark(shapes=FIELD_SHAPES, tier="headline")
def sumcheck_prove_field(shape: Shape) -> Case:
    """Prove the sum of a product of two tables over ``F_(p^2)``."""
    return prove_case(shape)


@benchmark(shapes=RING_SHAPES, reps=3)
def sumcheck_verify_ring(shape: Shape) -> Case:
    """Verify a sumcheck proof over ``R_q``, oracle queries included."""
    return verify_case(shape)


@benchmark(shapes=FIELD_SHAPES)
def sumcheck_verify_field(shape: Shape) -> Case:
    """Verify a sumcheck proof over ``F_(p^2)``, oracle queries included."""
    return verify_case(shape)
