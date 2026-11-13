# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The sumcheck protocol for the sum of a product of two multilinear tables.

The claim is ``sum_b f(b) g(b) = v`` over the hypercube, proved by
`SumcheckProd` under Fiat-Shamir and ending in one evaluation claim per
factor. Over ``R_q`` the ring has N = 64 and one 49-bit prime, split into 16
residue fields; over ``F_(p^2)``, p is the 50-bit prime of polycom's tests.
Ring runs are timed over three rounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Statement, TableOracle
from vfhe.piop.protocols import SumcheckProd
from vfhe.piop.relations import Relation_SumProd

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

RING_SETS = {
    "rq-n6-1p-v16": {"log_n": 6, "prime_bits": [49], "split_degree": 4, "num_vars": 16},
    "rq-n6-1p-v20": {"log_n": 6, "prime_bits": [49], "split_degree": 4, "num_vars": 20},
}
FIELD_SETS = {
    "fp2-q50-v16": {"prime": 562949948178433, "w": 5, "num_vars": 16},
    "fp2-q50-v20": {"prime": 562949948178433, "w": 5, "num_vars": 20},
}
rings = pytest.mark.parametrize("p", list(RING_SETS.values()), ids=list(RING_SETS))
fields = pytest.mark.parametrize("p", list(FIELD_SETS.values()), ids=list(FIELD_SETS))
three_rounds = pytest.mark.benchmark(min_rounds=3)
FACTORS = 2

#: A ring table repeats this many random elements, each entry its own copy:
#: the memory traffic of a table of distinct elements, at a fraction of the
#: setup cost, and a claimed sum computed from one period.
PERIOD = 256


class Setting:
    """Two factor tables, their oracles and the true sum of their product."""

    def __init__(self, p: dict) -> None:
        num_vars = p["num_vars"]
        variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
        n = 1 << num_vars
        mles: list[MLE] = []
        if "log_n" in p:
            ring = Ring(
                1 << p["log_n"],
                prime_size=p["prime_bits"],
                split_degree=p["split_degree"],
            )
            self.domain: Any = ring
            periods = [
                [ring.random_element() for _ in range(PERIOD)] for _ in range(FACTORS)
            ]
            mles.extend(
                MLE(
                    ring=ring,
                    variables=variables,
                    evaluations=[period[i % PERIOD].copy() for i in range(n)],
                )
                for period in periods
            )
            value = periods[0][0] * periods[1][0]
            for j in range(1, PERIOD):
                value += periods[0][j] * periods[1][j]
            value *= n // PERIOD
        else:
            field = Field(p["prime"], 2, p["w"])
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


def _prove(benchmark: BenchmarkFixture, p: dict) -> None:
    """The prover: the round messages and the folds of both tables.

    Each call builds a fresh IOP, since a run consumes its transcript; the
    first fold is out of place, so the tables are unchanged between calls.
    """
    setting = Setting(p)
    assert setting.verify(benchmark(setting.prove))


def _verify(benchmark: BenchmarkFixture, p: dict) -> None:
    """The verifier, including its query to each factor's oracle.

    The factors are plain table oracles, so the verifier settles the final
    evaluation claims by evaluating both tables at the last point; at these
    sizes that evaluation is most of the time. Behind a commitment scheme an
    evaluation proof answers those claims instead.
    """
    setting = Setting(p)
    proof = setting.prove()
    assert benchmark(setting.verify, proof)


@three_rounds
@rings
def sumcheck_prove_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Prove the sum of a product of two tables over ``R_q``."""
    _prove(benchmark, p)


@fields
def sumcheck_prove_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Prove the sum of a product of two tables over ``F_(p^2)``."""
    _prove(benchmark, p)


@three_rounds
@rings
def sumcheck_verify_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Verify a sumcheck proof over ``R_q``, oracle queries included."""
    _verify(benchmark, p)


@fields
def sumcheck_verify_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Verify a sumcheck proof over ``F_(p^2)``, oracle queries included."""
    _verify(benchmark, p)
