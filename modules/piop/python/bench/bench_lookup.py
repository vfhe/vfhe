# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two protocols for the lookup claim: every entry of a table lies in
``{0, ..., beta - 1}``.

`Lookup` proves it by offline memory checking over four grand products,
`LogUpGKR` by logarithmic derivatives with GKR on trees of fractions. Both
run in the ideal model (the oracles they send are tables, not commitments)
under Fiat-Shamir, over a 208-bit pseudo-Mersenne field, on a table of 2^16
or 2^20 random entries below 2^8 or 2^11.
"""

from __future__ import annotations

import random
from array import array
from typing import Any

from harness import Case, Shape, benchmark
from vfhe.arith import MLE, FieldVector, MLE_Variable, PseudoMersenneField
from vfhe.piop import IOP, OracleKind, Proof, Relation_Eval, Statement, TableOracle
from vfhe.piop.protocols import LogUpGKR, Lookup, Sumcheck, VirtualEval
from vfhe.piop.relations import Relation_Lookup, Relation_Sum

SHAPES = [
    Shape("pmf208-v16-b8", bits=208, num_vars=16, log_beta=8),
    Shape("pmf208-v20-b8", bits=208, num_vars=20, log_beta=8),
    Shape("pmf208-v20-b11", bits=208, num_vars=20, log_beta=11),
]


class Setting:
    """A lookup claim on a table of random entries in range, and the IOP that
    proves it with one of the two protocols."""

    def __init__(self, shape: Shape, protocol: type) -> None:
        self.field = PseudoMersenneField.generate(shape["bits"])
        self.protocol = protocol
        n, beta = shape["num_vars"], 1 << shape["log_beta"]
        rng = random.Random(n)
        entries = FieldVector(self.field, 1 << n)
        entries.set_uint64(array("Q", (rng.randrange(beta) for _ in range(1 << n))))
        variables = [MLE_Variable(f"x{i}") for i in range(n)]
        self.oracle = TableOracle(
            MLE(field=self.field, variables=variables, evaluations=entries)
        )
        self.beta = beta

    def statement(self) -> Statement:
        return Statement(Relation_Lookup(self.beta), oracles=[self.oracle])

    def iop(self) -> IOP:
        iop = IOP(domain=self.field, fiat_shamir=True)
        iop.register(Relation_Lookup, self.protocol(self.field))
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
        return iop

    def prove(self) -> Proof:
        return self.iop().prove(self.statement())

    def verify(self, proof: Proof) -> bool:
        return self.iop().verify(self.statement(), proof)


def prove_case(shape: Shape, protocol: type) -> Case:
    """Times the prover, from the claim to the last message."""
    setting = Setting(shape, protocol)

    def check(proof: Proof) -> None:
        assert setting.verify(proof)

    return Case(setting.prove, check=check)


def verify_case(shape: Shape, protocol: type) -> Case:
    """Times the verifier. In the ideal model it settles the final evaluation
    claims by evaluating the tables it holds, the looked-up table included,
    so this is not the cost of a verifier behind a commitment scheme."""
    setting = Setting(shape, protocol)
    proof = setting.prove()

    def check(accepted: Any) -> None:
        assert accepted

    return Case(lambda: setting.verify(proof), check=check)


@benchmark(shapes=SHAPES, reps=3)
def lookup_prove_memory_checking(shape: Shape) -> Case:
    """Prove the lookup claim with `Lookup`."""
    return prove_case(shape, Lookup)


@benchmark(shapes=SHAPES, reps=3)
def lookup_prove_logup_gkr(shape: Shape) -> Case:
    """Prove the lookup claim with `LogUpGKR`."""
    return prove_case(shape, LogUpGKR)


@benchmark(shapes=SHAPES, reps=3)
def lookup_verify_memory_checking(shape: Shape) -> Case:
    """Verify a `Lookup` proof in the ideal model."""
    return verify_case(shape, Lookup)


@benchmark(shapes=SHAPES, reps=3)
def lookup_verify_logup_gkr(shape: Shape) -> Case:
    """Verify a `LogUpGKR` proof in the ideal model."""
    return verify_case(shape, LogUpGKR)
