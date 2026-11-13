# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The basefold polynomial commitment: commit, evaluation proof, verification, encoding.

Over ``R_q`` the committed multilinear polynomial has 64 ring elements
(6 variables): 2^14 coefficients per prime on the small ring, 2^16 on the
CKKS-like one. Over ``F_(p^2)``, with p the 50-bit prime of polycom's tests,
it has 2^12 or 2^20 entries. Every code has ``k0 = 16`` and ``c = 4``, so
the evaluation protocol's default 32 queries fit in its first level. Proofs
are non-interactive (Fiat-Shamir).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Relation_Eval, Statement
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS, FoldableRS

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

RING_SETS = {
    "rq-n8-1p-v6": {"log_n": 8, "prime_bits": [49], "num_vars": 6},
    "rq-n10-3p-v6": {"log_n": 10, "prime_bits": [60, 40, 40], "num_vars": 6},
}
FIELD_SETS = {
    "fp2-q50-v12": {"prime": 562949948178433, "w": 5, "num_vars": 12},
    "fp2-q50-v20": {"prime": 562949948178433, "w": 5, "num_vars": 20},
}
rings = pytest.mark.parametrize("p", list(RING_SETS.values()), ids=list(RING_SETS))
fields = pytest.mark.parametrize("p", list(FIELD_SETS.values()), ids=list(FIELD_SETS))
K0_LOG = 4
C = 4


class Setting:
    """A basefold scheme, a committed polynomial, and one true evaluation claim on it."""

    def __init__(self, p: dict) -> None:
        num_vars = p["num_vars"]
        self.variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
        if "log_n" in p:
            ring = Ring(1 << p["log_n"], prime_size=p["prime_bits"], split_degree=1)
            self.domain: Any = ring
            code: Any = FoldableRS(ring, k0=1 << K0_LOG, c=C, d=num_vars - K0_LOG)
            self.f = MLE(
                ring=ring,
                variables=self.variables,
                evaluations=[ring.random_element() for _ in range(1 << num_vars)],
            )
            self.point = {v: ring.random_exceptional() for v in self.variables}
        else:
            field = Field(p["prime"], 2, p["w"])
            self.domain = field
            code = FieldFoldableRS(field, k0=1 << K0_LOG, c=C, d=num_vars - K0_LOG)
            table = FieldVector(field, 1 << num_vars)
            table.sample_random(b"bench-basefold-table")
            self.f = MLE(field=field, variables=self.variables, evaluations=table)
            self.point = {
                v: field.random_element(f"bench-point-{i}".encode())
                for i, v in enumerate(self.variables)
            }
        self.scheme = Basefold(code)
        self.oracle, self.opening = self.scheme.commit(self.f)
        self.value = self.f.evaluate(self.point, in_place=False).constant()

    def statement(self) -> Statement:
        return Statement(
            Relation_Eval(), oracles=(self.oracle,), point=self.point, value=self.value
        )

    def iop(self, with_opening: bool) -> IOP:
        """A fresh Fiat-Shamir IOP running `BasefoldEval`; the prover's opening if asked."""
        iop = IOP(domain=self.domain, fiat_shamir=True)
        iop.register(Relation_Eval, BasefoldEval(self.scheme))
        if with_opening:
            iop.prover.witnesses[self.oracle.commitment] = self.opening
        return iop

    def prove(self) -> Proof:
        return self.iop(with_opening=True).prove(self.statement())

    def verify(self, proof: Proof) -> bool:
        return self.iop(with_opening=False).verify(self.statement(), proof)


def _commit(benchmark: BenchmarkFixture, p: dict) -> None:
    """`Basefold.commit`: the coefficient table, its encoding, the Merkle tree.

    The scheme records every opening it makes; each call drops it again, so
    repeated commits do not accumulate codewords.
    """
    setting = Setting(p)

    def op():
        oracle, _ = setting.scheme.commit(setting.f)
        setting.scheme.openings.clear()
        return oracle

    assert benchmark(op).commitment.root == setting.oracle.commitment.root


def _prove(benchmark: BenchmarkFixture, p: dict) -> None:
    """The prover of one evaluation proof, on a commitment made beforehand; each call a fresh IOP."""
    setting = Setting(p)
    assert setting.verify(benchmark(setting.prove))


def _verify(benchmark: BenchmarkFixture, p: dict) -> None:
    """The verifier of one evaluation proof, from the commitment and the proof."""
    setting = Setting(p)
    proof = setting.prove()
    assert benchmark(setting.verify, proof)


def _encode(benchmark: BenchmarkFixture, p: dict) -> None:
    """The foldable code's encoding of the polynomial's coefficient table."""
    setting = Setting(p)
    code, message = setting.scheme.code, setting.f.to_coefficients().table
    assert benchmark(code.encode, message) == setting.opening.word


@rings
def basefold_commit_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Commit to a multilinear polynomial over ``R_q``."""
    _commit(benchmark, p)


@fields
def basefold_commit_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Commit to a multilinear polynomial over ``F_(p^2)``."""
    _commit(benchmark, p)


@rings
def basefold_prove_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Prove one evaluation of a committed polynomial over ``R_q``."""
    _prove(benchmark, p)


@fields
def basefold_prove_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Prove one evaluation of a committed polynomial over ``F_(p^2)``."""
    _prove(benchmark, p)


@rings
def basefold_verify_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Verify one evaluation proof over ``R_q``."""
    _verify(benchmark, p)


@fields
def basefold_verify_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Verify one evaluation proof over ``F_(p^2)``."""
    _verify(benchmark, p)


@rings
def foldable_code_encode_ring(benchmark: BenchmarkFixture, p: dict) -> None:
    """Encode a coefficient table with the foldable code over ``R_q``."""
    _encode(benchmark, p)


@fields
def foldable_code_encode_field(benchmark: BenchmarkFixture, p: dict) -> None:
    """Encode a coefficient table with the foldable code over ``F_(p^2)``."""
    _encode(benchmark, p)
