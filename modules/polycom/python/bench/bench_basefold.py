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

from typing import Any

from harness import Case, Shape, benchmark
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Relation_Eval, Statement
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS, FoldableRS

RING_SHAPES = [
    Shape("rq-n8-1p-v6", log_n=8, prime_bits=[49], num_vars=6),
    Shape("rq-n10-3p-v6", log_n=10, prime_bits=[60, 40, 40], num_vars=6),
]
FIELD_SHAPES = [
    Shape("fp2-q50-v12", prime=562949948178433, w=5, num_vars=12),
    Shape("fp2-q50-v20", prime=562949948178433, w=5, num_vars=20),
]
K0_LOG = 4
C = 4


class Setting:
    """A basefold scheme, a committed polynomial, and one true evaluation claim on it."""

    def __init__(self, shape: Shape) -> None:
        num_vars = shape["num_vars"]
        self.variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
        if "log_n" in shape:
            ring = Ring(
                1 << shape["log_n"], prime_size=shape["prime_bits"], split_degree=1
            )
            self.domain: Any = ring
            code: Any = FoldableRS(ring, k0=1 << K0_LOG, c=C, d=num_vars - K0_LOG)
            entries = [ring.random_element() for _ in range(1 << num_vars)]
            self.f = MLE(ring=ring, variables=self.variables, evaluations=entries)
            self.point = {v: ring.random_exceptional() for v in self.variables}
        else:
            field = Field(shape["prime"], 2, shape["w"])
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


def commit_case(shape: Shape) -> Case:
    """Times `Basefold.commit`: the coefficient table, its encoding, the Merkle tree.

    The scheme records every opening it makes; each call drops it again, so
    repeated commits do not accumulate codewords.
    """
    setting = Setting(shape)
    root = setting.oracle.commitment.root

    def op() -> Any:
        oracle, _ = setting.scheme.commit(setting.f)
        setting.scheme.openings.clear()
        return oracle

    def check(oracle: Any) -> None:
        assert oracle.commitment.root == root

    return Case(op, check=check)


def prove_case(shape: Shape) -> Case:
    """Times the prover of one evaluation proof, on a commitment made beforehand.

    Each call builds a fresh IOP, since a run consumes its transcript.
    """
    setting = Setting(shape)

    def check(proof: Proof) -> None:
        assert setting.verify(proof)

    return Case(setting.prove, check=check)


def verify_case(shape: Shape) -> Case:
    """Times the verifier of one evaluation proof, from the commitment and the proof."""
    setting = Setting(shape)
    proof = setting.prove()

    def check(accepted: bool) -> None:
        assert accepted

    return Case(lambda: setting.verify(proof), check=check)


def encode_case(shape: Shape) -> Case:
    """Times the foldable code's encoding of the polynomial's coefficient table."""
    setting = Setting(shape)
    code = setting.scheme.code
    message = setting.f.to_coefficients().table

    def check(word: Any) -> None:
        assert word == setting.opening.word

    return Case(lambda: code.encode(message), check=check)


@benchmark(shapes=RING_SHAPES, tier="headline")
def basefold_commit_ring(shape: Shape) -> Case:
    """Commit to a multilinear polynomial over ``R_q``."""
    return commit_case(shape)


@benchmark(shapes=FIELD_SHAPES, tier="headline")
def basefold_commit_field(shape: Shape) -> Case:
    """Commit to a multilinear polynomial over ``F_(p^2)``."""
    return commit_case(shape)


@benchmark(shapes=RING_SHAPES, tier="headline")
def basefold_prove_ring(shape: Shape) -> Case:
    """Prove one evaluation of a committed polynomial over ``R_q``."""
    return prove_case(shape)


@benchmark(shapes=FIELD_SHAPES, tier="headline")
def basefold_prove_field(shape: Shape) -> Case:
    """Prove one evaluation of a committed polynomial over ``F_(p^2)``."""
    return prove_case(shape)


@benchmark(shapes=RING_SHAPES)
def basefold_verify_ring(shape: Shape) -> Case:
    """Verify one evaluation proof over ``R_q``."""
    return verify_case(shape)


@benchmark(shapes=FIELD_SHAPES)
def basefold_verify_field(shape: Shape) -> Case:
    """Verify one evaluation proof over ``F_(p^2)``."""
    return verify_case(shape)


@benchmark(shapes=RING_SHAPES)
def foldable_code_encode_ring(shape: Shape) -> Case:
    """Encode a coefficient table with the foldable code over ``R_q``."""
    return encode_case(shape)


@benchmark(shapes=FIELD_SHAPES)
def foldable_code_encode_field(shape: Shape) -> Case:
    """Encode a coefficient table with the foldable code over ``F_(p^2)``."""
    return encode_case(shape)
