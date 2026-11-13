# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""A GKR proof of a layered arithmetic circuit over ``R_q``.

The circuit has ``2^width_bits`` inputs and ``depth`` layers of as many
fan-in-2 gates, each an addition or a multiplication of two random wires of
the layer below. The output is public; the input is a plain table oracle.
The ring has N = 64 and one 49-bit prime, split into 16 residue fields.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import IOP, OracleKind, Relation_Eval, Statement, TableOracle
from vfhe.piop.protocols import (
    GKR,
    GKR_Start,
    OutputTest,
    Sumcheck,
    SumDefStart,
    VirtualEval,
)
from vfhe.piop.relations import Relation_Equal, Relation_Layer, Relation_Sum
from vfhe.piop.relations.circuit import Relation_CircuitSparseIndex

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

SIZES = {"rq-n6-w6-d4": (6, 4), "rq-n6-w10-d8": (10, 8)}


def _iop(ring: Ring) -> IOP:
    """A Fiat-Shamir IOP with the protocols a GKR proof runs through."""
    iop = IOP(domain=ring, fiat_shamir=True)
    iop.register(Relation_CircuitSparseIndex, GKR())
    iop.register(Relation_CircuitSparseIndex.Start, GKR_Start())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


@pytest.mark.benchmark(min_rounds=3)
@pytest.mark.parametrize(("bits", "depth"), list(SIZES.values()), ids=list(SIZES))
def gkr_prove(benchmark: BenchmarkFixture, bits: int, depth: int) -> None:
    """Prove that the public output is the circuit's value on the committed input.

    Each call builds a fresh IOP, since a run consumes its transcript.
    """
    ring = Ring(64, prime_size=[49], split_degree=4)
    rng = random.Random(1)
    width = 1 << bits
    layers = [
        gkr.Layer(
            gates=[
                (add_gate if rng.random() < 0.5 else mul_gate)(
                    rng.randrange(width), rng.randrange(width)
                )
                for _ in range(width)
            ]
        )
        for _ in range(depth)
    ]
    circuit = gkr.Circuit(num_inputs=width, layers=layers)
    inputs = [rng.randrange(1, 1 << 20) for _ in range(width)]
    table = MLE(
        ring=ring,
        variables=[MLE_Variable(f"w{i}") for i in range(bits)],
        evaluations=inputs,
    )
    relation = Relation_CircuitSparseIndex.indexed(circuit, ring)
    output = MLE(
        ring=ring,
        variables=[MLE_Variable(f"o{i}") for i in range(bits)],
        evaluations=relation.layer_tables(table)[-1],
    )
    statement = Statement(
        relation, oracles=(TableOracle(output, public=True), TableOracle(table))
    )
    assert _iop(ring).verify(statement, benchmark(lambda: _iop(ring).prove(statement)))
