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

from harness import Case, Shape, benchmark
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import IOP, OracleKind, Proof, Relation_Eval, Statement, TableOracle
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

SHAPES = [
    Shape("rq-n6-w6-d4", width_bits=6, depth=4),
    Shape("rq-n6-w10-d8", width_bits=10, depth=8),
]


def gkr_iop(ring: Ring) -> IOP:
    """A Fiat-Shamir IOP with the protocols a GKR proof runs through."""
    iop = IOP(domain=ring, fiat_shamir=True)
    iop.register(Relation_CircuitSparseIndex, GKR())
    iop.register(Relation_CircuitSparseIndex.Start, GKR_Start())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


@benchmark(shapes=SHAPES)
def gkr_prove(shape: Shape) -> Case:
    """Prove that the public output is the circuit's value on the committed input.

    Each call builds a fresh IOP, since a run consumes its transcript.
    """
    ring = Ring(64, prime_size=[49], split_degree=4)
    rng = random.Random(1)
    bits, width = shape["width_bits"], 1 << shape["width_bits"]
    layers = [
        gkr.Layer(
            gates=[
                (add_gate if rng.random() < 0.5 else mul_gate)(
                    rng.randrange(width), rng.randrange(width)
                )
                for _ in range(width)
            ]
        )
        for _ in range(shape["depth"])
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

    def check(proof: Proof) -> None:
        assert gkr_iop(ring).verify(statement, proof)

    return Case(lambda: gkr_iop(ring).prove(statement), check=check)
