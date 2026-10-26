# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluating a layered arithmetic circuit and exporting its wiring.

The circuit has ``2^width_bits`` inputs and ``depth`` layers of as many
fan-in-2 gates, each an addition or a multiplication of two random wires of
the layer below, over the 31-bit prime 2013265921.
"""

from __future__ import annotations

import random

from harness import Case, Shape, benchmark
from vfhe.circuit import add_gate, export, gkr, mul_gate

SHAPES = [
    Shape("w8-d4", width_bits=8, depth=4),
    Shape("w12-d16", width_bits=12, depth=16),
]
MODULUS = 2013265921


def layered_circuit(shape: Shape) -> tuple[gkr.Circuit, list[int]]:
    """Returns a random layered circuit of the shape, and inputs for it."""
    rng = random.Random(1)  # noqa: S311 - benchmark data, not a key
    width = 1 << shape["width_bits"]
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
    circuit = gkr.Circuit(num_inputs=width, layers=layers, field_modulus=str(MODULUS))
    return circuit, [rng.randrange(MODULUS) for _ in range(width)]


@benchmark(shapes=SHAPES)
def circuit_evaluate(shape: Shape) -> Case:
    """Compute every layer's wire values from the inputs."""
    circuit, inputs = layered_circuit(shape)

    def check(values: list[list[int]]) -> None:
        # One gate of the last layer, recomputed from the layer below.
        gate, below = circuit.layers[-1].gates[0], values[-2]
        left, right = below[gate.left], below[gate.right]
        expected = left + right if gate.type == gkr.GATE_TYPE_ADD else left * right
        assert values[-1][0] == expected % MODULUS

    return Case(lambda: export.evaluate(circuit, inputs), check=check)


@benchmark(shapes=SHAPES)
def circuit_sparse_wiring(shape: Shape) -> Case:
    """Export every layer's wiring predicates in sparse form."""
    circuit, _ = layered_circuit(shape)

    def op() -> list[tuple[dict[int, int], dict[int, int]]]:
        return [export.sparse_wiring(circuit, i) for i in range(len(circuit.layers))]

    def check(wirings: list[tuple[dict[int, int], dict[int, int]]]) -> None:
        # An addition reads two wires and a multiplication one pair.
        for layer, (lin, xmult) in zip(circuit.layers, wirings, strict=True):
            adds = sum(g.type == gkr.GATE_TYPE_ADD for g in layer.gates)
            assert sum(lin.values()) == 2 * adds
            assert sum(xmult.values()) == len(layer.gates) - adds

    return Case(op, check=check)
