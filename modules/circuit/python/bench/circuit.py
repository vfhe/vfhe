# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluating a layered arithmetic circuit and exporting its wiring.

The circuit has ``2^width_bits`` inputs and ``depth`` layers of as many
fan-in-2 gates, each an addition or a multiplication of two random wires of
the layer below, over the 31-bit prime 2013265921.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
from vfhe.circuit import add_gate, export, gkr, mul_gate

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

SIZES = {"w8-d4": (8, 4), "w12-d16": (12, 16)}
sizes = pytest.mark.parametrize(
    ("width_bits", "depth"), list(SIZES.values()), ids=list(SIZES)
)
MODULUS = 2013265921


def _layered_circuit(width_bits: int, depth: int) -> tuple[gkr.Circuit, list[int]]:
    """Returns a random layered circuit of the size, and inputs for it."""
    rng = random.Random(1)  # noqa: S311 - benchmark data, not a key
    width = 1 << width_bits
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
    circuit = gkr.Circuit(num_inputs=width, layers=layers, field_modulus=str(MODULUS))
    return circuit, [rng.randrange(MODULUS) for _ in range(width)]


@sizes
def circuit_evaluate(benchmark: BenchmarkFixture, width_bits: int, depth: int) -> None:
    """Compute every layer's wire values from the inputs."""
    circuit, inputs = _layered_circuit(width_bits, depth)
    values = benchmark(export.evaluate, circuit, inputs)
    # One gate of the last layer, recomputed from the layer below.
    gate, below = circuit.layers[-1].gates[0], values[-2]
    left, right = below[gate.left], below[gate.right]
    assert (
        values[-1][0]
        == (left + right if gate.type == gkr.GATE_TYPE_ADD else left * right) % MODULUS
    )


@sizes
def circuit_sparse_wiring(
    benchmark: BenchmarkFixture, width_bits: int, depth: int
) -> None:
    """Export every layer's wiring predicates in sparse form."""
    circuit, _ = _layered_circuit(width_bits, depth)
    wiring = benchmark(
        lambda: [export.sparse_wiring(circuit, i) for i in range(len(circuit.layers))]
    )
    # An addition reads two wires and a multiplication one pair.
    for layer, (lin, xmult) in zip(circuit.layers, wiring, strict=True):
        adds = sum(g.type == gkr.GATE_TYPE_ADD for g in layer.gates)
        assert (
            sum(lin.values()) == 2 * adds
            and sum(xmult.values()) == len(layer.gates) - adds
        )
