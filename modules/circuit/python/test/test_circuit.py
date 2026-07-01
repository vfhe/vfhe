# SPDX-License-Identifier: Apache-2.0
from vfhe.circuit import add_gate, deserialize, gkr, mul_gate, serialize


def _sample_circuit() -> "gkr.Circuit":
    # 4 inputs -> layer of 2 gates (one add, one mul) -> 1 output add gate
    return gkr.Circuit(
        num_inputs=4,
        field_modulus="2013265921",  # a common 31-bit NTT-friendly prime
        layers=[
            gkr.Layer(gates=[add_gate(0, 1), mul_gate(2, 3)]),
            gkr.Layer(gates=[add_gate(0, 1)]),
        ],
    )


def test_gkr_protobuf_round_trip():
    circuit = _sample_circuit()
    restored = deserialize(serialize(circuit))
    assert restored == circuit
    assert restored.num_inputs == 4
    assert restored.layers[0].gates[1].type == gkr.GATE_TYPE_MUL
    assert restored.layers[1].gates[0].left == 0


def test_gate_helpers():
    assert add_gate(1, 2).type == gkr.GATE_TYPE_ADD
    assert mul_gate(3, 4).type == gkr.GATE_TYPE_MUL
