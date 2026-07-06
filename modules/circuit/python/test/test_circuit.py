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


def test_export_evaluate_and_mle():
    """Sanity for the exporter: evaluation, wiring tables, MLE consistency."""
    from vfhe.circuit import export

    circuit = _sample_circuit()
    p = int(circuit.field_modulus)

    # (0+1), (2*3) -> (1 + 6) = 7
    values = export.evaluate(circuit, [0, 1, 2, 3])
    assert values == [[0, 1, 2, 3], [1, 6], [7]]

    # Wiring tables: layer 0 has s_in=2, s_out=1; gate 0 = ADD(0,1), gate 1 = MUL(2,3).
    add_t, mul_t = export.wiring_tables(circuit, 0)
    assert len(add_t) == 1 << (1 + 2 * 2)
    assert add_t[(0 << 4) | (0 << 2) | 1] == 1 and sum(add_t) == 1
    assert mul_t[(1 << 4) | (2 << 2) | 3] == 1 and sum(mul_t) == 1

    # The MLE agrees with the table on hypercube points.
    for idx, want in enumerate(add_t):
        point = [(idx >> (4 - k)) & 1 for k in range(5)]
        assert export.mle_eval(add_t, point, p) == want


def test_export_to_arith_polynomials():
    """The bridge itself: tables land as coefficients of ring elements."""
    from vfhe.arith import Ring
    from vfhe.circuit import export

    circuit = _sample_circuit()
    ring = Ring(32, prime_size=[30, 30], split_degree=1)

    add_p, mul_p = export.wiring_polynomials(circuit, 0, ring)
    add_t, mul_t = export.wiring_tables(circuit, 0)
    assert add_p.get_polynomial() == add_t
    assert mul_p.get_polynomial() == mul_t

    w1 = export.values_polynomial(circuit, [0, 1, 2, 3], 1, ring)
    assert w1.get_polynomial()[:2] == [1, 6]
