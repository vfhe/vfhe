# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Relation_Circuit and the GKR protocol over vfhe.circuit's protobuf
circuits: the ideal model over plain tables, the ring domain under
Fiat-Shamir, and the shape of the reduction (claims per layer)."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    GKR,
    IOP,
    ImplicitEval,
    OracleKind,
    Protocol,
    Relation_Circuit,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Sumcheck,
    VirtualEval,
)


class _IntDomain:
    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random_exceptional(self):
        return self._rng.randrange(2, 1 << 20)


def _circuit():
    # inputs (a, b, c, d) -> layer 0: (a + b, c * d) -> layer 1: (a + b) + c * d
    return gkr.Circuit(
        num_inputs=4,
        layers=[
            gkr.Layer(gates=[add_gate(0, 1), mul_gate(2, 3)]),
            gkr.Layer(gates=[add_gate(0, 1)]),
        ],
    )


def _iop(domain, **kw):
    iop = IOP(domain=domain, **kw)
    iop.register(Relation_Circuit, GKR())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, ImplicitEval(), kind=OracleKind.implicit)
    return iop


def _instance(ring=None, inputs=(1, 2, 3, 4), output=None, public_output=True):
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    o = [MLE_Variable("o0")]
    a, b, c, d = inputs
    out = [a + b + c * d, 0] if output is None else output
    w_in = MLE(ring=ring, variables=w, evaluations=list(inputs))
    out_mle = MLE(ring=ring, variables=o, evaluations=out, public=public_output)
    return Statement(Relation_Circuit(_circuit()), oracles=[w_in], output=out_mle)


def test_relation_circuit_decider():
    assert _instance().check()
    assert not _instance(output=[16, 0]).check()
    (w_in,) = _instance().oracles
    tables = Relation_Circuit(_circuit()).layer_tables(w_in)
    assert tables == [[1, 2, 3, 4], [3, 12], [15, 0]]


def test_gkr_ideal_model():
    terminal = []

    class _Record(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    iop = _iop(_IntDomain())
    iop.register(Relation_Eval, _Record())
    stmt = _instance()
    assert iop.run(stmt)
    # Both claims on the input arrive together, on the input oracle.
    (w_in,) = stmt.oracles
    assert len(terminal) == 2 and all(s.oracles[0] is w_in for s in terminal)
    assert all(s.check() for s in terminal)
    order = iop.transcript.order
    assert order[0] == "gkr/r0"  # the output point, one challenge
    # layer 1 (one claim, no RLC) then layer 0 (two claims, two RLC challenges)
    assert "sumcheck/0/0/g0" in order
    assert any(label.endswith("/a0") for label in order)
    assert not _iop(_IntDomain()).run(_instance(output=[16, 0]))


def test_gkr_non_public_output_emits_two_claims():
    kinds = []

    class _Record(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            kinds.extend(Relation_Eval.kind(s) for s in statements)
            return []

    iop = _iop(_IntDomain())
    iop.register(Relation_Eval, _Record())
    stmt = _instance(public_output=False)
    assert iop.run(stmt)
    assert "gkr/out" in iop.transcript.order
    assert kinds.count(OracleKind.plain) == 3  # two on the input, one on the output


@pytest.mark.parametrize("public_output", [True, False])
def test_gkr_ring_fiat_shamir(public_output):
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _instance(ring=ring, public_output=public_output)
    assert stmt.check()
    proof = _iop(ring, fiat_shamir=True).prove(stmt)
    assert _iop(ring, fiat_shamir=True).verify(stmt, proof)
    assert proof.digest() == _iop(ring, fiat_shamir=True).prove(stmt).digest()
    bad = _instance(ring=ring, output=[16, 0], public_output=public_output)
    assert not _iop(ring, fiat_shamir=True).verify(bad, proof)
    assert not _iop(ring, fiat_shamir=True).run(bad)


def _dot_circuit():
    # inputs (a, b, c, d) -> one XMULT gate a*c + b*d, and its ADD/MUL expansion.
    from vfhe.circuit import xmult_gate

    xm = gkr.Circuit(
        num_inputs=4, layers=[gkr.Layer(gates=[xmult_gate([0, 1], [2, 3])])]
    )
    expanded = gkr.Circuit(
        num_inputs=4,
        layers=[
            gkr.Layer(gates=[mul_gate(0, 2), mul_gate(1, 3)]),
            gkr.Layer(gates=[add_gate(0, 1)]),
        ],
    )
    return xm, expanded


def test_xmult_gate_layer_tables_and_proof():
    xm, expanded = _dot_circuit()
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    w_in = MLE(variables=w, evaluations=[1, 2, 3, 4])
    assert Relation_Circuit(xm).layer_tables(w_in) == [[1, 2, 3, 4], [11, 0]]
    assert Relation_Circuit(expanded).layer_tables(w_in)[-1] == [11, 0]
    out = MLE(variables=[MLE_Variable("o0")], evaluations=[11, 0], public=True)
    stmt = Statement(Relation_Circuit(xm), oracles=[w_in], output=out)
    assert stmt.check()
    iop = _iop(_IntDomain())
    assert iop.run(stmt)
    # One layer, one sumcheck over (x, y): 4 rounds, all of degree 2.
    rounds = [k for k in iop.transcript.order if k.startswith("sumcheck") and "/g" in k]
    assert len(rounds) == 4
    assert all(len(iop.transcript.entries[k].result()) == 3 for k in rounds)
    bad = MLE(variables=[MLE_Variable("o0")], evaluations=[12, 0], public=True)
    assert not _iop(_IntDomain()).run(
        Statement(Relation_Circuit(xm), oracles=[w_in], output=bad)
    )


def test_gkr_dense_and_libra_strategies_agree(monkeypatch):
    from vfhe.piop.virtual import _LibraProver

    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _instance(ring=ring)
    (w_in,) = stmt.oracles
    assert isinstance(stmt.relation, Relation_Circuit)
    layer = GKR.layers(stmt.relation, w_in, ring)[0]
    (z,) = layer.variables
    body = layer.instantiate({z: ring.random_exceptional()})
    assert isinstance(body.prover_view({}), _LibraProver)
    from vfhe.piop import VirtualOracle

    digests = {}
    for strategy in ("dense", "libra"):
        monkeypatch.setattr(VirtualOracle, "strategy", strategy)
        digests[strategy] = _iop(ring, fiat_shamir=True).prove(stmt).digest()
    assert digests["dense"] == digests["libra"]


def test_gkr_wide_layers_run_in_linear_time():
    # 64 inputs, two layers of 64 and 32 random gates: the dense predicates
    # would have 2^(6 + 12) entries per layer; the two-phase prover touches
    # the wires and the 2^6 tables only.
    from vfhe.circuit import xmult_gate

    rng = random.Random(7)
    n = 64

    def layer(width, fan_in):
        gates = []
        for _ in range(width):
            kind = rng.choice(("add", "mul", "xmult"))
            if kind == "add":
                gates.append(add_gate(rng.randrange(fan_in), rng.randrange(fan_in)))
            elif kind == "mul":
                gates.append(mul_gate(rng.randrange(fan_in), rng.randrange(fan_in)))
            else:
                k = rng.randrange(2, 5)
                gates.append(
                    xmult_gate(
                        [rng.randrange(fan_in) for _ in range(k)],
                        [rng.randrange(fan_in) for _ in range(k)],
                    )
                )
        return gkr.Layer(gates=gates)

    circuit = gkr.Circuit(num_inputs=n, layers=[layer(64, 64), layer(32, 64)])
    w = [MLE_Variable(f"w{i}") for i in range(6)]
    inputs = [rng.randrange(1, 1000) for _ in range(n)]
    w_in = MLE(variables=w, evaluations=inputs)
    relation = Relation_Circuit(circuit)
    last = relation.layer_tables(w_in)[-1]
    out = MLE(
        variables=[MLE_Variable(f"o{i}") for i in range(5)],
        evaluations=last,
        public=True,
    )
    stmt = Statement(relation, oracles=[w_in], output=out)
    iop = _iop(_IntDomain())
    assert iop.run(stmt)
    last[0] += 1
    bad = MLE(variables=out.variables, evaluations=last, public=True)
    assert not _iop(_IntDomain()).run(Statement(relation, oracles=[w_in], output=bad))
