# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Relation_CircuitSparseIndex and the GKR protocol over vfhe.circuit's protobuf
circuits: the ideal model over plain tables, the ring domain under
Fiat-Shamir, the shape of the reduction (claims per layer), and the
message sequence, pinned."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    IOP,
    ImplicitOracle,
    OracleKind,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    TableOracle,
)
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
    iop.register(Relation_CircuitSparseIndex, GKR())
    iop.register(Relation_CircuitSparseIndex.Start, GKR_Start())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _instance(ring=None, inputs=(1, 2, 3, 4), output=None, public_output=True):
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    o = [MLE_Variable("o0")]
    a, b, c, d = inputs
    out = [a + b + c * d, 0] if output is None else output
    w_in = TableOracle(MLE(ring=ring, variables=w, evaluations=list(inputs)))
    out_mle = TableOracle(
        MLE(ring=ring, variables=o, evaluations=out), public=public_output
    )
    return Statement(
        Relation_CircuitSparseIndex.indexed(_circuit(), ring), oracles=(out_mle, w_in)
    )


def test_relation_circuit_decider():
    assert _instance().check()
    assert not _instance(output=[16, 0]).check()
    out, w_in = _instance().oracles
    relation = Relation_CircuitSparseIndex.indexed(_circuit(), None)
    assert relation.layer_tables(w_in.polynomial) == [[1, 2, 3, 4], [3, 12], [15, 0]]
    # The output as the head of a statement: computed from the input.
    head = ImplicitOracle(out.variables)
    headed = Statement(relation, oracles=(head, w_in), head=head)
    assert list(relation.compute_head(headed).table) == [15, 0]


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
    _, w_in = stmt.oracles
    assert len(terminal) == 2 and all(s.oracles[0] is w_in for s in terminal)
    assert all(s.check() for s in terminal)
    order = iop.transcript.order
    assert order[0] == "equal/0/r0"  # the output test's point, one challenge
    # layer 1 (one claim, no RLC) then layer 0 (two claims, two RLC challenges)
    assert order[1] == "sumcheck/1/0/0/g0"
    assert "sumdef/2/0/a0" in order and "sumdef/2/0/a1" in order
    # The prover computed and stored both layers' tables.
    heads = sorted(
        list(t.table)
        for o, t in iop.prover.witnesses.items()
        if isinstance(o, ImplicitOracle)
    )
    assert heads == [[3, 12], [15, 0]]
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
    assert "equal/0/out" in iop.transcript.order
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
    inputs = MLE(variables=w, evaluations=[1, 2, 3, 4])
    w_in = TableOracle(inputs)
    assert Relation_CircuitSparseIndex.indexed(xm, None).layer_tables(inputs) == [
        [1, 2, 3, 4],
        [11, 0],
    ]
    assert Relation_CircuitSparseIndex.indexed(expanded, None).layer_tables(inputs)[
        -1
    ] == [11, 0]
    out = TableOracle(
        MLE(variables=[MLE_Variable("o0")], evaluations=[11, 0]), public=True
    )
    stmt = Statement(Relation_CircuitSparseIndex.indexed(xm, None), oracles=(out, w_in))
    assert stmt.check()
    iop = _iop(_IntDomain())
    assert iop.run(stmt)
    # One layer, one sumcheck over (x, y): 4 rounds, all of degree 2.
    rounds = [k for k in iop.transcript.order if k.startswith("sumcheck") and "/g" in k]
    assert len(rounds) == 4
    assert all(len(iop.transcript.entries[k].result()) == 3 for k in rounds)
    bad = TableOracle(
        MLE(variables=[MLE_Variable("o0")], evaluations=[12, 0]), public=True
    )
    assert not _iop(_IntDomain()).run(
        Statement(Relation_CircuitSparseIndex.indexed(xm, None), oracles=(bad, w_in))
    )


def test_gkr_dense_and_libra_strategies_agree(monkeypatch):
    from vfhe.piop.protocols.sumcheck.prover_view import _LibraProver, prover_view

    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _instance(ring=ring)
    # The deepest layer's statement: its body reads the input table itself.
    *_, bottom = GKR().layers(stmt._fork(), _iop(ring))
    head, body = bottom[0].oracles
    (z,) = head.variables
    instantiated = body.evaluate({z: ring.random_exceptional()}, in_place=False)
    assert isinstance(prover_view(instantiated, {}), _LibraProver)

    digests = {}
    for strategy in ("dense", "libra"):
        monkeypatch.setattr(Sumcheck, "strategy", strategy)
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
    table = MLE(variables=w, evaluations=inputs)
    w_in = TableOracle(table)
    relation = Relation_CircuitSparseIndex.indexed(circuit, None)
    last = relation.layer_tables(table)[-1]
    out = TableOracle(
        MLE(variables=[MLE_Variable(f"o{i}") for i in range(5)], evaluations=last),
        public=True,
    )
    stmt = Statement(relation, oracles=(out, w_in))
    iop = _iop(_IntDomain())
    assert iop.run(stmt)
    last[0] += 1
    bad = TableOracle(MLE(variables=out.variables, evaluations=last), public=True)
    assert not _iop(_IntDomain()).run(Statement(relation, oracles=(bad, w_in)))


# The transcript values of the two-layer circuit under _IntDomain(0), with
# a public and with a plain output: what the protocol sent before it became
# a composition (labels changed, values did not).
_MESSAGES = [
    885442,
    (-2656323, -10625292, -18594261),
    403960,
    (-3219147373563, 0, 3219147373563),
    794774,
    (3635643, 7152969),
    933490,
    441003,
    (8003495100435, -1455177928458, -16734573667127),
    42452,
    (-30888334298413737, -5245219721000459968040, -10490902741755771067379),
    271495,
    (
        115646090843763747649818,
        -19637278236501905948295225080,
        -49093311237345608634485712518,
    ),
    1012365,
    (
        -117075939110956042625760349752,
        -5031492543592486215067923393532399106400,
        -10063004967161078387276906090334575977248,
    ),
    536112,
    (585443, 2084590),
]


@pytest.mark.parametrize("public_output", [True, False])
def test_gkr_message_sequence_is_pinned(public_output):
    iop = _iop(_IntDomain(0))
    assert iop.run(_instance(public_output=public_output))
    t = iop.transcript
    values = [t.entries[label].result() for label in t.order]
    expected = _MESSAGES if public_output else [_MESSAGES[0], -13281615, *_MESSAGES[1:]]
    assert values == expected


def test_gkr_start_proves_a_started_circuit():
    # A circuit statement started by claims on its output: the composable
    # form, as a larger reduction would use it. Here the started statement
    # is the root and the driver decides the claims on the input.
    stmt = _instance()
    out, w_in = stmt.oracles
    head = ImplicitOracle(out.variables)
    headed = Statement(
        Relation_CircuitSparseIndex.indexed(_circuit(), None),
        oracles=(head, w_in),
        head=head,
    )
    (o,) = out.variables
    claims = [
        Statement(Relation_Eval(), oracles=(head,), point={o: r}, value=(1 - r) * 15)
        for r in (3, 7)
    ]
    started = headed.relation.start(headed, claims)
    assert started.check()
    iop = _iop(_IntDomain())
    assert iop.run(started)
    assert iop.transcript.order[0] == "sumdef/0/0/0/a0"  # the two claims combined
    wrong = Statement(Relation_Eval(), oracles=(head,), point={o: 3}, value=1)
    bad = headed.relation.start(headed, [wrong, claims[1]])
    assert not bad.check()
    assert not _iop(_IntDomain(1)).run(bad)


def test_gkr_rejects_an_output_oracle_of_the_wrong_size():
    # check() refuses a 2-variable output for a 1-variable output layer, and
    # so must the protocol (the layer identity would silently pad).
    _, w_in = _instance().oracles
    out = TableOracle(
        MLE(variables=_vars("o0", "o1"), evaluations=[15, 0, 0, 0]), public=True
    )
    stmt = Statement(
        Relation_CircuitSparseIndex.indexed(_circuit(), None), oracles=(out, w_in)
    )
    assert not stmt.check()
    with pytest.raises(ValueError, match="output"):
        _iop(_IntDomain()).run(stmt)


def test_layer_compute_head_matches_the_generic_route():
    from vfhe.piop.relations import Relation_SumDef

    stmt = _instance()
    *_, bottom = GKR().layers(stmt._fork(), _iop(_IntDomain()))
    (layer0,) = bottom
    fast = layer0.relation.compute_head(layer0)
    generic = Relation_SumDef().compute_head(layer0)
    assert list(fast.table) == list(generic.table) == [3, 12]
    assert layer0.head is not None
    assert fast.variables == generic.variables == layer0.head.variables


def test_output_test_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _instance(ring=ring)
    equal, *_ = GKR().layers(stmt._fork(), _iop(ring))
    assert OutputTest().soundness_error(equal[0], ring) == 1 / ring.exceptional_set_size
    assert GKR().soundness_error(stmt, ring) == 1 / ring.exceptional_set_size
    assert GKR_Start().soundness_error(stmt, ring) == 0


# --- example (a) of the proposal: a circuit composed of three circuits -----

_C2 = gkr.Circuit(
    num_inputs=4, layers=[gkr.Layer(gates=[add_gate(0, 1), mul_gate(2, 3)])]
)
_C1 = gkr.Circuit(
    num_inputs=2,
    layers=[
        gkr.Layer(gates=[mul_gate(0, 1), add_gate(0, 1)]),
        gkr.Layer(gates=[add_gate(0, 1), mul_gate(0, 0)]),
    ],
)
_C0 = gkr.Circuit(num_inputs=2, layers=[gkr.Layer(gates=[mul_gate(0, 1)])])


def _chain_output(values):
    """y = C0(C1(C2(x))) on the table `values`."""
    from vfhe.circuit.export import layer_bit_sizes

    for c in (_C2, _C1, _C0):
        m = MLE(
            variables=_vars(*[f"t{i}" for i in range(layer_bit_sizes(c)[0])]),
            evaluations=values,
        )
        values = Relation_CircuitSparseIndex.indexed(c, None).layer_tables(m)[-1]
    return list(values)


class _ThreeCircuits(Relation):
    """y = C0(C1(C2(x))), with C0, C1, C2 headed at their outputs in turn."""

    name = "three_circuits"
    fields = ("oracles",)

    def check(self, statement):
        y, x = statement.oracles
        return list(y.polynomial.table) == _chain_output(list(x.polynomial.table))


def _three_layering(domain):
    """The layering, indexing the three circuits in `domain`."""
    from vfhe.circuit.export import layer_bit_sizes

    c0, c1, c2 = (
        Relation_CircuitSparseIndex.indexed(c, domain) for c in (_C0, _C1, _C2)
    )

    def layering(root):
        y, x = root.oracles
        m1 = ImplicitOracle(
            _vars(*[f"m1_{i}" for i in range(layer_bit_sizes(_C1)[-1])])
        )
        m2 = ImplicitOracle(
            _vars(*[f"m2_{i}" for i in range(layer_bit_sizes(_C2)[-1])])
        )
        return [
            [root.reduce_to(c0, oracles=(y, m1))],
            [root.reduce_to(c1, oracles=(m1, m2), head=m1)],
            [root.reduce_to(c2, oracles=(m2, x), head=m2)],
        ]

    return layering


def _three_instance(ring=None, tweak=0, public=True):
    from vfhe.piop import compose

    x = TableOracle(
        MLE(ring=ring, variables=_vars("x0", "x1"), evaluations=[1, 2, 3, 4])
    )
    out = _chain_output([1, 2, 3, 4])
    out[0] += tweak
    y = TableOracle(
        MLE(ring=ring, variables=_vars("y0"), evaluations=out), public=public
    )
    return Statement(_ThreeCircuits(), oracles=(y, x)), compose(
        _three_layering(ring), "three"
    )


@pytest.mark.parametrize("public", [True, False])
def test_three_circuits_compose(public):
    # C0 headless (GKR with its output test), C1 and C2 started (GKR_Start),
    # the intermediate oracles m1 = C1(m2), m2 = C2(x) implicit.
    stmt, protocol = _three_instance(public=public)
    assert stmt.check()
    iop = _iop(_IntDomain())
    iop.register(_ThreeCircuits, protocol)
    assert iop.run(stmt)
    heads = sorted(
        list(t.table)
        for o, t in iop.prover.witnesses.items()
        if isinstance(o, ImplicitOracle)
    )
    # m2 = C2(x) = [3, 12]; C1's two layers: [36, 15] then [51, 1296]; C0's
    # output layer [66096, 0]; the intermediate GKR layers are heads too.
    assert [3, 12] in heads and [36, 15] in heads and [51, 1296] in heads
    bad, protocol = _three_instance(tweak=1, public=public)
    iop = _iop(_IntDomain(1))
    iop.register(_ThreeCircuits, protocol)
    assert not iop.run(bad)


def test_three_circuits_fiat_shamir():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt, protocol = _three_instance(ring)

    def iop():
        fs = _iop(ring, fiat_shamir=True)
        fs.register(_ThreeCircuits, protocol)
        return fs

    proof = iop().prove(stmt)
    assert iop().verify(stmt, proof)
    bad, _ = _three_instance(ring, tweak=1)
    assert not iop().verify(bad, proof)
