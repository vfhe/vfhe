# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse index oracles committed through a dense one: `FamilyShape`, the
abstract `Relation_SpEval`, `dense_relation` / `dense_layering`, and a basic
realization of the sparse-evaluation relation that depends on nothing but
this module; the circuits of test_gkr.py through the dense
`Relation_Circuit`, the claim counts on a seven-predicate shape, and the
refusals.

The basic realization. `P_dense` is one dense table over the variables
`B || T || C`: `ell` bits `B` enumerate the nonzeros of a slot (`ell =
max(1, bits(max nnz))`), `T` the shape's slot bits, `C = bits(m)` column
bits for `m = shape.width`. Column 0 holds `val(b, i)`, the value of the
`b`-th nonzero of slot `i` (0 past its last nonzero); column `j + 1` holds
`ind(b, i, j)`, bit `j` of that nonzero's index within its table (0 past the
slot's width). No `ind*`, no packing. The decoder reads the `(b, i)` rows
back and scatters them into the master. The protocol runs once per claim
`P(bin(i), x) = y`: the registered sumcheck over the virtual oracle
`val(., i) * prod_j (ind(., i, j) x_j + (1 - ind(., i, j)) (1 - x_j))`,
degree `m + 1` over `B`, written as a products form over `m + 1`
constituents that are `P_dense` itself read through maps fixing the slot and
column bits; `VirtualEval` then leaves `m + 1` claims on `P_dense`.
"""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring, SparseMLE
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    GKR,
    IOP,
    GKR_Start,
    Implicit,
    OracleKind,
    OutputTest,
    Protocol,
    Relation,
    Relation_Circuit,
    Relation_Equal,
    Relation_Eval,
    Relation_Layer,
    Relation_Sum,
    Statement,
    Sumcheck,
    SumDefStart,
    VirtualEval,
    VirtualOracle,
    compose,
    with_start,
)
from vfhe.piop.circuit import Relation_CircuitSparseIndex, _domain_kw
from vfhe.piop.sparse import (
    FamilyShape,
    Relation_SpEval,
    dense_layering,
    dense_relation,
)
from vfhe.piop.virtual import _one_of, resolve_table


class _IntDomain:
    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random_exceptional(self):
        return self._rng.randrange(2, 1 << 20)


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _bits(n: int) -> int:
    """The number of bits indexing `n` positions (0 for a single one)."""
    return (n - 1).bit_length() if n > 1 else 0


# --- the basic realization -----------------------------------------------------


def _layout(dense, shape):
    """`(ell, c)`: the widths of the `B` and `C` blocks of a dense table."""
    c = shape.width.bit_length()
    return dense.num_vars - len(shape.selectors) - c, c


def _row(ell: int, t: int, b: int, i: int, col: int) -> int:
    return b | (i << ell) | (col << (ell + t))


def basic_encode(shape: FamilyShape, tables: dict, domain) -> MLE:
    """The encoder: the family's tables (the sparse index fields, as the
    inner relation's index holds them; a slot past a field's tuple is
    empty) into one dense table over fresh variables `B || T || C`."""
    members = []
    for name, position, _ in shape.slots:
        entry = tables.get(name)
        if entry is not None and position is not None:
            entry = entry[position] if position < len(entry) else None
        members.append(entry)
    nnz = max((len(m.evaluations) for m in members if m is not None), default=1)
    ell, t, c = max(1, _bits(nnz)), len(shape.selectors), shape.width.bit_length()
    entries = [0] * (1 << (ell + t + c))
    for i, member in enumerate(members):
        if member is None:
            continue
        width = shape.slots[i][2]
        for b, k in enumerate(sorted(member.evaluations)):
            entries[_row(ell, t, b, i, 0)] = member.evaluations[k]
            for j in range(width):
                entries[_row(ell, t, b, i, j + 1)] = (k >> j) & 1
    variables = (
        [MLE_Variable(f"b{i}") for i in range(ell)]
        + [MLE_Variable(f"t{i}") for i in range(t)]
        + [MLE_Variable(f"c{i}") for i in range(c)]
    )
    return MLE(variables=variables, evaluations=entries, **_domain_kw(domain))


@with_start
class BasicSpEval(Relation_SpEval):
    """The basic realization: `compute_head` decodes `P_dense`."""

    name = "basic_speval"

    def compute_head(self, statement):
        dense, shape = self.index["dense"], self.index["shape"]
        ell, _ = _layout(dense, shape)
        t = len(shape.selectors)
        evaluations: dict = {}
        for i, (_, _, width) in enumerate(shape.slots):
            for b in range(1 << ell):
                value = dense.table[_row(ell, t, b, i, 0)]
                if value == 0:
                    continue
                k = 0
                for j in range(width):
                    if dense.table[_row(ell, t, b, i, j + 1)] == 1:
                        k |= 1 << j
                index = shape.index_of(i, k)
                evaluations[index] = (
                    value if index not in evaluations else evaluations[index] + value
                )
        return SparseMLE(
            variables=self.index["sparse"].variables,
            evaluations=evaluations,
            ring=dense.ring,
            field=dense.field,
        )


class BasicSpEvalProtocol(Protocol):
    """The composable protocol of `BasicSpEval.Start`: one Sum claim per
    claim on the master, handed back for the registered sumcheck and
    `VirtualEval`. It never queries the master, draws nothing and reports
    no error of its own; any number of claims per slot."""

    name = "basic_speval"
    reduces_from = BasicSpEval.Start
    reduces_to = (Relation_Sum,)

    def soundness_error(self, statement, domain):
        return 0

    async def prove(self, prover, statements):
        return self._sums(statements)

    async def verify(self, verifier, statements):
        return self._sums(statements)

    def _sums(self, statements):
        (s,) = statements
        dense, shape = s.index["dense"], s.index["shape"]
        ell, _ = _layout(dense, shape)
        t, m = len(shape.selectors), shape.width
        bs, ts, cs = (
            dense.variables[:ell],
            dense.variables[ell : ell + t],
            dense.variables[ell + t :],
        )
        one = _one_of([dense])

        def constituent_map(i: int, col: int) -> dict:
            return {
                **{b: b for b in bs},
                **{tv: (i >> j) & 1 for j, tv in enumerate(ts)},
                **{cv: (col >> j) & 1 for j, cv in enumerate(cs)},
            }

        out = []
        for point, value in zip(s.points, s.values, strict=True):
            i = 0
            for j, tv in enumerate(shape.selectors):
                bit = point[tv]
                if not (bit == 0 or bit == 1):
                    raise ValueError("a claim on the master has its slot on the cube")
                i |= (1 if bit == 1 else 0) << j
            xs = [point[xv] for xv in shape.xs]
            xs = [one * x if isinstance(x, int) else x for x in xs]
            maps = [constituent_map(i, col) for col in range(m + 1)]
            terms = []
            for mask in range(1 << m):
                coeff = one
                for j, x in enumerate(xs):
                    coeff = coeff * ((x + x - one) if (mask >> j) & 1 else (one - x))
                terms.append(
                    (coeff, (0, *(j + 1 for j in range(m) if (mask >> j) & 1)))
                )
            G = VirtualOracle(list(bs), [dense] * (m + 1), terms, maps)
            out.append(self.reduce([s], oracles=[G], value=value))
        return out


# --- the circuits of test_gkr.py, through the dense relation --------------------


def _circuit():
    return gkr.Circuit(
        num_inputs=4,
        layers=[
            gkr.Layer(gates=[add_gate(0, 1), mul_gate(2, 3)]),
            gkr.Layer(gates=[add_gate(0, 1)]),
        ],
    )


def _dense_circuit(circuit, domain, shape=None, speval=BasicSpEval):
    """The dense circuit relation of `circuit`: the predicates indexed in
    `domain`, encoded by the basic encoder into a plain table."""
    sp = Relation_CircuitSparseIndex.indexed(circuit, domain)
    shape = shape or FamilyShape.of(sp.index, sp.sparse_index_fields)
    dense = basic_encode(shape, sp.index, domain)
    return Relation_Circuit(
        index={"circuit": circuit, "dense": dense, "shape": shape, "speval": speval}
    )


def _iop(domain, speval_protocol=None, **kw):
    iop = IOP(domain=domain, **kw)
    dense = compose(dense_layering, "dense")
    iop.register(Relation_Circuit, dense)
    iop.register(Relation_Circuit.Start, dense)
    iop.register(Relation_CircuitSparseIndex, GKR())
    iop.register(Relation_CircuitSparseIndex.Start, GKR_Start())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(BasicSpEval.Start, speval_protocol or BasicSpEvalProtocol())
    return iop


def _instance(ring=None, inputs=(1, 2, 3, 4), output=None, public_output=True, **kw):
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    o = [MLE_Variable("o0")]
    a, b, c, d = inputs
    out = [a + b + c * d, 0] if output is None else output
    w_in = MLE(ring=ring, variables=w, evaluations=list(inputs))
    out_mle = MLE(ring=ring, variables=o, evaluations=out, public=public_output)
    return Statement(_dense_circuit(_circuit(), ring, **kw), oracles=(out_mle, w_in))


class _Recording(BasicSpEvalProtocol):
    """Records the started statements it is handed."""

    def __init__(self):
        self.started = []

    def _sums(self, statements):
        self.started.extend(statements)
        return super()._sums(statements)


def _slots_claimed(started, shape) -> list[int]:
    """The slot of every claim of a started master statement."""
    return sorted(
        sum(point[tv] << j for j, tv in enumerate(shape.selectors))
        for point in started.points
    )


# --- FamilyShape, Relation_SpEval, the encoder ---------------------------------


def test_family_shape():
    shape = FamilyShape({"p": (5, 3), "q": 4})
    assert shape.count == 3 and shape.width == 5
    assert shape.slots == [("p", 0, 5), ("p", 1, 3), ("q", None, 4)]
    assert len(shape.selectors) == 2 and len(shape.xs) == 5
    P = shape.implicit()
    assert P.variables == shape.selectors + shape.xs
    assert not hasattr(shape, "variables")  # data, not an oracle
    assert shape.index_of(2, 5) == 2 | (5 << 2)
    # Member 1: T = bin(1), the two padded coordinates fixed to 0, the three
    # others renamed to variables of its own.
    r1 = shape.restriction(P, 1)
    assert r1.is_restriction() and len(r1.variables) == 3
    (m,) = r1.maps
    assert [m[tv] for tv in shape.selectors] == [1, 0]
    assert [m[x] for x in shape.xs] == [*r1.variables, 0, 0]
    placed = shape.place(P)
    assert set(placed) == {"p", "q"}
    assert isinstance(placed["p"], tuple) and len(placed["p"]) == 2
    assert placed["q"].is_restriction() and len(placed["q"].variables) == 4
    assert shape.digest() == FamilyShape({"p": (5, 3), "q": 4}).digest()
    assert shape.digest() != FamilyShape({"p": (5, 3), "q": 3}).digest()
    assert len(FamilyShape({"q": 2}).selectors) == 0  # one slot: no T
    with pytest.raises(ValueError):
        shape.restriction(P, 3)
    with pytest.raises(ValueError):
        FamilyShape({})
    # From an index holding the tables.
    sp = Relation_CircuitSparseIndex.indexed(_circuit(), None)
    of = FamilyShape.of(sp.index, sp.sparse_index_fields)
    assert of.widths == {"predicates": (5, 5, 3, 3)} and of.count == 4


def test_basic_realization_decodes_what_it_encodes():
    sp = Relation_CircuitSparseIndex.indexed(_circuit(), None)
    shape = FamilyShape.of(sp.index, sp.sparse_index_fields)
    dense = basic_encode(shape, sp.index, None)
    assert dense.num_vars == 1 + 2 + 3  # ell = 1 (two nonzeros at most), t, c
    P = shape.implicit()
    relation = BasicSpEval(index={"sparse": P, "dense": dense, "shape": shape})
    master = relation.compute_head(Statement(relation))
    assert isinstance(master, SparseMLE) and master.variables == P.variables
    for i, predicate in enumerate(sp.index["predicates"]):
        sliced = resolve_table(shape.restriction(master, i))
        assert sliced.evaluations == predicate.evaluations
    # The generic decider, over a table in the sparse entry.
    ideal = BasicSpEval(index={"sparse": master, "dense": dense, "shape": shape})
    assert Statement(ideal).check() and Statement(ideal).head is None
    wrong = master.copy()
    wrong.evaluations[0] = 7
    bad = BasicSpEval(index={"sparse": wrong, "dense": dense, "shape": shape})
    assert not Statement(bad).check()
    assert Relation_SpEval.fields == () and Relation_SpEval.head_index == "sparse"
    with pytest.raises(NotImplementedError):
        Relation_SpEval(
            index={"sparse": P, "dense": dense, "shape": shape}
        ).compute_head(Statement(relation))
    # An empty slot decodes to nothing.
    five = FamilyShape({"predicates": (5, 5, 3, 3, 3)})
    dense5 = basic_encode(five, sp.index, None)
    master5 = BasicSpEval(
        index={"sparse": five.implicit(), "dense": dense5, "shape": five}
    ).compute_head(Statement(relation))
    assert resolve_table(five.restriction(master5, 4)).evaluations == {}


def test_dense_circuit_relation():
    assert Relation_Circuit.inner is Relation_CircuitSparseIndex
    assert Relation_Circuit.Start.inner is Relation_CircuitSparseIndex
    assert Relation_Circuit.index_fields == ("circuit", "dense", "shape", "speval")
    assert Relation_Circuit.fields == ("oracles",) and Relation_Circuit.head_index == 0
    assert Relation_Circuit.__name__ == "Relation_Circuit"
    assert Relation_Circuit.name == "circuit" and Relation_Circuit.Start.name == (
        "circuit_start"
    )
    stmt = _instance()
    assert stmt.check()
    assert not _instance(output=[16, 0]).check()
    out, w_in = stmt.oracles
    relation = stmt.relation
    assert relation.index_oracles() == (relation.index["dense"],)
    head = Implicit(out.variables)
    headed = Statement(relation, oracles=(head, w_in), head=head)
    assert list(relation.compute_head(headed).table) == [15, 0]
    with pytest.raises(TypeError, match="sparse_index_fields"):
        dense_relation(Relation_Sum)


# --- (b) the two-layer circuit and the three circuits -------------------------


def test_dense_circuit_ideal_model():
    terminal = []

    class _Record(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    recording = _Recording()
    iop = _iop(_IntDomain(), speval_protocol=recording)
    iop.register(Relation_Eval, _Record())
    stmt = _instance()
    assert iop.run(stmt)
    _, w_in = stmt.oracles
    dense, shape = stmt.index["dense"], stmt.index["shape"]
    on_input = [s for s in terminal if s.oracles[0] is w_in]
    on_dense = [s for s in terminal if s.oracles[0] is dense]
    assert len(on_input) == 2 and len(terminal) == len(on_input) + len(on_dense)
    assert all(s.check() for s in terminal)
    # The master was started once, per party, with the claims of every
    # layer: one per predicate of the top layer, two per predicate below.
    assert len(recording.started) == 2
    started = recording.started[0]
    assert _slots_claimed(started, shape) == [0, 0, 1, 1, 2, 3]
    assert len(on_dense) == len(started.points) * (shape.width + 1)
    assert started.check()  # the ideal decider of the started master
    order = iop.transcript.order
    assert order[0].startswith("equal/")  # GKR's output test, in L_0
    assert any(k.startswith("sumcheck/0/1/") for k in order)  # the master's sumchecks
    # The prover's heads: the two layers and the master.
    kinds = sorted(type(t).__name__ for t in iop.prover.witnesses.values())
    assert kinds == ["MLE", "MLE", "SparseMLE"]
    # The same cheats as through the sparse-index relation, rejected once
    # each: the integer domain is seeded, so the runs are deterministic.
    assert not _iop(_IntDomain()).run(_instance(output=[16, 0]))
    assert not _iop(_IntDomain(1)).run(_instance(public_output=False, output=[16, 0]))
    assert _iop(_IntDomain(2)).run(_instance(public_output=False))
    wide = MLE(variables=_vars("o0", "o1"), evaluations=[15, 0, 0, 0], public=True)
    with pytest.raises(ValueError, match="output"):
        _iop(_IntDomain()).run(Statement(stmt.relation, oracles=(wide, w_in)))


def test_dense_circuit_with_an_empty_slot():
    # A shape with a fifth slot no layer reads: encoded empty, never claimed.
    five = FamilyShape({"predicates": (5, 5, 3, 3, 3)})
    recording = _Recording()
    iop = _iop(_IntDomain(), speval_protocol=recording)
    stmt = _instance(shape=five)
    assert stmt.check()
    assert iop.run(stmt)
    assert _slots_claimed(recording.started[0], five) == [0, 0, 1, 1, 2, 3]
    assert not _iop(_IntDomain()).run(_instance(shape=five, output=[16, 0]))


@pytest.mark.parametrize("public_output", [True, False])
def test_dense_circuit_fiat_shamir(public_output):
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _instance(ring=ring, public_output=public_output)
    assert stmt.check()
    proof = _iop(ring, fiat_shamir=True).prove(stmt)
    assert _iop(ring, fiat_shamir=True).verify(stmt, proof)
    assert proof.digest() == _iop(ring, fiat_shamir=True).prove(stmt).digest()
    bad = _instance(ring=ring, output=[16, 0], public_output=public_output)
    assert not _iop(ring, fiat_shamir=True).verify(bad, proof)
    assert not _iop(ring, fiat_shamir=True).run(bad)


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
    """y = C0(C1(C2(x))), each circuit through its dense relation."""

    name = "three_circuits"
    fields = ("oracles",)

    def check(self, statement):
        y, x = statement.oracles
        return list(y.table) == _chain_output(list(x.table))


def _three_layering(domain):
    from vfhe.circuit.export import layer_bit_sizes

    c0, c1, c2 = (_dense_circuit(c, domain) for c in (_C0, _C1, _C2))

    def layering(root):
        y, x = root.oracles
        m1 = Implicit(_vars(*[f"m1_{i}" for i in range(layer_bit_sizes(_C1)[-1])]))
        m2 = Implicit(_vars(*[f"m2_{i}" for i in range(layer_bit_sizes(_C2)[-1])]))
        return [
            [root.reduce_to(c0, oracles=(y, m1))],
            [root.reduce_to(c1, oracles=(m1, m2), head=m1)],
            [root.reduce_to(c2, oracles=(m2, x), head=m2)],
        ]

    return layering


def _three_instance(ring=None, tweak=0, public=True):
    x = MLE(ring=ring, variables=_vars("x0", "x1"), evaluations=[1, 2, 3, 4])
    out = _chain_output([1, 2, 3, 4])
    out[0] += tweak
    y = MLE(ring=ring, variables=_vars("y0"), evaluations=out, public=public)
    return Statement(_ThreeCircuits(), oracles=(y, x)), compose(
        _three_layering(ring), "three"
    )


@pytest.mark.parametrize("public", [True, False])
def test_three_circuits_through_the_dense_relation(public):
    # C0 headless, C1 and C2 started (the started-root path of the dense
    # layering); the claims each GKR leaves on its master return through the
    # nested compositions to the Composer that heads it.
    stmt, protocol = _three_instance(public=public)
    assert stmt.check()
    recording = _Recording()
    iop = _iop(_IntDomain(), speval_protocol=recording)
    iop.register(_ThreeCircuits, protocol)
    assert iop.run(stmt)
    heads = sorted(
        list(t.table)
        for o, t in iop.prover.witnesses.items()
        if isinstance(o, Implicit) and isinstance(t, MLE)
    )
    assert [3, 12] in heads and [36, 15] in heads and [51, 1296] in heads
    masters = [t for t in iop.prover.witnesses.values() if isinstance(t, SparseMLE)]
    assert len(masters) == 3  # one per circuit
    assert len(recording.started) == 6  # three per party
    bad, protocol = _three_instance(tweak=1, public=public)
    iop = _iop(_IntDomain(1))
    iop.register(_ThreeCircuits, protocol)
    assert not iop.run(bad)


def test_three_circuits_dense_fiat_shamir():
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


# --- (c) claim counts on a seven-predicate shape ---------------------------------


class _SevenSparseIndex(Relation):
    """A relation with seven sparse index tables and no circuit, whose
    protocol queries each once at the statement's coordinates."""

    name = "seven_sp"
    fields = ("coords",)
    index_fields = ("preds",)
    sparse_index_fields = ("preds",)

    def check(self, statement):
        return True


class _QuerySeven(Protocol):
    name = "seven"
    reduces_from = _SevenSparseIndex
    reduces_to = (Relation_Eval,)

    def _points(self, s):
        return [dict(zip(p.variables, s.coords, strict=True)) for p in s.index["preds"]]

    async def prove(self, prover, statements):
        (s,) = statements
        values = tuple(
            resolve_table(p, prover.witnesses).evaluate(pt, in_place=False).constant()
            for p, pt in zip(s.index["preds"], self._points(s), strict=True)
        )
        assert prover.iop is not None
        prover.iop.transcript.write(f"{self.name}{s.path}/vals", values)
        return self._claims(s, values)

    async def verify(self, verifier, statements):
        (s,) = statements
        assert verifier.iop is not None
        values = await verifier.iop.transcript.read(f"{self.name}{s.path}/vals")
        return self._claims(s, values)

    def _claims(self, s, values):
        return [
            self.reduce([s], oracles=[p], point=pt, value=v)
            for p, pt, v in zip(s.index["preds"], self._points(s), values, strict=True)
        ]


_Seven = dense_relation(_SevenSparseIndex)


def test_seven_predicates_make_seven_claims_on_one_master():
    rng = random.Random(3)
    m = 3
    preds = tuple(
        SparseMLE(
            variables=_vars(*[f"q{i}_{k}" for k in range(m)]),
            evaluations={k: rng.randrange(1, 9) for k in rng.sample(range(8), 3)},
        )
        for i in range(7)
    )
    shape = FamilyShape({"preds": (m,) * 7})
    dense = basic_encode(shape, {"preds": preds}, None)
    root = Statement(
        _Seven(index={"dense": dense, "shape": shape, "speval": BasicSpEval}),
        coords=[2, 3, 5],
    )
    assert root.check()
    sizes = []

    class _Bundle(Protocol):
        batching = True

        async def prove(self, prover, statements):
            sizes.append(len(statements))
            return []

        async def verify(self, verifier, statements):
            sizes.append(len(statements))
            assert all(s.oracles[0] is dense and s.check() for s in statements)
            return []

    recording = _Recording()
    iop = _iop(_IntDomain(), speval_protocol=recording)
    iop.register(_Seven, compose(dense_layering, "dense"))
    iop.register(_SevenSparseIndex, _QuerySeven())
    iop.register(Relation_Eval, _Bundle())
    assert iop.run(root)
    # One started master statement per party, with the seven claims at
    # T = bin(0), ..., bin(6); 7 (m + 1) claims on the dense table, bundled.
    assert len(recording.started) == 2
    assert _slots_claimed(recording.started[0], shape) == list(range(7))
    assert sizes == [7 * (m + 1)] * 2
    (master,) = [t for t in iop.prover.witnesses.values() if isinstance(t, SparseMLE)]
    for i, p in enumerate(preds):
        assert resolve_table(shape.restriction(master, i)).evaluations == p.evaluations


# --- (d) refusals ----------------------------------------------------------------


def test_refusals():
    stmt = _instance()
    out, w_in = stmt.oracles
    index = dict(stmt.index)
    # A layering without L_1: nothing computes the master's table, so the
    # prover's head pass fails on the first layer body reading it ...
    forgetful = compose(lambda root: dense_layering(root)[:1], "forgetful")
    iop = _iop(_IntDomain())
    iop.register(Relation_Circuit, forgetful)
    with pytest.raises(RuntimeError, match="no table yet"):
        iop.run(stmt)
    # ... and with the table supplied by hand, the claims on the master head
    # no statement and reach the driver, which refuses them.
    master = stmt.relation.decode()  # pyright: ignore[reportAttributeAccessIssue]
    iop = _iop(_IntDomain())

    def leaking(root):
        layers = dense_layering(root)
        iop.prover.witnesses[layers[1][0].head] = master
        return layers[:1]

    iop.register(Relation_Circuit, compose(leaking, "leaking"))
    with pytest.raises(RuntimeError, match="implicit oracle"):
        iop.run(stmt)
    # The dense table among the instance fields of the sparse-evaluation
    # relation: it has none.
    realization = BasicSpEval(
        index={
            "sparse": index["shape"].implicit(),
            **{k: index[k] for k in ("dense", "shape")},
        }
    )
    with pytest.raises(TypeError, match="no field"):
        Statement(realization, oracles=(index["dense"],))
    # An implicit oracle in a root's index that is not its head.
    with pytest.raises(TypeError, match="implicit"):
        _iop(_IntDomain()).run(
            Statement(
                Relation_Circuit(
                    index={**index, "dense": Implicit(index["dense"].variables)}
                ),
                oracles=(out, w_in),
            )
        )

    # A realization not decorated with `with_start`: the Composer cannot start it.
    class _Bare(Relation_SpEval):
        name = "bare"
        compute_head = BasicSpEval.compute_head

    with pytest.raises(NotImplementedError, match="start relation"):
        _iop(_IntDomain()).run(_instance(speval=_Bare))

    # Decorated, but its Start unregistered: refused at the Composer's discharge.
    @with_start
    class _Unregistered(Relation_SpEval):
        name = "unregistered"
        compute_head = BasicSpEval.compute_head

    with pytest.raises(RuntimeError, match="no protocol registered"):
        _iop(_IntDomain()).run(_instance(speval=_Unregistered))
