# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Layered arithmetic circuits as relations (piop.md section 4).

`Relation_CircuitSparseIndex` is circuit satisfiability: the input oracle
evaluates, under the circuit in the relation's `index`, to the output
oracle, its head. The index also holds the circuit's wiring predicates,
built once for a domain by the indexer (`wiring_predicates`, through
`indexed`), as an index is in [CHMMVW20], public sparse tables the verifier
evaluates; `Relation_Circuit` is the same relation with the predicates
committed through one dense index oracle (`relations/sparse.py`). `Relation_Layer`
is gate layer l as the sum definition of the layer's implicit oracle by the
generalized layer identity of Libra and zkCNN [XZZPS19; LXZ21, §4]

    W_{l+1}(z) = sum_{x,y} lin_l(z,x,y) W_l(x) + xmult_l(z,x,y) W_l(x) W_l(y)

over public *sparse* wiring predicates (`SparseMLE`, one nonzero per wire
read: `lin` for the linear summands, supported on y = 0; `xmult` for the
multiplied pairs -- a MUL gate is one pair, an XMULT inner-product gate
several) and the layer below; `Relation_Equal` ties an output oracle to the
output layer. Their protocols are in `protocols/gkr.py`.

Circuits are the protobuf messages of `vfhe.circuit` (imported lazily; the
predicates come from `vfhe.circuit.export.sparse_wiring`). A wiring index
packs `z || x || y` with y in the low bits, so a predicate's MLE variables
are listed `[y..., x..., z...]` and no reordering is needed against the
framework's LSB-first tables. A predicate has variables of its own; a layer
body reads it through a map renaming them to the body's `z`, `x` and `y`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vfhe.arith import Field, MLE_Variable, Ring, SparseMLE
from vfhe.piop.evaluation import with_start
from vfhe.piop.oracle import TableOracle
from vfhe.piop.relation import Relation, Statement, hypercube
from vfhe.piop.virtual import resolve_table, table_like

from .sparse import dense_relation
from .sumdef import Relation_SumDef

if TYPE_CHECKING:
    from vfhe.circuit import gkr


def _domain_kw(domain) -> dict:
    """The MLE constructor keyword for `domain` (a Ring, a Field, or None for
    plain Python values)."""
    if isinstance(domain, Ring):
        return {"ring": domain}
    if isinstance(domain, Field):
        return {"field": domain}
    return {}


def _sparse(domain, variables: list, entries: dict) -> TableOracle:
    """A public sparse predicate over `variables` in `domain`."""
    return TableOracle(
        SparseMLE(variables=variables, evaluations=entries, **_domain_kw(domain)),
        public=True,
    )


def wiring_predicates(circuit: gkr.Circuit, domain) -> tuple[TableOracle, ...]:
    """The indexer's predicates of `circuit` in `domain`: the tuple
    `(lin_0, xmult_0, ..., lin_{d-1}, xmult_{d-1})`, public sparse tables,
    those of layer `l` over variables `[y..., x..., z...]` of their own,
    created here (`s_in` of `y` and of `x`, `s_out` of `z`)."""
    from vfhe.circuit.export import layer_bit_sizes, sparse_wiring

    sizes = layer_bit_sizes(circuit)
    predicates: list[TableOracle] = []
    for layer in range(len(circuit.layers)):
        s_in, s_out = sizes[layer], sizes[layer + 1]
        variables = (
            [MLE_Variable(f"py{layer}_{i}") for i in range(s_in)]
            + [MLE_Variable(f"px{layer}_{i}") for i in range(s_in)]
            + [MLE_Variable(f"pz{layer}_{i}") for i in range(s_out)]
        )
        lin_t, xmult_t = sparse_wiring(circuit, layer)
        predicates.append(_sparse(domain, variables, lin_t))
        predicates.append(_sparse(domain, variables, xmult_t))
    return tuple(predicates)


def _eval_layer(layer, prev: list, size: int) -> list:
    """The padded wire values of one gate layer over the values `prev` of the
    layer below: entry i the gate i, zeros past the last gate up to `size`.
    Domain-agnostic: it uses the entries' own `+` and `*`."""
    from vfhe.circuit import gkr
    from vfhe.circuit.export import gate_pairs

    out = []
    for gate in layer.gates:
        if gate.type == gkr.GATE_TYPE_ADD:
            out.append(prev[gate.left] + prev[gate.right])
        elif gate.type in (gkr.GATE_TYPE_MUL, gkr.GATE_TYPE_XMULT):
            acc = None
            for a, b in gate_pairs(gate):
                t = prev[a] * prev[b]
                acc = t if acc is None else acc + t
            out.append(prev[0] * 0 if acc is None else acc)
        else:
            raise ValueError("unspecified gate type")
    zero = prev[0] * 0
    return out + [zero] * (size - len(out))


@with_start
class Relation_CircuitSparseIndex(Relation):
    """The input oracle evaluates, under the circuit in the index, to the
    output oracle: `oracles = (out, w_in)`, `out` a table over s_out
    variables (public, committed, or plain, or the implicit head of a headed
    statement) and `w_in` one over s_in variables. The index is the circuit
    description and its wiring predicates, `("circuit", "predicates")`, the
    latter the tuple `wiring_predicates` builds -- the sparse index field,
    public tables or restrictions of a master -- which `GKR`'s layer bodies
    read; `indexed(circuit, domain)` is the indexer. The ideal decider
    evaluates the circuit gate by gate over the input table; `compute_head`
    returns its output table, so a circuit can define an implicit oracle of
    a larger reduction."""

    name = "circuit_sp"
    fields = ("oracles",)
    index_fields = ("circuit", "predicates")
    sparse_index_fields = ("predicates",)
    head_index = 0

    @classmethod
    def indexed(cls, circuit: gkr.Circuit, domain):
        """The relation indexed by `circuit`, its predicates built in
        `domain` (a Ring, a Field, or None for plain Python values)."""
        return cls(
            index={"circuit": circuit, "predicates": wiring_predicates(circuit, domain)}
        )

    @property
    def circuit(self) -> gkr.Circuit:
        """The circuit this relation is indexed by."""
        return cast("gkr.Circuit", self.index["circuit"])

    @property
    def predicates(self) -> tuple[TableOracle, ...]:
        """The wiring predicates, `(lin_0, xmult_0, ..., lin_{d-1}, xmult_{d-1})`."""
        return self.index["predicates"]

    def layer_tables(self, table) -> list:
        """Every layer's padded wire values from the input `table` (a dense
        table): entry 0 is the input, entry l + 1 the gates of layer l (zeros
        past the last gate)."""
        from vfhe.circuit.export import layer_bit_sizes

        sizes = layer_bit_sizes(self.circuit)
        if table.num_vars != sizes[0]:
            raise ValueError(
                f"input oracle has {table.num_vars} variables, the circuit needs {sizes[0]}"
            )
        tables = [list(table.table)]
        for layer, s in zip(self.circuit.layers, sizes[1:], strict=True):
            tables.append(_eval_layer(layer, tables[-1], 1 << s))
        return tables

    def check(self, statement: Statement) -> bool:
        out, w_in = statement.oracles
        last = self.layer_tables(resolve_table(w_in))[-1]
        if out.num_vars != (len(last) - 1).bit_length():
            return False
        for b in hypercube(out.variables):
            idx = sum(bit << i for i, bit in enumerate(b[v] for v in out.variables))
            if out.query(b) != last[idx]:
                return False
        return True

    def compute_head(self, statement: Statement):
        out, w_in = statement.oracles
        table = resolve_table(w_in)
        return table_like([table], out.variables, self.layer_tables(table)[-1])


_Dense = dense_relation(Relation_CircuitSparseIndex)
_Dense.name = "circuit"
_Dense.__doc__ = """Circuit satisfiability with the wiring predicates committed
through one dense index oracle: `Relation_CircuitSparseIndex` with its
`"predicates"` replaced by `("dense", "shape", "speval")`, the index
`("circuit", "dense", "shape", "speval")`; `oracles = (out, w_in)` and the
head `out` as there. The decider decodes the predicates once through the
realization and decides the sparse-index relation; `compute_head` is its
(the output table from the input through the circuit). Its protocol is
`compose(dense_layering)`, registered against this class and its `Start`:
`GKR` runs in the first layer over restrictions of the master, the
realization's protocol in the second (`protocols/sparse.py`)."""
Relation_Circuit = with_start(_Dense)


@with_start
class Relation_Layer(Relation_SumDef):
    """Gate layer l of a circuit as a sum definition, indexed by
    `("circuit", "layer", "lin", "xmult")`, the circuit, l and the layer's
    two sparse predicates: `oracles = (W_{l+1}, G_l)` with G_l the layer
    identity over those predicates and the layer below under two renamings
    (`GKR` builds it). `compute_head` evaluates the gates on the table of
    the layer below, linear in the wires, where the generic route would
    materialize the predicates."""

    name = "layer"
    index_fields = ("circuit", "layer", "lin", "xmult")

    def compute_head(self, statement: Statement):
        head, body = statement.oracles
        circuit, layer = self.index["circuit"], self.index["layer"]
        below = resolve_table(body.constituents[2])  # W_l under the x renaming
        values = _eval_layer(
            circuit.layers[layer], list(below.table), 1 << head.num_vars
        )
        return table_like([below], head.variables, values)


class Relation_Equal(Relation):
    """Two oracles over the same number of variables agree on the hypercube,
    `oracles = (a, b)`: the headless statement that ties a circuit's output
    oracle to its output layer."""

    name = "equal"
    fields = ("oracles",)

    def check(self, statement: Statement) -> bool:
        a, b = statement.oracles
        if a.num_vars != b.num_vars:
            return False
        for i in range(1 << a.num_vars):
            pa = {v: (i >> k) & 1 for k, v in enumerate(a.variables)}
            pb = {v: (i >> k) & 1 for k, v in enumerate(b.variables)}
            va = a.query(pa)
            vb = b.query(pb)
            if va != vb:
                return False
        return True
