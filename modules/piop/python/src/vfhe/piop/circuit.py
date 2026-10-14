# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Layered arithmetic circuits as a relation, proven GKR-style (piop.md §4).

`Relation_Circuit` is circuit satisfiability: the input oracle evaluates,
under the circuit in the relation's `index`, to the output oracle, its head.
`GKR` is its protocol [GKR15; Tha22, §4.6], a composition (`composer.py`):
the *output test*, a headless statement `Relation_Equal(out, W_d)` whose
protocol `OutputTest` reduces "the output oracle equals the output layer" to
one evaluation claim on the layer at a random point; then, one layer per
gate layer, the sum definition `Relation_Layer` of the layer's implicit
oracle by the generalized layer identity of Libra and zkCNN [XZZPS19; LXZ21,
§4]

    W_{l+1}(z) = sum_{x,y} lin_l(z,x,y) W_l(x) + xmult_l(z,x,y) W_l(x) W_l(y)

over public *sparse* wiring predicates (`SparseMLE`, one nonzero per wire
read: `lin` for the linear summands, supported on y = 0; `xmult` for the
multiplied pairs — a MUL gate is one pair, an XMULT inner-product gate
several) and the layer below. The Composer computes every layer's table
(`Relation_Layer.compute_head`, the gates evaluated on the layer below) and,
layer by layer, starts each definition with the claims the layer above left
on it; `SumDefStart` combines them, and the registered sumcheck and
`VirtualEval` reduce them to claims on the layer below (two per layer, on
one oracle, at different points), down to the input oracle, whose claims are
decided by its own kind (public, committed, or plain). `GKR_Start` is the same composition without
the output test: the sequentially composable form, for a circuit statement
started by claims on its output. The prover's rounds run on the sparse
predicates through the Libra two-phase strategy of `virtual.py` (linear in
the number of wires plus the layer widths), which the dense strategy
reproduces message for message.

Circuits are the protobuf messages of `vfhe.circuit` (imported lazily; the
predicates come from `vfhe.circuit.export.sparse_wiring`). A wiring index
packs `z || x || y` with y in the low bits, so a predicate's MLE variables
are listed `[y..., x..., z...]` and no reordering is needed against the
framework's LSB-first tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vfhe.arith import Field, MLE_Variable, Ring, SparseMLE

from .composer import Composed
from .piop import (
    IOP,
    Implicit,
    Protocol,
    Prover,
    Relation,
    Relation_Eval,
    Statement,
    Verifier,
    _constant,
    _hypercube,
    with_start,
)
from .virtual import Relation_SumDef, VirtualOracle, _table_like, resolve_table

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


def _sparse(domain, variables: list, entries: dict) -> SparseMLE:
    """A public sparse predicate over `variables` in `domain`."""
    return SparseMLE(
        variables=variables, evaluations=entries, public=True, **_domain_kw(domain)
    )


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
class Relation_Circuit(Relation):
    """The input oracle evaluates, under the circuit `index`, to the output
    oracle: `oracles = (out, w_in)`, `out` a table over s_out variables
    (public, committed, or plain, or the implicit head of a headed
    statement) and `w_in` one over s_in variables. The ideal decider
    evaluates the circuit gate by gate over the input table; `compute_head`
    returns its output table, so a circuit can define an implicit oracle of
    a larger reduction."""

    name = "circuit"
    fields = ("oracles",)
    head_index = 0

    def __init__(self, index: gkr.Circuit) -> None:
        super().__init__(index=index)

    @property
    def circuit(self) -> gkr.Circuit:
        """The circuit this relation is indexed by."""
        return cast("gkr.Circuit", self.index)

    def layer_tables(self, w_in) -> list:
        """Every layer's padded wire values from the input table: entry 0 is
        the input, entry l + 1 the gates of layer l (zeros past the last
        gate)."""
        from vfhe.circuit.export import layer_bit_sizes

        sizes = layer_bit_sizes(self.circuit)
        if w_in.num_vars != sizes[0]:
            raise ValueError(
                f"input oracle has {w_in.num_vars} variables, the circuit needs {sizes[0]}"
            )
        tables = [list(w_in.table)]
        for layer, s in zip(self.circuit.layers, sizes[1:], strict=True):
            tables.append(_eval_layer(layer, tables[-1], 1 << s))
        return tables

    def check(self, statement: Statement) -> bool:
        out, w_in = statement.oracles
        last = self.layer_tables(w_in)[-1]
        if out.num_vars != (len(last) - 1).bit_length():
            return False
        for b in _hypercube(out.variables):
            idx = sum(bit << i for i, bit in enumerate(b[v] for v in out.variables))
            if _constant(out.evaluate(b, in_place=False)) != last[idx]:
                return False
        return True

    def compute_head(self, statement: Statement):
        out, w_in = statement.oracles
        return _table_like([w_in], out.variables, self.layer_tables(w_in)[-1])


@with_start
class Relation_Layer(Relation_SumDef):
    """Gate layer l of a circuit as a sum definition, `index = (circuit, l)`:
    `oracles = (W_{l+1}, G_l)` with G_l the layer identity over the layer's
    sparse predicates and the layer below under two renamings (`GKR` builds
    it). `compute_head` evaluates the gates on the table of the layer below,
    linear in the wires, where the generic route would materialize the
    predicates."""

    name = "layer"

    def __init__(self, index: tuple) -> None:
        super().__init__(index=index)

    def compute_head(self, statement: Statement):
        head, body = statement.oracles
        circuit, layer = cast("tuple", self.index)
        below = resolve_table(body.constituents[2])  # W_l under the x renaming
        values = _eval_layer(
            circuit.layers[layer], list(below.table), 1 << head.num_vars
        )
        return _table_like([below], head.variables, values)


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
            va = _constant(a.evaluate(pa, in_place=False))
            vb = _constant(b.evaluate(pb, in_place=False))
            if va != vb:
                return False
        return True


class OutputTest(Protocol):
    """`Relation_Equal(out, top)` -> `Eval(top, r, v)`, and `Eval(out, r, v)`
    unless `out` is public: the verifier draws r (one challenge per
    variable), v = out(r) is computed by the verifier when `out` is public
    and sent by the prover otherwise — the random-evaluation identity test
    of [Tha22, §4.6], soundness s / |A| by generalized Schwartz-Zippel
    [GNS23, Lem. 2]. The step that turns a sequentially composable protocol
    into a protocol for the relation with its head committed or public."""

    name = "equal"
    reduces_from: type[Relation] = Relation_Equal
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        from .sumcheck import _exceptional_set_size

        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return statement.oracles[0].num_vars / size

    def _point(self, iop: IOP, statement: Statement) -> list:
        label = f"{self.name}{statement.path}"
        out, _ = statement.oracles
        return [iop.verifier.challenge(f"{label}/r{i}") for i in range(out.num_vars)]

    def _outputs(self, statement: Statement, point: list, value) -> list[Statement]:
        out, top = statement.oracles
        claims = [
            self.reduce(
                [statement],
                oracles=[top],
                point=dict(zip(top.variables, point, strict=True)),
                value=value,
            )
        ]
        if not getattr(out, "public", False):
            claims.append(
                self.reduce(
                    [statement],
                    oracles=[out],
                    point=dict(zip(out.variables, point, strict=True)),
                    value=value,
                )
            )
        return claims

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = _iop_of(prover)
        out, top = statement.oracles
        point = self._point(iop, statement)
        table = resolve_table(top, prover.witnesses)
        value = _constant(
            table.evaluate(
                dict(zip(table.variables, point, strict=True)), in_place=False
            )
        )
        if not getattr(out, "public", False):
            iop.transcript.write(f"{self.name}{statement.path}/out", value)
        return self._outputs(statement, point, value)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = _iop_of(verifier)
        out, _ = statement.oracles
        point = self._point(iop, statement)
        if getattr(out, "public", False):
            value = _constant(
                out.evaluate(
                    dict(zip(out.variables, point, strict=True)), in_place=False
                )
            )
        else:
            value = await iop.transcript.read(f"{self.name}{statement.path}/out")
        return self._outputs(statement, point, value)


class GKR(Composed):
    """`Relation_Circuit` as a layered reduction: the output test in L_0, then
    the gate layers from the output layer down, one `Relation_Layer`
    statement per layer headed at the layer's implicit oracle; the input
    oracle is `w_in` itself. Registered protocols it relies on: `OutputTest`
    for `Relation_Equal`, `SumDefStart` for `Relation_Layer.Start`, the
    sumcheck and `VirtualEval`."""

    name = "gkr"
    reduces_from: type[Relation] = Relation_Circuit

    def __init__(self) -> None:
        super().__init__(None, self.name)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """The output test's own error; the layers' is the sumchecks'."""
        from .sumcheck import _exceptional_set_size

        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return statement.oracles[0].num_vars / size

    def layers(self, root: Statement, iop: IOP) -> list[list[Statement]]:
        if root.head is not None:
            raise TypeError("a started circuit statement is proven by GKR_Start")
        out, w_in = root.oracles
        top = Implicit(out.variables)
        test = root.reduce_to(Relation_Equal, oracles=(out, top))
        return [[test], *_layer_statements(root, w_in, top, iop.domain)]


class GKR_Start(GKR):
    """The sequentially composable form of `GKR`, for `Relation_Circuit.Start`:
    the gate layers alone, the top one started by the claims the statement
    carries on its head, the output oracle."""

    name = "gkr_start"
    reduces_from: type[Relation] = Relation_Circuit.Start

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """No random evaluation of its own: the layers' errors only."""
        return 0

    def layers(self, root: Statement, iop: IOP) -> list[list[Statement]]:
        if root.head is None:
            raise TypeError("GKR_Start proves a started circuit statement")
        _, w_in = root.oracles
        return _layer_statements(root, w_in, root.head, iop.domain)


def _layer_statements(
    root: Statement, w_in, top: Implicit, domain
) -> list[list[Statement]]:
    """One layer of the reduction per gate layer, output layer first: the
    sum definition of W_{l+1} (the head; `top` for the last layer) by the
    layer identity over the sparse predicates of layer l in `domain` and
    W_l under the x and y renamings, summed variables x first, then y (the
    round order the two-phase prover needs)."""
    from vfhe.circuit.export import layer_bit_sizes, sparse_wiring

    circuit = cast("gkr.Circuit", root.relation.index)
    sizes = layer_bit_sizes(circuit)
    depth = len(circuit.layers)
    if top.num_vars != sizes[-1] or w_in.num_vars != sizes[0]:
        raise ValueError(
            f"the circuit maps {sizes[0]} input variables to {sizes[-1]} output "
            f"variables; the oracles have {w_in.num_vars} and {top.num_vars}"
        )
    heads: list = [w_in]
    heads.extend(
        Implicit([MLE_Variable(f"z{layer}_{i}") for i in range(sizes[layer + 1])])
        for layer in range(depth - 1)
    )
    heads.append(top)
    statements = []
    for layer in reversed(range(depth)):
        below, head = heads[layer], heads[layer + 1]
        s_in = sizes[layer]
        z = list(head.variables)
        x = [MLE_Variable(f"x{layer}_{i}") for i in range(s_in)]
        y = [MLE_Variable(f"y{layer}_{i}") for i in range(s_in)]
        lin_t, xmult_t = sparse_wiring(circuit, layer)
        lin = _sparse(domain, y + x + z, lin_t)
        xmult = _sparse(domain, y + x + z, xmult_t)
        wx = dict(zip(below.variables, x, strict=True))
        wy = dict(zip(below.variables, y, strict=True))
        body = VirtualOracle(
            z + x + y,
            [lin, xmult, below, below],
            terms=[(1, (0, 2)), (1, (1, 2, 3))],
            maps=[
                {v: v for v in lin.variables},
                {v: v for v in xmult.variables},
                wx,
                wy,
            ],
        )
        relation = Relation_Layer(index=(circuit, layer))
        statements.append([root.reduce_to(relation, oracles=(head, body), head=head)])
    return statements


def _iop_of(party) -> IOP:
    iop = party.iop
    if iop is None:
        raise RuntimeError("this party is not bound to an IOP")
    return iop
