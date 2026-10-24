# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""GKR [GKR15; Tha22, §4.6] as a composition (piop.md section 4,
`composer.py`): the *output test*, a headless statement
`Relation_Equal(out, W_d)` whose protocol `OutputTest` reduces "the output
oracle equals the output layer" to one evaluation claim on the layer at a
random point; then, one layer per gate layer, the sum definition
`Relation_Layer` of the layer's implicit oracle. The Composer computes every
layer's table (`Relation_Layer.compute_head`, the gates evaluated on the
layer below) and, layer by layer, starts each definition with the claims
the layer above left on it; `SumDefStart` combines them, and the registered
sumcheck and `VirtualEval` reduce them to claims on the layer below (two per
layer, on one oracle, at different points), down to the input oracle, whose
claims are decided by its own kind (public, committed, or plain).
`GKR_Start` is the same composition without the output test: the
sequentially composable form, for a circuit statement started by claims on
its output. The prover's rounds run on the sparse predicates through the
Libra two-phase strategy of `prover_view.py` (linear in the number of wires
plus the layer widths), which the dense strategy reproduces message for
message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vfhe.arith import MLE_Variable
from vfhe.piop.composer import Composed
from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.oracle import ImplicitOracle
from vfhe.piop.protocol import Protocol, exceptional_set_size
from vfhe.piop.relations.circuit import (
    Relation_CircuitSparseIndex,
    Relation_Equal,
    Relation_Layer,
)
from vfhe.piop.virtual import VirtualOracle, resolve_table

if TYPE_CHECKING:
    from vfhe.circuit import gkr
    from vfhe.piop.iop import IOP
    from vfhe.piop.party import Prover, Verifier
    from vfhe.piop.relation import Relation, Statement


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
        size = exceptional_set_size(domain)
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
        if not out.public:
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
        iop = prover.iop
        out, top = statement.oracles
        point = self._point(iop, statement)
        table = resolve_table(top, prover.witnesses)
        value = table.evaluate(
            dict(zip(table.variables, point, strict=True)), in_place=False
        ).constant()
        if not out.public:
            iop.transcript.write(f"{self.name}{statement.path}/out", value)
        return self._outputs(statement, point, value)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        out, _ = statement.oracles
        point = self._point(iop, statement)
        if out.public:
            value = out.query(dict(zip(out.variables, point, strict=True)))
        else:
            value = await iop.transcript.read(f"{self.name}{statement.path}/out")
        return self._outputs(statement, point, value)


class GKR(Composed):
    """`Relation_CircuitSparseIndex` as a layered reduction: the output test in L_0, then
    the gate layers from the output layer down, one `Relation_Layer`
    statement per layer headed at the layer's implicit oracle; the input
    oracle is `w_in` itself. Registered protocols it relies on: `OutputTest`
    for `Relation_Equal`, `SumDefStart` for `Relation_Layer.Start`, the
    sumcheck and `VirtualEval`."""

    name = "gkr"
    reduces_from: type[Relation] = Relation_CircuitSparseIndex

    def __init__(self) -> None:
        super().__init__(None, self.name)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """The output test's own error; the layers' is the sumchecks'."""
        size = exceptional_set_size(domain)
        if size is None:
            return None
        return statement.oracles[0].num_vars / size

    def layers(self, root: Statement, iop: IOP) -> list[list[Statement]]:
        if root.head is not None:
            raise TypeError("a started circuit statement is proven by GKR_Start")
        out, w_in = root.oracles
        top = ImplicitOracle(out.variables)
        test = root.reduce_to(Relation_Equal, oracles=(out, top))
        return [[test], *_layer_statements(root, w_in, top)]


class GKR_Start(GKR):
    """The sequentially composable form of `GKR`, for
    `Relation_CircuitSparseIndex.Start`:
    the gate layers alone, the top one started by the claims the statement
    carries on its head, the output oracle."""

    name = "gkr_start"
    reduces_from: type[Relation] = Relation_CircuitSparseIndex.Start

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """No random evaluation of its own: the layers' errors only."""
        return 0

    def layers(self, root: Statement, iop: IOP) -> list[list[Statement]]:
        if root.head is None:
            raise TypeError("GKR_Start proves a started circuit statement")
        _, w_in = root.oracles
        return _layer_statements(root, w_in, root.head)


def _layer_statements(
    root: Statement, w_in, top: ImplicitOracle
) -> list[list[Statement]]:
    """One layer of the reduction per gate layer, output layer first: the
    sum definition of W_{l+1} (the head; `top` for the last layer) by the
    layer identity over the sparse predicates of layer l, read from the
    root's index and renamed to the body's variables, and W_l under the x
    and y renamings, summed variables x first, then y (the round order the
    two-phase prover needs)."""
    from vfhe.circuit.export import layer_bit_sizes

    circuit = cast("gkr.Circuit", root.index["circuit"])
    predicates = root.index["predicates"]
    sizes = layer_bit_sizes(circuit)
    depth = len(circuit.layers)
    if top.num_vars != sizes[-1] or w_in.num_vars != sizes[0]:
        raise ValueError(
            f"the circuit maps {sizes[0]} input variables to {sizes[-1]} output "
            f"variables; the oracles have {w_in.num_vars} and {top.num_vars}"
        )
    heads: list = [w_in]
    heads.extend(
        ImplicitOracle([MLE_Variable(f"z{layer}_{i}") for i in range(sizes[layer + 1])])
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
        lin, xmult = predicates[2 * layer], predicates[2 * layer + 1]
        wx = dict(zip(below.variables, x, strict=True))
        wy = dict(zip(below.variables, y, strict=True))
        body = VirtualOracle(
            z + x + y,
            [lin, xmult, below, below],
            terms=[(1, (0, 2)), (1, (1, 2, 3))],
            maps=[
                dict(zip(lin.variables, y + x + z, strict=True)),
                dict(zip(xmult.variables, y + x + z, strict=True)),
                wx,
                wy,
            ],
        )
        relation = Relation_Layer(
            index={"circuit": circuit, "layer": layer, "lin": lin, "xmult": xmult}
        )
        statements.append([root.reduce_to(relation, oracles=(head, body), head=head)])
    return statements
