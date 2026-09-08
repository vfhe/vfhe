# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Layered arithmetic circuits as a relation, proven GKR-style (piop.md §4).

`Relation_Circuit` is circuit satisfiability: the input oracle evaluates,
under the circuit in the relation's `index`, to the output oracle. `GKR` is
one protocol for it [GKR15; Tha22, §4.6]: every gate layer becomes an
implicit oracle (`virtual.ImplicitOracle`) defined by the layer identity

    W_{l+1}(z) = sum_{x,y} add_l(z,x,y) (W_l(x) + W_l(y)) + mul_l(z,x,y) W_l(x) W_l(y)

over public wiring predicates and the layer below, and the circuit claim
reduces to one evaluation claim on the output layer at a random point. The
rest is the framework's: `ImplicitEval` instantiates a layer's definition
at the claimed point, the sumcheck reduces it to claims on the layer below
(two per layer, on one oracle, at different points), the driver's parking
rule bundles them, and so on down to the input oracle, whose claims are
decided by its own kind (public, committed, or plain).

Circuits are the protobuf messages of `vfhe.circuit` (imported lazily; the
wiring tables come from `vfhe.circuit.export`). A wiring table is indexed
`z || x || y` with y in the low bits, so its MLE variables are listed
`[y..., x..., z...]` and no reordering is needed against the framework's
LSB-first tables. The tables are dense (2^(s_out + 2 s_in) entries) — the
correctness-first path; sparse predicates are a later concern.
"""

from __future__ import annotations

from vfhe.arith import Field, Ring

from .mle import MLE, MLE_Variable
from .piop import (
    IOP,
    Protocol,
    Prover,
    Relation,
    Relation_Eval,
    Statement,
    Verifier,
    _constant,
    _hypercube,
)
from .virtual import ImplicitOracle, VirtualOracle


def _table(domain, variables: list, values: list, public: bool = False) -> MLE:
    """A dense table over `variables` in `domain` (a Ring, a Field, or None
    for plain Python values), from a list of ints or domain elements."""
    if isinstance(domain, Ring):
        return MLE(ring=domain, variables=variables, evaluations=values, public=public)
    if isinstance(domain, Field):
        return MLE(field=domain, variables=variables, evaluations=values, public=public)
    return MLE(variables=variables, evaluations=values, public=public)


class Relation_Circuit(Relation):
    """The input oracle evaluates, under the circuit `index`, to the output
    oracle: `oracles = [W_in]` (one table over s_in variables) and `output`
    (a table over s_out variables — public, committed, or plain). The ideal
    decider evaluates the circuit gate by gate over the input table."""

    name = "circuit"
    fields = ("oracles", "output")

    def __init__(self, circuit):
        super().__init__(index=circuit)

    @property
    def circuit(self):
        return self.index

    def layer_tables(self, w_in) -> list:
        """Every layer's padded wire values from the input table: entry 0 is
        the input, entry l + 1 the gates of layer l (zeros past the last
        gate). Domain-agnostic: it uses the entries' own `+` and `*`."""
        from vfhe.circuit import gkr
        from vfhe.circuit.export import layer_bit_sizes

        sizes = layer_bit_sizes(self.circuit)
        if w_in.num_vars != sizes[0]:
            raise ValueError(
                f"input oracle has {w_in.num_vars} variables, the circuit needs {sizes[0]}"
            )
        tables = [list(w_in.table)]
        for layer, s in zip(self.circuit.layers, sizes[1:], strict=True):
            prev = tables[-1]
            out = []
            for gate in layer.gates:
                a, b = prev[gate.left], prev[gate.right]
                if gate.type == gkr.GATE_TYPE_ADD:
                    out.append(a + b)
                elif gate.type == gkr.GATE_TYPE_MUL:
                    out.append(a * b)
                else:
                    raise ValueError("unspecified gate type")
            zero = prev[0] * 0
            out += [zero] * ((1 << s) - len(out))
            tables.append(out)
        return tables

    def check(self, statement: Statement) -> bool:
        (w_in,) = statement.oracles
        last = self.layer_tables(w_in)[-1]
        out = statement.output
        if out.num_vars != (len(last) - 1).bit_length():
            return False
        for b in _hypercube(out.variables):
            idx = sum(bit << i for i, bit in enumerate(b[v] for v in out.variables))
            if _constant(out.evaluate(b, in_place=False)) != last[idx]:
                return False
        return True


class GKR(Protocol):
    """`Relation_Circuit -> Relation_Eval` on the output layer (and on the
    output oracle, unless it is public).

    Both parties build the same chain of implicit layer oracles from the
    index; the verifier draws the output point r (one challenge per output
    variable) and the claim becomes `W_out(r) = v`, with v computed by the
    verifier when the output is public, else sent by the prover together
    with the claim `output(r) = v` — the random-evaluation identity test of
    [Tha22, §4.6], soundness s_out / |A| by generalized Schwartz-Zippel
    [GNS23, Lem. 2]. The prover also computes every layer's table and
    records it under the layer oracle in `Prover.witnesses`, which is where
    the sumcheck provers of the later layer reductions find them.
    """

    name = "gkr"
    reduces_from: type[Relation] = Relation_Circuit
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        from .sumcheck import _exceptional_set_size

        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return statement.output.num_vars / size

    @staticmethod
    def layers(relation: Relation_Circuit, w_in, domain) -> list:
        """The implicit oracles W_1..W_d of the gate layers, W_0 = w_in; each
        defined by the layer identity over public wiring tables in `domain`
        and the layer below under two renamings (x and y)."""
        from vfhe.circuit.export import layer_bit_sizes, wiring_tables

        circuit = relation.circuit
        sizes = layer_bit_sizes(circuit)
        below = w_in
        layers = []
        for layer, s_out in enumerate(sizes[1:]):
            s_in = sizes[layer]
            z = [MLE_Variable(f"z{layer}_{i}") for i in range(s_out)]
            x = [MLE_Variable(f"x{layer}_{i}") for i in range(s_in)]
            y = [MLE_Variable(f"y{layer}_{i}") for i in range(s_in)]
            add_t, mul_t = wiring_tables(circuit, layer)
            add = _table(domain, y + x + z, add_t, public=True)
            mul = _table(domain, y + x + z, mul_t, public=True)
            wx = dict(zip(below.variables, x, strict=True))
            wy = dict(zip(below.variables, y, strict=True))
            body = VirtualOracle(
                z + x + y,
                [add, mul, below, below],
                terms=[(1, (0, 2)), (1, (0, 3)), (1, (1, 2, 3))],
                maps=[
                    {v: v for v in add.variables},
                    {v: v for v in mul.variables},
                    wx,
                    wy,
                ],
            )
            below = ImplicitOracle(z, body, x + y)
            layers.append(below)
        return layers

    def _outputs(self, statement: Statement, top, point: list, value):
        out = statement.output
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

    def _point(self, iop: IOP, statement: Statement) -> list:
        label = f"{self.name}{statement.path}"
        return [
            iop.verifier.challenge(f"{label}/r{i}")
            for i in range(statement.output.num_vars)
        ]

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = _iop_of(prover)
        relation = statement.relation
        assert isinstance(relation, Relation_Circuit)
        (w_in,) = statement.oracles
        layers = self.layers(relation, w_in, iop.domain)
        tables = relation.layer_tables(w_in)
        for layer, values in zip(layers, tables[1:], strict=True):
            prover.witnesses[layer] = _table(iop.domain, layer.variables, values)
        point = self._point(iop, statement)
        top = layers[-1]
        value = _constant(
            prover.witnesses[top].evaluate(
                dict(zip(top.variables, point, strict=True)), in_place=False
            )
        )
        if not getattr(statement.output, "public", False):
            iop.transcript.write(f"{self.name}{statement.path}/out", value)
        return self._outputs(statement, top, point, value)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = _iop_of(verifier)
        relation = statement.relation
        assert isinstance(relation, Relation_Circuit)
        (w_in,) = statement.oracles
        layers = self.layers(relation, w_in, iop.domain)
        point = self._point(iop, statement)
        out = statement.output
        if getattr(out, "public", False):
            value = _constant(
                out.evaluate(
                    dict(zip(out.variables, point, strict=True)), in_place=False
                )
            )
        else:
            value = await iop.transcript.read(f"{self.name}{statement.path}/out")
        return self._outputs(statement, layers[-1], point, value)


def _iop_of(party) -> IOP:
    iop = party.iop
    if iop is None:
        raise RuntimeError("this party is not bound to an IOP")
    return iop
