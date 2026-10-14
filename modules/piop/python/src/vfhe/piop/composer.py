# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The Composer: a protocol assembled from a layered reduction.

A generic protocol for a relation R is a *reduction* of R's statement to
statements of other relations, arranged in layers L_0, ..., L_D. A statement
is *headless* (an ordinary statement, discharged by its registered protocol)
or *headed* (`Statement.head`): it introduces an implicit oracle, the head,
whose table its relation computes from the statement's other oracles
(`Relation.compute_head`), and it is discharged by the sequentially
composable protocol of its relation, registered against `R.Start`, once
every evaluation claim on the head has been made. Two rules make the layers a
schedule: every implicit oracle heads exactly one statement (rule 1), and a
statement in L_l mentions, besides its own head, only non-implicit oracles
and heads of deeper layers (rule 2; through virtual oracles too).

`compose(layering)` turns such a reduction, written as a function from R's
statement to its layers, into a `Protocol`:

- on the prover's side it first computes every head's table, deepest layer
  first, from the tables of its statement's other oracles, and stores it in
  `Prover.witnesses` (where the subprotocols read it);
- then it runs the layers in order: a headless statement goes to its
  protocol, a headed one is *started* (`relation.start`) with the claims
  accumulated on its head and goes to its composable protocol;
- what a subprotocol leaves is routed: a claim on a head of this reduction
  waits for that head's layer; a statement with a non-bundle protocol (an
  evaluation claim on a virtual oracle, a sum a protocol handed back) is
  discharged inside the layer, so a claim on a head hidden behind a virtual
  oracle is caught in time; everything else (a terminal statement, a claim
  for the PCS bundle, a claim on a head of an enclosing composition) is an
  output of the composed protocol, for the driver.

A composed protocol is an ordinary protocol: the driver hands it one
statement and goes on with what it returns. A relation whose protocol is
composed can itself head a statement of another reduction: its `R.Start`
protocol is a composed protocol too, whose layering starts from the claims
the started statement carries (they seed the claims on its head), and whose
statements may read the heads of the enclosing reduction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from .piop import (
    Implicit,
    Party,
    Protocol,
    Prover,
    Relation_Eval,
    Statement,
    Verifier,
    oracle_closure,
)
from .virtual import VirtualOracle

Layering = Callable[[Statement], list[list[Statement]]]
"""R's reduction in executable form: given a statement of R, its layers, L_0
first, each a list of child statements (built with `statement.reduce_to`, so
paths and transcript labels agree on both parties), headless or headed."""


def compose(layering: Layering, name: str = "composed") -> Protocol:
    """The protocol for a relation whose reduction is `layering`."""
    return Composed(layering, name)


class Composed(Protocol):
    """The protocol `compose` builds; see the module docstring."""

    reduces_from = None
    reduces_to = ()

    def __init__(self, layering: Layering, name: str = "composed") -> None:
        self.layering = layering
        self.name = name

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        return await self._run(prover, statements, "prove")

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        return await self._run(verifier, statements, "verify")

    async def _run(
        self, party: Party, statements: list[Statement], half: str
    ) -> list[Statement]:
        (root,) = statements
        await root.resolved()
        layers = self.layering(root)  # the same layered DAG on both parties
        heads = _check_rules(layers)
        if half == "prove":
            assert isinstance(party, Prover)
            _compute_heads(party, layers)

        pending: dict[int, list[Statement]] = defaultdict(list)
        if root.head is not None:  # a started root: its own claims
            pending[id(root.head)] = _start_claims(root)
        outputs: list[Statement] = []
        for layer in layers:
            runs = []
            for stmt in layer:  # in list order: deterministic
                if stmt.head is None:
                    runs.append(stmt)
                    continue
                claims = pending.pop(id(stmt.head), [])
                if not claims:
                    raise RuntimeError(f"no claim on the head of {stmt!r}")
                for claim in claims:
                    await claim.resolved()
                runs.append(stmt.relation.start(stmt, claims))
            for stmt in runs:
                await self._discharge(party, half, stmt, heads, pending, outputs)
        if pending:
            raise RuntimeError(
                "claims left on implicit oracles that no later layer heads"
            )
        return outputs

    async def _discharge(
        self,
        party: Party,
        half: str,
        stmt: Statement,
        heads: dict[int, Statement],
        pending: dict[int, list[Statement]],
        outputs: list[Statement],
    ) -> None:
        """Run `stmt`'s protocol and route what it leaves."""
        iop = party.iop
        assert iop is not None
        protocol = iop.protocol_for(stmt)
        if protocol is None:  # terminal: the driver decides it ...
            if stmt.head is not None or any(
                id(o) in heads for o in oracle_closure(_oracles(stmt))
            ):  # ... unless it reads an implicit oracle, which it cannot
                raise RuntimeError(f"no protocol registered for {stmt!r}")
            outputs.append(stmt)
            return
        for claim in await getattr(protocol, half)(party, [stmt]):
            subject = _eval_subject(claim)
            if isinstance(subject, Implicit):
                if id(subject) in heads:
                    pending[id(subject)].append(claim)  # its head's layer
                else:
                    outputs.append(claim)  # a head of an enclosing one
                continue
            nxt = iop.protocol_for(claim)
            if nxt is None or nxt.batching:
                outputs.append(claim)  # terminal, or for the PCS bundle
            else:
                await self._discharge(party, half, claim, heads, pending, outputs)


def _oracles(statement: Statement) -> tuple:
    return tuple(statement.fields.get("oracles") or ())


def _check_rules(layers: list[list[Statement]]) -> dict[int, Statement]:
    """Rules 1 and 2 of a layering; returns `{id(head): its statement}`.

    Oracles are read through virtual definitions (`oracle_closure`), since an
    implicit oracle may hide behind a virtual one. An implicit oracle that
    heads no statement here belongs to an enclosing composition and counts as
    non-implicit for the rules.
    """
    heads: dict[int, Statement] = {}
    layer_of: dict[int, int] = {}
    for depth, layer in enumerate(layers):
        for stmt in layer:
            if stmt.head is None:
                continue
            if id(stmt.head) in heads:
                raise ValueError(f"rule 1: {stmt.head!r} heads two statements")
            heads[id(stmt.head)] = stmt
            layer_of[id(stmt.head)] = depth
    for depth, layer in enumerate(layers):
        for stmt in layer:
            for o in oracle_closure(_oracles(stmt)):
                if o is stmt.head or id(o) not in heads:
                    continue
                if layer_of[id(o)] <= depth:
                    raise ValueError(
                        f"rule 2: {stmt!r} in layer {depth} mentions {o!r}, "
                        f"a head of layer {layer_of[id(o)]}"
                    )
    return heads


def _compute_heads(prover: Prover, layers: list[list[Statement]]) -> None:
    """The pass Res: every head's table, once, deepest layer first, from the
    tables of its statement's other oracles, into `prover.witnesses`. A head
    an enclosing composition has already stored is not recomputed."""
    for layer in reversed(layers):
        for stmt in layer:
            head = stmt.head
            if head is None or head in prover.witnesses:
                continue
            table = stmt.relation.compute_head(_statement_view(stmt, prover.witnesses))
            variables = list(getattr(table, "variables", ()))
            if len(variables) != len(head.variables) or any(
                a is not b for a, b in zip(variables, head.variables, strict=True)
            ):
                raise RuntimeError(
                    f"compute_head of {stmt!r} returned a table over other "
                    f"variables than its head's"
                )
            prover.witnesses[head] = table


def _statement_view(statement: Statement, witnesses: dict) -> Statement:
    """`statement` as its relation's `compute_head` reads it: every oracle but
    the head replaced by what the prover holds of it. A stored table
    (`witnesses`) stands in for an implicit oracle; a virtual oracle becomes
    the same definition over its constituents so resolved; any other oracle
    (a dense or sparse table, public or committed) is its own table."""
    oracles = tuple(
        o if o is statement.head else _view_oracle(o, witnesses)
        for o in _oracles(statement)
    )
    fields = dict(statement.fields)
    fields["oracles"] = oracles
    return Statement(
        statement.relation,
        parents=statement.parents,
        path=statement.path,
        head=statement.head,
        **fields,
    )


def _view_oracle(oracle, witnesses: dict):
    if oracle in witnesses:
        return witnesses[oracle]
    if isinstance(oracle, Implicit):
        raise RuntimeError(f"{oracle!r} has no table yet (rule 2?)")
    if isinstance(oracle, VirtualOracle):
        return VirtualOracle(
            oracle.variables,
            [_view_oracle(c, witnesses) for c in oracle.constituents],
            oracle.terms,
            oracle.maps,
        )
    return oracle


def _start_claims(root: Statement) -> list[Statement]:
    """The evaluation claims a started root carries on its head."""
    assert root.head is not None
    return [
        root.reduce_to(Relation_Eval, oracles=(root.head,), point=p, value=v)
        for p, v in zip(root.points, root.values, strict=True)
    ]


def _eval_subject(claim: Statement):
    """The oracle an evaluation claim is about; None for any other statement."""
    if isinstance(claim.relation, Relation_Eval):
        return Relation_Eval.subject(claim)
    return None
