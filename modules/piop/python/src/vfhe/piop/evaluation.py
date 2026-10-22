# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The evaluation claim, and what the framework builds on it.

`Relation_Eval`, `f(z) = v` for one oracle `f`, is the terminal claim of the
sumcheck family and the one relation the framework itself is written
against (piop.md section 4): a polynomial IOP reduces everything to
evaluation claims, which are then discharged by what the verifier has of
the oracle -- its kind. The driver bundles, parks and routes any claim on
an oracle through `Relation.subject`; what needs the evaluation claim by
name is the head machinery: a head, an implicit oracle, is only ever the
subject of evaluation claims, so the start relation that `with_start`
derives decides them, and `start_claims` rebuilds them for a started root.
"""

from __future__ import annotations

from typing import TypeVar

from .oracle import Oracle, OracleKind, TableOracle
from .relation import Relation, Statement, child_statement, value_of


class Relation_Eval(Relation):
    """f(point) == value, for one oracle f: the terminal claim of the
    sumcheck family, and the one relation the framework itself is written
    against (`with_start`, the parking rule, the registry's kinds).

    What the verifier has of f is the oracle's kind (piop.md §4): a table
    it may query, a commitment -- "the commitment to f is simply the oracle
    pi_f" [ZCF24, §4], so a compiled claim is still this relation, R_Eval =
    {[(C, z, y); f] : f(z) = y and C opens to f} [ZCF24, Def. 8; BFS20],
    with f the witness, stored under the oracle in `Prover.witnesses` -- a
    definition, or nothing but the statement that heads it.

    Left terminal, the claim is decided by one oracle query (`check`), which
    needs the table: a claim on a committed oracle has no witness-free
    decider and must be discharged by a registered PCS evaluation protocol
    (e.g. `vfhe.polycom.BasefoldEval`).
    """

    name = "eval"
    fields = ("oracles", "point", "value")

    @staticmethod
    def kind(statement: Statement) -> OracleKind:
        """The kind of the oracle this claim is about (piop.md §4)."""
        (f,) = statement.oracles
        return f.kind

    def subject(self, statement: Statement) -> Oracle:
        """The one oracle the claim is about: what it is bundled, parked
        and routed by."""
        (f,) = statement.oracles
        return f

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        return f.query(dict(statement.point)) == value_of(statement.value)


def _with_head_table(statement: Statement, base: Relation, table) -> Statement:
    """`statement` as `base`'s ideal decider sees it once the head is known:
    the base relation's fields, with `table` (as a `TableOracle`) in the
    head's place -- among the oracles, or in the index entry, the relation
    then rebuilt over that index -- and no head (every oracle has a table
    now)."""
    fields = {n: statement.fields[n] for n in base.fields if n in statement.fields}
    head = TableOracle(table)
    if isinstance(base.head_index, str):
        base = type(base)(index={**base.index, base.head_index: head})
        return Statement(base, **fields)
    assert base.head_index is not None
    oracles = list(statement.oracles)
    oracles[base.head_index] = head
    fields["oracles"] = tuple(oracles)
    return Statement(base, **fields)


_R = TypeVar("_R", bound=Relation)


def with_start(cls: type[_R]) -> type[_R]:
    """Class decorator: derive the *start relation* of a relation with a head.

    For `R` with head `u` and other oracles `dep`, `R.Start` is the relation

        R_start(dep; (r_j, y_j)_j)  :=  R(F(dep); dep)  and  F(dep)~(r_j) = y_j
                                        for every j,  F = R.compute_head,

    that is `R` with its head *bound* to the oracle it defines, plus
    evaluation claims on that oracle (k >= 1 of them) at points the statement
    carries as data. Its statements are `R`'s with two more fields, `points`
    and `values`; the oracles, the index and the head are the same (a head
    named by an index field stays in the index, so a started statement of a
    relation with no instance fields is nothing but the index and the
    claims). The decider is
    generic: materialize the head with `compute_head`, decide `R` with that
    table in the head's place (the residual), then one `Relation_Eval` check
    per claim. The decorator also installs `R.start(statement, claims)`, which
    builds the started statement from a headed statement and the `Relation_Eval`
    claims on its head (the Composer calls it once per head, at its layer); the
    started statement is a child of the statement, with the claims as further
    parents. A sequentially composable PIOP for `R` is a PIOP for `R.Start`,
    registered against that class.
    """
    if cls.head_index is None:
        raise TypeError(f"{cls.__name__} declares no head_index; nothing to start")
    if isinstance(cls.head_index, str) and cls.head_index not in cls.index_fields:
        raise TypeError(
            f"{cls.__name__}.head_index names {cls.head_index!r}, which is not "
            f"one of its index fields {cls.index_fields}"
        )
    base_cls = cls

    class Start(Relation):
        name = base_cls.name + "_start"
        fields = (*base_cls.fields, "points", "values")
        index_fields = base_cls.index_fields
        head_index = base_cls.head_index
        base = base_cls

        def compute_head(self, statement: Statement):
            return base_cls(index=self.index).compute_head(statement)

        def check(self, statement: Statement) -> bool:
            points, values = statement.points, statement.values
            if not points or len(points) != len(values):  # k >= 1, paired
                return False
            base = base_cls(index=self.index)
            head = base.compute_head(statement)
            if not _with_head_table(statement, base, head).check():
                return False
            computed = TableOracle(head)
            return all(
                Statement(
                    Relation_Eval(), oracles=(computed,), point=dict(p), value=v
                ).check()
                for p, v in zip(points, values, strict=True)
            )

    Start.__name__ = base_cls.__name__ + ".Start"
    Start.__qualname__ = base_cls.__qualname__ + ".Start"
    Start.__doc__ = f"The start relation of {base_cls.__name__} (`with_start`)."
    if hasattr(base_cls, "inner"):  # a dense-index relation: its layering reads it
        Start.inner = base_cls.inner

    def start(self, statement: Statement, claims: list[Statement]) -> Statement:
        """The started statement: `statement` (headed) with the evaluation
        claims on its head, as a statement of `Start`."""
        if not isinstance(statement.relation, base_cls):
            raise TypeError(f"not a statement of {base_cls.__name__}")
        if not claims:
            raise ValueError("a started statement carries at least one claim")
        if statement.head is None:
            raise ValueError("only a headed statement can be started")
        for c in claims:
            if not isinstance(c.relation, Relation_Eval):
                raise TypeError("the claims on a head are Relation_Eval statements")
            if c.relation.subject(c) is not statement.head:
                raise ValueError("a claim is not on this statement's head")
        return child_statement(
            (statement, *claims),
            Start(index=self.index),
            head=statement.head,
            **statement.fields,
            points=[c.point for c in claims],
            values=[c.value for c in claims],
        )

    cls.Start = Start
    cls.start = start
    return cls


def start_claims(root: Statement) -> list[Statement]:
    """The evaluation claims a started root carries on its head, as
    statements again: what the Composer seeds the head's pending claims with
    when the statement it runs is a started one."""
    assert root.head is not None
    return [
        root.reduce_to(Relation_Eval, oracles=(root.head,), point=p, value=v)
        for p, v in zip(root.points, root.values, strict=True)
    ]
