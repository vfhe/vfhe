# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential composition: headed statements and implicit oracles, the
start relation derived by `with_start`, and the Composer."""

import pytest
from vfhe.arith import MLE, MLE_Variable
from vfhe.piop import (
    IOP,
    Implicit,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    element_digest,
    with_start,
)


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _table(variables, values):
    return MLE(variables=list(variables), evaluations=list(values))


def _at(f, point):
    return f.evaluate(dict(point), in_place=False).constant()


@with_start
class _Hadamard(Relation):
    """a = b ∘ c on the hypercube, head a."""

    name = "hadamard"
    fields = ("oracles",)
    head_index = 0

    def check(self, statement):
        a, b, c = statement.oracles
        return list(a.table) == [x * y for x, y in zip(b.table, c.table, strict=True)]

    def compute_head(self, statement):
        _, b, c = statement.oracles
        return _table(
            b.variables, [x * y for x, y in zip(b.table, c.table, strict=True)]
        )


@with_start
class _Doubled(Relation):
    """u = 2w on the hypercube and sum(w) = total: a definition plus a
    residual condition on the other oracle."""

    name = "doubled"
    fields = ("oracles", "total")
    head_index = 0

    def check(self, statement):
        u, w = statement.oracles
        return (
            list(u.table) == [2 * x for x in w.table]
            and sum(w.table) == statement.total
        )

    def compute_head(self, statement):
        _, w = statement.oracles
        return _table(w.variables, [2 * x for x in w.table])


class _Headless(Relation):
    name = "headless"
    fields = ("oracles",)

    def check(self, statement):
        return True


def _hadamard():
    """A headed Hadamard statement: `a` implicit, `b` and `c` tables."""
    x = _vars("x0", "x1")
    b = _table(x, [1, 2, 3, 4])
    c = _table(x, [5, 6, 7, 8])
    a = Implicit(x)
    return Statement(_Hadamard(), oracles=(a, b, c), head=a), a, b, c


def _hadamard_tables():
    """The same statement headless, every oracle a table."""
    x = _vars("x0", "x1")
    a = _table(x, [5, 12, 21, 32])
    b = _table(x, [1, 2, 3, 4])
    c = _table(x, [5, 6, 7, 8])
    return Statement(_Hadamard(), oracles=(a, b, c)), a, b, c


# --- headed statements -----------------------------------------------------


def test_head_is_the_oracle_at_head_index():
    stmt, a, _, _ = _hadamard()
    assert stmt.head is a and stmt.relation.head_index == 0
    assert _hadamard_tables()[0].head is None
    x = _vars("x0", "x1")
    m = Implicit(x)
    with pytest.raises(TypeError):  # the relation declares no head
        Statement(_Headless(), oracles=(m,), head=m)
    with pytest.raises(TypeError):  # a head is an Implicit handle
        t = _table(x, [1, 2, 3, 4])
        Statement(_Hadamard(), oracles=(t, t, t), head=t)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):  # the head sits at head_index
        Statement(_Hadamard(), oracles=(_table(x, [0] * 4), m, m), head=m)
    with pytest.raises(ValueError):  # the very object, not a look-alike
        Statement(_Hadamard(), oracles=(Implicit(x), m, m), head=m)


def test_head_travels_with_fork_and_reduce_and_is_not_digested():
    stmt, a, b, c = _hadamard()
    assert stmt._fork().head is a
    child = stmt.reduce_to(_Hadamard(), oracles=(a, b, c), head=a)
    assert child.head is a and child.parents == (stmt,) and child.path == "/0"
    assert stmt.reduce_to(_Hadamard(), oracles=(a, b, c)).head is None
    # Structure, not instance: the digest ignores the head attribute.
    headless = Statement(_Hadamard(), oracles=(a, b, c))
    assert element_digest(stmt) == element_digest(headless)
    # The handle itself digests by its variables, so it can sit in a root.
    assert element_digest(a) == element_digest(Implicit(a.variables))
    assert element_digest(a) != element_digest(Implicit(a.variables[:1]))


def test_implicit_is_a_handle():
    x = _vars("x0", "x1")
    m = Implicit(x)
    assert m.num_vars == 2 and not m.public
    assert not hasattr(m, "table")  # no content: nothing to evaluate or sum


def test_hadamard_relation_check():
    stmt, _, _, _ = _hadamard_tables()
    assert stmt.check()
    bad, _, _, _ = _hadamard_tables()
    bad.fields["oracles"] = (_table(_vars("x0", "x1"), [0] * 4), *bad.oracles[1:])
    assert not bad.check()


# --- the start relation ----------------------------------------------------


def test_with_start_needs_a_head():
    with pytest.raises(TypeError):
        with_start(_Headless)


def test_start_relation_shape():
    Start = _Hadamard.Start
    assert Start.name == "hadamard_start"
    assert Start.fields == ("oracles", "points", "values")
    assert Start.head_index == 0 and Start.base is _Hadamard
    assert _Doubled.Start.fields == ("oracles", "total", "points", "values")


def test_start_builds_the_started_statement():
    stmt, a, b, c = _hadamard()
    x = a.variables
    p1, p2 = {x[0]: 3, x[1]: 5}, {x[0]: 7, x[1]: 11}
    claims = [
        Statement(Relation_Eval(), oracles=(a,), point=p1, value=0),
        Statement(Relation_Eval(), oracles=(a,), point=p2, value=0),
    ]
    started = stmt.relation.start(stmt, claims)
    assert isinstance(started.relation, _Hadamard.Start)
    assert started.oracles == (a, b, c) and started.head is a
    assert started.points == [p1, p2] and started.values == [0, 0]
    assert started.parents == (stmt, *claims) and started.path == "/0"
    with pytest.raises(ValueError):
        stmt.relation.start(stmt, [])
    with pytest.raises(ValueError):  # a claim on another oracle
        stmt.relation.start(
            stmt, [Statement(Relation_Eval(), oracles=(b,), point=p1, value=0)]
        )
    with pytest.raises(ValueError):  # a headless statement cannot be started
        headless = Statement(_Hadamard(), oracles=(a, b, c))
        headless.relation.start(headless, claims)
    with pytest.raises(TypeError):  # a statement of another relation
        other = Statement(_Doubled(), oracles=(a, b), total=10, head=a)
        stmt.relation.start(other, claims)


def test_start_check_hadamard():
    stmt, a, _, _ = _hadamard()
    x = a.variables
    product = _table(x, [5, 12, 21, 32])
    points = [{x[0]: 3, x[1]: 5}, {x[0]: 7, x[1]: 11}]
    values = [_at(product, p) for p in points]
    claims = [
        Statement(Relation_Eval(), oracles=(a,), point=p, value=v)
        for p, v in zip(points, values, strict=True)
    ]
    started = stmt.relation.start(stmt, claims)
    assert started.check()
    assert started.relation.compute_head(started).table == product.table
    # One wrong evaluation claim fails the started statement.
    started.fields["values"] = [values[0], values[1] + 1]
    assert not started.check()
    started.fields["points"], started.fields["values"] = [], []  # k >= 1
    assert not started.check()


def test_start_check_residual():
    x = _vars("x0", "x1")
    w = _table(x, [1, 2, 3, 4])
    u = Implicit(x)
    doubled = _table(x, [2, 4, 6, 8])
    point = {x[0]: 3, x[1]: 5}
    claim = Statement(
        Relation_Eval(), oracles=(u,), point=point, value=_at(doubled, point)
    )
    good = Statement(_Doubled(), oracles=(u, w), total=10, head=u)
    assert good.relation.start(good, [claim]).check()
    # The evaluation claim is right, the residual (sum(w) = total) is not.
    bad = Statement(_Doubled(), oracles=(u, w), total=11, head=u)
    assert not bad.relation.start(bad, [claim]).check()


def test_start_relation_rebuilds_its_base_from_the_index():
    class _Scaled(Relation):
        name = "scaled"
        fields = ("oracles",)
        head_index = 0

        def check(self, statement):
            u, w = statement.oracles
            return list(u.table) == [self.index * x for x in w.table]

        def compute_head(self, statement):
            _, w = statement.oracles
            return _table(w.variables, [self.index * x for x in w.table])

    with_start(_Scaled)
    x = _vars("x0")
    w = _table(x, [1, 2])
    u = Implicit(x)
    stmt = Statement(_Scaled(index=3), oracles=(u, w), head=u)
    claim = Statement(Relation_Eval(), oracles=(u,), point={x[0]: 4}, value=3 * 5)
    started = stmt.relation.start(stmt, [claim])
    assert started.relation.index == 3 and started.check()


# --- the driver refuses stray claims on implicit oracles --------------------


def test_protocol_for_refuses_a_claim_on_an_implicit_oracle():
    class _Default(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            return []

    iop = IOP()
    iop.register(Relation_Eval, _Default())
    x = _vars("x0")
    claim = Statement(Relation_Eval(), oracles=(Implicit(x),), point={x[0]: 1}, value=0)
    with pytest.raises(RuntimeError, match="implicit oracle"):
        iop.protocol_for(claim)
    table = Statement(
        Relation_Eval(), oracles=(_table(x, [1, 2]),), point={x[0]: 1}, value=2
    )
    assert iop.protocol_for(table) is not None
