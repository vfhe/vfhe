# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequential composition: headed statements and implicit oracles, the
start relation derived by `with_start`, and the Composer."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.piop import (
    IOP,
    Implicit,
    OracleKind,
    Protocol,
    Relation,
    Relation_Eval,
    Relation_Sum,
    Relation_SumDef,
    Relation_Zero,
    Statement,
    Sumcheck,
    SumDefStart,
    VirtualEval,
    VirtualOracle,
    compose,
    element_digest,
    with_start,
)
from vfhe.piop.virtual import _eq_weights, _one_of


class _IntDomain:
    """Deterministic integer challenges (any nonzero integer is exceptional
    for exact integer arithmetic), so a rejection is reproducible."""

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random_exceptional(self):
        return self._rng.randrange(2, 1 << 20)


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _table(variables, values):
    return MLE(variables=list(variables), evaluations=list(values))


def _like(t, values, public=False):
    """A table over `t`'s variables and domain."""
    return MLE(
        ring=t.ring,
        field=t.field,
        variables=list(t.variables),
        evaluations=values,
        public=public,
    )


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
        return _like(b, [x * y for x, y in zip(b.table, c.table, strict=True)])


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


# --- the Composer ----------------------------------------------------------


class _HadamardStart(Protocol):
    """The Hadamard tail, a sequentially composable PIOP for `_Hadamard.Start`:
    each claim (b ∘ c)~(r) = y becomes the sum claim sum_i b(i) c(i) eq(i, r)
    = y, one sumcheck per claim (the simplest batching), which ends in claims
    on b and c (eq is public). Sound for any r: the identity is exact."""

    name = "hadamard_start"
    reduces_to = (Relation_Sum,)

    async def prove(self, prover, statements):
        return self._sums(statements)

    async def verify(self, verifier, statements):
        return self._sums(statements)

    def _sums(self, statements):
        (s,) = statements
        _, b, c = s.oracles
        ref = c if isinstance(b, Implicit) else b  # a table, for the domain
        out = []
        for point, value in zip(s.points, s.values, strict=True):
            r = [point[v] for v in s.head.variables]
            eq = _like(ref, _eq_weights(r, _one_of([ref])), public=True)
            g = VirtualOracle.product([b, c, eq])
            out.append(self.reduce([s], oracles=[g], value=value))
        return out


class _Chain(Relation):
    """sum_i b(i) (d ∘ e)(i) = total over tables b, d, e. Its protocol is a
    composition with the Hadamard product d ∘ e as an implicit oracle."""

    name = "chain"
    fields = ("oracles", "total")

    def check(self, statement):
        b, d, e = statement.oracles
        prods = [x * y * z for x, y, z in zip(b.table, d.table, e.table, strict=True)]
        return sum(prods) == statement.total


def _chain_layering(root):
    b, d, e = root.oracles
    c = Implicit(b.variables)
    product = VirtualOracle.product([b, c])
    return [
        [root.reduce_to(Relation_Sum, oracles=[product], value=root.total)],
        [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)],
    ]


def _chain_instance(ring=None, total=290, relation=None):
    x = _vars("x0", "x1")
    b = MLE(ring=ring, variables=x, evaluations=[1, 2, 3, 4])
    d = MLE(ring=ring, variables=x, evaluations=[5, 6, 7, 8])
    e = MLE(ring=ring, variables=x, evaluations=[2, 3, 4, 5])
    # d ∘ e = [10, 18, 28, 40]; sum b * (d ∘ e) = 10 + 36 + 84 + 160 = 290
    return Statement(relation or _Chain(), oracles=(b, d, e), total=total)


def _iop(domain, layering=_chain_layering, **kw):
    iop = IOP(domain=domain, **kw)
    iop.register(_Chain, compose(layering, name="chain"))
    iop.register(_Hadamard.Start, _HadamardStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


def test_composed_chain_ideal_model(monkeypatch):
    calls = []
    compute_head = _Hadamard.compute_head
    monkeypatch.setattr(
        _Hadamard,
        "compute_head",
        lambda self, s: calls.append(s) or compute_head(self, s),
    )
    stmt = _chain_instance()
    assert stmt.check()
    iop = _iop(_IntDomain())
    assert iop.run(stmt)  # the three terminal claims decided by the driver
    assert len(calls) == 1  # the prover, once
    (c,) = [o for o in iop.prover.witnesses if isinstance(o, Implicit)]
    assert list(iop.prover.witnesses[c].table) == [10, 18, 28, 40]
    order = iop.transcript.order
    # L0's sumcheck, its virtual-eval values, then the tail's sumcheck
    assert order[0] == "sumcheck/0/g0"
    assert order.index("virtual/0/0/vals") < order.index("sumcheck/1/0/0/g0")
    assert not _iop(_IntDomain(1)).run(_chain_instance(total=291))


class _ChainB(_Chain):
    """sum_i b(i) (b ∘ e)(i) = total: b is read by both layers."""

    name = "chain_b"

    def check(self, statement):
        b, _, e = statement.oracles
        prods = [x * x * z for x, z in zip(b.table, e.table, strict=True)]
        return sum(prods) == statement.total


def _chain_b_layering(root):
    b, _, e = root.oracles
    c = Implicit(b.variables)
    product = VirtualOracle.product([b, c])
    return [
        [root.reduce_to(Relation_Sum, oracles=[product], value=root.total)],
        [root.reduce_to(_Hadamard, oracles=(c, b, e), head=c)],
    ]


def test_composed_outputs_reach_the_driver():
    sizes = []

    class _Bundle(Protocol):
        reduces_from = Relation_Eval
        batching = True

        async def prove(self, prover, statements):
            sizes.append(len(statements))
            return []

        async def verify(self, verifier, statements):
            sizes.append(len(statements))
            assert all(s.check() for s in statements)
            return []

    iop = _iop(_IntDomain())
    iop.register(_ChainB, compose(_chain_b_layering, name="chain_b"))
    iop.register(Relation_Eval, _Bundle())
    # b ∘ e = [2, 6, 12, 20]; sum b * that = 2 + 12 + 36 + 80 = 130
    stmt = _chain_instance(total=130, relation=_ChainB())
    assert stmt.check()
    assert iop.run(stmt)
    # The composed protocol returns every claim on a table: two on b (one
    # per layer), one on e; the driver bundles them per oracle, both parties.
    assert sorted(sizes) == [1, 1, 2, 2]


def test_composed_chain_fiat_shamir():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _chain_instance(ring)
    proof = _iop(ring, fiat_shamir=True).prove(stmt)
    assert _iop(ring, fiat_shamir=True).verify(stmt, proof)
    assert proof.digest() == _iop(ring, fiat_shamir=True).prove(stmt).digest()
    bad = _chain_instance(ring, total=291)
    assert not _iop(ring, fiat_shamir=True).verify(bad, proof)
    assert not _iop(ring, fiat_shamir=True).run(bad)


def test_layering_rules_are_checked():
    def twice(root):  # rule 1: c heads two statements
        _, d, e = root.oracles
        c = Implicit(d.variables)
        return [
            [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)],
            [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)],
        ]

    def shallow(root):  # rule 2: the sum reads c (through the virtual
        b, d, e = root.oracles  # oracle) but sits below c's layer
        c = Implicit(b.variables)
        product = VirtualOracle.product([b, c])
        return [
            [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)],
            [root.reduce_to(Relation_Sum, oracles=[product], value=1)],
        ]

    def unclaimed(root):  # a head nobody queries
        _, d, e = root.oracles
        c = Implicit(d.variables)
        return [[root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)]]

    cases = ((twice, "rule 1"), (shallow, "rule 2"), (unclaimed, "no claim"))
    for bad, error in cases:
        with pytest.raises((ValueError, RuntimeError), match=error):
            _iop(_IntDomain(), layering=bad).run(_chain_instance())


@with_start
class _DoubleHadamard(Relation):
    """c = d ∘ (d ∘ e), head c; its composable protocol is itself composed,
    with the inner product d ∘ e as an implicit oracle of its own."""

    name = "double"
    fields = ("oracles",)
    head_index = 0

    def check(self, statement):
        c, d, e = statement.oracles
        want = [x * x * y for x, y in zip(d.table, e.table, strict=True)]
        return list(c.table) == want

    def compute_head(self, statement):
        _, d, e = statement.oracles
        return _like(d, [x * x * y for x, y in zip(d.table, e.table, strict=True)])


def _double_start_layering(started):
    # started: a statement of _DoubleHadamard.Start, head c, with claims on c
    c, d, e = started.oracles
    m = Implicit(d.variables)
    return [
        [started.reduce_to(_Hadamard, oracles=(c, d, m), head=c)],
        [started.reduce_to(_Hadamard, oracles=(m, d, e), head=m)],
    ]


class _Chain2(_Chain):
    """sum_i b(i) (d ∘ (d ∘ e))(i) = total."""

    name = "chain2"

    def check(self, statement):
        b, d, e = statement.oracles
        prods = [
            x * y * y * z for x, y, z in zip(b.table, d.table, e.table, strict=True)
        ]
        return sum(prods) == statement.total


def _chain2_layering(root):
    b, d, e = root.oracles
    c = Implicit(b.variables)
    product = VirtualOracle.product([b, c])
    return [
        [root.reduce_to(Relation_Sum, oracles=[product], value=root.total)],
        [root.reduce_to(_DoubleHadamard, oracles=(c, d, e), head=c)],
    ]


def _iop2(domain):
    iop = _iop(domain)
    iop.register(_Chain2, compose(_chain2_layering, name="chain2"))
    iop.register(_DoubleHadamard.Start, compose(_double_start_layering, "double"))
    return iop


def test_nested_composition(monkeypatch):
    calls = []
    for cls in (_Hadamard, _DoubleHadamard):
        original = cls.compute_head
        monkeypatch.setattr(
            cls,
            "compute_head",
            lambda self, s, f=original: calls.append(type(self).name) or f(self, s),
        )
    # d ∘ e = [10, 18, 28, 40]; d ∘ (d ∘ e) = [50, 108, 196, 320];
    # sum b * that = 50 + 216 + 588 + 1280 = 2134
    stmt = _chain_instance(total=2134, relation=_Chain2())
    assert stmt.check()
    iop = _iop2(_IntDomain())
    assert iop.run(stmt)
    # The outer pass computed c; the inner pass computed m and left c alone.
    assert sorted(calls) == ["double", "hadamard"]
    tables = sorted(
        list(t.table)
        for o, t in iop.prover.witnesses.items()
        if isinstance(o, Implicit)
    )
    assert tables == [[10, 18, 28, 40], [50, 108, 196, 320]]
    assert not _iop2(_IntDomain(1)).run(_chain_instance(total=2135, relation=_Chain2()))


def test_a_false_head_is_rejected(monkeypatch):
    # The prover computes a wrong table for c and claims the total that
    # table gives: the tail's sumcheck, bound to the real b and d, e through
    # the virtual oracle, catches it. Error about 8 / 2^20 per run; the
    # integer domain is deterministic, so the pinned seed makes it exact.
    def wrong(_self, statement):
        _, b, c = statement.oracles
        table = [x * y for x, y in zip(b.table, c.table, strict=True)]
        table[0] += 1
        return _like(b, table)

    monkeypatch.setattr(_Hadamard, "compute_head", wrong)
    assert not _iop(_IntDomain(0)).run(_chain_instance(total=291))


@with_start
class _Triple(Relation):
    """c = d ∘ (d ∘ m), head c, over tables d and an oracle m that may be a
    head of the enclosing reduction; composable protocol composed."""

    name = "triple"
    fields = ("oracles",)
    head_index = 0

    def check(self, statement):
        c, d, m = statement.oracles
        want = [x * x * y for x, y in zip(d.table, m.table, strict=True)]
        return list(c.table) == want

    def compute_head(self, statement):
        _, d, m = statement.oracles
        return _like(d, [x * x * y for x, y in zip(d.table, m.table, strict=True)])


def _triple_start_layering(started):
    c, d, m = started.oracles  # m: a head of the enclosing reduction
    k = Implicit(d.variables)
    return [
        [started.reduce_to(_Hadamard, oracles=(c, d, k), head=c)],
        [started.reduce_to(_Hadamard, oracles=(k, d, m), head=k)],
    ]


class _Chain3(_Chain):
    """sum_i b(i) (d ∘ (d ∘ (d ∘ e)))(i) = total."""

    name = "chain3"

    def check(self, statement):
        b, d, e = statement.oracles
        prods = [
            x * y * y * y * z for x, y, z in zip(b.table, d.table, e.table, strict=True)
        ]
        return sum(prods) == statement.total


def _chain3_layering(root):
    b, d, e = root.oracles
    c, m2 = Implicit(b.variables), Implicit(b.variables)
    product = VirtualOracle.product([b, c])
    return [
        [root.reduce_to(Relation_Sum, oracles=[product], value=root.total)],
        [root.reduce_to(_Triple, oracles=(c, d, m2), head=c)],
        [root.reduce_to(_Hadamard, oracles=(m2, d, e), head=m2)],
    ]


def _iop3(domain, **kw):
    iop = _iop(domain, **kw)
    iop.register(_Chain3, compose(_chain3_layering, name="chain3"))
    iop.register(_Triple.Start, compose(_triple_start_layering, "triple"))
    return iop


def test_nested_composition_reads_an_outer_head():
    # m2 = d ∘ e = [10, 18, 28, 40] is a head of the outer reduction, read by
    # the inner layering of _Triple.Start, whose own head is k = d ∘ m2 =
    # [50, 108, 196, 320]; c = d ∘ k = [250, 648, 1372, 2560];
    # sum b * c = 250 + 1296 + 4116 + 10240 = 15902.
    stmt = _chain_instance(total=15902, relation=_Chain3())
    assert stmt.check()
    iop = _iop3(_IntDomain())
    assert iop.run(stmt)
    tables = sorted(
        list(t.table)
        for o, t in iop.prover.witnesses.items()
        if isinstance(o, Implicit)
    )
    assert tables == [[10, 18, 28, 40], [50, 108, 196, 320], [250, 648, 1372, 2560]]
    ring = Ring(1024, prime_size=[49], split_degree=4)
    stmt = _chain_instance(ring, total=15902, relation=_Chain3())
    proof = _iop3(ring, fiat_shamir=True).prove(stmt)
    assert _iop3(ring, fiat_shamir=True).verify(stmt, proof)
    assert not _iop3(_IntDomain(1)).run(
        _chain_instance(total=15903, relation=_Chain3())
    )


def test_a_terminal_statement_reading_a_head_is_refused():
    # A layer statement with no protocol is an output for the driver, unless
    # it reads an implicit oracle, which the verifier could never decide.
    def bad(root):  # Zero has no protocol here, and reads c
        b, d, e = root.oracles
        c = Implicit(d.variables)
        zero = root.reduce_to(Relation_Zero, oracles=[VirtualOracle.product([b, c])])
        return [[zero], [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)]]

    with pytest.raises(RuntimeError, match="no protocol"):
        _iop(_IntDomain(), layering=bad).run(_chain_instance())


# --- sum definitions -------------------------------------------------------


def _eq_table(variables, ring=None):
    """eq(z, p) over variables [p..., z...] (same width each), public."""
    n = len(variables) // 2
    table = [int((b & ((1 << n) - 1)) == (b >> n)) for b in range(1 << len(variables))]
    return MLE(ring=ring, variables=variables, evaluations=table, public=True)


def _product_body(below, i):
    """The sum definition of one grand-product level: L(z) = sum_p eq(z, p)
    below(p, 0) below(p, 1) [Tha13, §5.3.1]; returns (L, G), L implicit."""
    n = below.num_vars - 1
    z = _vars(*[f"z{i}_{k}" for k in range(n)])
    p = _vars(*[f"p{i}_{k}" for k in range(n)])
    eq = _eq_table(p + z, ring=getattr(below, "ring", None))
    lo = dict(zip(below.variables[:-1], p, strict=True)) | {below.variables[-1]: 0}
    hi = dict(zip(below.variables[:-1], p, strict=True)) | {below.variables[-1]: 1}
    body = VirtualOracle(
        z + p,
        [eq, below, below],
        [(1, (0, 1, 2))],
        [{v: v for v in eq.variables}, lo, hi],
    )
    return Implicit(z), body


def _sum_tree(W, levels):
    """(L_i, G_i) pairs, deepest first: L_{levels-1}, ..., L_0, L_levels = W."""
    below = W
    pairs = []
    for i in range(levels):
        L, G = _product_body(below, i)
        pairs.append((L, G))
        below = L
    return pairs


def _messages(iop):
    """Every transcript value in order, challenges included."""
    t = iop.transcript
    return [t.entries[label].result() for label in t.order]


# The transcript values of the two runs below under _IntDomain(0), pinned:
# what the framework sent for the same claims before sum definitions became
# statements of a reduction, message for message.
ONE_LEVEL_MESSAGES = [(-100, 231, 1152, 2915), 885442, (885444, 1770889)]
TWO_LEVEL_MESSAGES = [
    (-9350, 31122, 172312, 468076),
    885442,
    (15052536, 42501301),
    403960,
    794774,
    (-67685686673310, 107545238116844, 463395593428950, 1042228612393776),
    933490,
    (
        1163832537013774607460054312480,
        4579598518976424980877088450070,
        10247319616473145718758580764968,
        18167017540661817691405424390862,
    ),
    441003,
    (411674146971, 4513009),
]


def _sumdef_iop(domain, **kw):
    iop = IOP(domain=domain, **kw)
    iop.register(Relation_SumDef.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


@pytest.mark.parametrize("ring", [None, Ring(1024, prime_size=[49], split_degree=4)])
def test_sumdef_relation_and_head(ring):
    W = MLE(ring=ring, variables=_vars("w0", "w1"), evaluations=[2, 3, 5, 7])
    ((L1, G1),) = _sum_tree(W, 1)  # L1(z) = W(z, 0) * W(z, 1) = [10, 21]
    stmt = Statement(Relation_SumDef(), oracles=(L1, G1), head=L1)
    assert Relation_SumDef.summed(stmt) == G1.variables[1:]
    table = stmt.relation.compute_head(stmt)  # constituents are tables already
    assert table.variables == L1.variables and table.ring is ring
    assert list(table.table) == [10, 21]
    assert Statement(Relation_SumDef(), oracles=(table, G1)).check()
    wrong = _like(table, [10, 22])
    assert not Statement(Relation_SumDef(), oracles=(wrong, G1)).check()
    plain = Statement(Relation_SumDef(), oracles=(L1, W), head=L1)
    with pytest.raises(TypeError):  # G is a virtual oracle
        plain.relation.compute_head(plain)


def test_sumdef_one_level_started_statement():
    # A started sum definition run on its own, the driver completing it
    # (sumcheck, then the virtual-oracle rewriting). Rejections are exact: an
    # honest prover on a false claim fails the first round's check whatever
    # the challenges.
    W = _table(_vars("w0", "w1"), [2, 3, 5, 7])
    ((L1, G1),) = _sum_tree(W, 1)
    (z,) = L1.variables
    r = 11
    v = (1 - r) * 10 + r * 21  # the MLE of the products, not the product of MLEs
    stmt = Statement(Relation_SumDef(), oracles=(L1, G1), head=L1)
    claim = Statement(Relation_Eval(), oracles=(L1,), point={z: r}, value=v)
    started = stmt.relation.start(stmt, [claim])
    assert started.check()
    iop = _sumdef_iop(_IntDomain())
    assert iop.run(started)
    # The started statement is the child "/0" of stmt; no RLC for one claim.
    assert iop.transcript.order[0] == "sumcheck/0/0/g0"
    assert len(iop.transcript.entries["sumcheck/0/0/g0"].result()) == 4  # degree 3
    bad = stmt.relation.start(
        stmt, [Statement(Relation_Eval(), oracles=(L1,), point={z: r}, value=v + 1)]
    )
    assert not bad.check()
    assert not _sumdef_iop(_IntDomain(1)).run(bad)
    # Two claims, one of them wrong: the random linear combination exposes it.
    v5 = (1 - 5) * 10 + 5 * 21
    other = Statement(Relation_Eval(), oracles=(L1,), point={z: 5}, value=v5)
    two = stmt.relation.start(stmt, [other, claim])
    assert two.check()
    assert _sumdef_iop(_IntDomain()).run(two)
    two_bad = stmt.relation.start(stmt, [other, bad.parents[1]])
    assert not two_bad.check()
    assert not _sumdef_iop(_IntDomain(1)).run(two_bad)
    ring = Ring(1024, prime_size=[49], split_degree=4)
    assert SumDefStart().soundness_error(started, ring) == 0
    assert SumDefStart().soundness_error(two, ring) == 1 / ring.exceptional_set_size
    assert _messages(iop) == ONE_LEVEL_MESSAGES


class _Query(Relation):
    """`L(point) = value` stated of an implicit oracle L by a headless
    statement of a reduction (as a circuit's output test does); its protocol
    emits the evaluation claim, with no message."""

    name = "query"
    fields = ("oracles", "point", "value")


class _Emit(Protocol):
    reduces_from = _Query
    reduces_to = (Relation_Eval,)

    async def prove(self, prover, statements):
        return self._claim(statements)

    async def verify(self, verifier, statements):
        return self._claim(statements)

    def _claim(self, statements):
        (s,) = statements
        return [self.reduce([s], point=s.point, value=s.value)]


class _Tree(Relation):
    """The two-level product tree of W evaluates to `value` at `point`."""

    name = "tree"
    fields = ("oracles", "point", "value")


def _tree_layering(root):
    (W,) = root.oracles
    (L1, G1), (L0, G0) = _sum_tree(W, 2)
    (z,) = L0.variables
    (zr,) = root.point.keys()
    point = {z: root.point[zr]}
    return [
        [root.reduce_to(_Query, oracles=[L0], point=point, value=root.value)],
        [root.reduce_to(Relation_SumDef, oracles=(L0, G0), head=L0)],
        [root.reduce_to(Relation_SumDef, oracles=(L1, G1), head=L1)],
    ]


def test_sumdef_two_levels_bundle_claims():
    # The two claims on L1 left by L0's sumcheck meet in one started
    # statement, combined by a random linear combination.
    W = _table(_vars("w0", "w1", "w2"), [2, 3, 5, 7, 11, 13, 17, 19])
    # L1(z) = W(z, 0) W(z, 1) = [22, 39, 85, 133]; L0(z) = L1(z, 0) L1(z, 1)
    # = [1870, 5187]; L0~(6) = (1 - 6) 1870 + 6 * 5187.
    r = 6
    v = -5 * 1870 + 6 * 5187
    terminal = []

    class _Record(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    iop = _sumdef_iop(_IntDomain())
    iop.register(_Tree, compose(_tree_layering, name="tree"))
    iop.register(_Query, _Emit())
    iop.register(Relation_Eval, _Record())
    root = Statement(_Tree(), oracles=[W], point={_vars("z")[0]: r}, value=v)
    assert iop.run(root)
    order = iop.transcript.order
    assert "sumdef/2/0/a0" in order and "sumdef/2/0/a1" in order  # the RLC
    assert len(terminal) == 2 and all(s.oracles[0] is W for s in terminal)
    assert all(s.check() for s in terminal)
    heads = sorted(
        (
            list(t.table)
            for o, t in iop.prover.witnesses.items()
            if isinstance(o, Implicit)
        ),
        key=len,
    )
    assert heads == [[1870, 5187], [22, 39, 85, 133]]
    assert _messages(iop) == TWO_LEVEL_MESSAGES
    bad = Statement(_Tree(), oracles=[W], point={_vars("z")[0]: r}, value=v + 1)
    iop = _sumdef_iop(_IntDomain(1))
    iop.register(_Tree, compose(_tree_layering, name="tree"))
    iop.register(_Query, _Emit())
    assert not iop.run(bad)


def test_a_head_waits_for_every_claim(monkeypatch):
    # Claims on c come from two layers (the sum in L0, the Hadamard headed at
    # k in L1); c is started once, in L2, with both. The layer order, not a
    # parking rule, decides when.
    starts = []
    start = _Hadamard.start

    def counting(self, statement, claims):
        starts.append(len(claims))
        return start(self, statement, claims)

    monkeypatch.setattr(_Hadamard, "start", counting)

    def layering(root):  # sum b (d o e) + sum b k, with k = (d o e) o e
        b, d, e = root.oracles
        c, k = Implicit(b.variables), Implicit(b.variables)
        bc = VirtualOracle.product([b, c])
        bk = VirtualOracle.product([b, k])
        return [
            [
                root.reduce_to(Relation_Sum, oracles=[bc], value=290),
                root.reduce_to(Relation_Sum, oracles=[bk], value=root.total - 290),
            ],
            [root.reduce_to(_Hadamard, oracles=(k, c, e), head=k)],
            [root.reduce_to(_Hadamard, oracles=(c, d, e), head=c)],
        ]

    # (d o e) o e = [20, 54, 112, 200]; sum b * that = 20 + 108 + 336 + 800
    stmt = _chain_instance(total=290 + 1264)
    iop = _iop(_IntDomain(), layering=layering)
    assert iop.run(stmt)
    assert starts == [1, 2, 1, 2]  # per party: k with one claim, c with both
