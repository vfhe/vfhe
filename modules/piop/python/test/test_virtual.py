# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Defined oracles: virtual oracles in a sumcheck and their free reduction
(VirtualEval), the oracle kinds, and Fiat-Shamir over statements carrying
them. Sum definitions and the Composer are in test_composition.py."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring, SparseMLE
from vfhe.piop import (
    IOP,
    Implicit,
    OracleKind,
    Protocol,
    Relation,
    Relation_Eval,
    Relation_Sum,
    Relation_Zero,
    Statement,
    Sumcheck,
    VirtualEval,
    VirtualOracle,
    element_digest,
    oracle_kind,
)
from vfhe.piop.virtual import resolve_table


class _IntDomain:
    """Deterministic integer challenges (any nonzero integer is exceptional
    for exact integer arithmetic)."""

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random_exceptional(self):
        return self._rng.randrange(2, 1 << 20)


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _iop(domain=None, **kw):
    iop = IOP(domain=domain or _IntDomain(), **kw)
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


def test_oracle_kinds():
    x = _vars("x0")
    f = MLE(variables=x, evaluations=[1, 2])
    g = VirtualOracle.product([f])
    assert oracle_kind(f) is OracleKind.plain
    assert (
        oracle_kind(MLE(variables=x, evaluations=[1, 2], public=True))
        is OracleKind.public
    )
    assert oracle_kind(f, commitment=object()) is OracleKind.committed
    assert oracle_kind(g) is OracleKind.virtual
    assert oracle_kind(Implicit(x)) is OracleKind.implicit
    v = _vars("y0", "y1", "y2")
    assert oracle_kind(SparseMLE(variables=v, evaluations={5: 3}, public=True)) is (
        OracleKind.public
    )
    # A table tagged by a commitment scheme is committed; one carrying both
    # marks is refused (public and committed are two discharges, not two
    # properties).
    tagged = MLE(variables=x, evaluations=[1, 2])
    tagged.kind = OracleKind.committed  # pyright: ignore[reportAttributeAccessIssue]
    assert oracle_kind(tagged) is OracleKind.committed
    both = MLE(variables=x, evaluations=[1, 2], public=True)
    both.kind = OracleKind.committed  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="public or committed"):
        oracle_kind(both)


def test_sparse_mle_digest_is_canonical():
    """A sparse table's digest depends on its entries, not on their insertion
    order — the framework keys parked claims on it."""
    v = _vars("x0", "x1", "x2")
    s = SparseMLE(variables=v, evaluations={5: 3, 2: 1})
    same = SparseMLE(variables=v, evaluations={2: 1, 5: 3})
    assert element_digest(s) == element_digest(same)
    assert element_digest(s) != element_digest(
        SparseMLE(variables=v, evaluations={2: 1})
    )


def test_virtual_products_form_sumcheck():
    # G(x0, x1) = f(x0, x1) * g(x0, x1) + 3 * h(x1): degree 2, a term over
    # a subset of the variables. The sumcheck runs on the products form and
    # ends in an Eval claim on G that VirtualEval opens into claims on f,
    # g and h, decided by the verifier in the ideal model.
    x = _vars("x0", "x1")
    x1 = x[1]
    f = MLE(variables=x, evaluations=[1, 2, 3, 4])
    g = MLE(variables=x, evaluations=[5, 6, 7, 8])
    h = MLE(variables=[x1], evaluations=[9, 10])
    G = VirtualOracle(x, [f, g, h], terms=[(1, (0, 1)), (3, (2,))])
    assert G.degree == 2
    total = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8 + 3 * (9 + 9 + 10 + 10)
    stmt = Statement(Relation_Sum(), oracles=[G], value=total)
    assert stmt.check()
    iop = _iop()
    assert iop.run(stmt)
    assert iop.transcript.order[:2] == ["sumcheck/g0", "sumcheck/r0"]
    assert len(iop.transcript.entries["sumcheck/g0"].result()) == 3  # degree 2
    assert "virtual/0/vals" in iop.transcript.order
    assert not _iop().run(Statement(Relation_Sum(), oracles=[G], value=total + 1))


def test_virtual_same_oracle_under_two_maps():
    # G(x, y) = W(x) * W(y): one oracle, two constituents, two renamings;
    # sum over the cube is (sum W)^2, and the reduction ends in two claims
    # on W at two points.
    w = _vars("w0", "w1")
    xs, ys = _vars("x0", "x1"), _vars("y0", "y1")
    W = MLE(variables=w, evaluations=[1, 2, 3, 4])
    maps = [dict(zip(w, xs, strict=True)), dict(zip(w, ys, strict=True))]
    G = VirtualOracle(xs + ys, [W, W], terms=[(1, (0, 1))], maps=maps)
    stmt = Statement(Relation_Sum(), oracles=[G], value=100)
    assert stmt.check()
    terminal = []

    class _Record(Protocol):
        reduces_from = Relation_Eval

        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    iop = _iop()
    iop.register(Relation_Eval, _Record())  # plain oracles
    assert iop.run(stmt)
    assert len(terminal) == 2 and all(s.oracles[0] is W for s in terminal)
    assert terminal[0].point != terminal[1].point
    assert all(s.check() for s in terminal)


def test_virtual_claims_dedupe_and_skip_public():
    x = _vars("x0", "x1")
    W = MLE(variables=x, evaluations=[1, 2, 3, 4])
    E = MLE(variables=x, evaluations=[1, 0, 0, 1], public=True)
    # G = E * W + W: three constituents, one public, W twice at the same point.
    G = VirtualOracle(x, [E, W, W], terms=[(1, (0, 1)), (1, (2,))])
    claims = G.claims({x[0]: 5, x[1]: 7})
    assert len(claims) == 1
    c, pt, idxs = claims[0]
    assert c is W and pt == {x[0]: 5, x[1]: 7} and idxs == [1, 2]
    assert not G.public
    assert VirtualOracle.product([E]).public
    # Symbolic binding: constituents untouched, sources become constants.
    H = G.evaluate({x[0]: 5})
    assert H.variables == [x[1]] and G.variables == x
    assert H.maps[1][x[0]] == 5 and H.constituents[1] is W


def test_parking_rule_waits_for_all_claims_on_an_oracle():
    # Zero -> an Eval claim on f and a Sum claim on f (synthetic), the Eval
    # first in the worklist. The Eval must wait for the Sum to be reduced to
    # its own Eval, so the bundle protocol for plain oracles sees both at
    # once: without parking it would see [1, 1] per party.
    sizes = []

    class _Split(Protocol):
        reduces_from = Relation_Zero
        reduces_to = (Relation_Eval, Relation_Sum)

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            (s,) = statements
            (f,) = s.oracles
            (x0, x1) = f.variables
            return [
                self.reduce(statements, Relation_Eval, point={x0: 1, x1: 1}, value=4),
                self.reduce(statements, Relation_Sum, value=sum(f.table)),
            ]

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

    x = _vars("x0", "x1")
    f = MLE(variables=x, evaluations=[1, 2, 3, 4])
    iop = _iop()
    iop.register(Relation_Zero, _Split())
    iop.register(Relation_Eval, _Bundle())
    assert iop.run(Statement(Relation_Zero(), oracles=[f]))
    assert sizes == [2, 2]


def test_parking_rule_sees_an_oracle_read_through_the_index():
    # As above, but the second frontier statement mentions f only through
    # its relation's index: a statement of `_Reads(index={"table": f})` with
    # no instance oracle, whose protocol emits a claim on that table. The
    # Eval on f must wait for it, or the bundle sees [1, 1] per party.
    sizes = []

    class _Reads(Relation):
        name = "reads"
        fields = ()
        index_fields = ("table",)

    class _Split(Protocol):
        reduces_from = Relation_Zero
        reduces_to = (Relation_Eval, _Reads)

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            (s,) = statements
            (f,) = s.oracles
            (x0, x1) = f.variables
            return [
                self.reduce(statements, Relation_Eval, point={x0: 1, x1: 1}, value=4),
                self.reduce(statements, _Reads(index={"table": f})),
            ]

    class _Open(Protocol):
        reduces_from = _Reads
        reduces_to = (Relation_Eval,)

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            (s,) = statements
            f = s.index["table"]
            (x0, x1) = f.variables
            return [
                self.reduce(
                    statements, oracles=[f], point={x0: 0, x1: 0}, value=f.table[0]
                )
            ]

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

    x = _vars("x0", "x1")
    f = MLE(variables=x, evaluations=[1, 2, 3, 4])
    iop = _iop()
    iop.register(Relation_Zero, _Split())
    iop.register(_Reads, _Open())
    iop.register(Relation_Eval, _Bundle())
    assert iop.run(Statement(Relation_Zero(), oracles=[f]))
    assert sizes == [2, 2]


def test_public_eval_is_terminal_even_with_a_default_protocol():
    calls = []

    class _Never(Protocol):
        batching = True

        async def prove(self, prover, statements):
            calls.append("prove")
            return []

        async def verify(self, verifier, statements):
            calls.append("verify")
            return []

    x = _vars("x0")
    e = MLE(variables=x, evaluations=[3, 4], public=True)
    iop = _iop()
    iop.register(Relation_Eval, _Never())
    assert iop.run(Statement(Relation_Eval(), oracles=[e], point={x[0]: 2}, value=5))
    assert calls == []


def test_fiat_shamir_over_defined_oracles():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    w = _vars("w0", "w1")
    W = MLE(ring=ring, variables=w, evaluations=[2, 3, 5, 7])
    E = MLE(ring=ring, variables=w, evaluations=[1, 0, 0, 1], public=True)
    G = VirtualOracle(w, [E, W, W], terms=[(1, (0, 1)), (1, (2,))])  # E W + W
    # The sumcheck's challenges come from the chain; VirtualEval ends it.
    total = (2 + 7) + (2 + 3 + 5 + 7)
    stmt = Statement(Relation_Sum(), oracles=[G], value=total)
    proofs = []
    for _ in range(2):
        iop = _iop(domain=ring, fiat_shamir=True)
        proofs.append(iop.prove(stmt))
    assert proofs[0].labels == proofs[1].labels
    assert proofs[0].digest() == proofs[1].digest()
    assert _iop(domain=ring, fiat_shamir=True).verify(stmt, proofs[0])
    assert any(label.startswith("sumcheck") for label in proofs[0].labels)
    bad = Statement(Relation_Sum(), oracles=[G], value=total + 1)
    assert not _iop(domain=ring, fiat_shamir=True).verify(bad, proofs[0])


# --- restrictions ------------------------------------------------------------


def test_sparse_restrict_is_the_dense_slice():
    rng = random.Random(11)
    v = _vars("t", "x0", "x1", "x2")
    entries = {k: rng.randrange(1, 9) for k in rng.sample(range(16), 7)}
    s = SparseMLE(variables=v, evaluations=entries, public=True)
    dense = s.materialize()
    for fixed in ({v[0]: 1}, {v[0]: 0, v[2]: 1}, {v[3]: 0}, {}):
        sliced = s.restrict(fixed)
        assert sliced.public and sliced.variables == [u for u in v if u not in fixed]
        expected = dense.evaluate(fixed, in_place=False)
        assert list(sliced.materialize().table) == list(expected.table)
    # The whole table fixed: one entry, or none.
    k = next(iter(entries))
    point = {u: (k >> i) & 1 for i, u in enumerate(v)}
    assert s.restrict(point).evaluations == {0: entries[k]}
    # A slice, not a partial evaluation; and only this table's variables.
    with pytest.raises(ValueError, match="0 or 1"):
        s.restrict({v[0]: 2})
    with pytest.raises(ValueError, match="not a variable"):
        s.restrict({_vars("w")[0]: 0})


def test_restriction_form_and_resolution():
    t, x = _vars("t"), _vars("x0", "x1")
    r = _vars("r0", "r1")
    P = SparseMLE(variables=t + x, evaluations={0: 5, 3: 7, 6: 11})  # (t, x0, x1)
    r0 = VirtualOracle.restriction(P, r, {t[0]: 0})
    r1 = VirtualOracle.restriction(P, r, {t[0]: 1})
    assert (
        r0.is_restriction()
        and r0.variables == r
        and r0.maps[0]
        == {
            t[0]: 0,
            x[0]: r[0],
            x[1]: r[1],
        }
    )
    assert oracle_kind(r0) is OracleKind.virtual and not r0.public
    # Resolved: the slice at t, renamed to the restriction's variables.
    s0, s1 = resolve_table(r0), resolve_table(r1)
    assert isinstance(s0, SparseMLE) and s0.variables == r
    assert s0.evaluations == {0: 5, 3: 11} and s1.evaluations == {1: 7}
    # Through a witness: an implicit master whose stored table is sparse.
    h = Implicit(t + x)
    assert resolve_table(
        VirtualOracle.restriction(h, r, {t[0]: 1}), {h: P}
    ).evaluations == {1: 7}
    # A dense constituent: the partial evaluation, renamed.
    D = MLE(variables=t + x, evaluations=list(range(8)))
    d1 = resolve_table(VirtualOracle.restriction(D, r, {t[0]: 1}))
    assert isinstance(d1, MLE) and d1.variables == r
    assert list(d1.table) == [1, 3, 5, 7]
    # No fixed coordinate: a renaming.
    rn = resolve_table(VirtualOracle.restriction(D, t + r, {}))
    assert rn.variables == t + r and list(rn.table) == list(range(8))
    # Not restrictions: a product, a scaled term, a slice at a non-bit.
    with pytest.raises(TypeError, match="restriction"):
        resolve_table(VirtualOracle.product([P, D]))
    scaled = VirtualOracle(r, [P], [(2, (0,))], [r0.maps[0]])
    assert not scaled.is_restriction()
    with pytest.raises(TypeError):
        resolve_table(scaled)
    assert not r0.evaluate({r[0]: 3}).is_restriction()
    with pytest.raises(ValueError):
        VirtualOracle.restriction(P, r, {t[0]: 2})
    with pytest.raises(ValueError):  # too few names for the free variables
        VirtualOracle.restriction(P, r[:1], {t[0]: 1})
    # The sparse table is never materialized by resolve_table.
    assert resolve_table(P) is P
