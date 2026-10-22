# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the relation / statement layer of vfhe.piop: the ideal deciders
of Relation_Sum / Relation_Zero / Relation_Eval over coefficient-basis and
ring-backed evaluation-basis MLE oracles, statement reduction
chaining, future resolution, and the index of a relation (its declared
fields, its oracles, its digest).
"""

import asyncio

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.piop import (
    IOP,
    CommittedOracle,
    ImplicitOracle,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    TableOracle,
    Value,
    element_digest,
)
from vfhe.piop.relation import statement_oracles
from vfhe.piop.relations import Relation_Sum, Relation_SumProd, Relation_Zero


def _coeff_mle():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    # f = 1 + 2*x0 + 3*x1 + 4*x0*x1; over {0,1}^2: 1 + 3 + 4 + 10 = 18.
    return v, MLE(variables=v, coefficients=[1, 2, 3, 4])


def test_relation_sum_coeff_basis():
    _, f = _coeff_mle()
    assert Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18).check()
    assert not Statement(Relation_Sum(), oracles=[TableOracle(f)], value=17).check()


def test_relation_zero_coeff_basis():
    v, f = _coeff_mle()
    zero = MLE(variables=v, coefficients=[0, 0, 0, 0])
    assert Statement(Relation_Zero(), oracles=[TableOracle(zero)]).check()
    assert not Statement(Relation_Zero(), oracles=[TableOracle(f)]).check()


def test_relation_eval_coeff_basis():
    v, f = _coeff_mle()
    point = {v[0]: 2, v[1]: 3}  # f(2, 3) = 1 + 4 + 9 + 24 = 38
    assert Statement(
        Relation_Eval(), oracles=[TableOracle(f)], point=point, value=38
    ).check()
    assert not Statement(
        Relation_Eval(), oracles=[TableOracle(f)], point=point, value=39
    ).check()


def test_relations_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    zero = MLE(ring=ring, variables=v, evaluations=[0, 0, 0, 0])

    assert Statement(Relation_Sum(), oracles=[TableOracle(f)], value=10).check()
    assert not Statement(Relation_Sum(), oracles=[TableOracle(f)], value=11).check()
    assert Statement(Relation_Zero(), oracles=[TableOracle(zero)]).check()
    assert not Statement(Relation_Zero(), oracles=[TableOracle(f)]).check()
    # f(2, 3) = 9 (see test_mle.py::test_mle_full_evaluation).
    point = {v[0]: 2, v[1]: 3}
    assert Statement(
        Relation_Eval(), oracles=[TableOracle(f)], point=point, value=9
    ).check()


def test_relation_sumprod():
    v, f = _coeff_mle()
    g = MLE(variables=v, coefficients=[2, 0, 1, 0])  # 2 + x1
    # f*g over {0,1}^2: 1*2 + 3*2 + 4*3 + 10*3 = 50
    assert Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=50
    ).check()
    assert not Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=49
    ).check()
    h = MLE(variables=v, coefficients=[0, 1, 0, 0])  # x0
    # f*g*h: 0 + 3*2*1 + 0 + 10*3*1 = 36
    assert Statement(
        Relation_SumProd(),
        oracles=[TableOracle(f), TableOracle(g), TableOracle(h)],
        value=36,
    ).check()


def test_relation_sumprod_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    # elementwise products summed: 1*2 + 2*2 + 3*3 + 4*3 = 27
    assert Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=27
    ).check()
    assert not Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=28
    ).check()


def test_reduce_to_chaining():
    v, f = _coeff_mle()
    sum_stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18)
    eval_stmt = sum_stmt.reduce_to(Relation_Eval(), point={v[0]: 2, v[1]: 3}, value=38)
    assert eval_stmt.parents == (sum_stmt,)
    assert eval_stmt.oracles == sum_stmt.oracles  # inherited by default
    assert (sum_stmt.path, eval_stmt.path) == ("", "/0")
    assert sum_stmt.reduce_to(Relation_Zero).path == "/1"  # next child
    assert eval_stmt.check() and sum_stmt.check()


def test_run_forks_root_so_child_paths_agree():
    # Regression: IOP.run hands each party its own fork of the root. With a
    # shared root, the second party's reduction would draw the next child
    # counter value ("/1" instead of "/0"), desynchronizing every transcript
    # label namespaced by a child path.
    class _Recorder(Protocol):  # terminal-ish: records the path it was given
        reduces_from = Relation_Eval
        reduces_to = ()

        async def prove(self, prover, statements):
            prover.state["path"] = statements[0].path
            return []

        async def verify(self, verifier, statements):
            verifier.state["path"] = statements[0].path
            return []

    class _Emit(Protocol):  # Sum -> one Eval child, no messages
        reduces_from = Relation_Sum
        reduces_to = (Relation_Eval,)

        async def prove(self, prover, statements):
            return [self.reduce(statements, point={}, value=0)]

        async def verify(self, verifier, statements):
            return [self.reduce(statements, point={}, value=0)]

    _, f = _coeff_mle()
    iop = IOP()
    iop.register(Relation_Sum, _Emit())
    iop.register(Relation_Eval, _Recorder())
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18)
    assert iop.run(stmt)
    assert iop.prover.state["path"] == iop.verifier.state["path"] == "/0"
    assert stmt._children == 0  # the parties drove forks, not the root


def test_statement_names_oracles_not_tables():
    _, f = _coeff_mle()
    with pytest.raises(TypeError, match="TableOracle"):
        Statement(Relation_Sum(), oracles=[f], value=18)


def test_statement_dynamic_fields():
    _, f = _coeff_mle()
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18)
    assert list(stmt.fields) == ["oracles", "value"]  # canonical order
    with pytest.raises(TypeError):
        Statement(Relation_Sum(), oracles=[TableOracle(f)], point={})  # not a Sum field
    assert (
        Statement(Relation_Sum(), oracles=[TableOracle(f)]).value is None
    )  # declared, unset
    with pytest.raises(AttributeError):
        _ = stmt.nonsense


def test_batching_driver_groups_frontier():
    sizes = []

    class _Split(Protocol):  # Zero -> two Sum claims (synthetic)
        reduces_from = Relation_Zero
        reduces_to = (Relation_Sum,)

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            return [self.reduce(statements, value=0), self.reduce(statements, value=0)]

    class _CountBatch(Protocol):  # swallows all frontier Sum claims at once
        reduces_from = Relation_Sum
        reduces_to = ()
        batching = True

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            sizes.append(len(statements))
            return []

    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    zero = MLE(variables=v, coefficients=[0, 0, 0, 0])
    iop = IOP()
    iop.register(Relation_Zero, _Split())
    iop.register(Relation_Sum, _CountBatch())
    assert iop.run(Statement(Relation_Zero(), oracles=[TableOracle(zero)]))
    assert sizes == [2, 2]  # both parties got the two Sum claims in one bundle


def test_relation_eval_on_a_committed_oracle_has_no_decider():
    # An eval claim on a committed oracle has no table to query: no
    # witness-free decider, so such claims must be discharged by a
    # registered PCS protocol.
    f = CommittedOracle([], commitment=object())
    stmt = Statement(Relation_Eval(), oracles=(f,), point={}, value=0)
    with pytest.raises(NotImplementedError, match="committed oracle"):
        stmt.check()


def test_transcript_write_read():
    iop = IOP()

    async def run():
        t = iop.transcript
        reader = asyncio.ensure_future(t.read("g0"))  # read scheduled first
        t.write("g0", 5)
        t.write("r0", 7)
        assert await reader == 5
        assert await t.read("r0") == 7
        assert t.order == ["g0", "r0"]  # write order, not read order
        with pytest.raises(asyncio.InvalidStateError):
            t.write("g0", 6)  # labels are single-use
        return True

    assert iop.loop.run_until_complete(run())


def test_statement_resolution():
    async def run():
        v, f = _coeff_mle()
        r0, r1, claimed = Value(), Value(), Value()
        stmt = Statement(
            Relation_Eval(),
            oracles=[TableOracle(f)],
            point={v[0]: r0, v[1]: r1},
            value=claimed,
        )
        with pytest.raises(RuntimeError):
            stmt.check()  # challenges still pending
        r0.set_result(2)
        r1.set_result(3)
        claimed.set_result(38)
        await stmt.resolved()
        assert stmt.point == {v[0]: 2, v[1]: 3} and stmt.value == 38
        return stmt.check()

    assert asyncio.run(run())


# --- the index of a relation --------------------------------------------------


class _Indexed(Relation):
    """A relation with two declared index fields and no decider."""

    name = "indexed"
    fields = ("oracles",)
    index_fields = ("a", "b")


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def test_relation_index_fields_take_a_mapping_over_exactly_those_names():
    r = _Indexed(index={"b": 2, "a": 1})
    assert list(r.index) == ["a", "b"]  # stored in the declared order
    assert r.index == {"a": 1, "b": 2}
    with pytest.raises(TypeError):  # a missing name
        _Indexed(index={"a": 1})
    with pytest.raises(TypeError):  # an extra name
        _Indexed(index={"a": 1, "b": 2, "c": 3})
    with pytest.raises(TypeError):  # not a mapping at all
        _Indexed(index=(1, 2))
    with pytest.raises(TypeError):
        _Indexed()
    # No declared fields: the index stays the opaque object it is given.
    assert Relation_Sum(index=(1, 2)).index == (1, 2)
    assert Relation_Sum().index is None and Relation_Sum().index_oracles() == ()
    with pytest.raises(TypeError):  # an opaque index has no declared digest
        Relation_Sum(index=3).index_digest()


def test_index_oracles_walk_tuples_lists_and_mappings():
    x = _vars("x0")
    t1 = TableOracle(MLE(variables=x, evaluations=[1, 2]))
    t2 = TableOracle(MLE(variables=x, evaluations=[3, 4]))
    h = ImplicitOracle(x)
    r = _Indexed(index={"a": (t1, {"k": [t2, 5]}, "data"), "b": h})
    assert r.index_oracles() == (t1, t2, h)  # declared order, data skipped
    assert _Indexed(index={"a": 1, "b": "two"}).index_oracles() == ()
    stmt = Statement(r, oracles=(t2,))
    assert stmt.index is r.index
    # The instance oracles first, then the index oracles.
    assert statement_oracles(stmt) == (t2, t1, t2, h)


def test_index_digest_follows_the_declared_order_and_is_computed_once(monkeypatch):
    import vfhe.piop.relation as relation

    r1 = _Indexed(index={"a": 1, "b": 2})
    r2 = _Indexed(index={"b": 2, "a": 1})  # another insertion order
    assert r1.index_digest() == r2.index_digest()
    assert r1.index_digest() != _Indexed(index={"a": 2, "b": 1}).index_digest()
    # A statement's digest goes through it, so it is insertion-order free too.
    x = _vars("x0")
    t = TableOracle(MLE(variables=x, evaluations=[1, 2]))
    assert element_digest(Statement(r1, oracles=(t,))) == element_digest(
        Statement(r2, oracles=(t,))
    )
    # Computed once per instance: a second call hashes nothing.
    calls = []
    digest = relation.element_digest
    monkeypatch.setattr(
        relation, "element_digest", lambda v: calls.append(v) or digest(v)
    )
    fresh = _Indexed(index={"a": 1, "b": 2})
    first = fresh.index_digest()
    assert len(calls) == 4  # two names, two values
    assert fresh.index_digest() == first and len(calls) == 4


def test_element_digest_of_a_class_is_by_its_qualified_name():
    class _A(Relation):
        name = "same"

    class _B(Relation):
        name = "same"

    assert element_digest(_A) == element_digest(_A)
    assert element_digest(_A) != element_digest(_B)  # not by `name`
    assert element_digest(Relation_Sum) != element_digest(Relation_SumProd)


def test_a_root_whose_index_holds_an_implicit_oracle_is_refused():
    class _Reads(Relation):
        name = "reads"
        fields = ("oracles",)
        index_fields = ("t",)

        def check(self, statement):
            return True

    x = _vars("x0")
    f = TableOracle(MLE(variables=x, evaluations=[1, 2]))
    stmt = Statement(_Reads(index={"t": ImplicitOracle(x)}), oracles=(f,))
    with pytest.raises(TypeError, match="implicit"):
        IOP().run(stmt)
    with pytest.raises(TypeError, match="implicit"):
        IOP(fiat_shamir=True).prove(stmt)
    with pytest.raises(TypeError, match="implicit"):
        IOP(fiat_shamir=True).verify(
            stmt,
            IOP(fiat_shamir=True).prove(
                Statement(_Reads(index={"t": f}), oracles=(f,))
            ),
        )
    assert IOP().run(Statement(_Reads(index={"t": f}), oracles=(f,)))
