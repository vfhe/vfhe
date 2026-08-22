# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the relation / statement layer of vfhe.piop: the ideal deciders
of Relation_Sum / Relation_Zero / Relation_Eval over coefficient-basis and
ring-backed evaluation-basis MLE oracles, statement reduction
chaining, and future resolution.
"""

import asyncio

import pytest
from vfhe.arith import Ring
from vfhe.piop import (
    IOP,
    MLE,
    MLE_Variable,
    Protocol,
    Relation_Eval,
    Relation_Sum,
    Relation_SumProd,
    Relation_Zero,
    Statement,
    Value,
)


def _coeff_mle():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    # f = 1 + 2*x0 + 3*x1 + 4*x0*x1; over {0,1}^2: 1 + 3 + 4 + 10 = 18.
    return v, MLE(variables=v, coefficients=[1, 2, 3, 4])


def test_relation_sum_coeff_basis():
    _, f = _coeff_mle()
    assert Statement(Relation_Sum(), oracles=[f], value=18).check()
    assert not Statement(Relation_Sum(), oracles=[f], value=17).check()


def test_relation_zero_coeff_basis():
    v, f = _coeff_mle()
    zero = MLE(variables=v, coefficients=[0, 0, 0, 0])
    assert Statement(Relation_Zero(), oracles=[zero]).check()
    assert not Statement(Relation_Zero(), oracles=[f]).check()


def test_relation_eval_coeff_basis():
    v, f = _coeff_mle()
    point = {v[0]: 2, v[1]: 3}  # f(2, 3) = 1 + 4 + 9 + 24 = 38
    assert Statement(Relation_Eval(), oracles=[f], point=point, value=38).check()
    assert not Statement(
        Relation_Eval(), oracles=[f], point=point, value=39
    ).check()


def test_relations_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    zero = MLE(ring=ring, variables=v, evaluations=[0, 0, 0, 0])

    assert Statement(Relation_Sum(), oracles=[f], value=10).check()
    assert not Statement(Relation_Sum(), oracles=[f], value=11).check()
    assert Statement(Relation_Zero(), oracles=[zero]).check()
    assert not Statement(Relation_Zero(), oracles=[f]).check()
    # f(2, 3) = 9 (see test_mle.py::test_mle_full_evaluation).
    point = {v[0]: 2, v[1]: 3}
    assert Statement(Relation_Eval(), oracles=[f], point=point, value=9).check()


def test_relation_sumprod():
    v, f = _coeff_mle()
    g = MLE(variables=v, coefficients=[2, 0, 1, 0])  # 2 + x1
    # f*g over {0,1}^2: 1*2 + 3*2 + 4*3 + 10*3 = 50
    assert Statement(Relation_SumProd(), oracles=[f, g], value=50).check()
    assert not Statement(Relation_SumProd(), oracles=[f, g], value=49).check()
    h = MLE(variables=v, coefficients=[0, 1, 0, 0])  # x0
    # f*g*h: 0 + 3*2*1 + 0 + 10*3*1 = 36
    assert Statement(Relation_SumProd(), oracles=[f, g, h], value=36).check()


def test_relation_sumprod_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    # elementwise products summed: 1*2 + 2*2 + 3*3 + 4*3 = 27
    assert Statement(Relation_SumProd(), oracles=[f, g], value=27).check()
    assert not Statement(Relation_SumProd(), oracles=[f, g], value=28).check()


def test_reduce_to_chaining():
    v, f = _coeff_mle()
    sum_stmt = Statement(Relation_Sum(), oracles=[f], value=18)
    eval_stmt = sum_stmt.reduce_to(
        Relation_Eval(), point={v[0]: 2, v[1]: 3}, value=38
    )
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
    stmt = Statement(Relation_Sum(), oracles=[f], value=18)
    assert iop.run(stmt)
    assert iop.prover.state["path"] == iop.verifier.state["path"] == "/0"
    assert stmt._children == 0  # the parties drove forks, not the root


def test_statement_dynamic_fields():
    _, f = _coeff_mle()
    stmt = Statement(Relation_Sum(), oracles=[f], value=18)
    assert list(stmt.fields) == ["oracles", "value"]  # canonical order
    with pytest.raises(TypeError):
        Statement(Relation_Sum(), oracles=[f], point={})  # not a Sum field
    assert Statement(Relation_Sum(), oracles=[f]).value is None  # declared, unset
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
    assert iop.run(Statement(Relation_Zero(), oracles=[zero]))
    assert sizes == [2, 2]  # both parties got the two Sum claims in one bundle


def test_relation_eval_commitment_only_has_no_decider():
    # An eval claim may carry a commitment instead of (or beside) the
    # oracle; without the oracle there is no witness-free decider — such
    # claims must be discharged by a registered PCS protocol.
    stmt = Statement(Relation_Eval(), commitment=object(), point={}, value=0)
    with pytest.raises(NotImplementedError, match="commitment-only"):
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
            Relation_Eval(), oracles=[f], point={v[0]: r0, v[1]: r1}, value=claimed
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
