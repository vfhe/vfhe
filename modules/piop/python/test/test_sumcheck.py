# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the protocol machinery (registry, transcript, driver loops) and
the sumcheck protocol over ML_Polynomial and MLE_Dense oracles: honest runs
accept, false claims are rejected in-round, and a prover that answers
consistently for a different polynomial is caught by the terminal oracle
query.
"""

from vfhe.arith import Ring
from vfhe.piop import (
    IOP,
    ML_Polynomial,
    MLE_Dense,
    MLE_Variable,
    Relation_Sum,
    Relation_SumProd,
    Statement,
    Sumcheck,
    SumcheckProd,
)


class _IntDomain:
    """Deterministic stand-in domain for integer-coefficient oracles."""

    def __init__(self, values=(2, 3, 5, 7)):
        self._values = iter(values)

    def random_exceptional(self):
        return next(self._values)


def _setup(domain, protocol=None) -> IOP:
    iop = IOP(domain=domain)
    iop.register(Relation_Sum, protocol or Sumcheck())
    return iop


def _mlp():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    # f = 1 + 2*x0 + 3*x1 + 4*x0*x1; sum over {0,1}^2 is 18.
    return v, ML_Polynomial(variables=v, coefficients=[1, 2, 3, 4])


def test_sumcheck_accepts_ml_polynomial():
    _, f = _mlp()
    iop = _setup(_IntDomain())
    stmt = Statement(Relation_Sum(), oracles=[f], value=18)
    assert iop.run(stmt)
    # Two rounds, in canonical write order: round polynomial, then challenge.
    assert iop.transcript.order == [
        "sumcheck/g0",
        "sumcheck/r0",
        "sumcheck/g1",
        "sumcheck/r1",
    ]


def test_sumcheck_rejects_false_claim():
    _, f = _mlp()
    iop = _setup(_IntDomain())
    stmt = Statement(Relation_Sum(), oracles=[f], value=17)
    assert not iop.run(stmt)


class _LyingSumcheck(Sumcheck):
    """Answers every round consistently — for a different polynomial."""

    async def prove(self, prover, statements):
        (statement,) = statements
        (f,) = statement.oracles
        # Hypercube values (0, 3, 4, 10): really sums to 17, so every round
        # check passes and only the terminal oracle query can catch the lie.
        fake = ML_Polynomial(variables=list(f.variables), coefficients=[0, 3, 4, 3])
        fake_statement = Statement(statement.relation, oracles=[fake], value=17)
        return await super().prove(prover, [fake_statement])


def test_sumcheck_terminal_query_catches_consistent_liar():
    _, f = _mlp()
    # All round checks pass (the fake polynomial really sums to 17); the
    # terminal Relation_Eval query against the real oracle must reject.
    iop = _setup(_IntDomain(), protocol=_LyingSumcheck())
    stmt = Statement(Relation_Sum(), oracles=[f], value=17)
    assert not iop.run(stmt)


def test_sumcheck_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]

    iop = _setup(ring)
    f = MLE_Dense(ring=ring, variables=v, evaluations=[1, 2, 3, 4])  # sum 10
    assert iop.run(Statement(Relation_Sum(), oracles=[f], value=10))
    # This run took the native (C kernel) path; check the round-0 message:
    # LSB pairs (T0,T1),(T2,T3), so g(0) = 1+3 = 4 and g(1) = 2+4 = 6.
    g0, g1 = iop.transcript.entries["sumcheck/g0"].result()
    assert g0 == 4 and g1 == 6
    # The shared oracle survives the prover's in-place folds untouched.
    assert f.num_vars == 2 and f.py_refs[0].get_polynomial()[0] == 1

    # An IOP is single-use: fresh transcript for the rejection run.
    iop = _setup(ring)
    assert not iop.run(Statement(Relation_Sum(), oracles=[f], value=11))


def test_verifier_challenge_samples_and_publishes():
    iop = _setup(_IntDomain())
    r = iop.verifier.challenge("r0")
    assert r == 2  # first value of the deterministic stub domain
    assert iop.transcript.entries["r0"].result() == 2
    assert iop.transcript.order == ["r0"]


def test_sumcheck_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f = _mlp()
    stmt = Statement(Relation_Sum(), oracles=[f], value=18)
    sc = Sumcheck()
    assert sc.soundness_error(stmt, ring) == 2 / min(ring.primes) ** ring.split_degree
    assert sc.soundness_error(stmt, _IntDomain()) is None


def _prod_setup(domain) -> IOP:
    iop = IOP(domain=domain)
    iop.register(Relation_SumProd, SumcheckProd())
    return iop


def _prod_oracles():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = ML_Polynomial(variables=v, coefficients=[1, 2, 3, 4])
    g = ML_Polynomial(variables=v, coefficients=[2, 0, 1, 0])  # 2 + x1
    return v, f, g  # sum of f*g over {0,1}^2 is 50 (see test_piop.py)


def test_sumcheckprod_accepts_ml_polynomial():
    _, f, g = _prod_oracles()
    iop = _prod_setup(_IntDomain())
    assert iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=50))
    # Degree-2 rounds then the per-factor values, in canonical write order.
    assert iop.transcript.order == [
        "sumcheckprod/g0",
        "sumcheckprod/r0",
        "sumcheckprod/g1",
        "sumcheckprod/r1",
        "sumcheckprod/vals",
    ]
    # One terminal Relation_Eval claim per factor.
    vals = iop.transcript.entries["sumcheckprod/vals"].result()
    assert len(vals) == 2


def test_sumcheckprod_three_factors():
    v, f, g = _prod_oracles()
    h = ML_Polynomial(variables=v, coefficients=[0, 1, 0, 0])  # x0
    iop = _prod_setup(_IntDomain())
    assert iop.run(Statement(Relation_SumProd(), oracles=[f, g, h], value=36))
    iop = _prod_setup(_IntDomain())
    assert not iop.run(Statement(Relation_SumProd(), oracles=[f, g, h], value=35))


def test_sumcheckprod_rejects_false_claim():
    _, f, g = _prod_oracles()
    iop = _prod_setup(_IntDomain())
    assert not iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=49))


def test_sumcheckprod_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE_Dense(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    g = MLE_Dense(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    iop = _prod_setup(ring)
    assert iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=27))
    # Native (C kernel) round-0 message, evaluations at t = 0, 1, 2:
    # g(0) = 1·2 + 3·3 = 11, g(1) = 2·2 + 4·3 = 16,
    # g(2) = (2·2-1)(2·2-2) + (2·4-3)(2·3-3) = 3·2 + 5·3 = 21.
    e0, e1, e2 = iop.transcript.entries["sumcheckprod/g0"].result()
    assert e0 == 11 and e1 == 16 and e2 == 21
    iop = _prod_setup(ring)
    assert not iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=28))


class _LyingSumcheckProd(SumcheckProd):
    """Answers every round consistently — for a different first factor."""

    async def prove(self, prover, statements):
        (statement,) = statements
        f, g = statement.oracles
        fake_f = ML_Polynomial(variables=list(f.variables), coefficients=[0, 2, 3, 4])
        # sum of fake_f*g over {0,1}^2 is 40 — consistent with the false claim.
        fake = Statement(statement.relation, oracles=[fake_f, g], value=40)
        return await super().prove(prover, [fake])


def test_sumcheckprod_terminal_evals_catch_consistent_liar():
    _, f, g = _prod_oracles()
    iop = _prod_setup(_IntDomain())
    iop.register(Relation_SumProd, _LyingSumcheckProd())
    stmt = Statement(Relation_SumProd(), oracles=[f, g], value=40)
    assert not iop.run(stmt)  # rounds pass; the per-factor eval queries fail


def test_sumcheckprod_rejects_wrong_final_values():
    # Consistent rounds but tampered per-factor values: the product check
    # (prod v_j == final claim) must reject before any eval claim is emitted.
    v = [MLE_Variable("x0")]
    f = ML_Polynomial(variables=v, coefficients=[1, 2])  # f(0)=1, f(1)=3, sum 4
    g = ML_Polynomial(variables=v, coefficients=[1, 0])  # constant 1
    iop = _prod_setup(_IntDomain())
    # Honest round message for f*g = (1 + 2X)·1 as evaluations at t=0,1,2,
    # then tampered values: r0 = 2 (stub domain), so f(r)=5, g(r)=1;
    # the claim becomes g_0(2)=5, but the tampered product is 5*2=10.
    iop.transcript.write("sumcheckprod/g0", (1, 3, 5))
    iop.transcript.write("sumcheckprod/vals", (5, 2))
    stmt = Statement(Relation_SumProd(), oracles=[f, g], value=4)
    assert not iop.loop.run_until_complete(iop.verifier.verify(stmt))


def test_native_delegation_predicates():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    dense = MLE_Dense(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    _, mlp = _mlp()

    sc, scp = Sumcheck(), SumcheckProd()
    ring_iop, int_iop = IOP(domain=ring), IOP(domain=_IntDomain())

    stmt = Statement(Relation_Sum(), oracles=[dense], value=10)
    assert sc.native_supported(ring_iop, stmt)
    assert not sc.native_supported(int_iop, stmt)  # unsupported domain
    assert not sc.native_supported(
        ring_iop, Statement(Relation_Sum(), oracles=[mlp], value=18)
    )  # oracle not MLE_Dense

    dense2 = MLE_Dense(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    prod2 = Statement(Relation_SumProd(), oracles=[dense, dense2], value=27)
    assert scp.native_supported(ring_iop, prod2)
    prod3 = Statement(
        Relation_SumProd(), oracles=[dense, dense2, dense], value=0
    )
    assert not scp.native_supported(ring_iop, prod3)  # native path is k=2 only


def test_sumcheckprod_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f, g = _prod_oracles()
    stmt = Statement(Relation_SumProd(), oracles=[f, g], value=50)
    scp = SumcheckProd()
    # degree k=2, n=2 variables.
    assert scp.soundness_error(stmt, ring) == 4 / min(ring.primes) ** ring.split_degree
