# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the protocol machinery (registry, transcript, driver loops) and
the sumcheck protocol over pure-Python and ring-backed MLE oracles:
honest runs
accept, false claims are rejected in-round, and a prover that answers
consistently for a different polynomial is caught by the terminal oracle
query.
"""

from vfhe.arith import Ring
from vfhe.piop import (
    IOP,
    MLE,
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


def _coeff_mle():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    # f = 1 + 2*x0 + 3*x1 + 4*x0*x1; sum over {0,1}^2 is 18.
    return v, MLE(variables=v, coefficients=[1, 2, 3, 4])


def test_sumcheck_accepts_coeff_basis():
    _, f = _coeff_mle()
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
    _, f = _coeff_mle()
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
        fake = MLE(variables=list(f.variables), coefficients=[0, 3, 4, 3])
        fake_statement = Statement(statement.relation, oracles=[fake], value=17)
        return await super().prove(prover, [fake_statement])


def test_sumcheck_terminal_query_catches_consistent_liar():
    _, f = _coeff_mle()
    # All round checks pass (the fake polynomial really sums to 17); the
    # terminal Relation_Eval query against the real oracle must reject.
    iop = _setup(_IntDomain(), protocol=_LyingSumcheck())
    stmt = Statement(Relation_Sum(), oracles=[f], value=17)
    assert not iop.run(stmt)


def test_sumcheck_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]

    iop = _setup(ring)
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])  # sum 10
    assert iop.run(Statement(Relation_Sum(), oracles=[f], value=10))
    # This run took the native (C kernel) path; check the round-0 message:
    # LSB pairs (T0,T1),(T2,T3), so g(0) = 1+3 = 4 and g(1) = 2+4 = 6.
    g0, g1 = iop.transcript.entries["sumcheck/g0"].result()
    assert g0 == 4 and g1 == 6
    # The shared oracle survives the prover's in-place folds untouched.
    assert f.num_vars == 2 and f.table[0].get_polynomial()[0] == 1

    # An IOP is single-use: fresh transcript for the rejection run.
    iop = _setup(ring)
    assert not iop.run(Statement(Relation_Sum(), oracles=[f], value=11))


def test_verifier_challenge_samples_and_publishes():
    iop = _setup(_IntDomain())
    r = iop.verifier.challenge("r0")
    assert r == 2  # first value of the deterministic stub domain
    assert iop.transcript.entries["r0"].result() == 2
    assert iop.transcript.order == ["r0"]


def test_verifier_challenge_bits_samples_and_publishes():
    # The second sampler: raw coins, published so protocols can expand them
    # into derived randomness on both sides. Compute-if-absent like
    # `challenge`, and shapeless — the byte length is the only structure.
    iop = _setup(_IntDomain())
    seed = iop.verifier.challenge_bits("q0")
    assert isinstance(seed, bytes) and len(seed) == 32  # 256 bits default
    assert iop.verifier.challenge_bits("q0") == seed  # recorded, not resampled
    assert iop.transcript.entries["q0"].result() == seed
    assert len(iop.verifier.challenge_bits("q1", bits=100)) == 13  # ceil(100/8)


def test_sumcheck_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f = _coeff_mle()
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
    f = MLE(variables=v, coefficients=[1, 2, 3, 4])
    g = MLE(variables=v, coefficients=[2, 0, 1, 0])  # 2 + x1
    return v, f, g  # sum of f*g over {0,1}^2 is 50 (see test_piop.py)


def test_sumcheckprod_accepts_coeff_basis():
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
    h = MLE(variables=v, coefficients=[0, 1, 0, 0])  # x0
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
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
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
        fake_f = MLE(variables=list(f.variables), coefficients=[0, 2, 3, 4])
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
    f = MLE(variables=v, coefficients=[1, 2])  # f(0)=1, f(1)=3, sum 4
    g = MLE(variables=v, coefficients=[1, 0])  # constant 1
    iop = _prod_setup(_IntDomain())
    # Honest round message for f*g = (1 + 2X)·1 as evaluations at t=0,1,2,
    # then tampered values: r0 = 2 (stub domain), so f(r)=5, g(r)=1;
    # the claim becomes g_0(2)=5, but the tampered product is 5*2=10.
    iop.transcript.write("sumcheckprod/g0", (1, 3, 5))
    iop.transcript.write("sumcheckprod/vals", (5, 2))
    stmt = Statement(Relation_SumProd(), oracles=[f, g], value=4)
    assert not iop.loop.run_until_complete(iop.verifier.verify(stmt))


def test_round_evals_any_variable_position():
    # The round-message kernels are generic over the round variable's
    # position: pairs (LSB), halves (MSB), and the strided generic fallback
    # (middle) must all agree with the pure-Python path.
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4, 5, 6, 7, 8])
    g = MLE(ring=ring, variables=v, evaluations=[2, 1, 3, 2, 4, 3, 5, 4])
    for var in v:
        native = Sumcheck.round_evals_native(f, var)
        python = Sumcheck._round_evals_python(f, var)
        assert all(a == b for a, b in zip(native, python, strict=True))
        native3 = SumcheckProd.prod2_round_evals_native(f, g, var)
        python3 = SumcheckProd._prod_round_evals_python([f, g], var)
        assert all(a == b for a, b in zip(native3, python3, strict=True))
    # The dispatching entry points default to the first variable.
    assert all(
        a == b
        for a, b in zip(
            Sumcheck.round_evals(f), Sumcheck.round_evals(f, v[0]), strict=True
        )
    )
    assert all(
        a == b
        for a, b in zip(
            SumcheckProd.prod_round_evals([f, g]),
            SumcheckProd.prod_round_evals([f, g], v[0]),
            strict=True,
        )
    )


def test_mixed_representation_factors_agree_with_native():
    # The round-message helpers own the native decision per call: a native
    # pair goes to the C kernel, while a mixed pair (one factor per basis /
    # backing) or three factors fall back to Python — with equal messages.
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    dense = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    dense2 = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    # Same polynomial as dense2, in the monomial basis (non-native).
    coeff2 = dense2.to_coefficients()
    native = SumcheckProd.prod_round_evals([dense, dense2])
    mixed = SumcheckProd.prod_round_evals([dense, coeff2])
    assert all(a == b for a, b in zip(native, mixed, strict=True))


def test_sumcheckprod_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f, g = _prod_oracles()
    stmt = Statement(Relation_SumProd(), oracles=[f, g], value=50)
    scp = SumcheckProd()
    # degree k=2, n=2 variables.
    assert scp.soundness_error(stmt, ring) == 4 / min(ring.primes) ** ring.split_degree
