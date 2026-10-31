# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the protocol machinery (registry, transcript, driver loops) and
the sumcheck protocol over pure-Python and ring-backed MLE oracles:
honest runs
accept, false claims are rejected in-round, and a prover that answers
consistently for a different polynomial is caught by the terminal oracle
query.
"""

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring
from vfhe.piop import IOP, Proof, Statement, TableOracle
from vfhe.piop.protocols import Sumcheck, SumcheckProd
from vfhe.piop.protocols.sumcheck import rounds
from vfhe.piop.protocols.sumcheck.rounds import (
    _prod_round_evals_python,
    _round_evals_python,
    prod2_round_evals_native,
    prod3_round_evals_native,
    prod_round_evals,
    round_evals,
    round_evals_native,
)
from vfhe.piop.relation import aligned_tables
from vfhe.piop.relations import Relation_Sum, Relation_SumProd


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
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18)
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
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=17)
    assert not iop.run(stmt)


class _LyingSumcheck(Sumcheck):
    """Answers every round consistently — for a different polynomial."""

    async def prove(self, prover, statements):
        (statement,) = statements
        (f,) = statement.oracles
        # Hypercube values (0, 3, 4, 10): really sums to 17, so every round
        # check passes and only the terminal oracle query can catch the lie.
        fake = MLE(variables=list(f.variables), coefficients=[0, 3, 4, 3])
        fake_statement = Statement(
            statement.relation, oracles=[TableOracle(fake)], value=17
        )
        return await super().prove(prover, [fake_statement])


def test_sumcheck_terminal_query_catches_consistent_liar():
    _, f = _coeff_mle()
    # All round checks pass (the fake polynomial really sums to 17); the
    # terminal Relation_Eval query against the real oracle must reject.
    iop = _setup(_IntDomain(), protocol=_LyingSumcheck())
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=17)
    assert not iop.run(stmt)


def test_sumcheck_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]

    iop = _setup(ring)
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])  # sum 10
    assert iop.run(Statement(Relation_Sum(), oracles=[TableOracle(f)], value=10))
    # This run took the native (C kernel) path; check the round-0 message:
    # LSB pairs (T0,T1),(T2,T3), so g(0) = 1+3 = 4 and g(1) = 2+4 = 6.
    g0, g1 = iop.transcript.entries["sumcheck/g0"].result()
    assert g0 == 4 and g1 == 6
    # The shared oracle survives the prover's in-place folds untouched.
    assert f.num_vars == 2 and f.table[0].get_polynomial()[0] == 1

    # An IOP is single-use: fresh transcript for the rejection run.
    iop = _setup(ring)
    assert not iop.run(Statement(Relation_Sum(), oracles=[TableOracle(f)], value=11))


def test_verifier_challenge_samples_and_publishes():
    iop = _setup(_IntDomain())
    r = iop.loop.run_until_complete(iop.verifier.challenge("r0"))
    assert r == 2  # first value of the deterministic stub domain
    assert iop.transcript.entries["r0"].result() == 2
    assert iop.transcript.order == ["r0"]


def test_verifier_challenge_bits_samples_and_publishes():
    # The second sampler: raw coins, published so protocols can expand them
    # into derived randomness on both sides. Compute-if-absent like
    # `challenge`, and shapeless — the byte length is the only structure.
    iop = _setup(_IntDomain())
    seed = iop.loop.run_until_complete(iop.verifier.challenge_bits("q0"))
    assert isinstance(seed, bytes) and len(seed) == 32  # 256 bits default
    assert (
        iop.loop.run_until_complete(iop.verifier.challenge_bits("q0")) == seed
    )  # recorded, not resampled
    assert iop.transcript.entries["q0"].result() == seed
    assert (
        len(iop.loop.run_until_complete(iop.verifier.challenge_bits("q1", bits=100)))
        == 13
    )  # ceil(100/8)


def test_sumcheck_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f = _coeff_mle()
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=18)
    sc = Sumcheck()
    assert sc.soundness_error(stmt, ring) == 2 / min(ring.primes) ** ring.split_degree
    assert sc.soundness_error(stmt, _IntDomain()) is None


def test_soundness_error_defaults_to_the_oracle_domain():
    ring = Ring(64, prime_size=[30, 30], split_degree=1)
    x = [MLE_Variable(f"x{i}") for i in range(3)]
    f = MLE(ring=ring, variables=x, evaluations=list(range(8)))
    stmt = Statement(Relation_Sum(), oracles=[TableOracle(f)], value=28)
    sc = Sumcheck()
    assert sc.soundness_error(stmt) == sc.soundness_error(stmt, ring)


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
    assert iop.run(
        Statement(
            Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=50
        )
    )
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
    assert iop.run(
        Statement(
            Relation_SumProd(),
            oracles=[TableOracle(f), TableOracle(g), TableOracle(h)],
            value=36,
        )
    )
    iop = _prod_setup(_IntDomain())
    assert not iop.run(
        Statement(
            Relation_SumProd(),
            oracles=[TableOracle(f), TableOracle(g), TableOracle(h)],
            value=35,
        )
    )


def test_sumcheckprod_rejects_false_claim():
    _, f, g = _prod_oracles()
    iop = _prod_setup(_IntDomain())
    assert not iop.run(
        Statement(
            Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=49
        )
    )


def test_sumcheckprod_mle_dense():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    f = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    g = MLE(ring=ring, variables=v, evaluations=[2, 2, 3, 3])
    iop = _prod_setup(ring)
    assert iop.run(
        Statement(
            Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=27
        )
    )
    # Native (C kernel) round-0 message, evaluations at t = 0, 1, 2:
    # g(0) = 1·2 + 3·3 = 11, g(1) = 2·2 + 4·3 = 16,
    # g(2) = (2·2-1)(2·2-2) + (2·4-3)(2·3-3) = 3·2 + 5·3 = 21.
    e0, e1, e2 = iop.transcript.entries["sumcheckprod/g0"].result()
    assert e0 == 11 and e1 == 16 and e2 == 21
    iop = _prod_setup(ring)
    assert not iop.run(
        Statement(
            Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=28
        )
    )


def test_ring_sumprod_check_in_one_pass():
    """Ring tables over the same variables in the same order are checked
    entry by entry (`aligned_tables`), not walked point by point; tables
    over the variables in another order are walked. Both agree with the sum
    computed from the entries, and no oracle's entries change."""
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    f, g, h = (
        MLE(
            ring=ring,
            variables=v,
            evaluations=[ring.random_element() for _ in range(8)],
        )
        for _ in range(3)
    )
    before = [[e.copy() for e in t.table] for t in (f, g, h)]

    products = [a * b * c for a, b, c in zip(f.table, g.table, h.table, strict=True)]
    total = sum(products[1:], products[0])
    relation = Relation_SumProd()
    oracles = [TableOracle(f), TableOracle(g), TableOracle(h)]
    assert aligned_tables(oracles) is not None
    assert relation.check(Statement(relation, oracles=oracles, value=total))
    assert not relation.check(Statement(relation, oracles=oracles, value=total + 1))
    one = Statement(
        Relation_Sum(), oracles=[TableOracle(f)], value=sum(f.table[1:], f.table[0])
    )
    assert Relation_Sum().check(one)

    # g read over its variables reversed is another polynomial: entry i of f
    # meets entry reverse(i) of g, so the one-pass path has to decline.
    g_reversed = MLE(ring=ring, variables=v[::-1], evaluations=list(g.table))
    walked = f.table[0] * g.table[0]
    for i in range(1, 8):
        walked = walked + f.table[i] * g.table[int(f"{i:03b}"[::-1], 2)]
    reversed_oracles = [TableOracle(f), TableOracle(g_reversed)]
    assert aligned_tables(reversed_oracles) is None
    assert relation.check(Statement(relation, oracles=reversed_oracles, value=walked))

    assert all(
        entry == old
        for t, olds in zip((f, g, h), before, strict=True)
        for entry, old in zip(t.table, olds, strict=True)
    )


class _LyingSumcheckProd(SumcheckProd):
    """Answers every round consistently — for a different first factor."""

    async def prove(self, prover, statements):
        (statement,) = statements
        f, g = statement.oracles
        fake_f = MLE(variables=list(f.variables), coefficients=[0, 2, 3, 4])
        # sum of fake_f*g over {0,1}^2 is 40 — consistent with the false claim.
        fake = Statement(statement.relation, oracles=[TableOracle(fake_f), g], value=40)
        return await super().prove(prover, [fake])


def test_sumcheckprod_terminal_evals_catch_consistent_liar():
    _, f, g = _prod_oracles()
    iop = _prod_setup(_IntDomain())
    iop.register(Relation_SumProd, _LyingSumcheckProd())
    stmt = Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=40
    )
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
    stmt = Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=4
    )
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
        native = round_evals_native(f, var)
        python = _round_evals_python(f, var)
        assert all(a == b for a, b in zip(native, python, strict=True))
        native3 = prod2_round_evals_native(f, g, var)
        python3 = _prod_round_evals_python([f, g], var)
        assert all(a == b for a, b in zip(native3, python3, strict=True))
    # The dispatching entry points default to the first variable.
    assert all(
        a == b for a, b in zip(round_evals(f), round_evals(f, v[0]), strict=True)
    )
    assert all(
        a == b
        for a, b in zip(
            prod_round_evals([f, g]),
            prod_round_evals([f, g], v[0]),
            strict=True,
        )
    )


def _random_tables(ring, variables, count):
    return [
        MLE(
            ring=ring,
            variables=variables,
            evaluations=[ring.random_element() for _ in range(1 << len(variables))],
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize("num_vars", [1, 2, 3, 5])
def test_prod3_round_evals_match_python_every_position(num_vars):
    # The three-factor kernels (pairs for the first variable, halves for the
    # last, the strided one in between) give the pure-Python path's message,
    # four nodes, for distinct factors and for a factor repeated.
    ring = Ring(1024, prime_size=[49, 49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    f, g, h = _random_tables(ring, v, 3)
    for factors in ([f, g, h], [f, g, g]):
        for var in v:
            native = prod3_round_evals_native(factors[0], factors[1], factors[2], var)
            python = _prod_round_evals_python(factors, var)
            assert len(native) == 4
            assert all(a == b for a, b in zip(native, python, strict=True))
            dispatched = prod_round_evals(factors, var)
            assert all(a == b for a, b in zip(dispatched, python, strict=True))


def test_prod3_dispatch_needs_the_same_variables():
    # Three native tables over the variables in different orders are not one
    # layout: the dispatch leaves them to the pure-Python path.
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    f, g, h = _random_tables(ring, v, 3)
    h_reversed = MLE(ring=ring, variables=v[::-1], evaluations=list(h.table))
    factors = [f, g, h_reversed]
    expected = _prod_round_evals_python(factors, v[1])
    assert all(
        a == b for a, b in zip(prod_round_evals(factors, v[1]), expected, strict=True)
    )


def _prod3_proof(ring, factors, value) -> Proof:
    iop = IOP(domain=ring, fiat_shamir=True)
    iop.register(Relation_SumProd, SumcheckProd())
    oracles = [TableOracle(f) for f in factors]
    return iop.prove(Statement(Relation_SumProd(), oracles=oracles, value=value))


def test_sumcheckprod_three_ring_factors_native_matches_python(monkeypatch):
    # A full degree-3 product sumcheck over R_q, sum_b W(b) l(b)^2: the
    # native rounds write the same proof as the pure-Python ones, and it
    # verifies; a false claim does not.
    ring = Ring(1024, prime_size=[49, 49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(4)]
    w, lc = _random_tables(ring, v, 2)
    factors = [w, lc, lc]
    products = [a * b * b for a, b in zip(w.table, lc.table, strict=True)]
    total = sum(products[1:], products[0])

    calls = []

    def counted(f, g, h, var=None):
        calls.append(var)
        return prod3_round_evals_native(f, g, h, var)

    monkeypatch.setattr(rounds, "prod3_round_evals_native", counted)
    native = _prod3_proof(ring, factors, total)
    assert len(calls) == len(v)  # every round took the kernel
    monkeypatch.setattr(
        rounds,
        "prod3_round_evals_native",
        lambda f, g, h, var=None: _prod_round_evals_python([f, g, h], var),
    )
    python = _prod3_proof(ring, factors, total)
    monkeypatch.undo()
    assert native.labels == python.labels
    assert native.digest() == python.digest()

    def verify(value, proof):
        iop = IOP(domain=ring, fiat_shamir=True)
        iop.register(Relation_SumProd, SumcheckProd())
        oracles = [TableOracle(f) for f in factors]
        return iop.verify(
            Statement(Relation_SumProd(), oracles=oracles, value=value), proof
        )

    assert verify(total, native)
    false = total + 1
    assert not verify(false, _prod3_proof(ring, factors, false))


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
    native = prod_round_evals([dense, dense2])
    mixed = prod_round_evals([dense, coeff2])
    assert all(a == b for a, b in zip(native, mixed, strict=True))


def test_sumcheckprod_soundness_error():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    _, f, g = _prod_oracles()
    stmt = Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=50
    )
    scp = SumcheckProd()
    # degree k=2, n=2 variables.
    assert scp.soundness_error(stmt, ring) == 4 / min(ring.primes) ** ring.split_degree
