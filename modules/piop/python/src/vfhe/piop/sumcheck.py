# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Sumcheck protocols [LFKN92]: Relation_Sum and Relation_SumProd both
reduce to Relation_Eval claims.

Round messages are the evaluations of the round polynomial g_i at 0..deg
(Libra's format [XZZPS19] — MLE_Dense tables are already in the evaluation
basis, so the linear-time kernels accumulate them directly). The verifier
updates its claim by Lagrange interpolation at the integer nodes 0..deg,
which over a ring requires {0..deg} to be an exceptional set [Sor22, fn.12]
— trivially true for R_q with large primes.

The prover halves delegate to the C kernels (sumcheck.c) when the domain is
in `supported_domains` and the oracles are MLE_Dense; otherwise they run the
naive pure-Python path. See piop.md §4-§6.
"""

from __future__ import annotations

from fractions import Fraction

from vfhe.arith import Polynomial, Ring
from vfhe.misc.libvfhe import lib

from .mle import MLE_Dense, _handle_array, _sync_repr
from .piop import (
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Relation_Sum,
    Relation_SumProd,
    Statement,
    Verifier,
    _constant,
    _hypercube,
    _hypercube_sum,
    _value_of,
)


def _exceptional_set_size(domain) -> int | None:
    """|A| for the domain's exceptional set; None if the domain is unknown.

    Duck-typed until the domain classes expose it themselves: for a Ring the
    residue fields have size p_i^split_degree, so |A| = min(p_i)^split_degree;
    for a Field the whole field F_{p^d} is exceptional.
    """
    if hasattr(domain, "primes") and hasattr(domain, "split_degree"):
        return min(domain.primes) ** domain.split_degree
    if hasattr(domain, "prime") and hasattr(domain, "d"):
        return domain.prime ** domain.d
    return None


def _interpolate(evals: tuple, r):
    """g(r) from the evaluations of g at the integer nodes 0..deg.

    Over a ring the Lagrange denominators are (products of) integers up to
    deg, inverted per RNS prime — sound because {0..deg} is an exceptional
    set there [Sor22, fn. 12]. Over plain integers the division is exact
    (g has integer coefficients), computed with Fractions.
    """
    k = len(evals) - 1
    if k == 1:  # linear: no division needed
        return evals[0] + r * (evals[1] - evals[0])
    if isinstance(evals[0], Polynomial):
        ring = evals[0].ring
        # The integer nodes as constant ring elements (Polynomial - int only
        # supports 0; arith is not modified from here).
        nodes = [Polynomial(ring).from_array([j]) for j in range(k + 1)]
        total = None
        for t in range(k + 1):
            num = None  # prod_{j != t} (r - j), a ring element
            denom = 1  # prod_{j != t} (t - j), a small integer
            for j in range(k + 1):
                if j == t:
                    continue
                factor = r - nodes[j]
                num = factor if num is None else num * factor
                denom *= t - j
            inv = [pow(denom % p, p - 2, p) for p in ring.primes]
            term = evals[t] * num * inv
            total = term if total is None else total + term
        return total
    total = Fraction(0)
    for t in range(k + 1):
        num, denom = 1, 1
        for j in range(k + 1):
            if j == t:
                continue
            num *= r - j
            denom *= t - j
        total += Fraction(evals[t]) * num / denom
    assert total.denominator == 1, "interpolation of integer evals must be exact"
    return int(total)


class Sumcheck(Protocol):
    """Reduces `sum_{b in {0,1}^n} f(b) == v` to `f(r) == v_n` in n rounds.

    Round i: the prover sends (g_i(0), g_i(1)) — the oracle is a single
    multilinear polynomial, so deg g_i = 1; the verifier checks
    g_i(0) + g_i(1) against the running claim, the challenge r_i is drawn
    from the domain's exceptional set, and the claim becomes g_i(r_i).
    """

    reduces_from: type[Relation] = Relation_Sum
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring,)

    def native_supported(self, iop, statement: Statement) -> bool:
        (f,) = statement.oracles
        return super().native_supported(iop, statement) and isinstance(f, MLE_Dense)

    def soundness_error(
        self, statement: Statement, domain, degree: int = 1
    ) -> float | None:
        """degree * num_vars / |A| (piop.md §6); None if |A| is unknown."""
        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return degree * statement.num_vars / size

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        if self.native_supported(prover.iop, statement):
            return self._prove_native(prover, statement)
        return self._prove_python(prover, statement)

    def _prove_python(self, prover: Prover, statement: Statement) -> list[Statement]:
        iop = prover.iop
        (f,) = statement.oracles
        label = f"sumcheck{statement.path}"
        point = {}
        cur = f
        for i, var in enumerate(list(f.variables)):
            g0 = _hypercube_sum(cur.evaluate({var: 0}, in_place=False))
            g1 = _hypercube_sum(cur.evaluate({var: 1}, in_place=False))
            iop.transcript.write(f"{label}/g{i}", (g0, g1))
            r = iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            # Fold: out of place once to keep the shared oracle intact,
            # in place afterwards.
            cur = cur.evaluate({var: r}, in_place=cur is not f)
        return [self.reduce([statement], point=point, value=cur.constant())]

    def _prove_native(self, prover: Prover, statement: Statement) -> list[Statement]:
        """Libra-style prover: the C round kernel accumulates the halves of
        the (evaluation-basis) table; folding is MLE evaluation."""
        iop = prover.iop
        (f,) = statement.oracles
        label = f"sumcheck{statement.path}"
        point = {}
        cur = f
        for i, var in enumerate(list(f.variables)):
            g0, g1 = Polynomial(cur.ring), Polynomial(cur.ring)
            lib.sumcheck_round(g0.obj, g1.obj, cur.data_ptr, 1 << cur.num_vars)
            _sync_repr([g0, g1], cur.py_refs[0])
            iop.transcript.write(f"{label}/g{i}", (g0, g1))
            r = iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            cur = cur.evaluate({var: r}, in_place=cur is not f)
        return [self.reduce([statement], point=point, value=cur.constant())]

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        # One verifier serves both prover paths: the wire format is shared,
        # and the per-round work is O(1) ring operations — no native
        # implementation exists (or is needed) yet.
        (statement,) = statements
        iop = verifier.iop
        (f,) = statement.oracles
        label = f"sumcheck{statement.path}"
        claim = _value_of(statement.value)
        point = {}
        for i, var in enumerate(list(f.variables)):
            evals = await iop.transcript.read(f"{label}/g{i}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {i}: g(0) + g(1) != claim")
            r = verifier.challenge(f"{label}/r{i}")
            point[var] = r
            claim = _interpolate(evals, r)
        return [self.reduce([statement], point=point, value=claim)]


class SumcheckProd(Protocol):
    """Libra-style product sumcheck [XZZPS19, Alg. 3]: reduces one
    `sum_{b in {0,1}^n} prod_j f_j(b) == v` claim to k = #factors
    `Relation_Eval` claims, all at the same random point r.

    Round i: each factor is linear in x_i, so the round polynomial
    g_i(X) = sum_b prod_j f_j(b, X) has degree k, sent as its evaluations at
    0..k (values above 1 extrapolated division-free from the table halves as
    lo + t·(hi - lo)). The verifier checks g_i(0) + g_i(1) against the
    running claim and interpolates g_i(r_i) as the next claim. After the
    last round the prover sends the per-factor values v_j = f_j(r); the
    verifier checks prod_j v_j against the final claim and each (f_j, r,
    v_j) becomes an output Relation_Eval statement.

    The native (C) path covers exactly two MLE_Dense factors over a Ring —
    the reference Libra shape; other arities or representations use the
    pure-Python path. Both paths produce identical transcripts.
    """

    reduces_from: type[Relation] = Relation_SumProd
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring,)

    def native_supported(self, iop, statement: Statement) -> bool:
        return (
            super().native_supported(iop, statement)
            and len(statement.oracles) == 2
            and all(isinstance(f, MLE_Dense) for f in statement.oracles)
        )

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """k * num_vars / |A| (piop.md §6); None if |A| is unknown."""
        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return len(statement.oracles) * statement.num_vars / size

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        if self.native_supported(prover.iop, statement):
            return self._prove_native(prover, statement)
        return self._prove_python(prover, statement)

    def _prove_python(self, prover: Prover, statement: Statement) -> list[Statement]:
        iop = prover.iop
        factors = list(statement.oracles)
        originals = tuple(factors)
        k = len(factors)
        label = f"sumcheckprod{statement.path}"
        point = {}
        for i, var in enumerate(list(factors[0].variables)):
            rest = [v for v in factors[0].variables if v is not var]
            evals = [None] * (k + 1)
            for b in _hypercube(rest):
                los = [
                    _constant(f.evaluate({**b, var: 0}, in_place=False))
                    for f in factors
                ]
                his = [
                    _constant(f.evaluate({**b, var: 1}, in_place=False))
                    for f in factors
                ]
                for t in range(k + 1):
                    prod = None
                    for lo, hi in zip(los, his, strict=True):
                        if t == 0:
                            val = lo
                        elif t == 1:
                            val = hi
                        else:
                            val = lo + t * (hi - lo)
                        prod = val if prod is None else prod * val
                    evals[t] = prod if evals[t] is None else evals[t] + prod
            iop.transcript.write(f"{label}/g{i}", tuple(evals))
            r = iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            factors = [
                f.evaluate({var: r}, in_place=f is not orig)
                for f, orig in zip(factors, originals, strict=True)
            ]
        values = tuple(f.constant() for f in factors)
        iop.transcript.write(f"{label}/vals", values)
        return [
            self.reduce([statement], oracles=[f], point=point, value=v)
            for f, v in zip(statement.oracles, values, strict=True)
        ]

    def _prove_native(self, prover: Prover, statement: Statement) -> list[Statement]:
        """Libra-style prover for two MLE_Dense factors: the C round kernel
        accumulates g(0), g(1), g(2) from the two tables; folding is MLE
        evaluation (out of place once, in place afterwards)."""
        iop = prover.iop
        f, g = statement.oracles
        label = f"sumcheckprod{statement.path}"
        point = {}
        cur_f, cur_g = f, g
        for i, var in enumerate(list(f.variables)):
            evals = [Polynomial(cur_f.ring) for _ in range(3)]
            lib.sumcheck_prod2_round(
                _handle_array(evals),
                cur_f.data_ptr,
                cur_g.data_ptr,
                1 << cur_f.num_vars,
            )
            _sync_repr(evals, cur_f.py_refs[0])
            iop.transcript.write(f"{label}/g{i}", tuple(evals))
            r = iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            cur_f = cur_f.evaluate({var: r}, in_place=cur_f is not f)
            cur_g = cur_g.evaluate({var: r}, in_place=cur_g is not g)
        values = (cur_f.constant(), cur_g.constant())
        iop.transcript.write(f"{label}/vals", values)
        return [
            self.reduce([statement], oracles=[o], point=point, value=v)
            for o, v in zip(statement.oracles, values, strict=True)
        ]

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        # One verifier serves both prover paths (shared wire format).
        (statement,) = statements
        iop = verifier.iop
        label = f"sumcheckprod{statement.path}"
        claim = _value_of(statement.value)
        point = {}
        for i, var in enumerate(list(statement.oracles[0].variables)):
            evals = await iop.transcript.read(f"{label}/g{i}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {i}: g(0) + g(1) != claim")
            r = verifier.challenge(f"{label}/r{i}")
            point[var] = r
            claim = _interpolate(evals, r)
        values = await iop.transcript.read(f"{label}/vals")
        prod = None
        for v in values:
            prod = v if prod is None else prod * v
        if not (prod == claim):
            raise Rejection(f"{label}: prod of claimed factor values != claim")
        return [
            self.reduce([statement], oracles=[f], point=point, value=v)
            for f, v in zip(statement.oracles, values, strict=True)
        ]
