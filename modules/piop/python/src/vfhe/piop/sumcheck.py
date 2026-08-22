# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Sumcheck protocols [LFKN92]: Relation_Sum and Relation_SumProd both
reduce to Relation_Eval claims.

Round messages are the evaluations of the round polynomial g_i at 0..deg
(Libra's format [XZZPS19] — MLE tables are already in the evaluation
basis, so the linear-time kernels accumulate them directly). The verifier
updates its claim by Lagrange interpolation at the integer nodes 0..deg,
which over a ring requires {0..deg} to be an exceptional set [Sor22, fn.12]
— trivially true for R_q with large primes.

Both protocols share the round machinery (`_SumcheckRounds`) and differ
only in their round message and closing step. The native/pure-Python
decision is made inside the round-message helpers (`Sumcheck.round_evals`,
`SumcheckProd.prod_round_evals`): native tables (`mle.native_table`) go to
the C kernels of sumcheck.c, anything else runs the naive pure-Python path
— identical messages, so identical transcripts, either way. See piop.md
§4-§6.
"""

from __future__ import annotations

from fractions import Fraction

from vfhe.arith import Polynomial, Ring
from vfhe.misc.libvfhe import lib

from .mle import MLE, handle_array, mark_ntt, native_table
from .piop import (
    IOP,
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
        return domain.prime**domain.d
    return None


def interpolate_evals(evals: tuple, r):
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
        # The integer nodes as constant ring elements (`Polynomial - int`
        # only supports 0).
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


class _SumcheckRounds(Protocol):
    """The round machinery shared by the sumcheck family.

    Per round i, over the first remaining variable: the prover writes the
    round polynomial's evaluations (`_round_message`) and folds every table
    by the challenge (out of place on round 0 to keep the shared oracles
    intact, in place afterwards); the verifier checks g_i(0) + g_i(1)
    against the running claim and interpolates g_i(r_i) as the next one.
    One verifier serves the native and pure-Python provers — the wire
    format is shared, and its per-round work is O(1) coefficient
    operations.

    Subclasses provide `name` (the transcript label prefix), the round
    message, and the closing step of each half (`_prove_tail` /
    `_verify_tail`).
    """

    name: str = "abstract"

    def _label(self, statement: Statement) -> str:
        return f"{self.name}{statement.path}"

    def _round_message(self, factors: list, var) -> tuple:
        """The evaluations of this round's polynomial at 0..deg."""
        raise NotImplementedError

    def _prove_tail(
        self, iop: IOP, statement: Statement, label: str, factors: list, point: dict
    ) -> list[Statement]:
        """Close the prover half over the fully-folded tables; returns the
        statements the claim reduces to."""
        raise NotImplementedError

    async def _verify_tail(
        self, iop: IOP, statement: Statement, label: str, point: dict, claim
    ) -> list[Statement]:
        """Close the verifier half against the final claim g_n(r_n);
        returns the statements the claim reduces to."""
        raise NotImplementedError

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = prover.iop
        factors = list(statement.oracles)
        originals = tuple(statement.oracles)
        label = self._label(statement)
        point = {}
        for i, var in enumerate(list(factors[0].variables)):
            iop.transcript.write(f"{label}/g{i}", self._round_message(factors, var))
            r = iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            factors = [
                f.evaluate({var: r}, in_place=f is not orig)
                for f, orig in zip(factors, originals, strict=True)
            ]
        return self._prove_tail(iop, statement, label, factors, point)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        label = self._label(statement)
        claim = _value_of(statement.value)
        point = {}
        for i, var in enumerate(list(statement.oracles[0].variables)):
            evals = await iop.transcript.read(f"{label}/g{i}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {i}: g(0) + g(1) != claim")
            r = verifier.challenge(f"{label}/r{i}")
            point[var] = r
            claim = interpolate_evals(evals, r)
        return await self._verify_tail(iop, statement, label, point, claim)


class Sumcheck(_SumcheckRounds):
    """Reduces `sum_{b in {0,1}^n} f(b) == v` to `f(r) == v_n` in n rounds.

    Round i: the prover sends (g_i(0), g_i(1)) — the oracle is a single
    multilinear polynomial, so deg g_i = 1; the verifier checks
    g_i(0) + g_i(1) against the running claim, the challenge r_i is drawn
    from the domain's exceptional set, and the claim becomes g_i(r_i).
    """

    name = "sumcheck"
    reduces_from: type[Relation] = Relation_Sum
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring,)

    @staticmethod
    def round_evals(f, var=None) -> tuple:
        """One sumcheck round message (g(0), g(1)) for oracle `f`, with the
        round variable `var` anywhere in f's variables (default: the first).

        The native/pure-Python decision is made here: a native table
        delegates to the C kernels (round_evals_native), anything else sums
        the hypercube in Python — identical messages either way."""
        if native_table(f):
            return Sumcheck.round_evals_native(f, var)
        return Sumcheck._round_evals_python(f, var)

    @staticmethod
    def _round_evals_python(f, var=None) -> tuple:
        """The pure-Python path of round_evals, over any MLE type."""
        var = f.variables[0] if var is None else var
        g0 = _hypercube_sum(f.evaluate({var: 0}, in_place=False))
        g1 = _hypercube_sum(f.evaluate({var: 1}, in_place=False))
        return (g0, g1)

    @staticmethod
    def round_evals_native(f: MLE, var=None) -> tuple:
        """The C-kernel path of round_evals, chosen by the round variable's
        position: adjacent pairs (LSB), the two table halves (MSB), or the
        stride-computed generic kernel (fallback)."""
        idx = 0 if var is None else f.variables.index(var)
        f.to_NTT()  # the kernels read RNS form; see MLE.to_NTT
        g0, g1 = Polynomial(f.ring), Polynomial(f.ring)
        size = 1 << f.num_vars
        if idx == 0:
            lib.sumcheck_round_pairs(g0.obj, g1.obj, f.table_ptr, size)
        elif idx == f.num_vars - 1:
            lib.sumcheck_round_halves(g0.obj, g1.obj, f.table_ptr, size)
        else:
            lib.sumcheck_round(g0.obj, g1.obj, f.table_ptr, size, idx)
        mark_ntt([g0, g1])
        return (g0, g1)

    def soundness_error(
        self, statement: Statement, domain, degree: int = 1
    ) -> float | None:
        """degree * num_vars / |A| (piop.md §6); None if |A| is unknown."""
        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return degree * statement.num_vars / size

    def _round_message(self, factors: list, var) -> tuple:
        (f,) = factors
        return self.round_evals(f, var)

    def _prove_tail(
        self, _iop: IOP, statement: Statement, _label: str, factors: list, point: dict
    ) -> list[Statement]:
        (cur,) = factors
        return [self.reduce([statement], point=point, value=cur.constant())]

    async def _verify_tail(
        self, _iop: IOP, statement: Statement, _label: str, point: dict, claim
    ) -> list[Statement]:
        return [self.reduce([statement], point=point, value=claim)]


class SumcheckProd(_SumcheckRounds):
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

    The native (C) path covers exactly two native tables over a Ring —
    the reference Libra shape; other arities or representations use the
    pure-Python path. Both paths produce identical transcripts.
    """

    name = "sumcheckprod"
    reduces_from: type[Relation] = Relation_SumProd
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring,)

    @staticmethod
    def prod_round_evals(factors: list, var=None) -> tuple:
        """One product-sumcheck round message: the evaluations at the integer
        nodes 0..k of g(t) = sum_b prod_j f_j(t, b), with the round variable
        `var` anywhere in the factors' variables (default: the first variable
        of factors[0]) and b ranging over the hypercube of the remaining ones
        (values above 1 extrapolated division-free from the table pairs as
        lo + t*(hi - lo)).

        The native/pure-Python decision is made here: exactly two native
        tables delegate to the C kernels (prod2_round_evals_native), anything
        else runs the pure-Python path — identical messages either way. This
        is the round building block for protocols that interleave product
        sumcheck rounds with other messages (e.g. basefold in vfhe.polycom).
        """
        if len(factors) == 2 and all(native_table(f) for f in factors):
            return SumcheckProd.prod2_round_evals_native(factors[0], factors[1], var)
        return SumcheckProd._prod_round_evals_python(factors, var)

    @staticmethod
    def _prod_round_evals_python(factors: list, var=None) -> tuple:
        """The pure-Python path of prod_round_evals, over any MLE type."""
        var = factors[0].variables[0] if var is None else var
        rest = [v for v in factors[0].variables if v is not var]
        k = len(factors)
        evals = [None] * (k + 1)
        for b in _hypercube(rest):
            los = [
                _constant(f.evaluate({**b, var: 0}, in_place=False)) for f in factors
            ]
            his = [
                _constant(f.evaluate({**b, var: 1}, in_place=False)) for f in factors
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
        return tuple(evals)

    @staticmethod
    def prod2_round_evals_native(f: MLE, g: MLE, var=None) -> tuple:
        """The C-kernel path of prod_round_evals: one two-factor round
        message over MLE tables (which must share their variable
        layout). The kernel is chosen by the round variable's position:
        adjacent pairs (LSB), the two table halves (MSB), or the
        stride-computed generic kernel (fallback)."""
        idx = 0 if var is None else f.variables.index(var)
        f.to_NTT()  # the kernels read RNS form; see MLE.to_NTT
        g.to_NTT()
        evals = [Polynomial(f.ring) for _ in range(3)]
        handles = handle_array(evals)
        size = 1 << f.num_vars
        if idx == 0:
            lib.sumcheck_prod2_round_pairs(handles, f.table_ptr, g.table_ptr, size)
        elif idx == f.num_vars - 1:
            lib.sumcheck_prod2_round_halves(handles, f.table_ptr, g.table_ptr, size)
        else:
            lib.sumcheck_prod2_round(handles, f.table_ptr, g.table_ptr, size, idx)
        mark_ntt(evals)
        return tuple(evals)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """k * num_vars / |A| (piop.md §6); None if |A| is unknown."""
        size = _exceptional_set_size(domain)
        if size is None:
            return None
        return len(statement.oracles) * statement.num_vars / size

    def _round_message(self, factors: list, var) -> tuple:
        return self.prod_round_evals(factors, var)

    def _outputs(self, statement: Statement, point: dict, values: tuple):
        """One Relation_Eval claim per factor, all at the same point."""
        return [
            self.reduce([statement], oracles=[f], point=point, value=v)
            for f, v in zip(statement.oracles, values, strict=True)
        ]

    def _prove_tail(
        self, iop: IOP, statement: Statement, label: str, factors: list, point: dict
    ) -> list[Statement]:
        values = tuple(f.constant() for f in factors)
        iop.transcript.write(f"{label}/vals", values)
        return self._outputs(statement, point, values)

    async def _verify_tail(
        self, iop: IOP, statement: Statement, label: str, point: dict, claim
    ) -> list[Statement]:
        values = await iop.transcript.read(f"{label}/vals")
        prod = None
        for v in values:
            prod = v if prod is None else prod * v
        if not (prod == claim):
            raise Rejection(f"{label}: prod of claimed factor values != claim")
        return self._outputs(statement, point, values)
