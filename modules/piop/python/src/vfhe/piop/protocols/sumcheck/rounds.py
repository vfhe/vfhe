# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The sumcheck round kernels [LFKN92; XZZPS19], as functions over tables,
and the verifier's claim update.

Round messages are the evaluations of the round polynomial g_i at 0..deg
(Libra's format [XZZPS19] -- MLE tables are already in the evaluation
basis, so the linear-time kernels accumulate them directly). The verifier
updates its claim by Lagrange interpolation at the integer nodes 0..deg,
which over a ring requires {0..deg} to be an exceptional set [Sor22, fn.12]
-- trivially true for R_q with large primes.

The native/pure-Python decision is made here, per call: native tables
(`mle.native_table`) go to the C kernels of sumcheck.c, field-backed tables
over their first variable to whole-vector operations, anything else to the
pure-Python path -- identical messages, so identical transcripts, either
way. These are the building blocks of the sumcheck protocols
(`sumcheck.py`), of the prover's round messages over a virtual oracle
(`prover_view.py`), and of protocols that interleave sumcheck rounds with
other messages (basefold in `vfhe.polycom`).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

from vfhe.arith import Polynomial, RNSPolynomial
from vfhe.arith.mle import MLE, element_array, mark_ntt, native_table, vector_table
from vfhe.piop.relation import hypercube
from vfhe.util.bindings import lib


def interpolate_evals(evals: tuple, r):
    """g(r) from the evaluations of g at the integer nodes 0..deg.

    Over a ring the Lagrange denominators are (products of) integers up to
    deg, inverted per RNS prime — sound because {0..deg} is an exceptional
    set there [Sor22, fn. 12]. Over a field they are inverted in the field.
    Over plain integers the division is exact (g has integer coefficients),
    computed with Fractions.
    """
    k = len(evals) - 1
    if k == 1:  # linear: no division needed
        return evals[0] + r * (evals[1] - evals[0])
    if not isinstance(evals[0], Polynomial | int) and hasattr(evals[0], "field"):
        total = None
        for t in range(k + 1):
            num = None  # prod_{j != t} (r - j), a field element
            denom = 1  # prod_{j != t} (t - j), a small integer
            for j in range(k + 1):
                if j == t:
                    continue
                factor = r - j
                num = factor if num is None else num * factor
                denom *= t - j
            term = evals[t] * num * (r.field.one * denom).inverse()
            total = term if total is None else total + term
        return total
    if isinstance(evals[0], RNSPolynomial):
        ring = evals[0].ring
        total = None
        for t in range(k + 1):
            num = None  # prod_{j != t} (r - j), a ring element
            denom = 1  # prod_{j != t} (t - j), a small integer
            for j in range(k + 1):
                if j == t:
                    continue
                factor = r - j
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


def _table_sum(table):
    """sum_{b in {0,1}^n} table(b) over a dense table, by its own binds: the
    reference the whole-vector and native paths must reproduce."""
    total = None
    for b in hypercube(table.variables):
        e = table.evaluate(b, in_place=False).constant()
        total = e if total is None else total + e
    return total


def round_evals(f, var=None) -> tuple:
    """One sumcheck round message (g(0), g(1)) for oracle `f`, with the
    round variable `var` anywhere in f's variables (default: the first).

    The native/pure-Python decision is made here: a native table
    delegates to the C kernels (round_evals_native), a field-backed
    table with the first variable as the round variable sums its two
    halves as vectors (round_evals_vector), anything else sums the
    hypercube in Python — identical messages either way."""
    if native_table(f):
        return round_evals_native(f, var)
    if vector_table(f) and (var is None or f.variables.index(var) == 0):
        return round_evals_vector(f)
    own: Any = getattr(f, "round_evals", None)
    if callable(own):  # a prover view of a virtual oracle (prover_view.py)
        return cast("tuple", own(var))
    return _round_evals_python(f, var)


def round_evals_vector(f: MLE) -> tuple:
    """The whole-vector path of round_evals over the first variable:
    g(0) and g(1) are the sums of the even and odd entries."""
    even, odd = f._vector.split_even_odd()
    return (even.sum(), odd.sum())


def _round_evals_python(f, var=None) -> tuple:
    """The pure-Python path of round_evals, over any MLE type."""
    var = f.variables[0] if var is None else var
    g0 = _table_sum(f.evaluate({var: 0}, in_place=False))
    g1 = _table_sum(f.evaluate({var: 1}, in_place=False))
    return (g0, g1)


def round_evals_native(f: MLE, var=None) -> tuple:
    """The C-kernel path of round_evals, chosen by the round variable's
    position: adjacent pairs (LSB), the two table halves (MSB), or the
    stride-computed generic kernel (fallback)."""
    idx = 0 if var is None else f.variables.index(var)
    f.to_NTT()  # the kernels read RNS form; see MLE.to_NTT
    ring = f.ring
    if ring is None:
        raise ValueError("a native round needs a ring-backed MLE")
    g0, g1 = Polynomial(ring), Polynomial(ring)
    size = 1 << f.num_vars
    if idx == 0:
        lib.sumcheck_round_pairs(
            ring.arith_ring, g0.as_element(), g1.as_element(), f.table_ptr, size
        )
    elif idx == f.num_vars - 1:
        lib.sumcheck_round_halves(
            ring.arith_ring, g0.as_element(), g1.as_element(), f.table_ptr, size
        )
    else:
        lib.sumcheck_round(
            ring.arith_ring,
            g0.as_element(),
            g1.as_element(),
            f.table_ptr,
            size,
            idx,
        )
    mark_ntt([g0, g1])
    return (g0, g1)


def prod_round_evals(factors: list, var=None) -> tuple:
    """One product-sumcheck round message: the evaluations at the integer
    nodes 0..k of g(t) = sum_b prod_j f_j(t, b), with the round variable
    `var` anywhere in the factors' variables (default: the first variable
    of factors[0]) and b ranging over the hypercube of the remaining ones
    (values above 1 extrapolated division-free from the table pairs as
    lo + t*(hi - lo)).

    The native/pure-Python decision is made here: exactly two native
    tables delegate to the C kernels (prod2_round_evals_native), two
    field-backed tables over their first variable run as whole-vector
    operations (prod2_round_evals_vector), anything else runs the
    pure-Python path — identical messages either way. This is the round
    building block for protocols that interleave product sumcheck rounds
    with other messages (e.g. basefold in vfhe.polycom).
    """
    if len(factors) == 2 and all(native_table(f) for f in factors):
        return prod2_round_evals_native(factors[0], factors[1], var)
    if (
        len(factors) == 2
        and all(vector_table(f) for f in factors)
        and (var is None or factors[0].variables.index(var) == 0)
    ):
        return prod2_round_evals_vector(factors[0], factors[1])
    return _prod_round_evals_python(factors, var)


def prod2_round_evals_vector(f: MLE, g: MLE) -> tuple:
    """The whole-vector path of prod_round_evals over the first variable
    of two field-backed tables: with (lo, hi) the even and odd halves,
    g(0) = sum lo_f lo_g, g(1) = sum hi_f hi_g and g(2) =
    sum (2 hi_f - lo_f)(2 hi_g - lo_g), the t = 2 extrapolation."""
    f_lo, f_hi = f._vector.split_even_odd()
    g_lo, g_hi = g._vector.split_even_odd()
    return (
        (f_lo * g_lo).sum(),
        (f_hi * g_hi).sum(),
        ((f_hi * 2 - f_lo) * (g_hi * 2 - g_lo)).sum(),
    )


def _prod_round_evals_python(factors: list, var=None) -> tuple:
    """The pure-Python path of prod_round_evals, over any MLE type."""
    var = factors[0].variables[0] if var is None else var
    rest = [v for v in factors[0].variables if v is not var]
    k = len(factors)
    evals = [None] * (k + 1)
    for b in hypercube(rest):
        los = [f.evaluate({**b, var: 0}, in_place=False).constant() for f in factors]
        his = [f.evaluate({**b, var: 1}, in_place=False).constant() for f in factors]
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
            acc = evals[t]
            evals[t] = prod if acc is None else acc + prod
    return tuple(evals)


def prod2_round_evals_native(f: MLE, g: MLE, var=None) -> tuple:
    """The C-kernel path of prod_round_evals: one two-factor round
    message over MLE tables (which must share their variable
    layout). The kernel is chosen by the round variable's position:
    adjacent pairs (LSB), the two table halves (MSB), or the
    stride-computed generic kernel (fallback)."""
    idx = 0 if var is None else f.variables.index(var)
    f.to_NTT()  # the kernels read RNS form; see MLE.to_NTT
    g.to_NTT()
    ring = f.ring
    if ring is None:
        raise ValueError("a native round needs a ring-backed MLE")
    evals = [Polynomial(ring) for _ in range(3)]
    handles = element_array(evals)
    size = 1 << f.num_vars
    if idx == 0:
        lib.sumcheck_prod2_round_pairs(
            ring.arith_ring, handles, f.table_ptr, g.table_ptr, size
        )
    elif idx == f.num_vars - 1:
        lib.sumcheck_prod2_round_halves(
            ring.arith_ring, handles, f.table_ptr, g.table_ptr, size
        )
    else:
        lib.sumcheck_prod2_round(
            ring.arith_ring, handles, f.table_ptr, g.table_ptr, size, idx
        )
    mark_ntt(evals)
    return tuple(evals)
