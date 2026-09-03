# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for field coefficient domains: MLE tables backed by a FieldVector
(both field implementations), their agreement with the pure-Python element
tables, the whole-vector sumcheck round messages against the Python ones,
field interpolation, field challenge sampling, and the sumcheck protocols
end to end (interactive and Fiat-Shamir) over a field."""

import random

import pytest
from vfhe.arith import Field, FieldVector, PseudoMersenneField
from vfhe.piop import (
    IOP,
    MLE,
    MLE_Basis,
    MLE_Variable,
    Relation_Sum,
    Relation_SumProd,
    Statement,
    Sumcheck,
    SumcheckProd,
)
from vfhe.piop.mle import native_table, vector_table
from vfhe.piop.sumcheck import interpolate_evals

# A 50-bit prime with 2-adicity 20 (x^2 - 5 is irreducible: 5 is a non-residue).
_PRIME = 562949948178433


@pytest.fixture(params=["extension", "pseudo_mersenne"])
def field(request):
    if request.param == "extension":
        return Field(_PRIME, 2, 5)
    return PseudoMersenneField.generate(260, two_adicity=8)


def _element(field, value: int):
    return type(field.one)(field, value)


def _random_elements(field, count: int, seed: int) -> list:
    rng = random.Random(seed)
    return [_element(field, rng.randrange(field.prime)) for _ in range(count)]


def _tables(field, num_vars: int, seed: int = 0, variables=None):
    """A field-backed table and the plain-Python element table with the
    same entries -- the reference semantics."""
    v = variables or [MLE_Variable(f"x{i}") for i in range(num_vars)]
    entries = _random_elements(field, 1 << num_vars, seed)
    return (
        v,
        MLE(field=field, variables=v, evaluations=entries),
        MLE(variables=v, evaluations=entries),
    )


def _point(field, variables) -> dict:
    return {var: field.random_exceptional() for var in variables}


def test_field_table_is_a_vector(field):
    v, f, _ = _tables(field, 3)
    assert isinstance(f.table, FieldVector) and len(f.table) == 8
    assert vector_table(f) and not native_table(f)
    assert not vector_table(f.to_coefficients())  # evaluation basis only
    # A vector may be passed in directly, and ints are lifted.
    vec = FieldVector(field, [1, 2, 3, 4])
    g = MLE(field=field, variables=v[:2], evaluations=vec)
    assert g.table is vec
    h = MLE(field=field, variables=v[:2], evaluations=[1, 2, 3, 4])
    assert h.table == vec
    with pytest.raises(TypeError, match="not both"):
        MLE(ring=object(), field=field, variables=v[:2])


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0)])
def test_evaluate_matches_python_table_in_any_order(field, order):
    v, f, ref = _tables(field, 4)
    point = _point(field, v)
    for i in order:
        f = f.evaluate({v[i]: point[v[i]]}, in_place=False)
        ref = ref.evaluate({v[i]: point[v[i]]}, in_place=False)
    assert f.variables == ref.variables
    assert f.table.to_list() == ref.table
    # Integer bindings (the hypercube) take the same path.
    assert f.evaluate(dict.fromkeys(f.variables, 1), in_place=False).constant() == (
        ref.evaluate(dict.fromkeys(ref.variables, 1), in_place=False).constant()
    )


def test_coefficient_basis_agrees(field):
    v, f, ref = _tables(field, 4)
    c = f.to_coefficients()
    assert c.basis is MLE_Basis.coeff
    assert c.table.to_list() == ref.to_coefficients().table
    point = _point(field, v)
    assert c.evaluate(point, in_place=False).constant() == (
        f.evaluate(point, in_place=False).constant()
    )
    # Untouched source, and a coefficient table converts to a copy.
    assert f.basis is MLE_Basis.eval
    assert c.to_coefficients().table == c.table


def test_arithmetic(field):
    v, f, ref_f = _tables(field, 3, seed=1)
    _, g, ref_g = _tables(field, 3, seed=2, variables=v)
    assert (f + g).table.to_list() == (ref_f + ref_g).table
    assert (f - g).table.to_list() == (ref_f - ref_g).table
    three = _element(field, 3)
    assert f.scale(three).table.to_list() == ref_f.scale(three).table
    assert f.scale(3).table == f.scale(three).table
    assert (3 * f).table == f.scale(three).table
    assert f.copy().table == f.table and f.copy().table is not f.table
    with pytest.raises(TypeError, match="not defined"):
        f * g


def test_eq_table(field):
    v, f, _ = _tables(field, 3)
    zs = [field.random_exceptional() for _ in v]
    eq = MLE.eq(field, zs, variables=v)
    assert vector_table(eq) and eq.variables == v
    one = field.one
    for b in range(8):
        expected = one
        for i, z in enumerate(zs):
            expected = expected * (z if (b >> i) & 1 else one - z)
        assert eq.table[b] == expected
    # sum_b f(b) eq(z, b) == f(z)
    assert (f.table * eq.table).sum() == (
        f.evaluate(dict(zip(v, zs, strict=True)), in_place=False).constant()
    )
    assert MLE.eq(field, [1, 0], variables=v[:2]).table == FieldVector(
        field, [0, 1, 0, 0]
    )


def test_vector_round_messages_match_python(field):
    # The pure-Python path (hypercube re-enumeration through the table's own
    # binds) is the reference the whole-vector shortcuts must reproduce.
    v, f, _ = _tables(field, 4, seed=3)
    _, g, _ = _tables(field, 4, seed=4, variables=v)
    assert Sumcheck.round_evals(f) == Sumcheck._round_evals_python(f)
    assert Sumcheck.round_evals(f) == Sumcheck.round_evals_vector(f)
    assert SumcheckProd.prod_round_evals([f, g]) == (
        SumcheckProd._prod_round_evals_python([f, g])
    )
    assert SumcheckProd.prod_round_evals([f, g]) == (
        SumcheckProd.prod2_round_evals_vector(f, g)
    )
    # A round variable that is not the first takes the Python path, over a
    # variable the vector layout cannot pair directly.
    assert Sumcheck.round_evals(f, v[2]) == Sumcheck._round_evals_python(f, v[2])
    assert SumcheckProd.prod_round_evals([f, g], v[3]) == (
        SumcheckProd._prod_round_evals_python([f, g], v[3])
    )
    assert f.num_vars == g.num_vars == 4  # the messages leave the tables alone


def test_interpolation_over_a_field(field):
    coeffs = _random_elements(field, 4, seed=5)  # g(t) = sum c_k t^k, deg 3
    r = field.random_exceptional()

    def g(t):
        total = None
        for k, c in enumerate(coeffs):
            term = c * (t**k if k else field.one)
            total = term if total is None else total + term
        return total

    for degree in (1, 2, 3):
        cs = coeffs[: degree + 1]

        def g_d(t, cs=cs):
            total = None
            for k, c in enumerate(cs):
                term = c * (t**k if k else field.one)
                total = term if total is None else total + term
            return total

        evals = tuple(g_d(_element(field, t)) for t in range(degree + 1))
        assert interpolate_evals(evals, r) == g_d(r)
    assert g(r) == g(r)  # the degree-3 helper is exercised above


def test_challenge_sampling(field):
    # The names the verifier and the Fiat-Shamir verifier call on a domain.
    a = field.random_exceptional()
    assert a.field is field or a.field == field
    assert a != field.random_exceptional()
    s1 = field.exceptional_from_seed(b"seed")
    assert s1 == field.exceptional_from_seed(b"seed")
    assert s1 != field.exceptional_from_seed(b"other")


@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_sumcheck_over_a_field(field, fiat_shamir):
    _, f, _ = _tables(field, 4, seed=6)
    total = f.table.sum()

    def run(value) -> bool:
        iop = IOP(domain=field, fiat_shamir=fiat_shamir)
        iop.register(Relation_Sum, Sumcheck())
        return iop.run(Statement(Relation_Sum(), oracles=[f], value=value))

    assert run(total)
    assert not run(total + field.one)
    assert f.num_vars == 4  # the shared oracle is folded out of place


@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_sumcheck_prod_over_a_field(field, fiat_shamir):
    v, f, _ = _tables(field, 4, seed=7)
    _, g, _ = _tables(field, 4, seed=8, variables=v)
    total = (f.table * g.table).sum()

    def run(value) -> bool:
        iop = IOP(domain=field, fiat_shamir=fiat_shamir)
        iop.register(Relation_SumProd, SumcheckProd())
        return iop.run(Statement(Relation_SumProd(), oracles=[f, g], value=value))

    assert run(total)
    assert not run(total + field.one)


def test_fiat_shamir_is_deterministic_over_a_field(field):
    from vfhe.piop import element_digest

    _, f, _ = _tables(field, 3, seed=9)
    runs = []
    for _ in range(2):
        iop = IOP(domain=field, fiat_shamir=True)
        iop.register(Relation_Sum, Sumcheck())
        assert iop.run(Statement(Relation_Sum(), oracles=[f], value=f.table.sum()))
        t = iop.transcript
        runs.append([(lbl, element_digest(t.entries[lbl].result())) for lbl in t.order])
    assert runs[0] == runs[1]
