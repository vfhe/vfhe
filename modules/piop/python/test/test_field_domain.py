# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for field coefficient domains: MLE tables backed by a FieldVector
(both field implementations), their agreement with the pure-Python element
tables, the whole-vector sumcheck round messages against the Python ones,
field interpolation, field challenge sampling, and the sumcheck protocols
end to end (interactive and Fiat-Shamir) over a field."""

import random

import pytest
from vfhe.arith import (
    MLE,
    Field,
    FieldVector,
    MLE_Basis,
    MLE_Variable,
    PseudoMersenneField,
)
from vfhe.arith.mle import native_table, vector_table
from vfhe.piop import (
    IOP,
    Relation_Sum,
    Relation_SumProd,
    Statement,
    Sumcheck,
    SumcheckProd,
)
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
        MLE(ring=object(), field=field, variables=v[:2])  # pyright: ignore[reportArgumentType]


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
        _ = f * g


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


@pytest.mark.parametrize("num_vars", [1, 2, 3, 4, 5, 8])
def test_eq_table_across_the_view_boundary(field, num_vars):
    """eq~ is built by doubling until the table is wide enough for its halves
    to be views, then in place. The switch happens at `padding_unit`, so the
    entries either side of it are what this spans -- the definition has to
    hold identically on both.
    """
    v = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    zs = [field.random_exceptional() for _ in v]
    eq = MLE.eq(field, zs, variables=v)
    one = field.one

    assert vector_table(eq) and len(eq.table) == 1 << num_vars
    for b in range(1 << num_vars):
        expected = one
        for i, z in enumerate(zs):
            expected = expected * (z if (b >> i) & 1 else one - z)
        assert eq.table[b] == expected, f"entry {b} of {num_vars} variables"

    # The table is a distribution over the cube: its entries sum to one.
    assert eq.table.sum() == one


def test_eq_table_spans_the_switch(field):
    """The boundary is not hypothetical: with the unit this build uses, some
    of the sizes above are built purely by doubling and some run the in-place
    half, and the test is only worth having if both happen."""
    unit = FieldVector(field, 1).padding_unit
    assert unit > 1, "a unit of 1 would leave the doubling path unexercised"
    assert unit < (1 << 8), "8 variables must reach the in-place half"


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


def test_sumprod_check_agrees_with_the_walk(field):
    """The whole-vector path and the point-by-point walk decide the same
    statements. `_tables` gives the same entries in both forms and only the
    field-backed one qualifies, so the plain-Python oracles *are* the
    reference: same entries, the old code path."""
    v, f_vec, f_py = _tables(field, 4, seed=11)
    _, g_vec, g_py = _tables(field, 4, seed=12, variables=v)
    _, h_vec, h_py = _tables(field, 4, seed=13, variables=v)
    zero = _element(field, 0)

    # Three oracles, so the in-place fold past the first product runs.
    total = sum(
        (f_py.table[i] * g_py.table[i] * h_py.table[i] for i in range(16)), zero
    )
    wrong = total + _element(field, 1)
    relation = Relation_SumProd()
    for oracles in ([f_py, g_py, h_py], [f_vec, g_vec, h_vec], [f_vec, g_py, h_vec]):
        assert relation.check(Statement(relation, oracles=oracles, value=total))
        assert not relation.check(Statement(relation, oracles=oracles, value=wrong))

    # The product is built in a vector of its own: no oracle was written to.
    assert [int(e) for e in f_vec.table] == [int(e) for e in f_py.table]
    assert [int(e) for e in g_vec.table] == [int(e) for e in g_py.table]

    # One oracle, and the plain sum relation, take the same route.
    assert Relation_Sum().check(
        Statement(Relation_Sum(), oracles=[f_vec], value=sum(f_py.table, zero))
    )
    assert Relation_SumProd().check(
        Statement(Relation_SumProd(), oracles=[f_vec], value=sum(f_py.table, zero))
    )


def test_sumprod_check_walks_when_the_tables_do_not_line_up(field):
    """Oracles over different variables cannot be read index for index, so
    the fast path has to decline rather than compare the wrong entries."""
    v, f_vec, _ = _tables(field, 3, seed=14)
    other = [MLE_Variable(f"y{i}") for i in range(3)]
    _, g_vec, g_py = _tables(field, 3, seed=15, variables=other)
    relation = Relation_SumProd()
    statement = Statement(relation, oracles=[f_vec, g_vec], value=_element(field, 0))
    # Decided by the walk over f's variables; the point is that it runs at all
    # and agrees with the same oracles read as Python tables.
    _, f_py, _ = (v, *_tables(field, 3, seed=14)[1:])
    assert relation.check(statement) == relation.check(
        Statement(relation, oracles=[f_py, g_py], value=_element(field, 0))
    )


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
