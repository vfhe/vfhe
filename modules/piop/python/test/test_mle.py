# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for vfhe.piop: MLE in both bases and over
both coefficient types - pure-Python tables (no ring) and the C-backed
ring tables (mle_dense_poly_* kernels over the cffi boundary) - plus
SparseMLE arithmetic. Evaluation points are always concrete values —
protocol futures live at the Transcript / Statement level (test_piop.py).
"""

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.piop import MLE, MLE_Basis, MLE_Variable, SparseMLE
from vfhe.piop.mle import native_table


def test_coeff_basis_arithmetic_and_eval():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    p1 = MLE(variables=v, coefficients=[1, 2, 3, 4])
    p2 = MLE(variables=v, coefficients=[10, 20, 30, 40])
    assert (p1 + p2).table == [11, 22, 33, 44]
    assert (p2 - p1).table == [9, 18, 27, 36]
    assert p1.scale(5).table == [5, 10, 15, 20]
    ex0 = p1.evaluate({v[0]: 2}, in_place=False)
    ex1 = p1.evaluate({v[1]: 3}, in_place=False)
    exf = p1.evaluate({v[0]: 2, v[1]: 3}, in_place=False)
    assert isinstance(ex0, MLE) and ex0.table == [5, 11]
    assert isinstance(ex1, MLE) and ex1.table == [10, 14]
    assert isinstance(exf, MLE) and exf.table == [38]


def test_sparse_mle_arithmetic():
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    s1 = SparseMLE(variables=v, evaluations={0: 5, 2: 12})
    s2 = SparseMLE(variables=v, evaluations={1: 7, 2: 3})
    assert (s1 + s2).evaluations == {0: 5, 1: 7, 2: 15}
    assert (s1 - s2).evaluations == {0: 5, 1: -7, 2: 9}
    assert s1.scale(2).evaluations == {0: 10, 2: 24}


def test_mle_arithmetic_and_eval():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    d1 = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    d2 = MLE(ring=ring, variables=v, evaluations=[10, 20, 30, 40])

    d3 = d1 + d2
    assert d3.table[0].get_polynomial()[0] == 11
    assert d3.table[3].get_polynomial()[0] == 44

    d4 = d2 - d1
    assert d4.table[0].get_polynomial()[0] == 9
    assert d4.table[3].get_polynomial()[0] == 36

    d5 = d1.scale(5)
    assert d5.table[0].get_polynomial()[0] == 5
    assert d5.table[3].get_polynomial()[0] == 20

    d6 = d1.scale(Polynomial(ring).from_array([10]))
    assert d6.table[0].get_polynomial()[0] == 10
    assert d6.table[3].get_polynomial()[0] == 40

    eval_x0 = d1.evaluate({v[0]: 2}, in_place=False)
    assert isinstance(eval_x0, MLE)
    assert eval_x0.num_vars == 1
    assert eval_x0.table[0].get_polynomial()[0] == 3
    assert eval_x0.table[1].get_polynomial()[0] == 5


def test_coeff_basis_bind_any_position():
    # 3 variables to exercise all three pair layouts: pairs (x0, LSB),
    # generic (x1, middle), halves (x2, MSB).
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    coeffs = [1, 2, 3, 4, 5, 6, 7, 8]
    f = MLE(variables=v, coefficients=coeffs)
    # Middle variable: pairs (idx, idx + 2) within blocks of 4.
    mid = f.evaluate({v[1]: 2}, in_place=False)
    assert mid.table == [7, 10, 19, 22]
    # MSB variable: the two coefficient halves.
    msb = f.evaluate({v[2]: 3}, in_place=False)
    assert msb.table == [16, 20, 24, 28]
    # Binding order does not change the result.
    orders = [
        {v[0]: 2, v[1]: 3, v[2]: 5},
        {v[2]: 5, v[0]: 2, v[1]: 3},
        {v[1]: 3, v[2]: 5, v[0]: 2},
    ]
    results = [f.evaluate(o, in_place=False).constant() for o in orders]
    assert results[0] == results[1] == results[2]


def test_mle_bind_any_position():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    f = MLE(ring=ring, variables=v, evaluations=list(range(1, 9)))
    two = Polynomial(ring).from_array([2])

    # Middle variable (generic kernel): pairs (b, b + 2) within blocks of 4.
    mid = f.evaluate({v[1]: 2}, in_place=False)
    assert [p.get_polynomial()[0] for p in mid.table] == [5, 6, 9, 10]
    # MSB variable (halves kernel), scalar and Polynomial values agree.
    msb = f.evaluate({v[2]: 2}, in_place=False)
    assert [p.get_polynomial()[0] for p in msb.table] == [9, 10, 11, 12]
    msb_poly = f.evaluate({v[2]: two}, in_place=False)
    assert all(
        a == b
        for a, b in zip(msb.table, msb_poly.table, strict=True)
    )
    # LSB variable (pairs kernel), both value types.
    lsb = f.evaluate({v[0]: 2}, in_place=False)
    assert [p.get_polynomial()[0] for p in lsb.table] == [3, 5, 7, 9]
    lsb_poly = f.evaluate({v[0]: two}, in_place=False)
    assert all(
        a == b
        for a, b in zip(lsb.table, lsb_poly.table, strict=True)
    )
    # Binding order does not change the result.
    zs = [ring.random_exceptional() for _ in range(3)]
    a = f.evaluate({v[0]: zs[0], v[1]: zs[1], v[2]: zs[2]}, in_place=False)
    b = f.evaluate({v[2]: zs[2], v[0]: zs[0], v[1]: zs[1]}, in_place=False)
    c = f.evaluate({v[1]: zs[1], v[2]: zs[2], v[0]: zs[0]}, in_place=False)
    assert a.constant() == b.constant() and b.constant() == c.constant()
    # The source table is untouched throughout.
    assert f.num_vars == 3 and f.table[0].get_polynomial()[0] == 1


def test_mle_to_coefficients():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    # Evaluations (1, 2, 3, 4) are the table of f = 1 + x0 + 2*x1.
    d1 = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    c = d1.to_coefficients()
    # A first-class MLE in the monomial basis, not a bare list.
    assert isinstance(c, MLE) and c.basis is MLE_Basis.coeff
    assert c.ring is ring and c.variables == v
    # Both bases describe the same polynomial: f(2, 3) = 1 + 2 + 6 = 9. Run
    # the native (kernel) evaluation before reading any entry of d1: reading
    # one converts it to coefficient form in place, and the kernels need the
    # whole table in NTT form (see the repr caveat in piop.md §7).
    point = {v[0]: 2, v[1]: 3}
    assert d1.evaluate(point, in_place=False).constant() == 9
    assert c.evaluate(point, in_place=False).constant() == 9
    # Source untouched, and no entry is shared with it.
    assert d1.basis is MLE_Basis.eval
    assert all(a is not b for a, b in zip(c.table, d1.table, strict=True))
    assert d1.table[3].get_polynomial()[0] == 4
    # The coefficients themselves, and: already monomial -> copy, no second
    # butterfly.
    assert c.table[0] == 1 and c.table[1] == 1
    assert c.table[2] == 2 and c.table[3] == 0
    again = c.to_coefficients()
    assert again.basis is MLE_Basis.coeff and again.table[2] == 2


def test_mle_basis_and_backing_are_independent():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    coeffs = [1, 2, 3, 4]  # f = 1 + 2*x0 + 3*x1 + 4*x0*x1
    # The same coefficient table over plain ints and over the ring agree,
    # and only the ring-backed evaluation table is C-kernel eligible.
    plain = MLE(variables=v, coefficients=coeffs)
    ringed = MLE(ring=ring, variables=v, coefficients=coeffs)
    assert plain.ring is None and plain.table_ptr is None
    assert ringed.ring is ring and ringed.table_ptr is not None
    assert not native_table(plain) and not native_table(ringed)  # coeff basis
    assert native_table(MLE(ring=ring, variables=v, evaluations=coeffs))

    point = {v[0]: 2, v[1]: 3}  # f(2, 3) = 1 + 4 + 9 + 24 = 38
    assert plain.evaluate(point, in_place=False).constant() == 38
    assert ringed.evaluate(point, in_place=False).constant() == 38
    # Arithmetic works on either backing, and preserves the basis.
    assert (plain + plain).table == [2, 4, 6, 8]
    assert (plain + plain).basis is MLE_Basis.coeff
    assert (ringed.scale(2)).table[1].get_polynomial()[0] == 4
    with pytest.raises(AssertionError):  # mixing bases is a bug, not a coercion
        _ = plain + MLE(variables=v, evaluations=coeffs)


def test_mle_survives_entry_inspection():
    # Reading an entry's value converts *that entry* to coefficient form in
    # place (arith mutates representation on read), leaving the table mixed.
    # The kernels read RNS form, so every native path normalizes with
    # to_NTT() first; without it they fold the wrong data and silently
    # return garbage. Regression for each kernel-backed operation.
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    point = {v[0]: 2, v[1]: 3, v[2]: 5}  # f(2,3,5) over the table below

    def poisoned():
        f = MLE(ring=ring, variables=v, evaluations=list(range(1, 9)))
        f.table[3].get_polynomial()  # poison one entry
        f.table[6] == 7  # noqa: B015 - and another, via __eq__
        return f

    clean = MLE(ring=ring, variables=v, evaluations=list(range(1, 9)))
    expected = clean.evaluate(point, in_place=False).constant()

    assert poisoned().evaluate(point, in_place=False).constant() == expected
    # Binding at every position (pairs / generic / halves kernels).
    for var in v:
        ref = MLE(ring=ring, variables=v, evaluations=list(range(1, 9)))
        want = ref.evaluate({var: 2}, in_place=False)
        got = poisoned().evaluate({var: 2}, in_place=False)
        assert all(a == b for a, b in zip(want.table, got.table, strict=True))
    # Elementwise kernels (add / sub / scale) and the coefficient butterfly.
    assert (poisoned() + poisoned()).table[0].get_polynomial()[0] == 2
    assert (poisoned() - poisoned()).table[5].get_polynomial()[0] == 0
    assert poisoned().scale(3).table[3].get_polynomial()[0] == 12
    assert poisoned().to_coefficients().table[7] == 0


def test_sparse_mle_is_not_an_mle():
    # The two are independent by design: a sparse map supports the linear
    # operations but none of the folding, so it is not an MLE subtype and
    # says so instead of half-implementing evaluate().
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    s = SparseMLE(variables=v, evaluations={0: 5, 2: 12})
    assert not isinstance(s, MLE) and not native_table(s)
    assert s.num_vars == 2  # same vocabulary, no shared base
    with pytest.raises(NotImplementedError):
        s.evaluate({v[0]: 1})
    with pytest.raises(TypeError):  # neither type multiplies with an MLE
        _ = s * MLE(variables=v, coefficients=[1, 2, 3, 4])
    with pytest.raises(TypeError):
        _ = MLE(variables=v, coefficients=[1, 2, 3, 4]) * s


def test_mle_eq():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    zs = [ring.random_exceptional(), ring.random_exceptional()]
    eq = MLE.eq(ring, zs, variables=v)
    f = MLE(
        ring=ring, variables=v, evaluations=[ring.random_element() for _ in range(4)]
    )
    # The defining identity: sum_b eq(z, b) * f(b) == f(z).
    total = None
    for e, t in zip(eq.table, f.table, strict=True):
        term = e * t
        total = term if total is None else total + term
    expected = f.evaluate({v[0]: zs[0], v[1]: zs[1]}, in_place=False).constant()
    assert total == expected


def test_mle_full_evaluation():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    d1 = MLE(ring=ring, variables=v, evaluations=[1, 2, 3, 4])
    out = d1.evaluate({v[0]: 2, v[1]: 3}, in_place=False)
    # Full evaluation -> a single (num_vars == 0) entry; f(2, 3) = 9.
    assert out.num_vars == 0
    assert out.constant().get_polynomial()[0] == 9
    # The original oracle is untouched (in_place=False).
    assert d1.num_vars == 2 and d1.table[0].get_polynomial()[0] == 1
