# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for vfhe.arith.mle: MLE in both bases and over
both coefficient types - pure-Python tables (no ring) and the C-backed
ring tables (mle_dense_poly_* kernels over the cffi boundary) - plus
SparseMLE arithmetic. Evaluation points are always concrete values —
protocol futures live at the Transcript / Statement level (test_piop.py).
"""

import pytest
from vfhe.arith import (
    MLE,
    Field,
    FieldVector,
    MLE_Basis,
    MLE_Variable,
    Polynomial,
    Ring,
    SparseMLE,
    repr,
    set_generic_only,
)
from vfhe.arith.mle import element_array, native_table
from vfhe.util import parallel, trace
from vfhe.util.bindings import ffi, lib


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
    assert all(a == b for a, b in zip(msb.table, msb_poly.table, strict=True))
    # LSB variable (pairs kernel), both value types.
    lsb = f.evaluate({v[0]: 2}, in_place=False)
    assert [p.get_polynomial()[0] for p in lsb.table] == [3, 5, 7, 9]
    lsb_poly = f.evaluate({v[0]: two}, in_place=False)
    assert all(a == b for a, b in zip(lsb.table, lsb_poly.table, strict=True))
    # Binding order does not change the result.
    zs = [ring.random_exceptional() for _ in range(3)]
    a = f.evaluate({v[0]: zs[0], v[1]: zs[1], v[2]: zs[2]}, in_place=False)
    b = f.evaluate({v[2]: zs[2], v[0]: zs[0], v[1]: zs[1]}, in_place=False)
    c = f.evaluate({v[1]: zs[1], v[2]: zs[2], v[0]: zs[0]}, in_place=False)
    assert a.constant() == b.constant() and b.constant() == c.constant()
    # The source table is untouched throughout.
    assert f.num_vars == 3 and f.table[0].get_polynomial()[0] == 1


@pytest.mark.parametrize("basis", ["evaluations", "coefficients"])
@pytest.mark.parametrize("num_vars", [1, 2, 5])
def test_field_table_bind_any_position(basis, num_vars):
    """A field table folds through whole-vector kernels at every variable
    position, in either basis. The reference is the same table over plain
    Python values, which takes the entry-at-a-time path."""
    field = Field((1 << 61) - 1, 4, w=3)
    values = FieldVector(field, 1 << num_vars)
    values.sample_random(b"mle-field-table")
    entries = values.to_list()
    v = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    point = [field.random_element(bytes([i])) for i in range(num_vars)]

    def build(backed: bool):
        """The same table over the field vector, or over plain values."""
        table = FieldVector(field, entries) if backed else list(entries)
        parent = field if backed else None
        if basis == "evaluations":
            return MLE(field=parent, variables=list(v), evaluations=table)
        return MLE(field=parent, variables=list(v), coefficients=table)

    for idx in range(num_vars):
        native, plain = build(True), build(False)
        native._bind(v[idx], point[idx])
        plain._bind(v[idx], point[idx])
        assert native.table.to_list() == plain.table

    # ...and binding everything, in either order, reaches the same value.
    forward, backward = build(True), build(True)
    for idx in range(num_vars):
        forward._bind(v[idx], point[idx])
    for idx in reversed(range(num_vars)):
        backward._bind(v[idx], point[idx])
    assert forward.constant() == backward.constant()


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
        _ = f.table[6] == 7  # via __eq__, which poisons the entry
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


def test_sparse_mle_evaluates_at_full_points():
    v = [MLE_Variable("x0"), MLE_Variable("x1"), MLE_Variable("x2")]
    s = SparseMLE(variables=v, evaluations={5: 3, 2: 1})
    dense = s.materialize()
    assert dense.table == [0, 0, 1, 0, 0, 3, 0, 0]
    for point in ([0, 0, 0], [1, 0, 1], [7, 11, 13], [2, -3, 5]):
        pt = dict(zip(v, point, strict=True))
        assert (
            s.evaluate(pt).constant() == dense.evaluate(pt, in_place=False).constant()
        )
    with pytest.raises(NotImplementedError):
        s.evaluate({v[0]: 1})


def test_sparse_mle_over_a_ring():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    v = [MLE_Variable("x0"), MLE_Variable("x1")]
    s = SparseMLE(ring=ring, variables=v, evaluations={1: 1, 3: 2})
    pt = {v[0]: ring.random_exceptional(), v[1]: ring.random_exceptional()}
    want = s.materialize().evaluate(pt, in_place=False).constant()
    assert s.evaluate(pt).constant() == want
    assert (s + s).evaluations == {1: 2, 3: 4} and (s + s).ring is ring
    assert s.scale(3).evaluations == {1: 3, 3: 6}


def test_field_table_digest_hashes_the_vector_once():
    field = Field((1 << 61) - 1, 2, w=5)
    vector = FieldVector(field, 8)
    vector.sample_random(b"\x01" * 32)
    table = MLE(field=field, num_vars=3, evaluations=vector)
    other = table.copy()
    assert other.digest() == table.digest()
    other.table[5] = other.table[5] + field.one
    assert other.digest() != table.digest()
    renamed = table.rename({table.variables[0]: MLE_Variable("y")})
    assert renamed.digest() != table.digest()
    assert table.to_coefficients().digest() != table.digest()


def test_ring_table_digest_streams_every_row():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    entries = [Polynomial(ring).from_array([i, 3 * i + 1]) for i in range(4)]
    table = MLE(ring=ring, num_vars=2, evaluations=entries)
    reprs = [p.repr for p in entries]
    digest = table.digest()
    assert [p.repr for p in entries] == reprs  # digesting converts copies
    in_ntt = [p.copy() for p in entries]
    for p in in_ntt:
        p.to_NTT()
    assert MLE(ring=ring, num_vars=2, evaluations=in_ntt).digest() == digest
    swapped = [entries[1], entries[0], *entries[2:]]
    assert MLE(ring=ring, num_vars=2, evaluations=swapped).digest() != digest
    assert type(entries[0]).digest_sequence(entries[:1]) == entries[0].digest()


def test_plain_table_digest_follows_the_entries():
    table = MLE(num_vars=2, evaluations=[1, -2, 3, 10**40])
    assert table.digest() == MLE(num_vars=2, evaluations=[1, -2, 3, 10**40]).digest()
    assert table.digest() != MLE(num_vars=2, evaluations=[1, -2, 3, 0]).digest()


# ---------------------------------------------------------------------------
# Linear combinations of tables
# ---------------------------------------------------------------------------


def _random_ring_table(ring, variables, ntt: bool = True) -> MLE:
    entries = [ring.random_element(ntt=ntt) for _ in range(1 << len(variables))]
    return MLE(ring=ring, variables=variables, evaluations=entries)


@pytest.mark.parametrize("split_degree", [1, 4])
def test_linear_combination_matches_a_python_loop(split_degree):
    """Known answers: each entry is the sum of products computed with ring
    operators, for ring-element and integer coefficients, zero and one among
    them, and tables given in coefficient form."""
    ring = Ring(64, prime_size=[30, 49, 55], split_degree=split_degree)
    v = [MLE_Variable(f"x{i}") for i in range(3)]
    tables = [_random_ring_table(ring, v, ntt=i % 2 == 0) for i in range(6)]
    one = Polynomial(ring).from_array([1])
    zero = Polynomial(ring).from_array([0])
    coefficients = [ring.random_element(), 0, 1, one, -5, 2**70 + 3]
    expected = [
        sum(
            (t.table[j] * c for t, c in zip(tables[1:], coefficients[1:], strict=True)),
            tables[0].table[j] * coefficients[0],
        )
        for j in range(8)
    ]
    out = MLE.linear_combination(tables, coefficients)
    assert out.variables == v and out.ring == ring and out.basis is MLE_Basis.eval
    assert all(a == b for a, b in zip(out.table, expected, strict=True))
    assert out.digest() == MLE(ring=ring, variables=v, evaluations=expected).digest()

    # One table: a coefficient of one copies it, of zero clears it.
    t = tables[0]
    assert MLE.linear_combination([t], [one]).digest() == t.digest()
    assert MLE.linear_combination([t], [1]).digest() == t.digest()
    assert all(e == 0 for e in MLE.linear_combination([t], [zero]).table)
    assert all(e == 0 for e in MLE.linear_combination([t], [0]).table)
    # The inputs are left as they were.
    assert all(a == b for a, b in zip(tables[0].table, t.table, strict=True))


@pytest.mark.parametrize("basis", ["evaluations", "coefficients"])
def test_linear_combination_over_a_field_and_plain_values(basis):
    """The field-backed and plain-value tables give the same entries."""
    field = Field((1 << 61) - 1, 4, w=3)
    v = [MLE_Variable(f"x{i}") for i in range(4)]
    vectors = []
    for i in range(3):
        vec = FieldVector(field, 16)
        vec.sample_random(bytes([i]))
        vectors.append(vec)
    coefficients = [field.random_element(b"c"), 1, -3]

    def build(vec, backed: bool) -> MLE:
        table = vec if backed else vec.to_list()
        return MLE(field=field if backed else None, variables=v, **{basis: table})

    native = MLE.linear_combination([build(x, True) for x in vectors], coefficients)
    plain = MLE.linear_combination([build(x, False) for x in vectors], coefficients)
    expected_basis = MLE_Basis.eval if basis == "evaluations" else MLE_Basis.coeff
    assert native.basis is plain.basis is expected_basis
    assert native.table.to_list() == plain.table
    a, b, c = (x.to_list() for x in vectors)
    assert plain.table == [
        coefficients[0] * x + y - 3 * z for x, y, z in zip(a, b, c, strict=True)
    ]
    single = MLE.linear_combination([build(vectors[0], True)], [0])
    assert all(e == field.zero for e in single.table.to_list())


def test_linear_combination_rejects_tables_that_do_not_line_up():
    ring = Ring(64, prime_size=[49], split_degree=1)
    v = [MLE_Variable(f"x{i}") for i in range(2)]
    f = _random_ring_table(ring, v)
    with pytest.raises(ValueError, match="variables"):
        MLE.linear_combination([f, _random_ring_table(ring, v[::-1])], [1, 1])
    with pytest.raises(ValueError, match="variables"):
        other = [MLE_Variable(f"x{i}") for i in range(2)]
        MLE.linear_combination([f, _random_ring_table(ring, other)], [1, 1])
    with pytest.raises(ValueError, match="ring"):
        other_ring = Ring(64, prime_size=[49, 49], split_degree=1)
        MLE.linear_combination([f, _random_ring_table(other_ring, v)], [1, 1])
    with pytest.raises(ValueError, match="coefficient"):
        MLE.linear_combination([f, f], [1])
    with pytest.raises(ValueError, match="coefficient"):
        MLE.linear_combination([], [])
    with pytest.raises(ValueError, match="ring"):
        other_ring = Ring(64, prime_size=[49, 49], split_degree=1)
        MLE.linear_combination([f], [other_ring.random_element()])


@pytest.mark.parametrize("split_degree", [1, 4])
def test_linear_combination_is_the_same_through_the_generic_interface(split_degree):
    """The one-reduction sum of products over RNS and the generic interface's
    products give the same words."""
    ring = Ring(64, prime_size=[30, 49, 55], split_degree=split_degree)
    v = [MLE_Variable(f"x{i}") for i in range(4)]
    tables = [_random_ring_table(ring, v) for _ in range(5)]
    coefficients = [ring.random_element() for _ in range(4)] + [7]
    fused = MLE.linear_combination(tables, coefficients)
    set_generic_only(True)
    try:
        generic = MLE.linear_combination(tables, coefficients)
    finally:
        set_generic_only(False)
    assert fused.digest() == generic.digest()


def test_linear_combination_kernel_with_no_terms_or_a_wrong_domain():
    """The native call's edges, which `MLE.linear_combination` never takes:
    no terms give zeros in the mul domain, and an operand outside the mul
    domain is refused before anything is written."""
    ring = Ring(64, prime_size=[30, 49], split_degree=1)
    v = [MLE_Variable(f"x{i}") for i in range(2)]
    f = _random_ring_table(ring, v)
    out = _random_ring_table(ring, v)
    before = out.digest()
    coefficient = ring.random_element(ntt=False)
    status = lib.mle_dense_poly_linear_combination(
        ring.arith_ring,
        out.table_ptr,
        ffi.new("ArithElement *[]", [f.table_ptr]),
        element_array([coefficient]),
        1,
        4,
    )
    assert status != 0
    assert out.digest() == before
    status = lib.mle_dense_poly_linear_combination(
        ring.arith_ring, out.table_ptr, ffi.NULL, ffi.NULL, 0, 4
    )
    assert status == 0
    assert all(e == 0 for e in out.table)


# ---------------------------------------------------------------------------
# The table kernels on several threads
# ---------------------------------------------------------------------------

# Three primes at N = 64 are 1.5 KiB an entry, so 2^11 entries (and the 2^10
# a bind leaves) are past the size below which the kernels stay on one
# thread, and split into a number of chunks that does not divide the table.
THREADED_VARS = 11


def _at_threads(n: int, compute):
    parallel.set_num_threads(n)
    try:
        return compute()
    finally:
        parallel.set_num_threads()


def _bind(position: int, integer: bool):
    def run(f: MLE, z) -> MLE:
        index = position if position >= 0 else f.num_vars + position
        return f.evaluate({f.variables[index]: 7 if integer else z}, in_place=False)

    return run


TABLE_OPERATIONS = {
    "add": lambda f, g, _z: f + g,
    "sub": lambda f, g, _z: f - g,
    "scale": lambda f, _g, z: f.scale(z),
    "scale_int": lambda f, _g, _z: f.scale(12345),
    "bind_pairs": lambda f, _g, z: _bind(0, False)(f, z),
    "bind_pairs_int": lambda f, _g, z: _bind(0, True)(f, z),
    "bind_halves": lambda f, _g, z: _bind(-1, False)(f, z),
    "bind_halves_int": lambda f, _g, z: _bind(-1, True)(f, z),
    "bind_strided": lambda f, _g, z: _bind(4, False)(f, z),
    "bind_strided_int": lambda f, _g, z: _bind(4, True)(f, z),
    "linear_combination": lambda f, g, z: MLE.linear_combination(
        [f, g, f], [z, 3, g.table[5]]
    ),
}


@pytest.fixture(params=[1, 4], ids=["split1", "split4"])
def large_tables(request):
    ring = Ring(64, prime_size=[30, 49, 55], split_degree=request.param)
    v = [MLE_Variable(f"x{i}") for i in range(THREADED_VARS)]
    return (
        _random_ring_table(ring, v),
        _random_ring_table(ring, v),
        ring.random_element(),
    )


@pytest.mark.parametrize("operation", list(TABLE_OPERATIONS))
def test_table_kernels_give_the_same_words_on_any_number_of_threads(
    large_tables, operation
):
    f, g, z = large_tables
    compute = TABLE_OPERATIONS[operation]
    serial = _at_threads(1, lambda: compute(f, g, z))
    threaded = _at_threads(8, lambda: compute(f, g, z))
    assert serial.digest() == threaded.digest()


@pytest.mark.parametrize("layout", ["pairs", "strided"])
def test_a_binding_over_its_own_input_is_the_same_on_any_number_of_threads(
    large_tables, layout
):
    """A pairs or strided binding written over its input reads entries other
    output entries overwrite, so it stays correct only in entry order."""
    f, _g, z = large_tables
    expected = _bind(0 if layout == "pairs" else 4, False)(f, z)

    def bind_in_place() -> MLE:
        h = f.copy()
        h.to_NTT()
        size = 1 << (THREADED_VARS - 1)
        if layout == "pairs":
            lib.mle_dense_poly_evaluate_pairs(
                f.ring.arith_ring, h.table_ptr, h.table_ptr, element_array([z]), size
            )
        else:
            lib.mle_dense_poly_evaluate(
                f.ring.arith_ring,
                h.table_ptr,
                h.table_ptr,
                element_array([z]),
                THREADED_VARS,
                4,
            )
        return h

    for threads in (1, 8):
        h = _at_threads(threads, bind_in_place)
        assert h.table[: 1 << (THREADED_VARS - 1)] == expected.table


def test_a_recorded_table_operation_runs_in_entry_order(large_tables):
    """While the tape records, the kernels stay on the calling thread, so the
    events come in entry order whatever the thread limit."""
    f, g, z = large_tables
    trace.set_verify(True)
    parallel.set_num_threads(8)
    try:
        with trace.recording() as t:
            out = MLE.linear_combination([f, g], [z, g.table[1]])
            bound = f.evaluate({f.variables[0]: z}, in_place=False)
        events = [e for e in t.events() if e.kind not in ("mark", "input", "copy")]
        size = 1 << THREADED_VARS
        combination, binding = events[: 2 * size], events[2 * size :]
        assert [e.kind for e in combination] == ["mul", "mul_addto"] * size
        assert [e.value for e in combination[1::2]] == out.table
        assert [e.kind for e in binding] == ["sub", "mul", "add"] * (size // 2)
        assert [e.value for e in binding[2::3]] == bound.table
        assert t.untraced == 0
    finally:
        parallel.set_num_threads()
        trace.enable().clear()


# (N, prime sizes, split degree): narrow (below 2^32) and wide rows, on rings
# split and not.
UINT64_SHAPES = [
    (64, [28, 50], 1),
    (64, [28, 31, 61], 4),
    (256, [29, 40], 16),
]


def _random_table(ring, num_vars: int, ntt: bool = True) -> MLE:
    entries = [ring.random_element(ntt=ntt) for _ in range(1 << num_vars)]
    return MLE(ring=ring, num_vars=num_vars, evaluations=entries)


@pytest.mark.parametrize(("n", "sizes", "d"), UINT64_SHAPES)
@pytest.mark.parametrize("representation", [repr.coeff, repr.ntt])
def test_table_uint64_rows_are_the_entries_rows(n, sizes, d, representation):
    # The buffer is the entries' own `to_uint64` rows one after another, and
    # reads back, entry by entry, as `from_uint64` reads each.
    ring = Ring(n, prime_size=sizes, split_degree=d)
    table = _random_table(ring, 3)
    words = table.to_uint64(repr=representation)
    width = ring.ell * ring.N
    assert words.tolist() == [
        w for p in table.table for w in p.to_uint64(repr=representation)
    ]
    back = MLE.from_uint64(ring, table.variables, words, repr=representation)
    assert native_table(back) and back.variables == table.variables
    for k, (got, want) in enumerate(zip(back.table, table.table, strict=True)):
        assert got.repr == repr.ntt
        assert got.to_uint64(repr=repr.ntt) == want.to_uint64(repr=repr.ntt)
        one = words[k * width : (k + 1) * width]
        assert got == Polynomial(ring).from_uint64(one, repr=representation)


@pytest.mark.parametrize(("n", "sizes", "d"), UINT64_SHAPES)
def test_table_to_uint64_leaves_the_entries_as_they_are(n, sizes, d):
    # Entries held in either representation are read through copies.
    ring = Ring(n, prime_size=sizes, split_degree=d)
    table = _random_table(ring, 2)
    table.table[1].to_coeff()
    held = [p.repr for p in table.table]
    for representation in (repr.coeff, repr.ntt):
        assert table.to_uint64(repr=representation).tolist() == [
            w for p in table.table for w in p.to_uint64(repr=representation)
        ]
    assert [p.repr for p in table.table] == held


def test_table_from_uint64_takes_a_numpy_array():
    np = pytest.importorskip("numpy")
    ring = Ring(64, prime_size=[28, 50], split_degree=4)
    table = _random_table(ring, 2)
    cube = np.frombuffer(table.to_uint64(), dtype=np.uint64).reshape(
        4, ring.ell, ring.N
    )
    back = MLE.from_uint64(ring, table.variables, cube.copy())
    assert all(a == b for a, b in zip(back.table, table.table, strict=True))
    with pytest.raises(TypeError, match="C-contiguous"):
        MLE.from_uint64(ring, table.variables, np.asfortranarray(cube))


def test_native_table_calls_refuse_entries_over_another_ring():
    """A table may hold entries of another ring; the calls that size their
    work by the table's ring refuse it rather than walk the wrong rows."""
    ring = Ring(64, prime_size=[30, 49], split_degree=1)
    wider = Ring(64, prime_size=[30, 49, 55], split_degree=1)
    v = [MLE_Variable(f"x{i}") for i in range(2)]
    mixed = MLE(
        ring=ring,
        variables=v,
        evaluations=[wider.random_element() for _ in range(4)],
    )
    with pytest.raises(ValueError, match="table's ring"):
        mixed.to_uint64()
    with pytest.raises(ValueError, match="table's ring"):
        MLE.linear_combination([_random_ring_table(ring, v), mixed], [1, 1])


def test_table_from_uint64_refuses_a_value_above_its_prime():
    from array import array

    ring = Ring(32, prime_size=[28, 50], split_degree=2)
    table = _random_table(ring, 2)
    words = table.to_uint64()
    words[(2 * ring.ell + 1) * ring.N + 7] = ring.primes[1]
    words[(3 * ring.ell + 0) * ring.N] = ring.primes[0] + 1
    with pytest.raises(ValueError, match="entry 2, row 1"):
        MLE.from_uint64(ring, table.variables, words)
    with pytest.raises(ValueError, match="values for"):
        MLE.from_uint64(ring, table.variables[:1], words)
    with pytest.raises(TypeError, match="64-bit"):
        MLE.from_uint64(ring, table.variables, array("I", [0] * len(words)))


@pytest.fixture
def threads():
    """Lets the test's parallel calls use up to 8 threads (vfhe defaults to 1)."""
    parallel.set_num_threads(8)
    yield
    parallel.set_num_threads()


@pytest.mark.usefixtures("threads")
@pytest.mark.parametrize("representation", [repr.coeff, repr.ntt])
def test_table_uint64_is_the_same_on_more_threads(representation):
    # 256 entries of three 4096-word rows: above the sizes where the copy and
    # the transform go to other threads, in items of four rows that straddle
    # the entries.
    ring = Ring(4096, prime_size=[30, 50, 60], split_degree=4)
    table = _random_table(ring, 8)
    with parallel.local_num_threads(1):
        serial = table.to_uint64(repr=representation)
        one = MLE.from_uint64(ring, table.variables, serial, repr=representation)
    assert table.to_uint64(repr=representation) == serial
    many = MLE.from_uint64(ring, table.variables, serial, repr=representation)
    for a, b in zip(many.table, one.table, strict=True):
        assert a.to_uint64(repr=repr.ntt) == b.to_uint64(repr=repr.ntt)
