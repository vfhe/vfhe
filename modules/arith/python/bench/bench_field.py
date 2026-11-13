# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""Element-wise products of field vectors: an extension of a word-sized prime field,
and pseudo-Mersenne fields of several words. And the fold kernels of field vectors
(inner products and sums of products of folded tables), each against the front's
generic body, over the fields that take each native body."""

from __future__ import annotations

from harness import Case, Shape, benchmark
from vfhe.arith import ExtensionField, Field, FieldVector, PseudoMersenneField

#: Pseudo-Mersenne fields of 256 and 260 bits, both five 52-bit words: the 260-bit
#: one fills its words, the 256-bit one also takes the reduction's top-bit step.
PMF_SHAPES = [
    Shape("pmf256-l16", bits=256, log_length=16),
    Shape("pmf260-l16", bits=260, log_length=16),
]


def non_residue(p: int) -> int:
    """Returns the smallest quadratic non-residue modulo the odd prime ``p``."""
    w = 2
    while pow(w, (p - 1) // 2, p) != p - 1:
        w += 1
    return w


def product_case(field: Field, length: int) -> Case:
    """Times ``out = a * b`` over random vectors, writing into an existing vector."""
    a, b, out = (
        FieldVector(field, length),
        FieldVector(field, length),
        FieldVector(field, length),
    )
    a.sample_random(b"bench-a")
    b.sample_random(b"bench-b")

    def check(result: FieldVector) -> None:
        for i in (0, length // 2, length - 1):
            assert result[i] == a[i] * b[i]

    return Case(lambda: a.mul(b, out=out), check=check)


@benchmark(shapes=["fp2-q49-l16", "fp2-q49-l20"], tier="headline")
def extension_field_product(shape: Shape) -> Case:
    """``a * b`` over the degree-2 extension of a 49-bit prime field.

    The longer vector is larger than a typical last-level cache.
    """
    p = shape["prime"]
    field = ExtensionField(p, shape["degree"], non_residue(p))
    return product_case(field, 1 << shape["log_length"])


@benchmark(shapes=PMF_SHAPES)
def pseudo_mersenne_product(shape: Shape) -> Case:
    """``a * b`` over a pseudo-Mersenne prime field."""
    field = PseudoMersenneField.generate(shape["bits"])
    return product_case(field, 1 << shape["log_length"])


#: The fold kernels' fields, by the native body each takes: a pseudo-Mersenne
#: field (the pmf kernels); F_(p^4) for a 48-bit p, the IFMA body on the
#: avx512ifma engine; F_(p^4) for p = 2^61 - 1, the element-wise body there.
#: On the portable engine both extension fields take the scalar body.
FOLD_SHAPES = [
    Shape("fp4-q48-l16", prime=140737974894593, degree=4, w=3, log_length=16),
    Shape("fp4-q48-l20", prime=140737974894593, degree=4, w=3, log_length=20),
    Shape("fp4-q61-l20", prime=(1 << 61) - 1, degree=4, w=3, log_length=20),
    Shape("pmf208-l20", bits=208, log_length=20),
]

#: The sums of a GKR layer over a tree of fractions, as a sumcheck runs them:
#: four tables (a(x, 0), a(x, 1), h(x, 0), h(x, 1)), eq-weighted, at three
#: nodes: a0 h1 + a1 h0, and h0 h1.
LAYER_SUMS = [[(1, (0, 3)), (1, (1, 2))], [(1, (2, 3))]]


def fold_field(shape: Shape) -> Field:
    """The field of a fold shape."""
    if "bits" in shape:
        return PseudoMersenneField.generate(shape["bits"])
    return ExtensionField(shape["prime"], shape["degree"], shape["w"])


def random_tables(field: Field, count: int, length: int, seed: bytes) -> list:
    """`count` random vectors of `length` elements."""
    tables = []
    for j in range(count):
        table = FieldVector(field, length)
        table.sample_random(seed + bytes([j]))
        tables.append(table)
    return tables


def inner_products_case(shape: Shape, generic: bool) -> Case:
    """Times the inner products of two folded vectors at 0, 1 and 2."""
    field = fold_field(shape)
    a, b = random_tables(field, 2, 1 << shape["log_length"], b"bench-fold-ip")
    expected = FieldVector.fold_inner_products(a, b)
    op = (
        (lambda: FieldVector.fold_inner_products(a, b))
        if generic
        else (lambda: a.fold_inner_products(b))
    )

    def check(result: tuple) -> None:
        assert result == expected

    return Case(op, check=check)


def fold_and_inner_products_case(shape: Shape, generic: bool) -> Case:
    """Times the folds of two vectors at r and the inner products of the folds."""
    field = fold_field(shape)
    a, b = random_tables(field, 2, 1 << shape["log_length"], b"bench-fold-fip")
    r = field.random_element(b"bench-fold-r")

    def generic_op() -> tuple:
        fa, fb = a.fold(r), b.fold(r)
        return fa, fb, FieldVector.fold_inner_products(fa, fb)

    expected = generic_op()[2]

    def check(result: tuple) -> None:
        assert result[2] == expected

    return Case(
        generic_op if generic else (lambda: a.fold_and_inner_products(b, r)),
        check=check,
    )


def product_sums_case(shape: Shape, generic: bool) -> Case:
    """Times `LAYER_SUMS` over four eq-weighted tables at three nodes."""
    field = fold_field(shape)
    length = 1 << shape["log_length"]
    tables = random_tables(field, 4, length, b"bench-fold-ps")
    (weights,) = random_tables(field, 1, length // 2, b"bench-fold-ps-w")

    def generic_op() -> list:
        # The front's generic body, timed against the kernel.
        return FieldVector._fold_product_sums(tables, LAYER_SUMS, 3, weights)  # noqa: SLF001

    def native_op() -> list:
        return FieldVector.fold_product_sums(tables, LAYER_SUMS, 3, weights)

    expected = generic_op()

    def check(result: list) -> None:
        assert result == expected

    return Case(generic_op if generic else native_op, check=check)


def fold_and_product_sums_case(shape: Shape, generic: bool) -> Case:
    """Times the folds of four tables at r and `LAYER_SUMS` over the folds."""
    field = fold_field(shape)
    length = 1 << shape["log_length"]
    tables = random_tables(field, 4, length, b"bench-fold-fps")
    (weights,) = random_tables(field, 1, length // 4, b"bench-fold-fps-w")
    r = field.random_element(b"bench-fold-r")

    def generic_op() -> tuple:
        folded = [t.fold(r) for t in tables]
        # The front's generic body, timed against the kernel.
        sums = FieldVector._fold_product_sums(folded, LAYER_SUMS, 3, weights)  # noqa: SLF001
        return folded, sums

    expected = generic_op()[1]

    def native_op() -> tuple:
        return FieldVector.fold_and_product_sums(tables, r, LAYER_SUMS, 3, weights)

    def check(result: tuple) -> None:
        assert result[1] == expected

    return Case(generic_op if generic else native_op, check=check)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_inner_products(shape: Shape) -> Case:
    """`fold_inner_products` through the front: the native kernel."""
    return inner_products_case(shape, generic=False)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_inner_products_generic(shape: Shape) -> Case:
    """`fold_inner_products` by the front's whole-vector body."""
    return inner_products_case(shape, generic=True)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_and_inner_products(shape: Shape) -> Case:
    """`fold_and_inner_products` through the front: the native kernel."""
    return fold_and_inner_products_case(shape, generic=False)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_and_inner_products_generic(shape: Shape) -> Case:
    """The two folds, then the front's whole-vector inner products."""
    return fold_and_inner_products_case(shape, generic=True)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_product_sums(shape: Shape) -> Case:
    """`fold_product_sums` of a GKR layer through the front: the native kernel."""
    return product_sums_case(shape, generic=False)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_product_sums_generic(shape: Shape) -> Case:
    """`fold_product_sums` of a GKR layer by the front's whole-vector body."""
    return product_sums_case(shape, generic=True)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_and_product_sums(shape: Shape) -> Case:
    """`fold_and_product_sums` of a GKR layer through the front: the native kernel."""
    return fold_and_product_sums_case(shape, generic=False)


@benchmark(shapes=FOLD_SHAPES)
def field_fold_and_product_sums_generic(shape: Shape) -> Case:
    """The folds, then the front's whole-vector product sums of a GKR layer."""
    return fold_and_product_sums_case(shape, generic=True)
