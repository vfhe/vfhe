# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Element-wise products of field vectors: an extension of a word-sized prime field,
and pseudo-Mersenne fields of several words."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import ExtensionField, Field, FieldVector, PseudoMersenneField

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: The degree-2 extension of a 49-bit prime field, x^2 - w with w the smallest
#: non-residue, over vectors of 2^log_length elements; polycom and piop run at
#: the same field.
FP2 = {
    "q49-l16": {"prime": 562949953421201, "degree": 2, "log_length": 16},
    "q49-l20": {"prime": 562949953421201, "degree": 2, "log_length": 20},
}
#: Pseudo-Mersenne fields of 256 and 260 bits, both five 52-bit words: the 260-bit
#: one fills its words, the 256-bit one also takes the reduction's top-bit step.
PMF = {
    "pmf256-l16": {"bits": 256, "log_length": 16},
    "pmf260-l16": {"bits": 260, "log_length": 16},
}


def _non_residue(p: int) -> int:
    """Returns the smallest quadratic non-residue modulo the odd prime ``p``."""
    w = 2
    while pow(w, (p - 1) // 2, p) != p - 1:
        w += 1
    return w


def _product(benchmark: BenchmarkFixture, field: Field, length: int) -> None:
    """``out = a * b`` over random vectors, writing into an existing vector."""
    a, b, out = (
        FieldVector(field, length),
        FieldVector(field, length),
        FieldVector(field, length),
    )
    a.sample_random(b"bench-a")
    b.sample_random(b"bench-b")
    result = benchmark(a.mul, b, out=out)
    for i in (0, length // 2, length - 1):
        assert result[i] == a[i] * b[i]


@pytest.mark.parametrize("p", list(FP2.values()), ids=list(FP2))
def extension_field_product(benchmark: BenchmarkFixture, p: dict) -> None:
    """``a * b`` over the degree-2 extension of a 49-bit prime field.

    The longer vector is larger than a typical last-level cache.
    """
    field = ExtensionField(p["prime"], p["degree"], _non_residue(p["prime"]))
    _product(benchmark, field, 1 << p["log_length"])


@pytest.mark.parametrize("p", list(PMF.values()), ids=list(PMF))
def pseudo_mersenne_product(benchmark: BenchmarkFixture, p: dict) -> None:
    """``a * b`` over a pseudo-Mersenne prime field."""
    _product(benchmark, PseudoMersenneField.generate(p["bits"]), 1 << p["log_length"])


#: The folds a sumcheck round makes, over the degree-4 extension of a 48- and a
#: 61-bit prime and a 208-bit pseudo-Mersenne field, on 2^log_length elements.
FOLDS = {
    "fp4-q48-l16": {"prime": 140737974894593, "degree": 4, "w": 3, "log_length": 16},
    "fp4-q48-l20": {"prime": 140737974894593, "degree": 4, "w": 3, "log_length": 20},
    "fp4-q61-l20": {"prime": (1 << 61) - 1, "degree": 4, "w": 3, "log_length": 20},
    "pmf208-l20": {"bits": 208, "log_length": 20},
}
folds = pytest.mark.parametrize("p", list(FOLDS.values()), ids=list(FOLDS))
#: The sums of a GKR layer over a tree of fractions, as a sumcheck runs them:
#: four tables (a(x, 0), a(x, 1), h(x, 0), h(x, 1)), eq-weighted, at three
#: nodes: a0 h1 + a1 h0, and h0 h1.
LAYER_SUMS = [[(1, (0, 3)), (1, (1, 2))], [(1, (2, 3))]]


def _fold_field(p: dict) -> Field:
    if "bits" in p:
        return PseudoMersenneField.generate(p["bits"])
    return ExtensionField(p["prime"], p["degree"], p["w"])


def _tables(field: Field, count: int, length: int, seed: bytes) -> list[FieldVector]:
    tables = []
    for j in range(count):
        table = FieldVector(field, length)
        table.sample_random(seed + bytes([j]))
        tables.append(table)
    return tables


@folds
def fold_inner_products(benchmark: BenchmarkFixture, p: dict) -> None:
    """The inner products of two folded vectors at 0, 1 and 2: the native kernel."""
    a, b = _tables(_fold_field(p), 2, 1 << p["log_length"], b"bench-fold-ip")
    assert benchmark(a.fold_inner_products, b) == FieldVector.fold_inner_products(a, b)


@folds
def fold_inner_products_generic(benchmark: BenchmarkFixture, p: dict) -> None:
    """The same inner products by the front's whole-vector body."""
    a, b = _tables(_fold_field(p), 2, 1 << p["log_length"], b"bench-fold-ip")
    assert benchmark(FieldVector.fold_inner_products, a, b) == a.fold_inner_products(b)


@folds
def fold_and_inner_products(benchmark: BenchmarkFixture, p: dict) -> None:
    """The folds of two vectors at r and the inner products of the folds, in one kernel."""
    field = _fold_field(p)
    a, b = _tables(field, 2, 1 << p["log_length"], b"bench-fold-fip")
    r = field.random_element(b"bench-fold-r")
    fa, fb = a.fold(r), b.fold(r)
    result = benchmark(a.fold_and_inner_products, b, r)
    assert result[2] == FieldVector.fold_inner_products(fa, fb)


@folds
def fold_and_inner_products_generic(benchmark: BenchmarkFixture, p: dict) -> None:
    """The two folds, then the front's whole-vector inner products."""
    field = _fold_field(p)
    a, b = _tables(field, 2, 1 << p["log_length"], b"bench-fold-fip")
    r = field.random_element(b"bench-fold-r")

    def op():
        fa, fb = a.fold(r), b.fold(r)
        return FieldVector.fold_inner_products(fa, fb)

    assert benchmark(op) == a.fold_and_inner_products(b, r)[2]


@folds
def fold_product_sums(benchmark: BenchmarkFixture, p: dict) -> None:
    """`LAYER_SUMS` over four eq-weighted tables at three nodes: the native kernel."""
    field = _fold_field(p)
    length = 1 << p["log_length"]
    tables = _tables(field, 4, length, b"bench-fold-ps")
    (weights,) = _tables(field, 1, length // 2, b"bench-fold-ps-w")
    expected = FieldVector._fold_product_sums(tables, LAYER_SUMS, 3, weights)  # noqa: SLF001
    assert (
        benchmark(FieldVector.fold_product_sums, tables, LAYER_SUMS, 3, weights)
        == expected
    )


@folds
def fold_product_sums_generic(benchmark: BenchmarkFixture, p: dict) -> None:
    """`LAYER_SUMS` by the front's whole-vector body."""
    field = _fold_field(p)
    length = 1 << p["log_length"]
    tables = _tables(field, 4, length, b"bench-fold-ps")
    (weights,) = _tables(field, 1, length // 2, b"bench-fold-ps-w")
    expected = FieldVector.fold_product_sums(tables, LAYER_SUMS, 3, weights)
    generic = FieldVector._fold_product_sums  # noqa: SLF001 - the front's body, timed against the kernel
    assert benchmark(generic, tables, LAYER_SUMS, 3, weights) == expected


@folds
def fold_and_product_sums(benchmark: BenchmarkFixture, p: dict) -> None:
    """The folds of four tables at r and `LAYER_SUMS` over the folds, in one kernel."""
    field = _fold_field(p)
    length = 1 << p["log_length"]
    tables = _tables(field, 4, length, b"bench-fold-fps")
    (weights,) = _tables(field, 1, length // 4, b"bench-fold-fps-w")
    r = field.random_element(b"bench-fold-r")
    folded = [t.fold(r) for t in tables]
    expected = FieldVector._fold_product_sums(folded, LAYER_SUMS, 3, weights)  # noqa: SLF001
    result = benchmark(
        FieldVector.fold_and_product_sums, tables, r, LAYER_SUMS, 3, weights
    )
    assert result[1] == expected


@folds
def fold_and_product_sums_generic(benchmark: BenchmarkFixture, p: dict) -> None:
    """The four folds, then `LAYER_SUMS` by the front's whole-vector body."""
    field = _fold_field(p)
    length = 1 << p["log_length"]
    tables = _tables(field, 4, length, b"bench-fold-fps")
    (weights,) = _tables(field, 1, length // 4, b"bench-fold-fps-w")
    r = field.random_element(b"bench-fold-r")

    def op():
        folded = [t.fold(r) for t in tables]
        return FieldVector._fold_product_sums(folded, LAYER_SUMS, 3, weights)  # noqa: SLF001

    assert (
        benchmark(op)
        == FieldVector.fold_and_product_sums(tables, r, LAYER_SUMS, 3, weights)[1]
    )
