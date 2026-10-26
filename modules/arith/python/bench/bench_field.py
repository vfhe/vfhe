# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Element-wise products of field vectors: an extension of a word-sized prime field,
and pseudo-Mersenne fields of several words."""

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
