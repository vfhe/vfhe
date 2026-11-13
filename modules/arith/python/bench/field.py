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
