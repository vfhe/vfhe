# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Counting over arrays of 64-bit integers."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from typing_extensions import Buffer


def _checked_length(values: Buffer, bound: int) -> int:
    """The number of `values`, a one-dimensional buffer of 64-bit unsigned
    integers, checked along with `bound`."""
    view = memoryview(values)
    if view.ndim != 1 or view.itemsize != 8 or view.format not in ("Q", "L"):
        raise TypeError(
            "values must be a one-dimensional buffer of 64-bit unsigned integers"
        )
    if bound < 0:
        raise ValueError(f"bound {bound} is negative")
    return len(view)


def histogram(values: Buffer, bound: int) -> array | None:
    """Count how many times each of `0, 1, ..., bound - 1` occurs in
    `values`.

    Returns `totals`, where `totals[x]` is the number of entries of `values`
    equal to `x`. For example, for `values = [2, 0, 2, 2]` and `bound = 4`,
    `totals = [1, 0, 3, 0]`. Returns None if an entry is `bound` or larger.

    `values` is a flat buffer of unsigned 64-bit integers, for instance
    ``array.array("Q", ...)`` (the standard-library array; typecode ``"Q"``
    means unsigned 64-bit), and `totals` is returned as such an array. It is
    the second result of `count_occurrences`, computed without the first.
    """
    count = _checked_length(values, bound)
    totals = array("Q", bytes(8 * bound))
    if count and not lib.histogram(
        ffi.from_buffer("uint64_t[]", totals),
        ffi.from_buffer("uint64_t[]", values),
        count,
        bound,
    ):
        return None
    return totals


def count_occurrences(values: Buffer, bound: int) -> tuple[array, array] | None:
    """Count the occurrences of each value in `values`, both up to each
    position and in total.

    Returns `(prior, totals)`:

    - `prior[i]` is how many times `values[i]` occurs before position `i`;
    - `totals[x]` is how many times `x` occurs in `values`, for every
      `x < bound`.

    For example, for `values = [2, 0, 2, 2]` and `bound = 4`, `prior =
    [0, 0, 1, 2]` and `totals = [1, 0, 3, 0]`. Returns None if an entry is
    `bound` or larger.

    `values` is a flat buffer of unsigned 64-bit integers, for instance
    ``array.array("Q", ...)`` (the standard-library array; typecode ``"Q"``
    means unsigned 64-bit), and `prior` and `totals` are returned as such
    arrays.
    """
    count = _checked_length(values, bound)
    prior = array("Q", bytes(8 * count))
    totals = array("Q", bytes(8 * bound))
    if count and not lib.count_occurrences(
        ffi.from_buffer("uint64_t[]", prior),
        ffi.from_buffer("uint64_t[]", totals),
        ffi.from_buffer("uint64_t[]", values),
        count,
        bound,
    ):
        return None
    return prior, totals
