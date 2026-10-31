# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Counting over arrays of 64-bit integers."""

from __future__ import annotations

from array import array

from vfhe.util.bindings import ffi, lib


def occurrence_counts(values, bound: int) -> tuple[array, array] | None:
    """For `values`, a buffer of 64-bit unsigned integers (``array('Q')``),
    the pair ``(prior, totals)`` of ``array('Q')``s: ``prior[i]`` is how many
    of ``values[:i]`` equal ``values[i]``, and ``totals[x]`` how many of the
    values equal `x`, for ``x < bound``.

    None when a value is `bound` or more.
    """
    view = memoryview(values)
    if view.itemsize != 8 or view.format not in ("Q", "L"):
        raise TypeError("values must be a buffer of 64-bit unsigned integers")
    if bound < 0:
        raise ValueError(f"bound {bound} is negative")
    count = len(view)
    prior = array("Q", bytes(8 * count))
    totals = array("Q", bytes(8 * bound))
    if count and not lib.vec_occurrence_counts(
        ffi.from_buffer("uint64_t[]", prior),
        ffi.from_buffer("uint64_t[]", totals),
        ffi.from_buffer("uint64_t[]", values),
        count,
        bound,
    ):
        return None
    return prior, totals
