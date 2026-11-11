# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""A matrix of signed 64-bit integers, laid out once for the native kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vfhe.util.bindings import ffi

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Buffer


class IntegerMatrix:
    """A matrix of integers, stored once in the layout the native
    kernels read, so that it can be applied to many vectors
    (:meth:`RNSRing.integer_linear_combinations`).

    ``rows`` is either a sequence of rows of Python integers or a
    two-dimensional buffer of signed 64-bit integers in row-major order (C
    order), for instance a NumPy array of dtype ``int64``. A buffer is used
    without a copy, so it must not change while the matrix is in use. Every
    entry must fit in a signed 64-bit integer.
    """

    def __init__(self, rows: Sequence[Sequence[int]] | Buffer) -> None:
        try:
            view = memoryview(rows)  # type: ignore[arg-type]
        except TypeError:
            view = None
        if view is not None:
            if view.ndim != 2 or view.itemsize != 8 or view.format not in ("q", "l"):
                raise TypeError("expected a two-dimensional buffer of int64")
            if not view.c_contiguous:
                raise TypeError("the buffer must be C-contiguous")
            self.n_rows, self.n_columns = view.shape or (0, 0)
            self._source = rows
            self.entries = ffi.from_buffer("int64_t[]", view.cast("B"))
        else:
            rows = cast("Sequence[Sequence[int]]", rows)
            self.n_rows = len(rows)
            self.n_columns = len(rows[0]) if rows else 0
            if any(len(row) != self.n_columns for row in rows):
                raise ValueError("expected rows of one length")
            flat = [int(w) for row in rows for w in row]
            if any(not -(2**63) <= w < 2**63 for w in flat):
                raise ValueError("every entry must fit in a signed 64-bit integer")
            self.entries = ffi.new("int64_t[]", flat)
        if self.n_rows == 0 or self.n_columns == 0:
            raise ValueError("expected at least one row and one column")
