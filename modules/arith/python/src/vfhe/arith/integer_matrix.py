# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""A matrix of signed 64-bit integers, laid out once for the native kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vfhe.util.bindings import ffi

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Buffer

    from .impl.rns.polynomial import RNSRing


class IntegerMatrix:
    """A matrix of integers, for multiplying vectors of ring elements by it
    (:meth:`RNSRing.integer_linear_combinations`).

    ``rows`` is either a sequence of rows of Python integers or a
    two-dimensional buffer of signed 64-bit integers in row-major order (C
    order), for instance a NumPy array of dtype ``int64``. A buffer is read
    in place, not copied, so it must not change while the matrix is in use.
    Every entry must fit in a signed 64-bit integer.

    The first product with elements over some primes prepares a native copy
    of the matrix for them (`prepared`), which later products reuse.
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
        #: Per base address, the prepared form and the mask of primes it covers.
        self._prepared: dict[int, tuple[Any, int]] = {}

    def prepared(self, ring: RNSRing) -> Any:
        """Return the native form of the matrix for elements of ``ring``: a
        ``RNS_IntegerMatrix`` (``rns_integer_matrix_new``) holding a copy of
        the matrix with its entries reduced modulo each of the ring's primes.

        There is one such form per base of primes (the primes of all the rings
        of one degree and split degree), kept for the life of the matrix. When
        ``ring`` has primes the form does not cover yet, it is rebuilt for
        those and the ones it covered, so a chain of levels shares one form.
        """
        key = int(ffi.cast("uintptr_t", ring.base))
        native, mask = self._prepared.get(key, (None, 0))
        if native is None or ring.mask & ~mask:
            mask |= ring.mask
            native = ffi.gc(
                ring.lib.rns_integer_matrix_new(
                    ring.base, mask, self.entries, self.n_rows, self.n_columns
                ),
                ring.lib.rns_integer_matrix_free,
            )
            self._prepared[key] = (native, mask)
        return native
