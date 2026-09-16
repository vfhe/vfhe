# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Vectors of `ExtensionField` elements, as d coefficient planes.

The plane layout -- coefficient j of every element contiguous -- is what makes
a vector worth having: each operation becomes a fixed number of calls into the
engine-tuned eltwise kernels over length-n runs, instead of n calls over one
d-word element. The C side states the layout contract at the `FieldVector`
declaration in ``arith.h``; this module is what allocates buffers meeting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.arith._alloc import aligned64, aligned64_unset
from vfhe.arith.base import FieldVector, _sample_positions, index_buffer
from vfhe.engine import ffi, lib

from .field import ExtensionFieldElement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .field import ExtensionField


class ExtensionFieldVector(FieldVector):
    """n elements of one `ExtensionField`, held in d coefficient planes."""

    #: The parent every element belongs to.
    field: ExtensionField

    def __init__(self, field: ExtensionField, values: int | Iterable) -> None:
        """
        Build from a length, or from a sequence of values.

        An int allocates that many zeros. A sequence builds one element per
        entry, each accepted in the forms `ExtensionFieldElement` takes: an element of
        the same field, an int, or a list of up to d coefficients.
        """
        self.field = field
        if isinstance(values, bool):
            raise TypeError("values is a length or a sequence, not a bool")
        if isinstance(values, int):
            if values < 0:
                raise ValueError(f"length must not be negative, got {values}")
            self._allocate(values, zeroed=True)
            return
        values = list(values)
        # Not zeroed: `_write_range` below writes every one of them.
        self._allocate(len(values))
        if values:
            self._write_range(0, values)

    def _allocate(self, n: int, zeroed: bool = False) -> None:
        """Reserve d padded planes and the struct the kernels read them from.

        **Undefined by default**, because almost every vector here is about to
        be written whole -- by a kernel, or by `_write_range`. `zeroed` is for
        the one case that is not: a caller asking for a vector of zeros.

        The padding past `n` is zeroed either way. The arithmetic reads and
        writes it, so it must hold reduced values whatever the data words
        hold; `field_vec_clear_padding` does it in one call over d planes
        rather than a slice assignment per plane from Python.
        """
        field = self.field
        self._n = n
        self._allocated_n = lib.field_vec_padded_length(n)
        # Kept alive as attributes: the struct holds borrowed pointers into them.
        allocator = aligned64 if zeroed else aligned64_unset
        self._planes = [
            allocator("uint64_t[]", self._allocated_n) for _ in range(field.d)
        ]
        self._plane_ptrs = ffi.new("uint64_t*[]", self._planes)
        self._struct = ffi.new("FieldVector")
        self._struct.coeffs = self._plane_ptrs
        self._struct.n = n
        self._struct.allocated_n = self._allocated_n
        self._struct.d = field.d
        self._struct.w = field.w
        self._struct.mod = field.mod
        if not zeroed:
            lib.field_vec_clear_padding(self._struct)

    def _like(self, n: int | None = None) -> ExtensionFieldVector:
        """A destination over the same field, this length unless told another.

        Left **unzeroed**: every caller here hands it straight to a kernel
        that writes all of it. That is the internal counterpart of
        `FieldVector(field, n)`, which does promise zeros -- so an operation
        producing a result uses this and a caller asking for a vector gets
        that one.
        """
        result = ExtensionFieldVector.__new__(ExtensionFieldVector)
        result.field = self.field
        result._allocate(self._n if n is None else n)
        return result

    def _coerce_element(self, value) -> ExtensionFieldElement:
        """Promote `value` to an element of this field, or raise."""
        if isinstance(value, ExtensionFieldElement):
            if value.field is not self.field:
                raise ValueError("element belongs to a different field")
            return value
        if isinstance(value, (int, list, tuple)) and not isinstance(value, bool):
            return ExtensionFieldElement(
                self.field, list(value) if not isinstance(value, int) else value
            )
        raise TypeError(f"cannot use {type(value).__name__} as a field element")

    def _write_range(self, start: int, values: list) -> None:
        """Transpose `values` into the planes with one call, not one per value.

        Plain integers take a shortcut. They are scalars of F_p, so only plane
        0 is nonzero, and a whole run of them goes in with one slice assignment
        per plane -- the copying then happens in C. The general path below
        builds an element per value, which is the right cost for a handful and
        the wrong one for a table of n constants: that was most of what
        building a code's twist tables used to spend.
        """
        d = self.field.d
        count = len(values)
        if count and all(type(value) is int for value in values):
            prime = self.field.prime
            self._planes[0][start : start + count] = [v % prime for v in values]
            zeros = [0] * count
            for j in range(1, d):
                self._planes[j][start : start + count] = zeros
            return
        flat = ffi.new("uint64_t[]", count * d)
        for i, value in enumerate(values):
            element = self._coerce_element(value)
            for j in range(d):
                flat[i * d + j] = element.value[j]
        lib.field_vec_set_range(self._struct, start, flat, count)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        """Yield each element, in index order. Each is a copy (see `__getitem__`)."""
        for i in range(self._n):
            yield self[i]

    def __getitem__(self, index: int) -> ExtensionFieldElement:
        """
        The element at `index`, as a detached copy.

        Writing to the returned element does not touch the vector; assign
        through `__setitem__` to change one.
        """
        index = self._checked_index(index)
        out = ffi.new("uint64_t[]", self.field.d)
        lib.field_vec_get_element(out, self._struct, index)
        return ExtensionFieldElement(self.field, out)

    def __setitem__(self, index: int, value) -> None:
        """Replace the element at `index`."""
        index = self._checked_index(index)
        element = self._coerce_element(value)
        lib.field_vec_set_element(self._struct, index, element.value)

    def _checked_index(self, index: int) -> int:
        """Normalize a negative index and reject one out of range."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"index must be an int, not {type(index).__name__}")
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"index out of range for a vector of {self._n}")
        return index

    def to_list(self) -> list[ExtensionFieldElement]:
        """Every element, in index order."""
        d = self.field.d
        if self._n == 0:
            return []
        flat = ffi.new("uint64_t[]", self._n * d)
        lib.field_vec_get_range(flat, self._struct, 0, self._n)
        # One memmove an element rather than a word at a time: the d words are
        # already contiguous and adjacent in `flat`, so the split is a copy,
        # not a gather, and Python has no business doing it a word at a time.
        width = d * 8
        source = ffi.cast("char *", flat)
        elements = []
        for i in range(self._n):
            out = ffi.new("uint64_t[]", d)
            ffi.memmove(out, source + i * width, width)
            elements.append(ExtensionFieldElement(self.field, out))
        return elements

    def copy(self) -> ExtensionFieldVector:
        """An independent vector with the same contents."""
        result = self._like()
        lib.field_vec_copy(result._struct, self._struct)
        return result

    def _destinations(
        self, out, n: int
    ) -> tuple[ExtensionFieldVector, ExtensionFieldVector]:
        """The two destinations `split_even_odd` writes, checked as a pair."""
        if out is None:
            return self._like(n), self._like(n)
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a pair of vectors, one per half")
        return self._destination(out[0], n), self._destination(out[1], n)

    def _destination(self, out, n: int | None = None) -> ExtensionFieldVector:
        """Where a result goes: `out` when given, a fresh vector otherwise.

        An `out` is what lets a chunked expression reuse one buffer instead of
        allocating per operation -- the half of the fusion story that `view`
        does not cover.
        """
        if out is None:
            return self._like(n)
        if not isinstance(out, ExtensionFieldVector):
            raise TypeError(f"out must be a FieldVector, not {type(out).__name__}")
        if out.field is not self.field:
            raise ValueError("out belongs to a different field")
        want = self._n if n is None else n
        if len(out) != want:
            raise ValueError(f"out holds {len(out)} elements, not {want}")
        return out

    def _binary(self, other, vector_kernel, scalar_kernel, out=None):
        """Apply the elementwise kernel, or the broadcast one for an element."""
        result = self._destination(out)
        if isinstance(other, ExtensionFieldVector):
            if other.field is not self.field:
                raise ValueError("vectors belong to different fields")
            if len(other) != self._n:
                raise ValueError(f"length mismatch: {self._n} and {len(other)}")
            vector_kernel(result._struct, self._struct, other._struct)
            return result
        element = self._coerce_element(other)
        scalar_kernel(result._struct, self._struct, element.value)
        return result

    def add(self, other, out=None) -> ExtensionFieldVector:
        """`self + other` into `out` when given -- `__add__` with a destination."""
        return self._binary(other, lib.field_vec_add, lib.field_vec_add_scalar, out)

    def sub(self, other, out=None) -> ExtensionFieldVector:
        """`self - other` into `out` when given."""
        return self._binary(other, lib.field_vec_sub, lib.field_vec_sub_scalar, out)

    def mul(self, other, out=None) -> ExtensionFieldVector:
        """`self * other` into `out` when given."""
        return self._binary(other, lib.field_vec_mul, lib.field_vec_scale, out)

    def rsub(self, other, out=None) -> ExtensionFieldVector:
        """`other - self` for one element `other`, in one pass."""
        element = self._coerce_element(other)
        result = self._destination(out)
        lib.field_vec_scalar_sub(result._struct, element.value, self._struct)
        return result

    def neg(self, out=None) -> ExtensionFieldVector:
        """`-self` into `out` when given."""
        result = self._destination(out)
        lib.field_vec_neg(result._struct, self._struct)
        return result

    def fma(self, b, c, out=None) -> ExtensionFieldVector:
        """`self + b * c`, the product formed and consumed in one pass.

        Four streams through memory rather than the six a multiply and an add
        take, and no intermediate vector at all. `c` is a vector or one
        element; with `out` and `view` it is what makes a chunked expression
        allocation-free.
        """
        if not isinstance(b, ExtensionFieldVector):
            raise TypeError(f"b must be a FieldVector, not {type(b).__name__}")
        if b.field is not self.field:
            raise ValueError("vectors belong to different fields")
        if len(b) != self._n:
            raise ValueError(f"length mismatch: {self._n} and {len(b)}")
        result = self._destination(out)
        if isinstance(c, ExtensionFieldVector):
            if c.field is not self.field:
                raise ValueError("vectors belong to different fields")
            if len(c) != self._n:
                raise ValueError(f"length mismatch: {self._n} and {len(c)}")
            lib.field_vec_fma(result._struct, self._struct, b._struct, c._struct)
        else:
            element = self._coerce_element(c)
            lib.field_vec_fma_scalar(
                result._struct, self._struct, b._struct, element.value
            )
        return result

    def __add__(self, other) -> ExtensionFieldVector:
        """Elementwise sum, or the sum with one element broadcast."""
        return self._binary(other, lib.field_vec_add, lib.field_vec_add_scalar)

    def __radd__(self, other) -> ExtensionFieldVector:
        """``other + self``; addition commutes, so this is `__add__`."""
        return self.__add__(other)

    def __sub__(self, other) -> ExtensionFieldVector:
        """Elementwise difference, or the difference with one element."""
        return self._binary(other, lib.field_vec_sub, lib.field_vec_sub_scalar)

    def __rsub__(self, other) -> ExtensionFieldVector:
        """
        ``other - self``, where `other` is one element.

        NOT symmetric with `__sub__`: subtraction does not commute, so this is
        the reversed-operand kernel rather than a delegation.
        """
        return self.rsub(other)

    def __neg__(self) -> ExtensionFieldVector:
        """Elementwise negation."""
        result = self._like()
        lib.field_vec_neg(result._struct, self._struct)
        return result

    def __mul__(self, other) -> ExtensionFieldVector:
        """
        The elementwise (Hadamard) product, or the product with one element.

        A vector operand multiplies position by position; an element (or an
        int) multiplies every position, the same as `scale`.
        """
        return self._binary(other, lib.field_vec_mul, lib.field_vec_scale)

    def __rmul__(self, other) -> ExtensionFieldVector:
        """``other * self``; multiplication commutes, so this is `__mul__`."""
        return self.__mul__(other)

    def scale(self, value, out=None) -> ExtensionFieldVector:
        """Every element multiplied by one field element, into `out` when given."""
        element = self._coerce_element(value)
        result = self._destination(out)
        lib.field_vec_scale(result._struct, self._struct, element.value)
        return result

    def sum(self) -> ExtensionFieldElement:
        """The sum of every element; zero for an empty vector."""
        out = ffi.new("uint64_t[]", self.field.d)
        lib.field_vec_sum(out, self._struct)
        return ExtensionFieldElement(self.field, out)

    def inverse(self) -> ExtensionFieldVector:
        """
        The elementwise inverse, by Montgomery's trick.

        One inversion and three multiplications per element rather than n
        inversions. Raises ValueError if any element is zero.
        """
        result = self._like()
        if lib.field_vec_inv(result._struct, self._struct) == 0:
            raise ValueError("vector contains an element that is not invertible")
        return result

    def split_even_odd(
        self,
        block: int = 1,
        out: tuple[ExtensionFieldVector, ExtensionFieldVector] | None = None,
    ) -> tuple[ExtensionFieldVector, ExtensionFieldVector]:
        """
        Deinterleave into the even-indexed and odd-indexed halves.

        Requires an even length; each half holds n / 2 elements. `block`
        deinterleaves runs of that many elements instead of single ones.
        """
        block = self._checked_block(block)
        half = self._n // 2
        even, odd = self._destinations(out, half)
        lib.field_vec_split_blocks(even._struct, odd._struct, self._struct, block)
        return even, odd

    @staticmethod
    def interleave(even, odd, out=None) -> ExtensionFieldVector:
        """Even positions from `even`, odd from `odd`: the inverse of `split_even_odd`."""
        if not isinstance(even, ExtensionFieldVector) or not isinstance(
            odd, ExtensionFieldVector
        ):
            raise TypeError("interleave takes two ExtensionFieldVectors")
        if even.field is not odd.field:
            raise ValueError("cannot interleave vectors over different fields")
        if len(even) != len(odd):
            raise ValueError(f"length mismatch: {len(even)} and {len(odd)}")
        result = even._destination(out, 2 * len(even))
        lib.field_vec_interleave(result._struct, even._struct, odd._struct)
        return result

    @staticmethod
    def concat(vectors: list, out=None) -> ExtensionFieldVector:
        """One vector holding every element of `vectors`, in order."""
        vectors = list(vectors)
        if not vectors:
            raise ValueError("concat needs at least one vector")
        first = vectors[0]
        for vector in vectors:
            if not isinstance(vector, ExtensionFieldVector):
                raise TypeError(f"cannot concatenate a {type(vector).__name__}")
            if vector.field is not first.field:
                raise ValueError("cannot concatenate vectors over different fields")
        result = first._destination(out, sum(len(v) for v in vectors))
        parts = ffi.new("FieldVector[]", [v._struct for v in vectors])
        lib.field_vec_concat(result._struct, parts, len(vectors))
        return result

    def query(self, indices) -> ExtensionFieldVector:
        """The elements at `indices`, gathered; the front states the contract."""
        view = index_buffer(indices)
        if view is None:
            positions = [self._checked_index(i) for i in indices]
            result = self._like(len(positions))
            if positions:
                # _checked_index has already rejected everything out of range,
                # so the kernel's own answer cannot be false here.
                lib.field_vec_gather(
                    result._struct,
                    self._struct,
                    ffi.new("uint64_t[]", positions),
                    len(positions),
                )
            return result

        # Already the kernel's layout: nothing copied, and the bound tested
        # there rather than once per index here.
        count = len(view)
        result = self._like(count)
        if count and not lib.field_vec_gather(
            result._struct, self._struct, ffi.from_buffer("uint64_t[]", view), count
        ):
            raise IndexError(
                f"index out of range for a vector of {self._n}: "
                f"{next(i for i in view if i >= self._n)}"
            )
        return result

    def fold(self, r, block: int = 1) -> ExtensionFieldVector:
        """``lo + r * (hi - lo)`` per pair, in one kernel pass.

        The pair partner sits `block` positions away -- adjacent by default,
        the two halves at ``block = n // 2``.
        """
        block = self._checked_block(block)
        element = self._coerce_element(r)
        result = self._like(self._n // 2)
        lib.field_vec_fold_blocks(result._struct, self._struct, block, element.value)
        return result

    def chunk_length(self, live: int = 4) -> int:
        """The front states the contract."""
        return self._chunk_length(self.field.d * 8, live)

    def chain_bytes(self, live: int = 4) -> int:
        """The front states the contract; here an element is its planes."""
        return len(self) * self.field.d * 8 * max(live, 1)

    @property
    def padding_unit(self) -> int:
        """The eltwise kernels' vector width; the front says what it governs."""
        return lib.field_vec_padded_length(1)

    def view(self, start: int = 0, length: int | None = None) -> ExtensionFieldVector:
        """`length` elements from `start`, over this vector's own planes.

        The padding unit is the eltwise kernels' vector width, which is what
        `field_vec_padded_length` rounds to; the front states what `start`
        and `length` must satisfy against it. The view holds a reference to
        this vector, so the planes outlive it.
        """
        start, length = self._checked_view(
            start, length, lib.field_vec_padded_length(1)
        )
        result = ExtensionFieldVector.__new__(ExtensionFieldVector)
        result.field = self.field
        result._n = length
        result._allocated_n = lib.field_vec_padded_length(length)
        # The planes are the parent's, offset; `_planes` holds it alive rather
        # than owning buffers of its own.
        result._planes = self._planes
        result._plane_ptrs = ffi.new(
            "uint64_t*[]", [plane + start for plane in self._planes]
        )
        result._struct = ffi.new("FieldVector")
        result._struct.coeffs = result._plane_ptrs
        result._struct.n = length
        result._struct.allocated_n = result._allocated_n
        result._struct.d = self.field.d
        result._struct.w = self.field.w
        result._struct.mod = self.field.mod
        return result

    def frobenius(self, k: int = 1) -> ExtensionFieldVector:
        """`ExtensionFieldElement.frobenius` on every element, as the d
        whole-plane scalings it is."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"k must be an int, not {type(k).__name__}")
        if k < 0:
            raise ValueError("k must not be negative; the group is cyclic of order d")
        result = self._like()
        lib.field_vec_frobenius(result._struct, self._struct, k)
        return result

    def sample_random_at(self, seed: bytes, indices) -> None:
        """The front states the contract."""
        view = index_buffer(indices)
        if view is None:
            positions = _sample_positions(indices, self._n)
            buffer = ffi.new("uint64_t[]", positions) if positions else ffi.NULL
        else:
            if len(view) != self._n:
                raise ValueError(f"expected {self._n} indices, got {len(view)}")
            buffer = ffi.from_buffer("uint64_t[]", view) if self._n else ffi.NULL
        if self._n:
            lib.field_vec_sample_random_at(
                self._struct, seed, len(seed), buffer, self._n
            )

    def sample_random(self, seed: bytes, start: int = 0) -> None:
        """Fill with uniform elements drawn from `seed`, in place, taking the
        sequence from position `start`."""
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(f"start must be an int, not {type(start).__name__}")
        if start < 0:
            raise ValueError(f"start must not be negative, got {start}")
        lib.field_vec_sample_random(self._struct, seed, len(seed), start)

    def hash(self) -> bytes:
        """BLAKE3 over every element in index order, 32 bytes."""
        out = ffi.new("uint8_t[32]")
        lib.field_vec_hash(out, self._struct)
        return bytes(out)

    def _digests(self, count: int, kernel, group: int, stride: int) -> memoryview:
        """The kernel's own output buffer, viewed rather than copied.

        It already writes one digest per window, contiguously, which is the
        packed layout the front promises -- so there is nothing to do but
        hand it over. The view holds the buffer alive and is read-only, which
        is what makes it safe to pass somewhere that will not copy it either.
        """
        if count == 0:
            return memoryview(b"")
        out = ffi.new("uint8_t[]", count * 32)
        kernel(out, self._struct, group, stride)
        return memoryview(ffi.buffer(out)).toreadonly()

    def hash_elements(self, group: int = 1, stride: int = 1) -> memoryview:
        """Every window's digest, packed; the front states the contract."""
        if group < 1 or stride < 1:
            raise ValueError(
                f"group and stride must be positive, got {group}, {stride}"
            )
        count = lib.field_vec_hash_count(self._struct, group, stride)
        return self._digests(count, lib.field_vec_hash_elements, group, stride)

    def hash_fibers(self, group: int = 1, stride: int = 1) -> memoryview:
        """Every fiber's digest, packed; the front states the contract."""
        if group < 1 or stride < 1:
            raise ValueError(
                f"group and stride must be positive, got {group}, {stride}"
            )
        count = lib.field_vec_hash_fiber_count(self._struct, group, stride)
        return self._digests(count, lib.field_vec_hash_fibers, group, stride)

    def __eq__(self, other: object) -> bool:
        """Equal length and equal elements, over the same field."""
        if not isinstance(other, ExtensionFieldVector):
            return NotImplemented
        if other.field is not self.field:
            return False
        return bool(lib.field_vec_is_equal(self._struct, other._struct))

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __repr__(self) -> str:
        return f"ExtensionFieldVector(n={self._n}, d={self.field.d})"
