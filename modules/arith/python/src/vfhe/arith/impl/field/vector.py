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

from vfhe.arith._alloc import aligned64
from vfhe.arith.base import FieldVector
from vfhe.engine import ffi, lib

from .field import FieldElement


class ExtensionFieldVector(FieldVector):
    """n elements of one `ExtensionField`, held in d coefficient planes."""

    def __init__(self, field, values) -> None:
        """
        Build from a length, or from a sequence of values.

        An int allocates that many zeros. A sequence builds one element per
        entry, each accepted in the forms `FieldElement` takes: an element of
        the same field, an int, or a list of up to d coefficients.
        """
        self.field = field
        if isinstance(values, int) and not isinstance(values, bool):
            if values < 0:
                raise ValueError(f"length must not be negative, got {values}")
            self._allocate(values)
            return
        values = list(values)
        self._allocate(len(values))
        if values:
            self._write_range(0, values)

    def _allocate(self, n: int) -> None:
        """Reserve d padded planes and the struct the kernels read them from."""
        field = self.field
        self._n = n
        self._allocated_n = lib.field_vec_padded_length(n)
        # Kept alive as attributes: the struct holds borrowed pointers into them.
        self._planes = [
            aligned64("uint64_t[]", self._allocated_n) for _ in range(field.d)
        ]
        self._plane_ptrs = ffi.new("uint64_t*[]", self._planes)
        self._struct = ffi.new("FieldVector")
        self._struct.coeffs = self._plane_ptrs
        self._struct.n = n
        self._struct.allocated_n = self._allocated_n
        self._struct.d = field.d
        self._struct.w = field.w
        self._struct.mod = field.mod

    def _like(self, n: int | None = None) -> ExtensionFieldVector:
        """A fresh vector over the same field, this length unless told another."""
        return ExtensionFieldVector(self.field, self._n if n is None else n)

    def _coerce_element(self, value) -> FieldElement:
        """Promote `value` to an element of this field, or raise."""
        if isinstance(value, FieldElement):
            if value.field is not self.field:
                raise ValueError("element belongs to a different field")
            return value
        if isinstance(value, (int, list, tuple)) and not isinstance(value, bool):
            return FieldElement(
                self.field, list(value) if not isinstance(value, int) else value
            )
        raise TypeError(f"cannot use {type(value).__name__} as a field element")

    def _write_range(self, start: int, values: list) -> None:
        """Transpose `values` into the planes with one call, not one per value."""
        d = self.field.d
        flat = ffi.new("uint64_t[]", len(values) * d)
        for i, value in enumerate(values):
            element = self._coerce_element(value)
            for j in range(d):
                flat[i * d + j] = element.value[j]
        lib.field_vec_set_range(self._struct, start, flat, len(values))

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        """Yield each element, in index order. Each is a copy (see `__getitem__`)."""
        for i in range(self._n):
            yield self[i]

    def __getitem__(self, index: int) -> FieldElement:
        """
        The element at `index`, as a detached copy.

        Writing to the returned element does not touch the vector; assign
        through `__setitem__` to change one.
        """
        index = self._checked_index(index)
        out = ffi.new("uint64_t[]", self.field.d)
        lib.field_vec_get_element(out, self._struct, index)
        return FieldElement(self.field, out)

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

    def to_list(self) -> list[FieldElement]:
        """Every element, in index order."""
        d = self.field.d
        if self._n == 0:
            return []
        flat = ffi.new("uint64_t[]", self._n * d)
        lib.field_vec_get_range(flat, self._struct, 0, self._n)
        elements = []
        for i in range(self._n):
            out = ffi.new("uint64_t[]", d)
            for j in range(d):
                out[j] = flat[i * d + j]
            elements.append(FieldElement(self.field, out))
        return elements

    def copy(self) -> ExtensionFieldVector:
        """An independent vector with the same contents."""
        result = self._like()
        lib.field_vec_copy(result._struct, self._struct)
        return result

    def _binary(self, other, vector_kernel, scalar_kernel):
        """Apply the elementwise kernel, or the broadcast one for an element."""
        if isinstance(other, ExtensionFieldVector):
            if other.field is not self.field:
                raise ValueError("vectors belong to different fields")
            if len(other) != self._n:
                raise ValueError(f"length mismatch: {self._n} and {len(other)}")
            result = self._like()
            vector_kernel(result._struct, self._struct, other._struct)
            return result
        element = self._coerce_element(other)
        result = self._like()
        scalar_kernel(result._struct, self._struct, element.value)
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
        element = self._coerce_element(other)
        result = self._like()
        lib.field_vec_scalar_sub(result._struct, element.value, self._struct)
        return result

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

    def scale(self, value) -> ExtensionFieldVector:
        """Every element multiplied by one field element."""
        element = self._coerce_element(value)
        result = self._like()
        lib.field_vec_scale(result._struct, self._struct, element.value)
        return result

    def sum(self) -> FieldElement:
        """The sum of every element; zero for an empty vector."""
        out = ffi.new("uint64_t[]", self.field.d)
        lib.field_vec_sum(out, self._struct)
        return FieldElement(self.field, out)

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

    def split_even_odd(self) -> tuple[ExtensionFieldVector, ExtensionFieldVector]:
        """
        Deinterleave into the even-indexed and odd-indexed halves.

        Requires an even length; each half holds n / 2 elements.
        """
        if self._n % 2:
            raise ValueError(f"length {self._n} is odd; cannot split into halves")
        half = self._n // 2
        even, odd = self._like(half), self._like(half)
        lib.field_vec_split_even_odd(even._struct, odd._struct, self._struct)
        return even, odd

    def sample_random(self, seed: bytes) -> None:
        """Fill with uniform elements drawn from `seed`, in place."""
        lib.field_vec_sample_random(self._struct, seed, len(seed))

    def hash(self) -> bytes:
        """BLAKE3 over every element in index order, 32 bytes."""
        out = ffi.new("uint8_t[32]")
        lib.field_vec_hash(out, self._struct)
        return bytes(out)

    def hash_elements(self, group: int = 1, stride: int = 1) -> list[bytes]:
        """
        One digest per window of `group` elements, taken every `stride` indices.

        Window k covers elements ``k * stride`` through
        ``k * stride + group - 1``, and only whole windows count -- so the
        Merkle-leaf case of adjacent pairs is ``group=2, stride=2``. Empty
        when no whole window fits.
        """
        if group < 1 or stride < 1:
            raise ValueError(
                f"group and stride must be positive, got {group}, {stride}"
            )
        count = lib.field_vec_hash_count(self._struct, group, stride)
        if count == 0:
            return []
        out = ffi.new("uint8_t[]", count * 32)
        lib.field_vec_hash_elements(out, self._struct, group, stride)
        raw = bytes(ffi.buffer(out))
        return [raw[k * 32 : (k + 1) * 32] for k in range(count)]

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
