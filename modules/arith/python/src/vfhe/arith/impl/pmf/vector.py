# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Vectors of `PseudoMersenneField` elements, as L limb planes.

A single element fills one AVX-512 register with its own limbs, so its carries
run across lanes. A vector inverts that: plane j holds limb j of every element,
one register holds one limb of eight different elements, and carries move
between planes at a fixed lane. That is what a vector buys here -- SIMD across
elements rather than within one -- on top of the single boundary crossing.

The C side states the layout contract at the `PMFVector` declaration in
``arith.h``; this module allocates buffers meeting it.
"""

from __future__ import annotations

from vfhe.arith._alloc import aligned64
from vfhe.arith.base import FieldVector
from vfhe.engine import ffi, lib

from .pseudo_mersenne import _LANES, PseudoMersenneElement


class PseudoMersenneVector(FieldVector):
    """n elements of one `PseudoMersenneField`, held in L limb planes."""

    def __init__(self, field, values) -> None:
        """
        Build from a length, or from a sequence of values.

        An int allocates that many zeros. A sequence builds one element per
        entry, each either an element of the same field or an int, which is
        reduced into it.
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
        """Reserve L padded planes and the struct the kernels read them from."""
        field = self.field
        self._n = n
        self._allocated_n = lib.pmf_vec_padded_length(n)
        # Kept alive as attributes: the struct holds borrowed pointers into them.
        self._planes = [
            aligned64("uint64_t[]", self._allocated_n) for _ in range(field.limbs)
        ]
        self._plane_ptrs = ffi.new("uint64_t*[]", self._planes)
        self._struct = ffi.new("PMFVector")
        self._struct.limbs = self._plane_ptrs
        self._struct.n = n
        self._struct.allocated_n = self._allocated_n
        self._struct.params = field._params

    def _like(self, n: int | None = None) -> PseudoMersenneVector:
        """A fresh vector over the same field, this length unless told another."""
        return PseudoMersenneVector(self.field, self._n if n is None else n)

    def _coerce_element(self, value) -> PseudoMersenneElement:
        """Promote `value` to an element of this field, or raise."""
        if isinstance(value, PseudoMersenneElement):
            if value.field.prime != self.field.prime:
                raise ValueError("element belongs to a different field")
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return PseudoMersenneElement(self.field, value)
        raise TypeError(f"cannot use {type(value).__name__} as a field element")

    def _write_range(self, start: int, values: list) -> None:
        """Transpose `values` into the planes with one call, not one per value."""
        flat = ffi.new("uint64_t[]", len(values) * _LANES)
        for i, value in enumerate(values):
            element = self._coerce_element(value)
            for k in range(self.field.limbs):
                flat[i * _LANES + k] = element._buf[k]
        lib.pmf_vec_set_range(self._struct, start, flat, len(values))

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        """Yield each element, in index order. Each is a copy (see `__getitem__`)."""
        for i in range(self._n):
            yield self[i]

    def __getitem__(self, index: int) -> PseudoMersenneElement:
        """
        The element at `index`, as a detached copy.

        Elements are immutable, so this is a copy only in the sense that it
        stops tracking the vector: a later `__setitem__` does not change it.
        """
        index = self._checked_index(index)
        buf = self.field._new_buffer()
        lib.pmf_vec_get_element(buf, self._struct, index)
        return self.field._wrap(buf)

    def __setitem__(self, index: int, value) -> None:
        """Replace the element at `index`."""
        index = self._checked_index(index)
        element = self._coerce_element(value)
        lib.pmf_vec_set_element(self._struct, index, element._buf)

    def _checked_index(self, index: int) -> int:
        """Normalize a negative index and reject one out of range."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"index must be an int, not {type(index).__name__}")
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"index out of range for a vector of {self._n}")
        return index

    def to_list(self) -> list[PseudoMersenneElement]:
        """Every element, in index order."""
        if self._n == 0:
            return []
        flat = ffi.new("uint64_t[]", self._n * _LANES)
        lib.pmf_vec_get_range(flat, self._struct, 0, self._n)
        elements = []
        for i in range(self._n):
            # Copied into an aligned buffer of its own: an element's buffer is
            # what the kernels load as one register, so it cannot be a slice.
            buf = self.field._new_buffer()
            for k in range(_LANES):
                buf[k] = flat[i * _LANES + k]
            elements.append(self.field._wrap(buf))
        return elements

    def copy(self) -> PseudoMersenneVector:
        """An independent vector with the same contents."""
        result = self._like()
        lib.pmf_vec_copy(result._struct, self._struct)
        return result

    def _binary(self, other, vector_kernel, scalar_kernel):
        """Apply the elementwise kernel, or the broadcast one for an element."""
        if isinstance(other, PseudoMersenneVector):
            if other.field.prime != self.field.prime:
                raise ValueError("vectors belong to different fields")
            if len(other) != self._n:
                raise ValueError(f"length mismatch: {self._n} and {len(other)}")
            result = self._like()
            vector_kernel(result._struct, self._struct, other._struct)
            return result
        element = self._coerce_element(other)
        result = self._like()
        scalar_kernel(result._struct, self._struct, element._buf)
        return result

    def __add__(self, other) -> PseudoMersenneVector:
        """Elementwise sum, or the sum with one element broadcast."""
        return self._binary(other, lib.pmf_vec_add, lib.pmf_vec_add_scalar)

    def __radd__(self, other) -> PseudoMersenneVector:
        """``other + self``; addition commutes, so this is `__add__`."""
        return self.__add__(other)

    def __sub__(self, other) -> PseudoMersenneVector:
        """Elementwise difference, or the difference with one element."""
        return self._binary(other, lib.pmf_vec_sub, lib.pmf_vec_sub_scalar)

    def __rsub__(self, other) -> PseudoMersenneVector:
        """
        ``other - self``, where `other` is one element.

        NOT symmetric with `__sub__`: subtraction does not commute, so this is
        the reversed-operand kernel rather than a delegation.
        """
        element = self._coerce_element(other)
        result = self._like()
        lib.pmf_vec_scalar_sub(result._struct, element._buf, self._struct)
        return result

    def __neg__(self) -> PseudoMersenneVector:
        """Elementwise negation."""
        result = self._like()
        lib.pmf_vec_neg(result._struct, self._struct)
        return result

    def __mul__(self, other) -> PseudoMersenneVector:
        """
        The elementwise (Hadamard) product, or the product with one element.

        A vector operand multiplies position by position; an element (or an
        int) multiplies every position, the same as `scale`.
        """
        return self._binary(other, lib.pmf_vec_mul, lib.pmf_vec_scale)

    def __rmul__(self, other) -> PseudoMersenneVector:
        """``other * self``; multiplication commutes, so this is `__mul__`."""
        return self.__mul__(other)

    def scale(self, value) -> PseudoMersenneVector:
        """Every element multiplied by one field element."""
        element = self._coerce_element(value)
        result = self._like()
        lib.pmf_vec_scale(result._struct, self._struct, element._buf)
        return result

    def sum(self) -> PseudoMersenneElement:
        """The sum of every element; zero for an empty vector."""
        buf = self.field._new_buffer()
        lib.pmf_vec_sum(buf, self._struct)
        return self.field._wrap(buf)

    def split_even_odd(self) -> tuple[PseudoMersenneVector, PseudoMersenneVector]:
        """
        Deinterleave into the even-indexed and odd-indexed halves.

        Requires an even length; each half holds n / 2 elements.
        """
        if self._n % 2:
            raise ValueError(f"length {self._n} is odd; cannot split into halves")
        half = self._n // 2
        even, odd = self._like(half), self._like(half)
        lib.pmf_vec_split_even_odd(even._struct, odd._struct, self._struct)
        return even, odd

    def sample_random(self, seed: bytes) -> None:
        """Fill with uniform elements drawn from `seed`, in place."""
        lib.pmf_vec_sample_random(self._struct, seed, len(seed))

    def hash(self) -> bytes:
        """BLAKE3 over every element's canonical encoding, in order, 32 bytes."""
        out = ffi.new("uint8_t[32]")
        lib.pmf_vec_hash(out, self._struct)
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
        count = lib.pmf_vec_hash_count(self._struct, group, stride)
        if count == 0:
            return []
        out = ffi.new("uint8_t[]", count * 32)
        lib.pmf_vec_hash_elements(out, self._struct, group, stride)
        raw = bytes(ffi.buffer(out))
        return [raw[k * 32 : (k + 1) * 32] for k in range(count)]

    def __eq__(self, other: object) -> bool:
        """Equal length and equal elements, over the same field."""
        if not isinstance(other, PseudoMersenneVector):
            return NotImplemented
        if other.field.prime != self.field.prime:
            return False
        return bool(lib.pmf_vec_is_equal(self._struct, other._struct))

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __repr__(self) -> str:
        return (
            f"PseudoMersenneVector(n={self._n}, 2^{self.field.bits} - {self.field.c})"
        )
