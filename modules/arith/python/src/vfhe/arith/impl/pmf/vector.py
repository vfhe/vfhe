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

from typing import TYPE_CHECKING, cast

from vfhe.arith._alloc import aligned64, aligned64_unset
from vfhe.arith.base import FieldVector, index_buffer
from vfhe.engine import ffi, lib

from .pseudo_mersenne import _LANES, PseudoMersenneElement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .pseudo_mersenne import PseudoMersenneField


class PseudoMersenneVector(FieldVector):
    """n elements of one `PseudoMersenneField`, held in L limb planes."""

    #: The parent every element belongs to.
    field: PseudoMersenneField

    def __init__(self, field: PseudoMersenneField, values: int | Iterable) -> None:
        """
        Build from a length, or from a sequence of values.

        An int allocates that many zeros. A sequence builds one element per
        entry, each either an element of the same field or an int, which is
        reduced into it.
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
        """Reserve L padded planes and the struct the kernels read them from.

        Undefined by default and zeroed only when a caller asks for zeros; the
        padding is cleared either way, by `pmf_vec_clear_padding`. See the
        field vector, which says why at more length.
        """
        field = self.field
        self._n = n
        self._allocated_n = lib.pmf_vec_padded_length(n)
        # Kept alive as attributes: the struct holds borrowed pointers into them.
        allocator = aligned64 if zeroed else aligned64_unset
        self._planes = [
            allocator("uint64_t[]", self._allocated_n) for _ in range(field.limbs)
        ]
        self._plane_ptrs = ffi.new("uint64_t*[]", self._planes)
        self._struct = ffi.new("PMFVector")
        self._struct.limbs = self._plane_ptrs
        self._struct.n = n
        self._struct.allocated_n = self._allocated_n
        self._struct.params = field._params
        if not zeroed:
            lib.pmf_vec_clear_padding(self._struct)

    def _like(self, n: int | None = None) -> PseudoMersenneVector:
        """A destination over the same field, this length unless told another.

        Left **unzeroed**, as the field vector's is and for the same reason:
        every caller hands it straight to a kernel that writes all of it.
        `PseudoMersenneVector(field, n)` is the one that promises zeros.
        """
        result = PseudoMersenneVector.__new__(PseudoMersenneVector)
        result.field = self.field
        result._allocate(self._n if n is None else n)
        return result

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

    def _destinations(
        self, out, n: int
    ) -> tuple[PseudoMersenneVector, PseudoMersenneVector]:
        """The two destinations `split_even_odd` writes, checked as a pair."""
        if out is None:
            return self._like(n), self._like(n)
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a pair of vectors, one per half")
        return self._destination(out[0], n), self._destination(out[1], n)

    def _destination(self, out, n: int | None = None) -> PseudoMersenneVector:
        """Where a result goes: `out` when given, a fresh vector otherwise."""
        if out is None:
            return self._like(n)
        if not isinstance(out, PseudoMersenneVector):
            raise TypeError(f"out must be a FieldVector, not {type(out).__name__}")
        if out.field.prime != self.field.prime:
            raise ValueError("out belongs to a different field")
        want = self._n if n is None else n
        if len(out) != want:
            raise ValueError(f"out holds {len(out)} elements, not {want}")
        return out

    def add(self, other, out=None) -> PseudoMersenneVector:
        """`self + other` into `out` when given."""
        return self._binary(other, lib.pmf_vec_add, lib.pmf_vec_add_scalar, out)

    def sub(self, other, out=None) -> PseudoMersenneVector:
        """`self - other` into `out` when given."""
        return self._binary(other, lib.pmf_vec_sub, lib.pmf_vec_sub_scalar, out)

    def mul(self, other, out=None) -> PseudoMersenneVector:
        """`self * other` into `out` when given."""
        return self._binary(other, lib.pmf_vec_mul, lib.pmf_vec_scale, out)

    def rsub(self, other, out=None) -> PseudoMersenneVector:
        """`other - self` for one element `other`, in one pass."""
        element = self._coerce_element(other)
        result = self._destination(out)
        lib.pmf_vec_scalar_sub(result._struct, element._buf, self._struct)
        return result

    def neg(self, out=None) -> PseudoMersenneVector:
        """`-self` into `out` when given."""
        result = self._destination(out)
        lib.pmf_vec_neg(result._struct, self._struct)
        return result

    def fma(self, b, c, out=None) -> PseudoMersenneVector:
        """`self + b * c`, the long way round.

        There is no fused kernel over this field, so the product is formed
        into `out` and the addend folded in after: the right answer, one pass
        more than a fused one, and still free of the intermediate a caller
        would otherwise allocate.

        Except when `out` is `self`, which is the shape a caller accumulating
        in place writes. Forming the product there first would overwrite the
        addend before it is added, so that one case takes a vector for the
        product -- the answer the aliasing contract promises, at the cost the
        fused path does not pay.
        """
        if not isinstance(b, PseudoMersenneVector):
            raise TypeError(f"b must be a FieldVector, not {type(b).__name__}")
        result = self._destination(out)
        if result is self:
            return b.mul(c).add(self, out=result)
        b.mul(c, out=result)
        return result.add(self, out=result)

    def _binary(self, other, vector_kernel, scalar_kernel, out=None):
        """Apply the elementwise kernel, or the broadcast one for an element."""
        result = self._destination(out)
        if isinstance(other, PseudoMersenneVector):
            if other.field.prime != self.field.prime:
                raise ValueError("vectors belong to different fields")
            if len(other) != self._n:
                raise ValueError(f"length mismatch: {self._n} and {len(other)}")
            vector_kernel(result._struct, self._struct, other._struct)
            return result
        element = self._coerce_element(other)
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
        return self.rsub(other)

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

    def scale(self, value, out=None) -> PseudoMersenneVector:
        """Every element multiplied by one field element, into `out` when given."""
        element = self._coerce_element(value)
        result = self._destination(out)
        lib.pmf_vec_scale(result._struct, self._struct, element._buf)
        return result

    def sum(self) -> PseudoMersenneElement:
        """The sum of every element; zero for an empty vector."""
        buf = self.field._new_buffer()
        lib.pmf_vec_sum(buf, self._struct)
        return self.field._wrap(buf)

    def split_even_odd(
        self,
        block: int = 1,
        out: tuple[PseudoMersenneVector, PseudoMersenneVector] | None = None,
    ) -> tuple[PseudoMersenneVector, PseudoMersenneVector]:
        """
        Deinterleave into the even-indexed and odd-indexed halves.

        Requires an even length; each half holds n / 2 elements. Runs of
        `block` elements above 1 are gathered rather than deinterleaved:
        there is no kernel for that layout here, only for the adjacent one.
        """
        if block != 1:
            lo, hi = self._block_split_indices(block)
            return self.query(lo), self.query(hi)
        self._checked_block(block)
        half = self._n // 2
        even, odd = self._destinations(out, half)
        lib.pmf_vec_split_even_odd(even._struct, odd._struct, self._struct)
        return even, odd

    @staticmethod
    def interleave(even, odd, out=None) -> PseudoMersenneVector:
        """Even positions from `even`, odd from `odd`: the inverse of `split_even_odd`."""
        if not isinstance(even, PseudoMersenneVector) or not isinstance(
            odd, PseudoMersenneVector
        ):
            raise TypeError("interleave takes two PseudoMersenneVectors")
        if even.field.prime != odd.field.prime:
            raise ValueError("cannot interleave vectors over different fields")
        if len(even) != len(odd):
            raise ValueError(f"length mismatch: {len(even)} and {len(odd)}")
        result = even._destination(out, 2 * len(even))
        lib.pmf_vec_interleave(result._struct, even._struct, odd._struct)
        return result

    @staticmethod
    def concat(vectors: list, out=None) -> PseudoMersenneVector:
        """One vector holding every element of `vectors`, in order."""
        vectors = list(vectors)
        if not vectors:
            raise ValueError("concat needs at least one vector")
        first = vectors[0]
        for vector in vectors:
            if not isinstance(vector, PseudoMersenneVector):
                raise TypeError(f"cannot concatenate a {type(vector).__name__}")
            if vector.field.prime != first.field.prime:
                raise ValueError("cannot concatenate vectors over different fields")
        result = first._destination(out, sum(len(v) for v in vectors))
        parts = ffi.new("PMFVector[]", [v._struct for v in vectors])
        lib.pmf_vec_concat(result._struct, parts, len(vectors))
        return result

    def query(self, indices) -> PseudoMersenneVector:
        """The elements at `indices`, gathered; the front states the contract."""
        view = index_buffer(indices)
        if view is None:
            positions = [self._checked_index(i) for i in indices]
            result = self._like(len(positions))
            if positions:
                # Already range-checked one at a time, so the kernel's answer
                # cannot be false on this path.
                lib.pmf_vec_gather(
                    result._struct,
                    self._struct,
                    ffi.new("uint64_t[]", positions),
                    len(positions),
                )
            return result

        count = len(view)
        result = self._like(count)
        if count and not lib.pmf_vec_gather(
            result._struct, self._struct, ffi.from_buffer("uint64_t[]", view), count
        ):
            raise IndexError(
                f"index out of range for a vector of {self._n}: "
                f"{next(i for i in view if i >= self._n)}"
            )
        return result

    def fold(self, r, block: int = 1) -> PseudoMersenneVector:
        """``lo + r * (hi - lo)`` per pair, in one kernel pass.

        Only the adjacent layout (`block` 1) has a kernel; a wider block
        gathers the two operands first and then runs the same three
        whole-vector operations the front spells out.
        """
        if block != 1:
            # The front's formula over the gathered operands; it is written
            # against the base type, which this one is.
            return cast("PseudoMersenneVector", super().fold(r, block))
        self._checked_block(block)
        element = self._coerce_element(r)
        result = self._like(self._n // 2)
        lib.pmf_vec_fold(result._struct, self._struct, element._buf)
        return result

    def chunk_length(self, live: int = 4) -> int:
        """The front states the contract."""
        return self._chunk_length(self.field.limbs * 8, live)

    def chain_bytes(self, live: int = 4) -> int:
        """The front states the contract; here an element is its planes."""
        return len(self) * self.field.limbs * 8 * max(live, 1)

    @property
    def padding_unit(self) -> int:
        """The group width; the front says what it governs."""
        return lib.pmf_vec_padded_length(1)

    def view(self, start: int = 0, length: int | None = None) -> PseudoMersenneVector:
        """`length` elements from `start`, over this vector's own planes.

        The padding unit is the group width `pmf_vec_padded_length` rounds
        to; the front states what `start` and `length` must satisfy against
        it. The view holds a reference to this vector, so the planes outlive
        it.
        """
        start, length = self._checked_view(start, length, lib.pmf_vec_padded_length(1))
        result = PseudoMersenneVector.__new__(PseudoMersenneVector)
        result.field = self.field
        result._n = length
        result._allocated_n = lib.pmf_vec_padded_length(length)
        # The planes are the parent's, offset; `_planes` holds it alive rather
        # than owning buffers of its own.
        result._planes = self._planes
        result._plane_ptrs = ffi.new(
            "uint64_t*[]", [plane + start for plane in self._planes]
        )
        result._struct = ffi.new("PMFVector")
        result._struct.limbs = result._plane_ptrs
        result._struct.n = length
        result._struct.allocated_n = result._allocated_n
        result._struct.params = self.field._params
        return result

    def sample_random(self, seed: bytes, start: int = 0) -> None:
        """Fill with uniform elements drawn from `seed`, in place.

        `start` is replayed rather than seeked: this sampler's draw stream is
        walked from the beginning, so a non-zero `start` costs the values
        before it as well (`_sample_random_replayed`).
        """
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(f"start must be an int, not {type(start).__name__}")
        if start < 0:
            raise ValueError(f"start must not be negative, got {start}")
        if start:
            self._sample_random_replayed(seed, start)
            return
        lib.pmf_vec_sample_random(self._struct, seed, len(seed))

    def hash(self) -> bytes:
        """BLAKE3 over every element's canonical encoding, in order, 32 bytes."""
        out = ffi.new("uint8_t[32]")
        lib.pmf_vec_hash(out, self._struct)
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
        count = lib.pmf_vec_hash_count(self._struct, group, stride)
        return self._digests(count, lib.pmf_vec_hash_elements, group, stride)

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
