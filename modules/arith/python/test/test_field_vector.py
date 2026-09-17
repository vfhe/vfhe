# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Vectors of field elements, checked against the same work done one at a time.

Every arithmetic test here is differential: the vector kernel against a Python
loop over `FieldElement`, which is the thing the vector exists to replace. The
lengths are chosen around the padding boundary -- the eltwise kernels process
whole SIMD vectors with no tail, so a length that is not a multiple of the
vector width is the case that catches a wrong `allocated_n`.
"""

from __future__ import annotations

import random
from array import array
from typing import Any, ClassVar

import pytest
from vfhe.arith import (
    ExtensionField,
    ExtensionFieldElement,
    ExtensionFieldVector,
    Field,
    FieldElement,
    FieldVector,
    Spec,
)

PRIME = (1 << 61) - 1
W = 3
SEED = b"test_seed_vector"

#: Around the eltwise kernels' vector width of 8: below it (the scalar path),
#: astride it, and a multiple of it.
LENGTHS = [1, 5, 8, 13, 32]
DEGREES = [1, 2, 4, 8]


def make_field(d: int = 4) -> ExtensionField:
    return ExtensionField(PRIME, d, W)


def digest_list(packed: bytes | memoryview) -> list[bytes]:
    """A packed digest buffer as one bytes per window -- what a caller that
    really wants an object per window writes for itself."""
    return [bytes(packed[i : i + 32]) for i in range(0, len(packed), 32)]


def random_elements(field, n, seed=42):
    rng = random.Random(seed)  # noqa: S311 - test data, not a key
    return [
        FieldElement(field, [rng.randrange(PRIME) for _ in range(field.d)])
        for _ in range(n)
    ]


class TestDispatch:
    """The front resolves through the field's spec, as `Polynomial` does."""

    def test_builds_the_field_vector_type(self):
        vector = FieldVector(make_field(), 4)
        assert type(vector) is ExtensionFieldVector
        assert isinstance(vector, FieldVector)

    def test_a_field_without_vectors_says_so(self):
        """Both implementations have one today, so the refusal needs a stub."""

        class _NoVectors(Field):
            def __init__(self) -> None:
                """A parent with a spec and nothing else: what a vector needs
                to look at before it refuses."""

            spec = Spec(
                implementation="field",
                backend="_test_no_vectors",
                parent_cls=ExtensionField,
            )

        with pytest.raises(TypeError, match="has no vector type"):
            FieldVector(_NoVectors(), 4)

    def test_vector_cls_is_on_the_spec(self):
        assert make_field().spec.vector_cls is ExtensionFieldVector

    def test_a_vector_is_not_hashable(self):
        with pytest.raises(TypeError):
            hash(FieldVector(make_field(), 4))


class TestConstruction:
    """A vector is built from a length or from a sequence of values."""

    @pytest.mark.parametrize("n", [1, 2, 7, 13, 17])
    def test_padding_is_zero_however_the_vector_was_made(self, n):
        """Buffers are allocated undefined, so the padding is established by
        `field_vec_clear_padding` rather than by the allocator. The kernels
        read and write those words, so they have to be reduced -- and nothing
        above the C boundary can see them, which is why this reaches for the
        planes directly."""
        from vfhe.engine import lib

        field = make_field()
        values = random_elements(field, n, seed=160)
        a = FieldVector(field, values)
        b = FieldVector(field, random_elements(field, n, seed=161))

        for vector in (a, a + b, a * b, a.copy(), -a, FieldVector(field, n)):
            allocated = lib.field_vec_padded_length(len(vector))
            for plane in vector._planes:
                assert list(plane[len(vector) : allocated]) == [0] * (
                    allocated - len(vector)
                ), f"padding of a {len(vector)}-element vector"

    def test_a_result_vector_is_still_fully_defined(self):
        """Destinations are allocated unzeroed, so every operation that makes
        one has to write all of it. Checked against the same computation on a
        vector that was zeroed -- the contents must not depend on what the
        allocator happened to hand over."""
        field = make_field()
        a = FieldVector(field, random_elements(field, 13, seed=150))
        b = FieldVector(field, random_elements(field, 13, seed=151))
        for got, want in (
            (a + b, a.add(b, out=FieldVector(field, 13))),
            (a * b, a.mul(b, out=FieldVector(field, 13))),
            (a - b, a.sub(b, out=FieldVector(field, 13))),
            (-a, a.scale(-1)),
            (a.copy(), a),
            (a.query([3, 0, 12, 12]), FieldVector(field, [a[3], a[0], a[12], a[12]])),
        ):
            assert got.to_list() == want.to_list()

        # An even length of its own: the halves only exist for one.
        c = FieldVector(field, random_elements(field, 12, seed=152))
        even, odd = c.split_even_odd()
        assert even.to_list() == [c[i] for i in range(0, 12, 2)]
        assert odd.to_list() == [c[i] for i in range(1, 12, 2)]

    @pytest.mark.parametrize("n", LENGTHS)
    def test_a_length_gives_zeros(self, n):
        field = make_field()
        vector = FieldVector(field, n)
        assert len(vector) == n
        assert all(element == field.zero for element in vector)

    @pytest.mark.parametrize("n", LENGTHS)
    def test_a_sequence_round_trips(self, n):
        field = make_field()
        elements = random_elements(field, n)
        vector = FieldVector(field, elements)
        assert vector.to_list() == elements
        assert list(vector) == elements

    def test_values_may_be_ints_or_coefficient_lists(self):
        field = make_field()
        vector = FieldVector(field, [7, [1, 2, 3, 4], field.one])
        assert vector[0] == FieldElement(field, 7)
        assert vector[1] == FieldElement(field, [1, 2, 3, 4])
        assert vector[2] == field.one

    def test_empty_is_allowed(self):
        field = make_field()
        empty = FieldVector(field, 0)
        assert len(empty) == 0
        assert empty.to_list() == []
        assert empty.sum() == field.zero
        assert len(empty.inverse()) == 0

    def test_a_negative_length_is_rejected(self):
        with pytest.raises(ValueError, match="must not be negative"):
            FieldVector(make_field(), -1)


class TestIndexing:
    """Integer indexing, with `__getitem__` handing back a detached copy."""

    def test_get_and_set(self):
        field = make_field()
        vector = FieldVector(field, 4)
        value = FieldElement(field, [9, 8, 7, 6])
        vector[2] = value
        assert vector[2] == value
        assert vector[1] == field.zero

    def test_negative_indices_count_from_the_end(self):
        field = make_field()
        elements = random_elements(field, 5)
        vector = FieldVector(field, elements)
        assert vector[-1] == elements[4]
        assert vector[-5] == elements[0]

    def test_out_of_range_raises(self):
        vector = FieldVector(make_field(), 3)
        for index in (3, -4):
            with pytest.raises(IndexError):
                vector[index]

    def test_a_read_element_is_detached(self):
        """Writing to what `__getitem__` returned must not reach the vector."""
        field = make_field()
        vector = ExtensionFieldVector(field, [5, 6, 7])
        element = vector[1]
        element.value[0] = 999
        assert vector[1] == FieldElement(field, 6)


class TestArithmetic:
    """Each kernel against the same work as a loop of element operations."""

    @pytest.mark.parametrize("n", LENGTHS)
    @pytest.mark.parametrize("d", DEGREES)
    def test_elementwise_ops(self, n, d):
        field = make_field(d)
        left = random_elements(field, n, seed=1)
        right = random_elements(field, n, seed=2)
        a, b = FieldVector(field, left), FieldVector(field, right)

        assert (a + b).to_list() == [x + y for x, y in zip(left, right, strict=True)]
        assert (a - b).to_list() == [x - y for x, y in zip(left, right, strict=True)]
        assert (a * b).to_list() == [x * y for x, y in zip(left, right, strict=True)]
        assert (-a).to_list() == [-x for x in left]

    @pytest.mark.parametrize("n", LENGTHS)
    @pytest.mark.parametrize("d", DEGREES)
    def test_broadcast_against_one_element(self, n, d):
        field = make_field(d)
        values = random_elements(field, n, seed=3)
        scalar = random_elements(field, 1, seed=4)[0]
        vector = FieldVector(field, values)

        assert (vector + scalar).to_list() == [x + scalar for x in values]
        assert (scalar + vector).to_list() == [x + scalar for x in values]
        assert (vector - scalar).to_list() == [x - scalar for x in values]
        assert (scalar - vector).to_list() == [scalar - x for x in values]
        assert (vector * scalar).to_list() == [x * scalar for x in values]
        assert (scalar * vector).to_list() == [x * scalar for x in values]
        assert vector.scale(scalar).to_list() == [x * scalar for x in values]

    def test_scale_agrees_with_a_constant_vector(self):
        field = make_field()
        values = random_elements(field, 13, seed=5)
        scalar = random_elements(field, 1, seed=6)[0]
        vector = FieldVector(field, values)
        constant = FieldVector(field, [scalar] * 13)
        assert vector.scale(scalar) == vector * constant

    @pytest.mark.parametrize("n", LENGTHS)
    def test_sum(self, n):
        field = make_field()
        values = random_elements(field, n, seed=7)
        total = field.zero
        for value in values:
            total = total + value
        assert FieldVector(field, values).sum() == total

    @pytest.mark.parametrize("n", LENGTHS)
    @pytest.mark.parametrize("d", DEGREES)
    def test_batch_inverse(self, n, d):
        field = make_field(d)
        values = random_elements(field, n, seed=8)
        inverses = FieldVector(field, values).inverse()
        assert inverses.to_list() == [value.inverse() for value in values]

    def test_batch_inverse_rejects_a_zero(self):
        field = make_field()
        values = random_elements(field, 6, seed=9)
        values[3] = field.zero
        with pytest.raises(ValueError, match="not invertible"):
            FieldVector(field, values).inverse()

    #: Around the point `inverse` stops working an element at a time. It runs
    #: its two sweeps as whole-vector multiplies over rows, which needs a row
    #: to be at least a SIMD vector wide with 256 of them -- so under about
    #: 2048 elements it stays serial. `LENGTHS` is entirely on that side of
    #: the line, which is why these exist: 5000 and 8200 also leave a tail of
    #: elements the whole rows do not cover.
    ROW_SPLIT_LENGTHS: ClassVar[list[int]] = [2047, 2048, 2049, 5000, 8200]

    @pytest.mark.parametrize("n", ROW_SPLIT_LENGTHS)
    @pytest.mark.parametrize("d", [1, 4])
    def test_batch_inverse_across_the_row_split(self, n, d):
        field = make_field(d)
        a = FieldVector(field, n)
        a.sample_random(b"row-split")
        inverses = a.inverse()

        assert (a * inverses).to_list() == [field.one] * n
        # And the values themselves at the seams: the ends, the middle, and
        # either side of where the last whole row stops.
        entries, got = a.to_list(), inverses.to_list()
        for i in (0, 1, n // 2, n - 2, n - 1):
            assert got[i] == entries[i].inverse(), f"position {i} of {n}"

    @pytest.mark.parametrize("n", [2048, 2049, 5000])
    @pytest.mark.parametrize("where", [0, 1, "middle", "last"])
    def test_batch_inverse_rejects_a_zero_in_any_chain(self, n, where):
        """The chains are independent, so a zero has to be caught whichever
        one it lands in -- including the leftover the whole rows leave over,
        which is a chain of its own."""
        field = make_field()
        a = FieldVector(field, n)
        a.sample_random(b"row-split")
        a[{"middle": n // 2, "last": n - 1}.get(where, where)] = 0
        with pytest.raises(ValueError, match="not invertible"):
            a.inverse()

    def test_batch_inverse_result_may_be_its_input(self):
        """The C entry allows `out == a`. The row sweep has to read a row
        before writing over it, which the element-at-a-time form got for free
        and this one does not."""
        from vfhe.engine import lib

        field = make_field()
        a = FieldVector(field, 5000)
        a.sample_random(b"in-place")
        want = a.inverse().to_list()
        b = a.copy()
        assert lib.field_vec_inv(b._struct, b._struct) == 1
        assert b.to_list() == want

    def test_an_operand_may_be_the_result_of_another(self):
        """Chained expressions: no operand is written through."""
        field = make_field()
        values = random_elements(field, 9, seed=10)
        a = FieldVector(field, values)
        b = a + a
        assert a.to_list() == values
        assert b.to_list() == [x + x for x in values]

    def test_length_and_field_mismatches_are_rejected(self):
        field = make_field()
        with pytest.raises(ValueError, match="length mismatch"):
            _ = FieldVector(field, 4) + FieldVector(field, 5)
        with pytest.raises(ValueError, match="different fields"):
            _ = FieldVector(field, 4) + FieldVector(make_field(), 4)


class TestFallbackTier:
    """The operations the front derives once for any implementation."""

    @pytest.mark.parametrize("exponent", [0, 1, 2, 5, 17])
    def test_pow_matches_element_pow(self, exponent):
        field = make_field()
        values = random_elements(field, 11, seed=11)
        result = FieldVector(field, values) ** exponent
        assert result.to_list() == [value**exponent for value in values]

    def test_pow_rejects_a_negative_exponent(self):
        with pytest.raises(ValueError, match="negative exponent"):
            _ = FieldVector(make_field(), 4) ** -1

    def test_concat(self):
        field = make_field()
        first = random_elements(field, 5, seed=12)
        second = random_elements(field, 3, seed=13)
        joined = FieldVector.concat(
            [FieldVector(field, first), FieldVector(field, second)]
        )
        assert joined.to_list() == first + second

    def test_query_gathers(self):
        field = make_field()
        values = random_elements(field, 8, seed=14)
        gathered = FieldVector(field, values).query([5, 0, 5, 2])
        assert gathered.to_list() == [values[i] for i in (5, 0, 5, 2)]


class TestMovement:
    """Copy, split and equality, which read exactly n elements."""

    def test_copy_is_independent(self):
        field = make_field()
        values = random_elements(field, 7, seed=15)
        original = FieldVector(field, values)
        duplicate = original.copy()
        duplicate[0] = field.one
        assert original[0] == values[0]
        assert duplicate[0] == field.one

    @pytest.mark.parametrize("n", [2, 8, 14])
    def test_split_even_odd(self, n):
        field = make_field()
        values = random_elements(field, n, seed=16)
        even, odd = FieldVector(field, values).split_even_odd()
        assert even.to_list() == values[0::2]
        assert odd.to_list() == values[1::2]

    def test_split_needs_an_even_length(self):
        with pytest.raises(ValueError, match="odd"):
            FieldVector(make_field(), 5).split_even_odd()

    def test_equality(self):
        field = make_field()
        values = random_elements(field, 6, seed=17)
        assert FieldVector(field, values) == FieldVector(field, values)
        assert FieldVector(field, values) != FieldVector(field, values[:5])
        other = list(values)
        other[2] = other[2] + field.one
        assert FieldVector(field, values) != FieldVector(field, other)

    def test_equality_ignores_the_padding(self):
        """A short vector's padding is scratch; only the n elements count."""
        field = make_field()
        values = random_elements(field, 3, seed=18)
        left = FieldVector(field, values)
        right = FieldVector(field, values)
        # Dirty one padding word behind the API's back.
        right._planes[0][5] = 12345
        assert left == right


class TestPadding:
    """The invariant the tuned kernels depend on, checked directly."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_planes_are_padded_and_start_zero(self, n):
        field = make_field()
        vector = FieldVector(field, random_elements(field, n, seed=19))
        assert vector._allocated_n % 8 == 0
        assert vector._allocated_n >= n
        for plane in vector._planes:
            assert all(plane[i] == 0 for i in range(n, vector._allocated_n))

    def test_padding_stays_reduced_through_arithmetic(self):
        """Kernels touch the padding; it must never leave [0, q)."""
        field = make_field()
        a = FieldVector(field, random_elements(field, 13, seed=20))
        b = FieldVector(field, random_elements(field, 13, seed=21))
        for result in (a + b, a - b, a * b, -a, a.scale(field.two)):
            for plane in result._planes:
                assert all(plane[i] < PRIME for i in range(result._allocated_n))

    def test_results_are_unaffected_by_dirty_padding(self):
        field = make_field()
        values = random_elements(field, 5, seed=22)
        clean = FieldVector(field, values)
        dirty = FieldVector(field, values)
        for plane in dirty._planes:
            for i in range(5, dirty._allocated_n):
                plane[i] = PRIME - 1
        assert (dirty * dirty).to_list() == (clean * clean).to_list()
        assert dirty.sum() == clean.sum()
        assert dirty.hash() == clean.hash()


class TestSamplingAndHashing:
    """Sampling is a pure function of the seed; digests cover exactly n."""

    def test_sampling_is_uniform_across_the_whole_vector(self):
        """One draw stream, so no coefficient repeats by construction."""
        field = make_field()
        vector = ExtensionFieldVector(field, 16)
        vector.sample_random(SEED)
        coefficients = [element.value[j] for element in vector for j in range(field.d)]
        assert len(set(coefficients)) == len(coefficients)
        assert all(0 <= c < PRIME for c in coefficients)

    def test_sampling_is_a_pure_function_of_the_seed(self):
        field = make_field()
        first, second = FieldVector(field, 8), FieldVector(field, 8)
        first.sample_random(SEED)
        second.sample_random(SEED)
        assert first == second
        third = FieldVector(field, 8)
        third.sample_random(SEED + b"other")
        assert first != third

    def test_hash_is_stable_and_content_dependent(self):
        field = make_field()
        values = random_elements(field, 9, seed=23)
        vector = FieldVector(field, values)
        assert len(vector.hash()) == 32
        assert vector.hash() == FieldVector(field, values).hash()
        assert vector.hash() != FieldVector(field, values[:8]).hash()

    def test_digest_elements_is_the_vector_digest(self):
        """`field.digest_elements` skips the vector; it must not skip any of
        the bytes, so it has to equal both the vector's own digest and the
        matching window of `hash_elements`."""
        field = make_field()
        values = random_elements(field, 8, seed=140)
        vector = FieldVector(field, values)

        assert field.digest_elements(values) == vector.hash()
        assert (
            field.digest_elements(values[2:4]) == FieldVector(field, values[2:4]).hash()
        )
        assert (
            field.digest_elements(values[2:4])
            == digest_list(vector.hash_elements(group=2, stride=2))[1]
        )
        assert (
            field.digest_elements([values[3]]) == digest_list(vector.hash_elements())[3]
        )

        # Order binds: the same two elements the other way round differ.
        assert field.digest_elements(values[2:4]) != field.digest_elements(
            values[3:1:-1]
        )

    def test_digest_elements_rejects_a_foreign_element(self):
        field = make_field()
        other = make_field(2)
        with pytest.raises(ValueError, match="different field"):
            field.digest_elements(random_elements(other, 2, seed=141))

    def test_hash_elements_windows(self):
        field = make_field()
        values = random_elements(field, 8, seed=24)
        vector = FieldVector(field, values)

        singles = digest_list(vector.hash_elements())
        assert len(singles) == 8
        assert singles[3] == FieldVector(field, [values[3]]).hash()

        pairs = digest_list(vector.hash_elements(group=2, stride=2))
        assert len(pairs) == 4
        assert pairs[1] == FieldVector(field, values[2:4]).hash()

        sliding = digest_list(vector.hash_elements(group=3, stride=1))
        assert len(sliding) == 6
        assert sliding[2] == FieldVector(field, values[2:5]).hash()

    def test_hash_elements_drops_a_partial_window(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 7, seed=25))
        assert len(vector.hash_elements(group=2, stride=2)) == 3 * 32
        assert vector.hash_elements(group=9, stride=1) == b""

    def test_hash_elements_rejects_a_zero_step(self):
        vector = FieldVector(make_field(), 4)
        with pytest.raises(ValueError, match="must be positive"):
            vector.hash_elements(group=0)


def test_a_twisted_pair_fold_is_expressible_in_vector_operations():
    """A twisted pair fold, written over whole vectors.

    ``folded[i] = hi + coeff * twist + r * coeff`` with
    ``coeff = (lo - hi) * twist2_inv``, per position. Written once over
    vectors and once as the loop it replaces, so the API is shown to cover
    the case before any consumer is moved onto it.
    """
    field = make_field()
    n = 16
    word = random_elements(field, n, seed=26)
    twists = random_elements(field, n // 2, seed=27)
    twists2_inv = random_elements(field, n // 2, seed=28)
    r = random_elements(field, 1, seed=29)[0]

    expected = []
    for i in range(n // 2):
        lo, hi = word[2 * i], word[2 * i + 1]
        coeff = (lo - hi) * twists2_inv[i]
        expected.append(hi + coeff * twists[i] + r * coeff)

    lo_vec, hi_vec = FieldVector(field, word).split_even_odd()
    coeff_vec = (lo_vec - hi_vec) * FieldVector(field, twists2_inv)
    folded = hi_vec + coeff_vec * FieldVector(field, twists) + coeff_vec.scale(r)

    assert folded.to_list() == expected


class TestAliasingContract:
    """The header promises arithmetic outputs may alias their inputs.

    Nothing in the Python API aliases -- every operation allocates its result
    -- but the C entry points are the currency other modules' kernels are
    written against, so the promise is exercised here at that level.
    """

    @staticmethod
    def _call(kernel, *args):
        from vfhe.engine import lib

        getattr(lib, kernel)(*args)

    @pytest.mark.parametrize(
        "kernel", ["field_vec_add", "field_vec_sub", "field_vec_mul"]
    )
    def test_output_may_be_an_input(self, kernel):
        field = make_field()
        left = random_elements(field, 13, seed=30)
        right = random_elements(field, 13, seed=31)
        a, b = FieldVector(field, left), FieldVector(field, right)
        expected = getattr(
            a,
            {"field_vec_add": "__add__", "field_vec_sub": "__sub__"}.get(
                kernel, "__mul__"
            ),
        )(b)

        self._call(kernel, a._struct, a._struct, b._struct)
        assert a.to_list() == expected.to_list()
        assert b.to_list() == right  # the other operand is untouched

    def test_inverse_output_may_be_its_input(self):
        field = make_field()
        values = random_elements(field, 9, seed=32)
        vector = FieldVector(field, values)
        self._call("field_vec_inv", vector._struct, vector._struct)
        assert vector.to_list() == [value.inverse() for value in values]


def test_padded_length_is_the_single_source_of_the_plane_size():
    from vfhe.engine import lib

    for n in (0, 1, 7, 8, 9, 16, 13):
        padded = lib.field_vec_padded_length(n)
        assert padded >= n and padded % 8 == 0 and padded - n < 8
        assert FieldVector(make_field(), n)._allocated_n == padded


class TestNativeMovementAndFold:
    """Interleave, concat, query and the fused fold, against the generic forms.

    The generic forms are the base-class defaults written over `split_even_odd`
    and element access; the native kernels must agree with them at every
    length around the padding boundary, including the odd multiples of the
    pair width where a half-length rounds up past the input's allocation.
    """

    rng = random.Random(0x5EED)  # noqa: S311 - test data, not a key

    def values(self, field, n):
        return [[self.rng.randrange(PRIME) for _ in range(field.d)] for _ in range(n)]

    @pytest.mark.parametrize("n", [2, 8, 18, 26])
    def test_interleave_inverts_split(self, n):
        field = make_field()
        vector = FieldVector(field, self.values(field, n))
        even, odd = vector.split_even_odd()
        assert FieldVector.interleave(even, odd) == vector
        assert ExtensionFieldVector.interleave(even, odd) == vector

    @pytest.mark.parametrize("n", [1, 8, 18, 26, 200])
    @pytest.mark.parametrize("scalars", [(False, False), (True, False), (True, True)])
    def test_fma_interleave_matches_two_fmas_and_an_interleave(self, n, scalars):
        field = make_field()
        a = FieldVector(field, self.values(field, n))
        b = FieldVector(field, self.values(field, n))

        def multiplier(as_scalar):
            if as_scalar:
                return FieldElement(field, self.values(field, 1)[0])
            return FieldVector(field, self.values(field, n))

        c_even, c_odd = multiplier(scalars[0]), multiplier(scalars[1])
        fused = FieldVector.fma_interleave(a, b, c_even, c_odd)
        assert len(fused) == 2 * n
        assert fused == FieldVector.interleave(a.fma(b, c_even), a.fma(b, c_odd))

    def test_fma_interleave_takes_a_destination_and_checks_it(self):
        field = make_field()
        a = FieldVector(field, self.values(field, 8))
        b = FieldVector(field, self.values(field, 8))
        out = FieldVector(field, 16)
        assert FieldVector.fma_interleave(a, b, b, b, out=out) is out
        with pytest.raises(ValueError, match="not 16"):
            FieldVector.fma_interleave(a, b, b, b, out=FieldVector(field, 8))
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector.fma_interleave(a, FieldVector(field, 5), b, b)

    @pytest.mark.parametrize("n", [1, 8, 18, 26, 200])
    def test_fold_twisted_matches_the_pairwise_formula(self, n):
        field = make_field()
        word = FieldVector(field, self.values(field, 2 * n))
        twist2_inv = FieldVector(field, self.values(field, n))
        twist = FieldVector(field, self.values(field, n))
        r = FieldElement(field, self.values(field, 1)[0])

        folded = word.fold_twisted(twist2_inv, twist, r)

        assert len(folded) == n
        for i in range(n):
            coeff = (word[2 * i] - word[2 * i + 1]) * twist2_inv[i]
            assert folded[i] == word[2 * i + 1] + coeff * twist[i] + coeff * r

    # More elements than the kernel holds in one window, so the loop that
    # walks them runs several times and the last one is short.
    @pytest.mark.parametrize("n", [9000, 20001])
    def test_the_fused_pair_crosses_the_kernels_window(self, n):
        field = make_field()
        a = FieldVector(field, n)
        b = FieldVector(field, n)
        c_even = FieldVector(field, n)
        c_odd = FieldVector(field, n)
        for vector, seed in ((a, b"a"), (b, b"b"), (c_even, b"e"), (c_odd, b"o")):
            vector.sample_random(seed)

        assert FieldVector.fma_interleave(
            a, b, c_even, c_odd
        ) == FieldVector.interleave(a.fma(b, c_even), a.fma(b, c_odd))

        word = FieldVector(field, 2 * n)
        word.sample_random(b"w")
        r = field.random_element(b"r")
        lo, hi = word.split_even_odd()
        lo.sub(hi, out=lo)
        lo.mul(c_even, out=lo)
        want = hi.fma(lo, c_odd)
        assert word.fold_twisted(c_even, c_odd, r) == want.fma(lo, r, out=want)

    @pytest.mark.parametrize("n", [1, 8, 18, 26, 200, 9000])
    def test_lift_twisted_is_fma_interleave_with_the_table_and_its_negation(self, n):
        field = make_field()
        a = FieldVector(field, self.values(field, n))
        b = FieldVector(field, self.values(field, n))
        twist = FieldVector(field, self.values(field, n))
        lifted = FieldVector.lift_twisted(a, b, twist)
        assert len(lifted) == 2 * n
        assert lifted == FieldVector.fma_interleave(a, b, twist, -twist)
        out = FieldVector(field, 2 * n)
        assert FieldVector.lift_twisted(a, b, twist, out=out) is out and out == lifted
        assert ExtensionFieldVector.lift_twisted(a, b, twist) == lifted

    @pytest.mark.parametrize("period", [4, 8, 64, 512, 4096, 16384])
    def test_lift_twisted_reads_a_short_table_cyclically(self, period):
        """A table of `period` positions is read `n / period` times over --
        including one shorter than the kernel's vector width, which the front
        tiles itself, and ones longer than its window."""
        field = make_field()
        n = 16384
        a, b = FieldVector(field, n), FieldVector(field, n)
        twist = FieldVector(field, period)
        for vector, seed in ((a, b"a"), (b, b"b"), (twist, b"t")):
            vector.sample_random(seed)
        tiled = FieldVector.concat([twist] * (n // period))
        assert FieldVector.lift_twisted(a, b, twist) == FieldVector.fma_interleave(
            a, b, tiled, -tiled
        )
        with pytest.raises(ValueError, match="does not divide"):
            FieldVector.lift_twisted(a, b, FieldVector(field, 24))

    @pytest.mark.parametrize("n", [8, 26, 9000])
    def test_prime_subfield_tables_multiply_plane_wise(self, n):
        """A table over `F_p` (the degree-1 field with the same prime) is
        accepted by `mul`, `fma`, `lift_twisted` and `fold_twisted`, and gives
        exactly what the same table lifted to the extension gives."""
        field = make_field()
        subfield = ExtensionField(PRIME, 1)
        ints = [value[0] or 1 for value in self.values(field, n)]
        table = FieldVector(subfield, ints)
        lifted = FieldVector(field, ints)
        a = FieldVector(field, self.values(field, n))
        b = FieldVector(field, self.values(field, n))
        assert a.mul(table) == a.mul(lifted) and a * table == a * lifted
        assert a.fma(b, table) == a.fma(b, lifted)
        out = FieldVector(field, n)
        assert a.fma(b, table, out=out) is out and out == a.fma(b, lifted)
        assert FieldVector.lift_twisted(a, b, table) == FieldVector.lift_twisted(
            a, b, lifted
        )
        word = FieldVector(field, self.values(field, 2 * n))
        r = FieldElement(field, self.values(field, 1)[0])
        inverse = table.scale(2).inverse()
        assert word.fold_twisted(inverse, table, r) == word.fold_twisted(
            FieldVector(field, [int(e) for e in inverse]), lifted, r
        )
        # The lift folds back with the same tables.
        assert (
            FieldVector.lift_twisted(a, b, table).fold_twisted(
                inverse, table, field.one
            )
            == a + b
        )
        other = FieldVector(ExtensionField(562949953421201, 1), [1] * n)
        with pytest.raises(ValueError, match="different field"):
            a.mul(other)
        with pytest.raises(ValueError, match="different field"):
            word.fold_twisted(other, other, r)

    def test_a_short_prime_subfield_table_is_tiled_and_lifted(self):
        field = make_field()
        subfield = ExtensionField(PRIME, 1)
        n, period = 64, 4
        a, b = FieldVector(field, n), FieldVector(field, n)
        a.sample_random(b"a")
        b.sample_random(b"b")
        table = FieldVector(subfield, [3, 5, 7, 11])
        tiled = FieldVector(field, [3, 5, 7, 11] * (n // period))
        assert FieldVector.lift_twisted(a, b, table) == FieldVector.fma_interleave(
            a, b, tiled, -tiled
        )

    def test_fold_twisted_checks_the_table_lengths(self):
        field = make_field()
        word = FieldVector(field, self.values(field, 16))
        good = FieldVector(field, self.values(field, 8))
        with pytest.raises(ValueError, match="not 8"):
            word.fold_twisted(FieldVector(field, 4), good, 1)
        with pytest.raises(ValueError, match="not 8"):
            word.fold_twisted(good, FieldVector(field, 9), 1)

    def test_interleave_rejects_mismatches(self):
        field = make_field()
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector.interleave(FieldVector(field, 4), FieldVector(field, 5))
        with pytest.raises(TypeError):
            FieldVector.interleave(FieldVector(field, 4), [0] * 4)

    def test_concat_matches_the_element_lists(self):
        field = make_field()
        parts = [self.values(field, n) for n in (5, 0, 8, 3)]
        vectors = [FieldVector(field, part) for part in parts]
        joined = FieldVector.concat(vectors)
        assert len(joined) == 16
        assert joined.to_list() == [e for v in vectors for e in v.to_list()]
        assert FieldVector.concat(vectors[:1]) == vectors[0]
        with pytest.raises(ValueError, match="at least one"):
            FieldVector.concat([])
        with pytest.raises(ValueError, match="different fields"):
            FieldVector.concat([vectors[0], FieldVector(make_field(2), 3)])

    def test_query_gathers_with_negative_indices(self):
        field = make_field()
        values = self.values(field, 9)
        vector = FieldVector(field, values)
        gathered = vector.query([8, 0, -1, 3, 3])
        elements = vector.to_list()
        assert gathered.to_list() == [elements[i] for i in (8, 0, 8, 3, 3)]
        assert len(vector.query([])) == 0
        with pytest.raises(IndexError):
            vector.query([9])

    def test_query_takes_a_prebuilt_index_buffer(self):
        """The fast path: an unsigned 64-bit buffer goes to the kernel as it
        stands. Same elements as the sequence form, which is the whole
        contract -- what changes is only that nothing is checked per index."""
        field = make_field()
        vector = FieldVector(field, self.values(field, 9))
        positions = [8, 0, 8, 3, 3]
        assert vector.query(array("Q", positions)) == vector.query(positions)
        assert len(vector.query(array("Q", []))) == 0

        # A memoryview of the same bytes is the same buffer.
        buffer = array("Q", positions)
        assert vector.query(memoryview(buffer)) == vector.query(positions)

    def test_a_buffer_index_out_of_range_is_still_an_IndexError(self):
        """The bound moved into the kernel; the error it reports must not."""
        field = make_field()
        vector = FieldVector(field, self.values(field, 9))
        with pytest.raises(IndexError, match="out of range"):
            vector.query(array("Q", [0, 9]))

    def test_a_signed_buffer_is_read_as_a_sequence(self):
        """Signed items are deliberately not the fast path: -1 has to keep
        meaning the last element rather than becoming a huge index."""
        field = make_field()
        vector = FieldVector(field, self.values(field, 9))
        assert vector.query(array("q", [8, 0, -1, 3])) == vector.query([8, 0, -1, 3])

    @pytest.mark.parametrize("n", [2, 8, 18, 26, 50])
    def test_fold_matches_the_split_formula(self, n):
        field = make_field()
        vector = FieldVector(field, self.values(field, n))
        r = FieldElement(field, self.values(field, 1)[0])
        even, odd = vector.split_even_odd()
        expected = even + (odd - even).scale(r)
        folded = vector.fold(r)
        assert len(folded) == n // 2
        assert folded == expected
        assert folded.to_list() == [
            e + r * (o - e) for e, o in zip(even.to_list(), odd.to_list(), strict=True)
        ]
        # ...and against the generic default, which is the formula itself.
        assert FieldVector.fold(vector, r) == expected

    def test_fold_accepts_an_int_and_needs_an_even_length(self):
        field = make_field()
        vector = FieldVector(field, self.values(field, 8))
        assert vector.fold(5) == vector.fold(FieldElement(field, 5))
        with pytest.raises(ValueError, match="odd"):
            FieldVector(field, 5).fold(3)

    def test_fold_leaves_the_padding_reduced(self):
        field = make_field()
        folded = FieldVector(field, self.values(field, 18)).fold(7)
        for plane in folded._planes:
            assert all(
                plane[i] < PRIME for i in range(len(folded), folded._allocated_n)
            )


class TestBlockSplitAndFold:
    """The strided layouts: `block` moves the pair partner further away.

    The reference is the index arithmetic itself -- runs of `block` elements
    alternating between the two halves -- so a kernel that reads the wrong
    stride disagrees with a list comprehension rather than with itself.
    """

    @staticmethod
    def blocks(values, block):
        """The two operand lists `split_even_odd(block)` produces."""
        n = len(values)
        lo = [values[at + i] for at in range(0, n, 2 * block) for i in range(block)]
        hi = [
            values[at + block + i]
            for at in range(0, n, 2 * block)
            for i in range(block)
        ]
        return lo, hi

    @pytest.mark.parametrize("n", [2, 8, 16, 64])
    def test_split_matches_the_index_arithmetic(self, n):
        field = make_field()
        values = random_elements(field, n, seed=30)
        vector = FieldVector(field, values)
        block = 1
        while block <= n // 2:
            lo, hi = vector.split_even_odd(block)
            assert (lo.to_list(), hi.to_list()) == self.blocks(values, block)
            block *= 2

    @pytest.mark.parametrize("n", [2, 8, 16, 64])
    def test_fold_matches_the_split_formula_at_every_block(self, n):
        field = make_field()
        values = random_elements(field, n, seed=31)
        vector = FieldVector(field, values)
        r = random_elements(field, 1, seed=32)[0]
        block = 1
        while block <= n // 2:
            lo, hi = self.blocks(values, block)
            folded = vector.fold(r, block)
            assert len(folded) == n // 2
            assert folded.to_list() == [
                left + r * (right - left) for left, right in zip(lo, hi, strict=True)
            ]
            block *= 2

    def test_the_contiguous_halves_case_does_not_disturb_its_input(self):
        """`block == n // 2` reads the two halves where they lie; the source
        must come back unchanged."""
        field = make_field()
        values = random_elements(field, 32, seed=33)
        vector = FieldVector(field, values)
        vector.fold(random_elements(field, 1, seed=34)[0], 16)
        assert vector.to_list() == values

    def test_fold_leaves_the_padding_reduced(self):
        field = make_field()
        folded = FieldVector(field, random_elements(field, 20, seed=35)).fold(7, 2)
        for plane in folded._planes:
            assert all(
                plane[i] < PRIME for i in range(len(folded), folded._allocated_n)
            )

    @pytest.mark.parametrize("block", [0, -1, 3, 6, 16])
    def test_a_block_that_does_not_divide_the_half_length_is_rejected(self, block):
        vector = FieldVector(make_field(), 16)
        with pytest.raises(ValueError, match="block"):
            vector.split_even_odd(block)
        with pytest.raises(ValueError, match="block"):
            vector.fold(3, block)


class TestPaddingUnit:
    """`padding_unit` is the number `view` states its rule against, so the
    two have to agree: a view on a multiple of it is accepted and one off it
    is not."""

    def test_it_is_what_view_accepts(self):
        field = make_field()
        vector = FieldVector(field, 64)
        unit = vector.padding_unit
        assert unit >= 1 and 64 % unit == 0

        assert len(vector.view(unit, unit)) == unit
        if unit > 1:
            with pytest.raises(ValueError, match="multiple"):
                vector.view(1, unit)
            with pytest.raises(ValueError, match="multiple"):
                vector.view(unit, unit + 1)

    def test_a_view_may_run_to_the_end(self):
        """The one length that need not be a multiple: the parent's own
        padding covers the rounding."""
        field = make_field()
        vector = FieldVector(field, 20)
        unit = vector.padding_unit
        assert len(vector.view(unit)) == 20 - unit


class TestView:
    """A view shares the parent's planes, so the two see each other's writes."""

    def test_a_view_reads_the_parent(self):
        field = make_field()
        values = random_elements(field, 64, seed=36)
        vector = FieldVector(field, values)
        assert vector.view(8, 16).to_list() == values[8:24]
        assert vector.view().to_list() == values
        assert vector.view(56).to_list() == values[56:]

    def test_a_write_through_a_view_reaches_the_parent(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 32, seed=37))
        window = vector.view(8, 8)
        window[0] = field.one
        assert vector[8] == field.one
        # ...and arithmetic on a view is arithmetic on those elements.
        assert (window * window).to_list() == [e * e for e in vector.to_list()[8:16]]

    def test_a_view_of_a_view_starts_where_the_view_does(self):
        field = make_field()
        values = random_elements(field, 64, seed=39)
        vector = FieldVector(field, values)
        nested = vector.view(32, 32).view(8, 8)
        assert nested.to_list() == values[40:48]
        nested[0] = field.one
        assert vector[40] == field.one

    def test_a_view_keeps_the_parent_alive(self):
        field = make_field()
        values = random_elements(field, 16, seed=38)
        window = FieldVector(field, values).view(8, 8)
        assert window.to_list() == values[8:16]

    @pytest.mark.parametrize(
        ("start", "length"), [(1, 8), (8, 7), (0, 65), (-8, 8), (60, 8)]
    )
    def test_the_alignment_and_padding_contract_is_enforced(self, start, length):
        """Only a multiple of the padding unit may start a view, and only the
        tail of the parent may end on a partial one."""
        vector = FieldVector(make_field(), 64)
        with pytest.raises(ValueError):
            vector.view(start, length)

    def test_a_partial_length_is_allowed_at_the_end(self):
        field = make_field()
        values = random_elements(field, 20, seed=39)
        assert FieldVector(field, values).view(8).to_list() == values[8:]


class TestFrobenius:
    """The coefficient map against the exponentiation that defines it."""

    @pytest.mark.parametrize("d", [1, 2, 4, 8])
    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_matches_repeated_exponentiation(self, d, k):
        field = make_field(d)
        vector = FieldVector(field, random_elements(field, 13, seed=40))
        expected = []
        for element in vector.to_list():
            raised = element
            for _ in range(k % d):
                raised = raised**PRIME
            expected.append(raised)
        assert vector.frobenius(k).to_list() == expected

    def test_agrees_with_the_element_and_with_the_generic_default(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 9, seed=41))
        for k in (1, 2, 3):
            assert vector.frobenius(k).to_list() == [
                e.frobenius(k) for e in vector.to_list()
            ]
            assert vector.frobenius(k) == FieldVector.frobenius(vector, k)

    def test_is_a_field_automorphism(self):
        field = make_field()
        a = FieldVector(field, random_elements(field, 8, seed=42))
        b = FieldVector(field, random_elements(field, 8, seed=43))
        assert (a * b).frobenius() == a.frobenius() * b.frobenius()
        assert (a + b).frobenius() == a.frobenius() + b.frobenius()
        assert a.frobenius(field.d) == a  # the group is cyclic of order d


class TestIndexedSampling:
    """One sequence, reachable at any position without the ones before it."""

    def test_a_window_is_the_slice_of_the_whole_fill(self):
        field = make_field()
        whole = FieldVector(field, 64)
        whole.sample_random(SEED)
        window = FieldVector(field, 8)
        window.sample_random(SEED, 40)
        assert window.to_list() == whole.to_list()[40:48]

    def test_the_fill_does_not_depend_on_the_vector_length(self):
        field = make_field()
        short, long = FieldVector(field, 8), FieldVector(field, 64)
        short.sample_random(SEED)
        long.sample_random(SEED)
        assert long.to_list()[:8] == short.to_list()

    def test_element_from_seed_is_the_same_sequence(self):
        field = make_field()
        vector = FieldVector(field, 32)
        vector.sample_random(SEED)
        elements = vector.to_list()
        for index in (0, 1, 17, 31):
            assert field.element_from_seed(SEED, index) == elements[index]

    def test_element_from_seed_reaches_past_any_vector_ever_built(self):
        """The point of the indexed form: a position no fill materialized."""
        field = make_field()
        far = field.element_from_seed(SEED, 1 << 40)
        window = FieldVector(field, 1)
        window.sample_random(SEED, 1 << 40)
        assert window[0] == far

    #: A prime sitting just above a power of two, so the rejection mask throws
    #: away about half of every draw and positions keep falling through to
    #: later attempt streams. `make_field`'s Mersenne prime almost never
    #: rejects, so it never reaches that part of the sampler at all.
    REJECTING_PRIME, REJECTING_W = 1099511627791, 3

    def rejecting_field(self):
        return ExtensionField(self.REJECTING_PRIME, 2, self.REJECTING_W)

    @pytest.mark.parametrize("index", [0, 1, 2047, 2048, 2049, 4095, 4096, 4097, 9000])
    def test_the_fill_agrees_with_the_indexed_form_across_a_bulk_boundary(self, index):
        """A long fill is drawn a run at a time, so the run boundaries are
        where a batched sampler stops agreeing with the definition: a value
        must depend on its index and on nothing about how the fill was cut up.
        Checked on a prime that rejects about half of every draw, so positions
        really do fall through to later streams."""
        field = self.rejecting_field()
        vector = FieldVector(field, 9001)
        vector.sample_random(SEED)
        assert vector[index] == field.element_from_seed(SEED, index)

    def test_a_window_is_the_slice_of_the_long_fill_it_names(self):
        """`start` moves the window, not the sequence -- including when the
        window and the fill are cut into runs differently."""
        field = self.rejecting_field()
        whole = FieldVector(field, 9001)
        whole.sample_random(SEED)
        entries = whole.to_list()
        # The last runs to the end of the fill: 9001 - 5000.
        for start, length in ((0, 10), (4090, 20), (4096, 4096), (5000, 4001)):
            window = FieldVector(field, length)
            window.sample_random(SEED, start)
            assert window.to_list() == entries[start : start + length], (
                f"window {start}..{start + length}"
            )

    def test_sample_random_at_reads_the_fill_at_those_positions(self):
        """Scattered, repeated and unsorted positions of the same sequence a
        fill holds -- so a whole fill is the reference for all of them."""
        field = make_field()
        whole = FieldVector(field, 300)
        whole.sample_random(SEED)
        entries = whole.to_list()

        positions = [7, 0, 299, 7, 42, 1, 299]
        got = FieldVector(field, len(positions))
        got.sample_random_at(SEED, positions)
        assert got.to_list() == [entries[i] for i in positions]
        assert got.to_list() == [field.element_from_seed(SEED, i) for i in positions]

    def test_sample_random_at_merges_consecutive_positions(self):
        """Consecutive indices are drawn as one run rather than one at a
        time. A different code path, so it owes the same values -- checked
        against the window `sample_random(start)` fills and against the
        scattered path reading the same positions out of order."""
        field = make_field()
        run = list(range(500, 700))
        got = FieldVector(field, len(run))
        got.sample_random_at(SEED, run)

        window = FieldVector(field, len(run))
        window.sample_random(SEED, 500)
        assert got.to_list() == window.to_list()

        shuffled = run[::-1]
        other = FieldVector(field, len(shuffled))
        other.sample_random_at(SEED, shuffled)
        assert other.to_list() == got.to_list()[::-1]

        # A run broken in the middle exercises both halves of the merge.
        split = [*range(10, 20), 900, *range(20, 30)]
        broken = FieldVector(field, len(split))
        broken.sample_random_at(SEED, split)
        assert broken.to_list() == [field.element_from_seed(SEED, i) for i in split]

    def test_sample_random_at_takes_an_index_buffer(self):
        field = make_field()
        positions = [9, 3, 3, 77]
        sequence = FieldVector(field, len(positions))
        sequence.sample_random_at(SEED, positions)
        buffered = FieldVector(field, len(positions))
        buffered.sample_random_at(SEED, array("Q", positions))
        assert buffered.to_list() == sequence.to_list()

    def test_sample_random_at_reaches_past_any_vector_ever_built(self):
        field = make_field()
        far = FieldVector(field, 2)
        far.sample_random_at(SEED, [1 << 40, (1 << 40) + 1])
        window = FieldVector(field, 2)
        window.sample_random(SEED, 1 << 40)
        assert far.to_list() == window.to_list()

    def test_sample_random_at_checks_its_arguments(self):
        field = make_field()
        with pytest.raises(ValueError, match="expected 3 indices"):
            FieldVector(field, 3).sample_random_at(SEED, [1, 2])
        with pytest.raises(ValueError, match="expected 2 indices"):
            FieldVector(field, 2).sample_random_at(SEED, array("Q", [1, 2, 3]))
        with pytest.raises(ValueError, match="negative"):
            FieldVector(field, 2).sample_random_at(SEED, [1, -1])

    def test_the_element_sampler_is_a_separate_stream(self):
        """`random_element` is domain-separated from the vector fill, so one
        seed feeding both gives independent values."""
        field = make_field()
        vector = FieldVector(field, 4)
        vector.sample_random(SEED)
        assert field.random_element(SEED) != vector[0]

    def test_a_negative_start_or_index_is_rejected(self):
        field = make_field()
        with pytest.raises(ValueError, match="negative"):
            FieldVector(field, 4).sample_random(SEED, -1)
        with pytest.raises(ValueError, match="negative"):
            field.element_from_seed(SEED, -1)


class TestFusedProduct:
    """The extension product where the fused kernel takes over.

    `field_fused_applies` gates on the modulus family, so **the prime below is
    what selects the path**: under 2^50 it takes the 2^52 Shoup family, which
    is the only one with a widening multiply that accumulates. Every other test
    in this file uses a 61-bit prime and therefore exercises the schoolbook
    passes instead -- these two primes are how both paths stay covered, and
    raising this one above 2^50 would silently stop testing the kernel.

    The reference is `FieldElement.__mul__`, which is the scalar coefficient
    product in `field.c`: a different implementation, not a rearrangement of
    the same one.
    """

    #: 49 bits, 1 mod 4, with 3 a non-residue -- so x^d - 3 is irreducible.
    FUSED_PRIME = 562949953421201
    FUSED_W = 3

    def field(self, d: int) -> ExtensionField:
        return ExtensionField(self.FUSED_PRIME, d, self.FUSED_W)

    def values(self, field, n, seed) -> list[ExtensionFieldElement]:
        """The concrete element type, not the front: these tests reach for
        `value`, the native coefficient array, which is this implementation's
        and not part of the declared API."""
        rng = random.Random(seed)  # noqa: S311 - test data, not a key
        return [
            ExtensionFieldElement(
                field, [rng.randrange(self.FUSED_PRIME) for _ in range(field.d)]
            )
            for _ in range(n)
        ]

    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 8, 9])
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 100])
    def test_matches_the_element_product(self, d, n):
        """Degrees on both sides of the instantiations (2, 4, 8) and of the
        cap, and lengths on both sides of the kernel's vector width."""
        field = self.field(d)
        left = self.values(field, n, seed=50 + d)
        right = self.values(field, n, seed=80 + d)
        a, b = FieldVector(field, left), FieldVector(field, right)
        assert (a * b).to_list() == [x * y for x, y in zip(left, right, strict=True)]
        assert (a * right[0]).to_list() == [x * right[0] for x in left]
        assert a.scale(right[0]).to_list() == [x * right[0] for x in left]

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_the_padding_stays_reduced_and_carries_no_meaning(self, d):
        """The kernel writes whole vectors, so it touches the padding; and a
        dirty padding must not reach the answer."""
        field = self.field(d)
        left = self.values(field, 13, seed=60)
        right = self.values(field, 13, seed=61)
        a, b = FieldVector(field, left), FieldVector(field, right)
        product = a * b
        for plane in product._planes:
            assert all(plane[i] < self.FUSED_PRIME for i in range(product._allocated_n))

        dirty = FieldVector(field, right)
        for plane in dirty._planes:
            for i in range(13, dirty._allocated_n):
                plane[i] = self.FUSED_PRIME - 1
        assert (a * dirty).to_list() == product.to_list()

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_the_output_may_alias_an_input(self, d):
        """The header's promise, on this path too: the kernel loads a block's
        operands before it writes any of that block's outputs, which is the
        only reason it holds."""
        from vfhe.engine import lib

        field = self.field(d)
        left = self.values(field, 13, seed=70 + d)
        right = self.values(field, 13, seed=71 + d)
        a, b = FieldVector(field, left), FieldVector(field, right)
        expected = (a * b).to_list()

        lib.field_vec_mul(a._struct, a._struct, b._struct)
        assert a.to_list() == expected
        assert b.to_list() == right

        a = FieldVector(field, left)
        expected = a.scale(right[0]).to_list()
        lib.field_vec_scale(a._struct, a._struct, right[0].value)
        assert a.to_list() == expected

    def test_matches_the_product_worked_out_over_the_integers(self):
        """An oracle the kernel shares no code with.

        `test_matches_the_element_product` checks it against `field.c`'s scalar
        product, which is a different implementation but still the same
        definition compiled the same way. This one forms the polynomial product
        and the fold on x^d = w in Python over plain integers, and only then
        reduces -- so a shared misunderstanding of the quotient ring would show
        up here and nowhere else.
        """
        d, n = 4, 24
        prime = self.FUSED_PRIME
        rng = random.Random(99)  # noqa: S311 - test data, not a key
        left = [[rng.randrange(prime) for _ in range(d)] for _ in range(n)]
        right = [[rng.randrange(prime) for _ in range(d)] for _ in range(n)]

        field = self.field(d)
        a = ExtensionFieldVector(field, [ExtensionFieldElement(field, v) for v in left])
        b = ExtensionFieldVector(
            field, [ExtensionFieldElement(field, v) for v in right]
        )
        got = [[e.value[j] for j in range(d)] for e in (a * b).to_list()]

        for i in range(n):
            wide = [0] * (2 * d - 1)
            for p_i in range(d):
                for r_i in range(d):
                    wide[p_i + r_i] += left[i][p_i] * right[i][r_i]
            for k in range(2 * d - 2, d - 1, -1):
                wide[k - d] += self.FUSED_W * wide[k]
            assert got[i] == [c % prime for c in wide[:d]]


#: Below 2^50 the fma takes the fused kernel; PRIME (61 bits) does not, so
#: parametrising over both is what covers the two implementations.
_FUSED_PRIME, _FUSED_W = 562949953421201, 3
_DESTINATION_FIELDS = [
    pytest.param(_FUSED_PRIME, _FUSED_W, id="fused"),
    pytest.param(PRIME, W, id="generic"),
]


class TestFusedAccumulation:
    """``x.fma(b, c, out=x)``: the destination is also the addend.

    The shape a caller accumulating in place writes, and the one an
    implementation without a fused kernel gets wrong -- forming the product in
    the destination overwrites the addend before it is added. The C header
    promises outputs may alias inputs, so both families owe the same answer
    here whether or not a fused kernel exists.
    """

    def test_the_destination_may_be_the_addend(self):
        field = make_field()
        a = FieldVector(field, random_elements(field, 13, seed=120))
        b = FieldVector(field, random_elements(field, 13, seed=121))
        c = FieldVector(field, random_elements(field, 13, seed=122))
        want = (a + b * c).to_list()
        assert a.fma(b, c, out=a).to_list() == want
        assert a.to_list() == want

    def test_the_destination_may_be_a_factor(self):
        field = make_field()
        a = FieldVector(field, random_elements(field, 13, seed=123))
        b = FieldVector(field, random_elements(field, 13, seed=124))
        c = FieldVector(field, random_elements(field, 13, seed=125))
        want = (a + b * c).to_list()
        assert a.fma(b, c, out=b).to_list() == want

    def test_the_broadcast_form_may_accumulate_in_place(self):
        field = make_field()
        a = FieldVector(field, random_elements(field, 13, seed=126))
        b = FieldVector(field, random_elements(field, 13, seed=127))
        scalar = random_elements(field, 1, seed=128)[0]
        want = (a + b.scale(scalar)).to_list()
        assert a.fma(b, scalar, out=a).to_list() == want


class TestDestinations:
    """The operations with a destination, and the fused multiply-add.

    Together with `view` these are what let an expression be written per chunk
    without allocating: the intermediates are the same shape every time and
    want one buffer. The checks are against the allocating forms, which are
    the same arithmetic and are what a caller would otherwise write.
    """

    FUSED_PRIME, FUSED_W = _FUSED_PRIME, _FUSED_W

    @staticmethod
    def vectors(prime, w, n, count):
        field = ExtensionField(prime, 4, w)
        out = []
        for i in range(count):
            vector = FieldVector(field, n)
            vector.sample_random(b"destination-%d" % i)
            out.append(vector)
        return field, out

    @pytest.mark.parametrize(("prime", "w"), _DESTINATION_FIELDS)
    @pytest.mark.parametrize("n", [8, 13, 64])
    def test_out_gets_the_result_and_is_returned(self, prime, w, n):
        field, (a, b) = self.vectors(prime, w, n, 2)
        element = b[0]
        for got, want in (
            (lambda d: a.add(b, out=d), a + b),
            (lambda d: a.add(element, out=d), a + element),
            (lambda d: a.sub(b, out=d), a - b),
            (lambda d: a.rsub(element, out=d), element - a),
            (lambda d: a.mul(b, out=d), a * b),
            (lambda d: a.neg(out=d), -a),
            (lambda d: a.scale(element, out=d), a.scale(element)),
        ):
            dest = FieldVector(field, n)
            assert got(dest) is dest
            assert dest.to_list() == want.to_list()

    @pytest.mark.parametrize(("prime", "w"), _DESTINATION_FIELDS)
    @pytest.mark.parametrize("n", [8, 13, 64])
    def test_fma_is_the_expression_it_names(self, prime, w, n):
        """``a + b * c``, against the two-operation form it replaces."""
        field, (a, b, c) = self.vectors(prime, w, n, 3)
        assert a.fma(b, c).to_list() == (a + b * c).to_list()
        element = c[0]
        assert a.fma(b, element).to_list() == (a + b.scale(element)).to_list()
        dest = FieldVector(field, n)
        assert a.fma(b, c, out=dest) is dest
        assert dest.to_list() == (a + b * c).to_list()

    @pytest.mark.parametrize(("prime", "w"), _DESTINATION_FIELDS)
    def test_a_destination_may_be_an_operand(self, prime, w):
        """The kernels promise it, and a chunked expression will do it."""
        _, (a, b, c) = self.vectors(prime, w, 16, 3)
        want = (a + b * c).to_list()
        into_a = a.copy()
        assert into_a.fma(b, c, out=into_a).to_list() == want
        into_b = b.copy()
        assert a.fma(into_b, c, out=into_b).to_list() == want
        sum_want = (a + b).to_list()
        into = a.copy()
        assert into.add(b, out=into).to_list() == sum_want

    @pytest.mark.parametrize(("prime", "w"), _DESTINATION_FIELDS)
    def test_a_destination_is_checked(self, prime, w):
        field, (a, b) = self.vectors(prime, w, 16, 2)
        other = ExtensionField(prime + 4, 4, w)
        with pytest.raises(ValueError, match="holds"):
            a.add(b, out=FieldVector(field, 8))
        with pytest.raises(ValueError, match="different field"):
            a.add(b, out=FieldVector(other, 16))
        with pytest.raises(TypeError, match="out must be"):
            a.add(b, out=[0] * 16)
        with pytest.raises(TypeError, match="b must be"):
            a.fma([0] * 16, b)

    def test_a_chunked_expression_matches_the_whole_vector_one(self):
        """What the destination and `view` are for: the same expression, one
        buffer reused per chunk, no allocation inside the loop."""
        field, (a, b, u) = self.vectors(self.FUSED_PRIME, self.FUSED_W, 256, 3)
        lam = u[0]
        whole = a + b * (u + lam)

        chunk = 64
        out = FieldVector(field, 256)
        scratch = FieldVector(field, chunk)
        for start in range(0, 256, chunk):
            u.view(start, chunk).add(lam, out=scratch)
            a.view(start, chunk).fma(
                b.view(start, chunk), scratch, out=out.view(start, chunk)
            )
        assert out.to_list() == whole.to_list()


class TestFiberHashing:
    """`hash_fibers`: the gather `hash_elements` deliberately cannot do.

    Fiber k is ``{k, k + stride, ...}``, so a vector held as `group` blocks of
    `stride` gets one digest per position across all of them. Checked against
    the digest of the same elements gathered by hand, which is the definition,
    and against the front's generic form, which is a second implementation.
    """

    @pytest.mark.parametrize(("group", "stride"), [(1, 8), (2, 4), (4, 8), (8, 4)])
    def test_matches_the_gathered_digest(self, group, stride):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 32, seed=90))
        got = digest_list(vector.hash_fibers(group, stride))
        assert len(got) == stride
        for k in range(stride):
            fiber = [vector[k + j * stride] for j in range(group)]
            assert got[k] == FieldVector(field, fiber).hash()

    def test_agrees_with_the_generic_form(self):
        from vfhe.arith.base import FieldVector as Front

        field = make_field()
        vector = FieldVector(field, random_elements(field, 24, seed=91))
        assert vector.hash_fibers(3, 8) == Front.hash_fibers(vector, 3, 8)

    def test_is_not_the_contiguous_windowing(self):
        """The two window shapes are different digests, which is the point of
        having both -- a fiber is a gather, a window is a run."""
        field = make_field()
        vector = FieldVector(field, random_elements(field, 16, seed=92))
        assert vector.hash_fibers(2, 8) != vector.hash_elements(2, 8)

    def test_a_fiber_set_that_does_not_fit_yields_nothing(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 16, seed=93))
        assert vector.hash_fibers(3, 8) == b""  # 24 elements needed, 16 present
        with pytest.raises(ValueError, match="must be positive"):
            vector.hash_fibers(0, 4)


class TestMovementDestinations:
    """`interleave`, `concat` and `split_even_odd` with somewhere to write.

    They are the movement half of a fused expression: every other operation
    can write into a reused buffer now, so an expression that ends by
    interleaving or concatenating two results was left allocating for that
    step alone.
    """

    def test_interleave_and_split_take_destinations(self):
        field = make_field()
        values = random_elements(field, 32, seed=94)
        vector = FieldVector(field, values)
        lo, hi = FieldVector(field, 16), FieldVector(field, 16)
        even, odd = vector.split_even_odd(out=(lo, hi))
        assert even is lo and odd is hi
        assert lo.to_list() == values[0::2]
        assert hi.to_list() == values[1::2]

        dest = FieldVector(field, 32)
        assert FieldVector.interleave(lo, hi, out=dest) is dest
        assert dest.to_list() == values

    def test_concat_takes_a_destination(self):
        field = make_field()
        left = random_elements(field, 5, seed=95)
        right = random_elements(field, 3, seed=96)
        dest = FieldVector(field, 8)
        got = FieldVector.concat(
            [FieldVector(field, left), FieldVector(field, right)], out=dest
        )
        assert got is dest
        assert dest.to_list() == left + right

    def test_the_front_dispatches_to_the_implementation(self):
        """`FieldVector.interleave(...)` is the documented spelling, so it must
        reach the kernel rather than the generic body the front keeps for an
        implementation that has none."""
        field = make_field()
        vector = FieldVector(field, random_elements(field, 8, seed=97))
        even, odd = vector.split_even_odd()
        # A destination is what the generic path would have to emulate; the
        # implementation's own takes it directly.
        dest = FieldVector(field, 8)
        assert FieldVector.interleave(even, odd, out=dest).to_list() == vector.to_list()
        assert FieldVector.concat([even, odd], out=dest).to_list() == (
            even.to_list() + odd.to_list()
        )

    def test_a_destination_of_the_wrong_shape_is_refused(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 8, seed=98))
        even, odd = vector.split_even_odd()
        with pytest.raises(ValueError, match="holds"):
            FieldVector.interleave(even, odd, out=FieldVector(field, 4))
        not_a_pair: Any = FieldVector(field, 4)  # the check is a runtime one
        with pytest.raises(TypeError, match="pair of vectors"):
            vector.split_even_odd(out=not_a_pair)


class TestDigestsAreAViewNotACopy:
    """`hash_elements` hands back the kernel's own buffer.

    A view rather than a copy, because a long vector's digests are tens of
    megabytes and copying them costs more than hashing them. It has to behave
    like the bytes it views for everything a caller does with digests, and it
    has to be read-only -- a tree that will not copy it either is only safe if
    nobody can write through it.
    """

    @staticmethod
    def vector():
        field = make_field()
        return FieldVector(field, random_elements(field, 32, seed=130))

    def test_it_reads_as_the_bytes_it_views(self):
        digests = self.vector().hash_elements(2, 2)
        assert isinstance(digests, memoryview)
        assert len(digests) == 16 * 32
        copied = bytes(digests)
        assert digests == copied
        assert digests[32:64] == copied[32:64]
        assert b"".join([digests[:32], digests[32:64]]) == copied[:64]

    def test_it_is_read_only(self):
        digests = self.vector().hash_elements(2, 2)
        assert digests.readonly
        with pytest.raises(TypeError):
            digests[0] = 0  # pyright: ignore[reportIndexIssue]

    def test_it_outlives_the_vector_it_came_from(self):
        """The view owns a reference to the C buffer, so dropping the vector
        must not leave it reading freed memory."""
        import gc

        vector = self.vector()
        digests = vector.hash_elements(2, 2)
        expected = bytes(digests)
        del vector
        gc.collect()
        assert bytes(digests) == expected

    def test_the_empty_case_is_still_a_view(self):
        assert self.vector().hash_elements(33, 1) == b""
        assert len(self.vector().hash_elements(33, 1)) == 0


class TestDigestsReachATreeUntouched:
    """The digests come back as one buffer, and that buffer is what a Merkle
    tree takes -- no Python object per window at either end.

    What each window digests is pinned down in `TestSamplingAndHashing` and
    `TestFiberHashing`; what is left here is that the buffer reaches a tree
    unchanged, and gives the root the per-leaf path gives.
    """

    @pytest.mark.parametrize(("group", "stride"), [(1, 1), (2, 2), (4, 4)])
    def test_the_buffer_is_what_a_tree_takes(self, group, stride):
        from vfhe.crypto import Merkle

        field = make_field()
        vector = FieldVector(field, random_elements(field, 32, seed=104))
        digests = vector.hash_elements(group, stride)
        packed = Merkle.from_digests(digests)
        objects = Merkle(digest_list(digests), hash=lambda leaf: leaf)
        assert packed.root == objects.root
        assert len(packed) == len(vector) // stride
