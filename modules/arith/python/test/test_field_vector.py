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

import pytest
from vfhe.arith import (
    ExtensionField,
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


def make_field(d=4):
    return Field(PRIME, d, W)


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

        class _NoVectors:
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
        vector = FieldVector(field, [5, 6, 7])
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
            FieldVector(field, 4) + FieldVector(field, 5)
        with pytest.raises(ValueError, match="different fields"):
            FieldVector(field, 4) + FieldVector(make_field(), 4)


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
            FieldVector(make_field(), 4) ** -1

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
        vector = FieldVector(field, 16)
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

    def test_hash_elements_windows(self):
        field = make_field()
        values = random_elements(field, 8, seed=24)
        vector = FieldVector(field, values)

        singles = vector.hash_elements()
        assert len(singles) == 8
        assert singles[3] == FieldVector(field, [values[3]]).hash()

        pairs = vector.hash_elements(group=2, stride=2)
        assert len(pairs) == 4
        assert pairs[1] == FieldVector(field, values[2:4]).hash()

        sliding = vector.hash_elements(group=3, stride=1)
        assert len(sliding) == 6
        assert sliding[2] == FieldVector(field, values[2:5]).hash()

    def test_hash_elements_drops_a_partial_window(self):
        field = make_field()
        vector = FieldVector(field, random_elements(field, 7, seed=25))
        assert len(vector.hash_elements(group=2, stride=2)) == 3
        assert vector.hash_elements(group=9, stride=1) == []

    def test_hash_elements_rejects_a_zero_step(self):
        vector = FieldVector(make_field(), 4)
        with pytest.raises(ValueError, match="must be positive"):
            vector.hash_elements(group=0)


def test_the_codeword_fold_is_expressible_in_vector_operations():
    """A Reed-Solomon fold, the shape the first consumer needs.

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
    -- but the C entry points are the currency the piop and polycom kernels
    will be written against, so the promise is exercised here at that level.
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
