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
from typing import Any

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
        got = vector.hash_fibers(group, stride)
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
        assert vector.hash_fibers(3, 8) == []  # 24 elements needed, 16 present
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
