# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Vectors of pseudo-Mersenne elements, against two independent oracles.

Every arithmetic result is checked twice: against Python big ints, and against
the same work done one element at a time through the scalar kernels. The two
disagree in different ways -- the big ints catch an algorithm that is wrong,
the element loop catches a plane layout that is wrong -- and the vector
kernels share no code with either.

Running this file under ``VFHE_ENGINE=portable`` re-runs the same plane
algorithm one lane wide, which is what turns a lane or mask mistake into a
failure rather than a silent difference between engines.

The three fields cover both limb counts and both reduction paths: a shift of
zero skips the top-bit step, and the 256-bit field is the one that exercises
it.
"""

from __future__ import annotations

import random

import pytest
from vfhe.arith import FieldVector, PseudoMersenneField, PseudoMersenneVector
from vfhe.misc.libvfhe import lib

rng = random.Random(0xB0A710)

LIMB_BITS = 52
LIMB_MASK = (1 << LIMB_BITS) - 1
LANES = 8

#: Around the group width of 8: below it, astride it, and multiples of it.
LENGTHS = [0, 1, 5, 8, 13, 32]


@pytest.fixture(
    params=[260, 312, 256], ids=["5-limb-aligned", "6-limb-aligned", "5-limb-shifted"]
)
def field(request):
    return PseudoMersenneField.generate(request.param, two_adicity=8)


def edge_values(f):
    """Values on a boundary of the representation or of the reduction."""
    p = f.prime
    return [
        0,
        1,
        f.c,
        f.fold,
        p - 1,
        p - 2,
        LIMB_MASK,
        1 << LIMB_BITS,
        (1 << (LIMB_BITS * f.limbs)) - 1,
        (1 << f.bits) - 1,
    ]


def random_values(f, n, seed=1):
    local = random.Random(seed)
    return [local.randrange(f.prime) for _ in range(n)]


def assert_canonical(element, f):
    """The representation contract every element of a result must satisfy."""
    limbs = element.to_limbs()
    assert 0 <= int(element) < f.prime, "value outside [0, p)"
    assert all(limb <= LIMB_MASK for limb in limbs[: f.limbs]), "limb exceeds 52 bits"
    assert limbs[f.limbs :] == [0] * (LANES - f.limbs), "padding lanes not zero"


def assert_vector(vector, expected_ints, f):
    """Every element canonical, and equal to the big-int oracle."""
    assert len(vector) == len(expected_ints)
    for element, expected in zip(vector.to_list(), expected_ints, strict=True):
        assert_canonical(element, f)
        assert int(element) == expected % f.prime


class TestDispatch:
    """The front resolves through the field's spec, as for the extension field."""

    def test_builds_the_pmf_vector_type(self, field):
        vector = FieldVector(field, 4)
        assert type(vector) is PseudoMersenneVector
        assert isinstance(vector, FieldVector)

    def test_vector_cls_is_on_the_spec(self, field):
        assert field.spec.vector_cls is PseudoMersenneVector

    def test_a_vector_is_not_hashable(self, field):
        with pytest.raises(TypeError):
            hash(FieldVector(field, 4))


class TestConstruction:
    """A vector is built from a length or from a sequence of values."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_a_length_gives_zeros(self, field, n):
        vector = FieldVector(field, n)
        assert len(vector) == n
        assert all(int(element) == 0 for element in vector)

    @pytest.mark.parametrize("n", LENGTHS)
    def test_a_sequence_round_trips(self, field, n):
        values = random_values(field, n)
        vector = FieldVector(field, values)
        assert_vector(vector, values, field)
        assert [int(x) for x in vector] == values

    def test_values_may_be_elements_or_ints(self, field):
        vector = FieldVector(field, [7, field(9), field.prime + 3])
        assert [int(x) for x in vector] == [7, 9, 3]

    def test_edge_values_round_trip(self, field):
        values = edge_values(field)
        assert_vector(FieldVector(field, values), values, field)

    def test_empty_is_allowed(self, field):
        empty = FieldVector(field, 0)
        assert len(empty) == 0
        assert empty.to_list() == []
        assert int(empty.sum()) == 0
        assert len(empty.inverse()) == 0

    def test_a_negative_length_is_rejected(self, field):
        with pytest.raises(ValueError, match="must not be negative"):
            FieldVector(field, -1)


class TestIndexing:
    """Integer indexing over the plane layout."""

    def test_get_and_set(self, field):
        vector = FieldVector(field, 4)
        vector[2] = field(12345)
        assert int(vector[2]) == 12345
        assert int(vector[1]) == 0
        assert_canonical(vector[2], field)

    def test_negative_indices_count_from_the_end(self, field):
        values = random_values(field, 5)
        vector = FieldVector(field, values)
        assert int(vector[-1]) == values[4]
        assert int(vector[-5]) == values[0]

    def test_out_of_range_raises(self, field):
        vector = FieldVector(field, 3)
        for index in (3, -4):
            with pytest.raises(IndexError):
                vector[index]

    def test_a_read_element_stops_tracking_the_vector(self, field):
        vector = FieldVector(field, [5, 6, 7])
        element = vector[1]
        vector[1] = field(99)
        assert int(element) == 6


class TestArithmetic:
    """Each kernel against big ints and against the scalar element kernels."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_elementwise_ops(self, field, n):
        left, right = random_values(field, n, 1), random_values(field, n, 2)
        a, b = FieldVector(field, left), FieldVector(field, right)

        assert_vector(a + b, [x + y for x, y in zip(left, right, strict=True)], field)
        assert_vector(a - b, [x - y for x, y in zip(left, right, strict=True)], field)
        assert_vector(a * b, [x * y for x, y in zip(left, right, strict=True)], field)
        assert_vector(-a, [-x for x in left], field)

        # ...and against the scalar kernels, which share no code with these.
        elements = [field(x) for x in left], [field(y) for y in right]
        assert (a + b).to_list() == [x + y for x, y in zip(*elements, strict=True)]
        assert (a * b).to_list() == [x * y for x, y in zip(*elements, strict=True)]
        assert (a - b).to_list() == [x - y for x, y in zip(*elements, strict=True)]
        assert (-a).to_list() == [-x for x in elements[0]]

    def test_edge_values_through_every_op(self, field):
        """The boundaries of the reduction, in every pairing."""
        values = edge_values(field)
        pairs = [(x, y) for x in values for y in values]
        a = FieldVector(field, [x for x, _ in pairs])
        b = FieldVector(field, [y for _, y in pairs])

        assert_vector(a + b, [x + y for x, y in pairs], field)
        assert_vector(a - b, [x - y for x, y in pairs], field)
        assert_vector(a * b, [x * y for x, y in pairs], field)
        assert_vector(-a, [-x for x, _ in pairs], field)

    def test_negating_zero_gives_zero(self, field):
        """Not p, which is the trap the per-lane zero test exists for."""
        vector = FieldVector(field, [0, 1, 0, field.prime - 1, 0])
        assert_vector(-vector, [0, -1, 0, 1, 0], field)

    @pytest.mark.parametrize("n", LENGTHS)
    def test_broadcast_against_one_element(self, field, n):
        values = random_values(field, n, 3)
        scalar = rng.randrange(field.prime)
        vector, element = FieldVector(field, values), field(scalar)

        assert_vector(vector + element, [x + scalar for x in values], field)
        assert_vector(element + vector, [x + scalar for x in values], field)
        assert_vector(vector - element, [x - scalar for x in values], field)
        assert_vector(element - vector, [scalar - x for x in values], field)
        assert_vector(vector * element, [x * scalar for x in values], field)
        assert_vector(element * vector, [x * scalar for x in values], field)
        assert_vector(vector.scale(element), [x * scalar for x in values], field)
        # an int coerces the same way
        assert_vector(vector + scalar, [x + scalar for x in values], field)

    def test_scale_agrees_with_a_constant_vector(self, field):
        values = random_values(field, 13, 4)
        scalar = field(rng.randrange(field.prime))
        vector = FieldVector(field, values)
        assert vector.scale(scalar) == vector * FieldVector(field, [scalar] * 13)

    @pytest.mark.parametrize("n", LENGTHS)
    def test_sum(self, field, n):
        values = random_values(field, n, 5)
        total = FieldVector(field, values).sum()
        assert int(total) == sum(values) % field.prime
        assert_canonical(total, field)

    def test_sum_across_the_normalize_cadence(self, field):
        """More groups than the accumulator can hold before it must reduce."""
        n = 8 * 300
        values = [field.prime - 1] * n
        assert (
            int(FieldVector(field, values).sum())
            == (n * (field.prime - 1)) % field.prime
        )

    def test_an_operand_is_never_written_through(self, field):
        values = random_values(field, 9, 6)
        a = FieldVector(field, values)
        b = a + a
        assert [int(x) for x in a] == values
        assert_vector(b, [x + x for x in values], field)

    def test_length_and_field_mismatches_are_rejected(self, field):
        other = PseudoMersenneField.generate(312 if field.bits != 312 else 260)
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector(field, 4) + FieldVector(field, 5)
        with pytest.raises(ValueError, match="different fields"):
            FieldVector(field, 4) + FieldVector(other, 4)


class TestFallbackTier:
    """The operations the front derives once, over the element kernels."""

    @pytest.mark.parametrize("exponent", [0, 1, 2, 5, 17])
    def test_pow_matches_the_oracle(self, field, exponent):
        values = random_values(field, 11, 7)
        result = FieldVector(field, values) ** exponent
        assert_vector(result, [pow(x, exponent, field.prime) for x in values], field)

    def test_pow_rejects_a_negative_exponent(self, field):
        with pytest.raises(ValueError, match="negative exponent"):
            FieldVector(field, 4) ** -1

    def test_batch_inverse(self, field):
        p = field.prime
        values = [1, 2, field.c, p - 1, *random_values(field, 9, 8)]
        vector = FieldVector(field, values)
        inverses = vector.inverse()
        assert_vector(inverses, [pow(x, p - 2, p) for x in values], field)
        assert_vector(vector * inverses, [1] * len(values), field)

    def test_batch_inverse_rejects_a_zero(self, field):
        values = random_values(field, 6, 9)
        values[3] = 0
        with pytest.raises(ZeroDivisionError):
            FieldVector(field, values).inverse()

    def test_concat(self, field):
        first, second = random_values(field, 5, 10), random_values(field, 3, 11)
        joined = FieldVector.concat(
            [FieldVector(field, first), FieldVector(field, second)]
        )
        assert_vector(joined, first + second, field)

    def test_query_gathers(self, field):
        values = random_values(field, 8, 12)
        gathered = FieldVector(field, values).query([5, 0, 5, 2])
        assert_vector(gathered, [values[i] for i in (5, 0, 5, 2)], field)


class TestMovement:
    """Copy, split and equality, which read exactly n elements."""

    def test_copy_is_independent(self, field):
        values = random_values(field, 7, 13)
        original = FieldVector(field, values)
        duplicate = original.copy()
        duplicate[0] = field.one
        assert int(original[0]) == values[0]
        assert int(duplicate[0]) == 1

    @pytest.mark.parametrize("n", [2, 8, 14])
    def test_split_even_odd(self, field, n):
        values = random_values(field, n, 14)
        even, odd = FieldVector(field, values).split_even_odd()
        assert_vector(even, values[0::2], field)
        assert_vector(odd, values[1::2], field)

    def test_split_needs_an_even_length(self, field):
        with pytest.raises(ValueError, match="odd"):
            FieldVector(field, 5).split_even_odd()

    def test_equality(self, field):
        values = random_values(field, 6, 15)
        assert FieldVector(field, values) == FieldVector(field, values)
        assert FieldVector(field, values) != FieldVector(field, values[:5])
        other = list(values)
        other[2] = (other[2] + 1) % field.prime
        assert FieldVector(field, values) != FieldVector(field, other)

    def test_equality_ignores_the_padding(self, field):
        """A short vector's padding is scratch; only the n elements count."""
        values = random_values(field, 3, 16)
        left, right = FieldVector(field, values), FieldVector(field, values)
        right._planes[0][5] = 12345
        assert left == right


class TestPadding:
    """The invariant the group-at-a-time kernels depend on, checked directly."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_planes_are_padded_and_start_zero(self, field, n):
        vector = FieldVector(field, random_values(field, n, 17))
        assert vector._allocated_n % LANES == 0
        assert vector._allocated_n >= n
        assert len(vector._planes) == field.limbs
        for plane in vector._planes:
            assert all(plane[i] == 0 for i in range(n, vector._allocated_n))

    def test_padding_stays_canonical_through_arithmetic(self, field):
        """Kernels touch the padding; it must stay a valid element."""
        a = FieldVector(field, random_values(field, 13, 18))
        b = FieldVector(field, random_values(field, 13, 19))
        element = field(rng.randrange(field.prime))
        for result in (a + b, a - b, a * b, -a, a.scale(element), a + element):
            for index in range(len(a), result._allocated_n):
                value = sum(
                    int(result._planes[k][index]) << (k * LIMB_BITS)
                    for k in range(field.limbs)
                )
                assert value < field.prime, "padding left a non-canonical element"

    def test_results_are_unaffected_by_dirty_padding(self, field):
        values = random_values(field, 5, 20)
        clean, dirty = FieldVector(field, values), FieldVector(field, values)
        filler = field(field.prime - 1)
        for index in range(5, dirty._allocated_n):
            lib.pmf_vec_set_element(dirty._struct, index, filler._buf)
        assert (dirty * dirty).to_list() == (clean * clean).to_list()
        assert dirty.sum() == clean.sum()
        assert dirty.hash() == clean.hash()
        assert dirty == clean


class TestAliasingContract:
    """The header promises arithmetic outputs may alias their inputs.

    Nothing in the Python API aliases -- every operation allocates its result
    -- but the C entry points are the currency a consumer's kernels would be
    written against, so the promise is exercised at that level.
    """

    @pytest.mark.parametrize("kernel", ["pmf_vec_add", "pmf_vec_sub", "pmf_vec_mul"])
    def test_output_may_be_an_input(self, field, kernel):
        left, right = random_values(field, 13, 21), random_values(field, 13, 22)
        a, b = FieldVector(field, left), FieldVector(field, right)
        operator = {"pmf_vec_add": "__add__", "pmf_vec_sub": "__sub__"}.get(
            kernel, "__mul__"
        )
        expected = getattr(a, operator)(b)

        getattr(lib, kernel)(a._struct, a._struct, b._struct)
        assert a.to_list() == expected.to_list()
        assert [int(x) for x in b] == right  # the other operand is untouched

    def test_squaring_in_place_aliases_both_inputs(self, field):
        values = random_values(field, 9, 23)
        vector = FieldVector(field, values)
        lib.pmf_vec_mul(vector._struct, vector._struct, vector._struct)
        assert_vector(vector, [x * x for x in values], field)

    def test_negation_in_place(self, field):
        values = random_values(field, 9, 24)
        vector = FieldVector(field, values)
        lib.pmf_vec_neg(vector._struct, vector._struct)
        assert_vector(vector, [-x for x in values], field)


class TestSamplingAndHashing:
    """Sampling is a pure function of the seed; digests cover exactly n."""

    def test_sampling_fills_the_vector_with_distinct_elements(self, field):
        """One draw stream, so no element repeats by construction."""
        vector = FieldVector(field, 16)
        vector.sample_random(b"pmf-vector-seed")
        values = [int(element) for element in vector]
        assert len(set(values)) == len(values)
        for element in vector:
            assert_canonical(element, field)

    def test_sampling_is_a_pure_function_of_the_seed(self, field):
        first, second, third = (FieldVector(field, 8) for _ in range(3))
        first.sample_random(b"seed")
        second.sample_random(b"seed")
        third.sample_random(b"other")
        assert first == second
        assert first != third

    def test_hash_is_stable_and_content_dependent(self, field):
        values = random_values(field, 9, 25)
        vector = FieldVector(field, values)
        assert len(vector.hash()) == 32
        assert vector.hash() == FieldVector(field, values).hash()
        assert vector.hash() != FieldVector(field, values[:8]).hash()

    def test_hash_covers_the_canonical_encodings(self, field):
        """The digest is over the encoding, so it is representation-free."""
        values = random_values(field, 4, 26)
        vector = FieldVector(field, values)
        assert vector.hash_elements()[2] == field(values[2]).digest()

    def test_hash_elements_windows(self, field):
        values = random_values(field, 8, 27)
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

    def test_hash_elements_drops_a_partial_window(self, field):
        vector = FieldVector(field, random_values(field, 7, 28))
        assert len(vector.hash_elements(group=2, stride=2)) == 3
        assert vector.hash_elements(group=9, stride=1) == []

    def test_hash_elements_rejects_a_zero_step(self, field):
        with pytest.raises(ValueError, match="must be positive"):
            FieldVector(field, 4).hash_elements(group=0)


def test_padded_length_is_the_single_source_of_the_plane_size(field):
    for n in (0, 1, 7, 8, 9, 16, 13):
        padded = lib.pmf_vec_padded_length(n)
        assert padded >= n and padded % LANES == 0 and padded - n < LANES
        assert FieldVector(field, n)._allocated_n == padded


def test_the_codeword_fold_is_expressible_in_vector_operations(field):
    """A Reed-Solomon fold, the shape a consumer needs.

    ``folded[i] = hi + coeff * twist + r * coeff`` with
    ``coeff = (lo - hi) * twist2_inv``, per position. Written once over
    vectors and once as the loop it replaces.
    """
    p, n = field.prime, 16
    word = random_values(field, n, 29)
    twists = random_values(field, n // 2, 30)
    twists2_inv = random_values(field, n // 2, 31)
    r = rng.randrange(p)

    expected = []
    for i in range(n // 2):
        lo, hi = word[2 * i], word[2 * i + 1]
        coeff = (lo - hi) * twists2_inv[i] % p
        expected.append((hi + coeff * twists[i] + r * coeff) % p)

    lo_vec, hi_vec = FieldVector(field, word).split_even_odd()
    coeff_vec = (lo_vec - hi_vec) * FieldVector(field, twists2_inv)
    folded = hi_vec + coeff_vec * FieldVector(field, twists) + coeff_vec.scale(field(r))

    assert_vector(folded, expected, field)
