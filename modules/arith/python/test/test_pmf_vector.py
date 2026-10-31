# SPDX-FileCopyrightText: 2026 The vFHE Authors
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
from array import array
from typing import Any

import pytest
from vfhe.arith import FieldVector, PseudoMersenneField, PseudoMersenneVector
from vfhe.crypto import hash_bytes
from vfhe.util.bindings import lib

rng = random.Random(0xB0A710)  # noqa: S311 - test data, not a key

LIMB_BITS = 52
LIMB_MASK = (1 << LIMB_BITS) - 1
LANES = 8

#: Around the group width of 8: below it, astride it, and multiples of it.
LENGTHS = [0, 1, 5, 8, 13, 32]


@pytest.fixture(
    params=[156, 208, 193, 260, 312, 256],
    ids=[
        "3-limb-aligned",
        "4-limb-aligned",
        "4-limb-shifted",
        "5-limb-aligned",
        "6-limb-aligned",
        "5-limb-shifted",
    ],
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


def digest_list(packed: bytes | memoryview) -> list[bytes]:
    """A packed digest buffer as one bytes per window."""
    return [bytes(packed[i : i + 32]) for i in range(0, len(packed), 32)]


def random_values(f, n, seed=1):
    local = random.Random(seed)  # noqa: S311 - test data, not a key
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


def _dirty(vector):
    """`vector` with its padding set to p - 1, which a kernel must not read
    as values."""
    filler = vector.field(vector.field.prime - 1)
    for index in range(len(vector), vector._allocated_n):
        lib.pmf_vec_set_element(vector._struct, index, filler._buf)
    return vector


def _product_sums(values, sums, nodes, weights, p):
    """`FieldVector.fold_product_sums` over big ints: `values` are the
    tables' entries, `weights` the pairs' weights or None."""
    rows = []
    for products in sums:
        row = []
        for t in range(nodes):
            total = 0
            for i in range(len(values[0]) // 2):
                inner = 0
                for sign, factors in products:
                    term = 1
                    for j in factors:
                        u, v = values[j][2 * i], values[j][2 * i + 1]
                        term *= u + t * (v - u)
                    inner += sign * term
                total += inner * (1 if weights is None else weights[i])
            row.append(total % p)
        rows.append(row)
    return rows


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
            _ = FieldVector(field, 4) + FieldVector(field, 5)
        with pytest.raises(ValueError, match="different fields"):
            _ = FieldVector(field, 4) + FieldVector(other, 4)


class TestFallbackTier:
    """The operations the front derives once, over the element kernels."""

    @pytest.mark.parametrize("exponent", [0, 1, 2, 5, 17])
    def test_pow_matches_the_oracle(self, field, exponent):
        values = random_values(field, 11, 7)
        result = FieldVector(field, values) ** exponent
        assert_vector(result, [pow(x, exponent, field.prime) for x in values], field)

    def test_pow_rejects_a_negative_exponent(self, field):
        with pytest.raises(ValueError, match="negative exponent"):
            _ = FieldVector(field, 4) ** -1

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


#: Around the group width, and around the 2048 elements from which the kernel
#: cuts the vector into rows: exactly whole rows, one element of tail, more
#: than 256 rows (4095), and rows wider than one block (6149).
INVERSE_LENGTHS = [0, 1, 5, 8, 13, 2048, 2049, 4095, 6149]


class TestInverse:
    """`pmf_vec_inv`, against big ints and against the front's element loop."""

    def test_batch_inverse(self, field):
        p = field.prime
        values = [1, 2, field.c, p - 1, *random_values(field, 9, 8)]
        vector = FieldVector(field, values)
        inverses = vector.inverse()
        assert_vector(inverses, [pow(x, p - 2, p) for x in values], field)
        assert_vector(vector * inverses, [1] * len(values), field)

    @pytest.mark.parametrize("n", INVERSE_LENGTHS)
    def test_agrees_with_the_front(self, field, n):
        p = field.prime
        values = [x or 1 for x in random_values(field, n, n)]
        vector = FieldVector(field, values)
        inverses = vector.inverse()
        assert type(inverses) is PseudoMersenneVector
        assert inverses == FieldVector.inverse(vector)
        assert_vector(inverses, [pow(x, -1, p) for x in values], field)
        assert_vector(vector, values, field)

    @pytest.mark.parametrize("n", [13, 2049, 6149])
    def test_in_place(self, field, n):
        values = [x or 1 for x in random_values(field, n, n + 1)]
        vector = FieldVector(field, values)
        expected = vector.inverse()
        assert lib.pmf_vec_inv(vector._struct, vector._struct) == 1
        assert vector == expected

    def test_dirty_padding_is_not_read(self, field):
        values = [x or 1 for x in random_values(field, 2053, 3)]
        expected = FieldVector(field, values).inverse()
        assert _dirty(FieldVector(field, values)).inverse() == expected

    @pytest.mark.parametrize(
        ("n", "position"),
        [(6, 3), (2049, 0), (2049, 1500), (2049, 2047), (2049, 2048), (6149, 6100)],
    )
    def test_a_zero_is_rejected(self, field, n, position):
        values = [x or 1 for x in random_values(field, n, 9)]
        values[position] = 0
        vector = FieldVector(field, values)
        with pytest.raises(ValueError, match="not invertible"):
            vector.inverse()
        out = FieldVector(field, [7] * n)
        assert lib.pmf_vec_inv(out._struct, vector._struct) == 0
        assert out == FieldVector(field, [7] * n), "out was written"

    def test_empty(self, field):
        result = FieldVector(field, 0).inverse()
        assert type(result) is PseudoMersenneVector
        assert len(result) == 0


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

    def test_digest_is_the_tagged_hash(self, field):
        values = random_values(field, 9, 27)
        vector = FieldVector(field, values)
        assert vector.digest() == FieldVector(field, values).digest()
        assert vector.digest() != FieldVector(field, values[:8]).digest()
        assert vector.digest() != vector.hash()
        assert FieldVector(field, values[:1]).digest() != field(values[0]).digest()

    def test_hash_covers_the_canonical_encodings(self, field):
        """The digest is over the encoding, so it is representation-free."""
        values = random_values(field, 4, 26)
        vector = FieldVector(field, values)
        assert digest_list(vector.hash_elements())[2] == field(values[2]).digest()

    def test_digest_elements_is_the_vector_digest(self, field):
        """The generic route: this family has no flat-buffer path, so the
        front's default builds the vector -- and must still agree."""
        values = random_values(field, 8, 29)
        vector = FieldVector(field, values)
        elements = vector.to_list()
        assert field.digest_elements(elements) == vector.hash()
        assert (
            field.digest_elements(elements[2:4])
            == digest_list(vector.hash_elements(group=2, stride=2))[1]
        )

    def test_hash_elements_windows(self, field):
        values = random_values(field, 8, 27)
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

    @pytest.mark.parametrize("group", [1, 2, 3, 5])
    def test_digests_are_blake3_of_the_big_endian_encodings(self, field, group):
        """Against BLAKE3 of Python's own encodings, not of another digest.
        1100 elements cross a 32 KiB staging buffer at every width, and
        windows of 1 to 5 encodings end in a partial 64-byte block at every
        width timed here."""
        width = (field.bits + 7) // 8
        values = random_values(field, 1100, 31)
        values[:3] = [0, field.prime - 1, 1]
        encoded = [v.to_bytes(width, "big") for v in values]
        vector = FieldVector(field, values)
        assert vector.hash() == hash_bytes(b"".join(encoded))
        windows = digest_list(vector.hash_elements(group=group, stride=group))
        assert windows == [
            hash_bytes(b"".join(encoded[k : k + group]))
            for k in range(0, len(values) - group + 1, group)
        ]

    def test_hash_elements_drops_a_partial_window(self, field):
        vector = FieldVector(field, random_values(field, 7, 28))
        assert len(vector.hash_elements(group=2, stride=2)) == 3 * 32
        assert vector.hash_elements(group=9, stride=1) == b""

    def test_hash_elements_rejects_a_zero_step(self, field):
        with pytest.raises(ValueError, match="must be positive"):
            FieldVector(field, 4).hash_elements(group=0)


def test_padded_length_is_the_single_source_of_the_plane_size(field):
    for n in (0, 1, 7, 8, 9, 16, 13):
        padded = lib.pmf_vec_padded_length(n)
        assert padded >= n and padded % LANES == 0 and padded - n < LANES
        assert FieldVector(field, n)._allocated_n == padded


def test_a_twisted_pair_fold_is_expressible_in_vector_operations(field):
    """A twisted pair fold, written over whole vectors.

    ``folded[i] = hi + coeff * twist + r * coeff`` with
    ``coeff = (lo - hi) * untwist``, per position. Written once over
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


class TestNativeMovementAndFold:
    """Interleave, concat, query and the fused fold, against the generic forms.

    The lengths sit around the group width of 8 and include the odd multiples
    of the pair width (18, 50) where the folded half-length pads past the
    input's allocation, which is the boundary the pair loader has to respect.
    """

    @pytest.mark.parametrize("n", [2, 8, 18, 26])
    def test_interleave_inverts_split(self, field, n):
        vector = FieldVector(field, random_values(field, n, 40 + n))
        even, odd = vector.split_even_odd()
        assert FieldVector.interleave(even, odd) == vector
        assert PseudoMersenneVector.interleave(even, odd) == vector

    def test_interleave_rejects_mismatches(self, field):
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector.interleave(FieldVector(field, 4), FieldVector(field, 5))
        with pytest.raises(TypeError):
            FieldVector.interleave(FieldVector(field, 4), [0] * 4)

    def test_concat_matches_the_element_lists(self, field):
        parts = [random_values(field, n, 50 + n) for n in (5, 0, 8, 3)]
        vectors = [FieldVector(field, part) for part in parts]
        joined = FieldVector.concat(vectors)
        assert_vector(joined, [x for part in parts for x in part], field)
        assert FieldVector.concat(vectors[:1]) == vectors[0]
        with pytest.raises(ValueError, match="at least one"):
            FieldVector.concat([])
        other = PseudoMersenneField.generate(312 if field.bits != 312 else 260)
        with pytest.raises(ValueError, match="different fields"):
            FieldVector.concat([vectors[0], FieldVector(other, 3)])

    def test_query_gathers_with_negative_indices(self, field):
        values = random_values(field, 9, 60)
        vector = FieldVector(field, values)
        gathered = vector.query([8, 0, -1, 3, 3])
        assert_vector(gathered, [values[i] for i in (8, 0, 8, 3, 3)], field)
        assert len(vector.query([])) == 0
        with pytest.raises(IndexError):
            vector.query([9])

    def test_sample_random_at_on_the_generic_route(self, field):
        """This family has no kernel for it, so the front's default runs --
        and must read the same sequence a fill holds."""
        whole = FieldVector(field, 60)
        whole.sample_random(b"seed-at")
        entries = whole.to_list()
        positions = [5, 0, 59, 5, 17]
        got = FieldVector(field, len(positions))
        got.sample_random_at(b"seed-at", positions)
        assert [int(e) for e in got] == [int(entries[i]) for i in positions]
        with pytest.raises(ValueError, match="expected 2 indices"):
            FieldVector(field, 2).sample_random_at(b"seed-at", [1])

    def test_query_takes_a_prebuilt_index_buffer(self, field):
        """The fast path, on the other vector family: same elements as the
        sequence form, with nothing checked per index."""
        values = random_values(field, 9, 61)
        vector = FieldVector(field, values)
        positions = [8, 0, 8, 3, 3]
        assert vector.query(array("Q", positions)) == vector.query(positions)
        assert len(vector.query(array("Q", []))) == 0
        with pytest.raises(IndexError, match="out of range"):
            vector.query(array("Q", [0, 9]))
        assert vector.query(array("q", [8, -1])) == vector.query([8, -1])

    @pytest.mark.parametrize("n", [2, 8, 18, 26, 50])
    def test_fold_matches_the_split_formula(self, field, n):
        p = field.prime
        values = random_values(field, n, 70 + n)
        r = rng.randrange(p)
        vector = FieldVector(field, values)
        folded = vector.fold(field(r))
        expected = [
            (values[2 * i] + r * (values[2 * i + 1] - values[2 * i])) % p
            for i in range(n // 2)
        ]
        assert_vector(folded, expected, field)
        # ...and against the generic default, which is the formula itself.
        assert FieldVector.fold(vector, field(r)) == folded

    @pytest.mark.parametrize("n", [0, 2, 6, 16, 18, 50, 4102])
    def test_fold_inner_products_match_big_ints(self, field, n):
        """The three sums against big ints and against the front's
        whole-vector body, around the group width (pairs in whole groups,
        past them, or both) and past the 255 groups the lazy sums hold
        between reductions."""
        p = field.prime
        xs, ys = random_values(field, n, 90 + n), random_values(field, n, 91 + n)
        a, b = FieldVector(field, xs), FieldVector(field, ys)
        expected = [
            sum(
                (xs[2 * i] + t * (xs[2 * i + 1] - xs[2 * i]))
                * (ys[2 * i] + t * (ys[2 * i + 1] - ys[2 * i]))
                for i in range(n // 2)
            )
            % p
            for t in (0, 1, 2)
        ]
        sums = a.fold_inner_products(b)
        assert [int(e) for e in sums] == expected
        for e in sums:
            assert_canonical(e, field)
        assert sums == FieldVector.fold_inner_products(a, b)

    def test_fold_inner_products_of_a_vector_with_itself(self, field):
        p = field.prime
        xs = random_values(field, 50, 92)
        a = FieldVector(field, xs)
        expected = [
            sum((xs[2 * i] + t * (xs[2 * i + 1] - xs[2 * i])) ** 2 for i in range(25))
            % p
            for t in (0, 1, 2)
        ]
        assert [int(e) for e in a.fold_inner_products(a)] == expected

    def test_fold_inner_products_reject_mismatches(self, field):
        with pytest.raises(ValueError, match="odd"):
            FieldVector(field, 5).fold_inner_products(FieldVector(field, 5))
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector(field, 4).fold_inner_products(FieldVector(field, 6))
        other = PseudoMersenneField.generate(312 if field.bits != 312 else 260)
        with pytest.raises(ValueError, match="different fields"):
            FieldVector(field, 4).fold_inner_products(FieldVector(other, 4))

    @pytest.mark.parametrize("n", [4, 8, 12, 32, 36, 100, 4104])
    def test_fold_and_inner_products_match_the_separate_calls(self, field, n):
        """The folds, their padding, and the sums of the folds, against the
        three calls they stand for and big ints: folded lengths below a
        group, in whole groups of pairs, and with pairs past them."""
        p = field.prime
        xs, ys = random_values(field, n, 93 + n), random_values(field, n, 94 + n)
        r = rng.randrange(p)
        a, b = FieldVector(field, xs), FieldVector(field, ys)
        a_bound, b_bound, sums = a.fold_and_inner_products(b, field(r))
        assert a_bound == a.fold(r) and b_bound == b.fold(r)
        assert sums == a_bound.fold_inner_products(b_bound)
        assert (a_bound, b_bound, sums) == FieldVector.fold_and_inner_products(
            a, b, field(r)
        )
        fa = [(xs[2 * i] + r * (xs[2 * i + 1] - xs[2 * i])) % p for i in range(n // 2)]
        fb = [(ys[2 * i] + r * (ys[2 * i + 1] - ys[2 * i])) % p for i in range(n // 2)]
        assert [int(e) for e in sums] == [
            sum(
                (fa[2 * i] + t * (fa[2 * i + 1] - fa[2 * i]))
                * (fb[2 * i] + t * (fb[2 * i + 1] - fb[2 * i]))
                for i in range(n // 4)
            )
            % p
            for t in (0, 1, 2)
        ]
        for folded in (a_bound, b_bound):
            for index in range(len(folded), folded._allocated_n):
                value = sum(
                    int(folded._planes[k][index]) << (LIMB_BITS * k)
                    for k in range(field.limbs)
                )
                assert value < field.prime

    def test_fold_and_inner_products_reject_mismatches(self, field):
        with pytest.raises(ValueError, match="multiple of 4"):
            FieldVector(field, 6).fold_and_inner_products(FieldVector(field, 6), 3)
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector(field, 4).fold_and_inner_products(FieldVector(field, 8), 3)

    @pytest.mark.parametrize("n", [2, 6, 16, 18, 50, 4102])
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_fold_product_sums_match_big_ints(self, field, n, weighted):
        """Every node count, against big ints and the front's whole-vector
        body: pairs below a group, in whole groups and past them, and past
        the 255 groups the lazy sums hold between reductions, with the
        padding dirty. A table repeated in one product and shared between sums, products of one to
        three factors, both signs, a negated first product."""
        p = field.prime
        values = [random_values(field, n, 100 + 7 * n + j) for j in range(5)]
        weights = random_values(field, n // 2, 99 + n)
        tables = [_dirty(FieldVector(field, v)) for v in values]
        w = _dirty(FieldVector(field, weights)) if weighted else None
        sums = [
            [(1, (0,)), (-1, (1, 2))],
            [(-1, (3, 4, 1)), (1, (2, 2))],
            [(1, (4,))],
        ]
        for nodes in (1, 2, 3, 4):
            got = FieldVector.fold_product_sums(tables, sums, nodes, w)
            assert got == FieldVector._fold_product_sums(tables, sums, nodes, w)
            assert [[int(e) for e in row] for row in got] == _product_sums(
                values, sums, nodes, weights if weighted else None, p
            )
            for row in got:
                for e in row:
                    assert_canonical(e, field)

    @pytest.mark.parametrize("n", [4, 8, 12, 32, 36, 100, 4104])
    def test_fold_and_product_sums_match_the_separate_calls(self, field, n):
        """The folds, their padding, and the sums over the folds, against
        the calls they stand for and big ints, from tables and weights whose
        padding is dirty."""
        p = field.prime
        values = [random_values(field, n, 200 + 5 * n + j) for j in range(4)]
        weights = random_values(field, n // 4, 199 + n)
        r = rng.randrange(p)
        tables = [_dirty(FieldVector(field, v)) for v in values]
        sums = [[(1, (0,)), (-1, (1, 2))], [(1, (3,)), (-1, (3, 0, 1))]]
        for w in (None, _dirty(FieldVector(field, weights))):
            folded, got = FieldVector.fold_and_product_sums(
                tables, field(r), sums, 3, w
            )
            assert folded == [t.fold(r) for t in tables]
            assert got == FieldVector.fold_product_sums(folded, sums, 3, w)
            bound = [
                [(v[2 * i] + r * (v[2 * i + 1] - v[2 * i])) % p for i in range(n // 2)]
                for v in values
            ]
            assert [[int(e) for e in row] for row in got] == _product_sums(
                bound, sums, 3, None if w is None else weights, p
            )
            for table in folded:
                for index in range(len(table), table._allocated_n):
                    value = sum(
                        int(table._planes[k][index]) << (LIMB_BITS * k)
                        for k in range(field.limbs)
                    )
                    assert value < p

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_fold_product_sums_at_the_lazy_sum_bound(self, field, weighted):
        """Every summand p - 1, over 4200 groups: entries p - 1 and weights 1,
        or entries p - 1 under a negated product (summand 1) and weights
        p - 1; through both kernels."""
        p = field.prime
        n = 2 * 8 * 4200
        table = FieldVector(field, n) - field.one
        if weighted:
            w = FieldVector(field, n // 2) - field.one
            sums = [[(-1, (0,))]]
        else:
            w, sums = None, [[(1, (0,))]]
        expected = [(n // 2) * (p - 1) % p] * 4
        got = FieldVector.fold_product_sums([table], sums, 4, w)
        assert [int(e) for e in got[0]] == expected
        if weighted:
            w = FieldVector(field, n // 4) - field.one
        _, got = FieldVector.fold_and_product_sums([table], 5, sums, 4, w)
        assert [int(e) for e in got[0]] == [(n // 4) * (p - 1) % p] * 4

    def test_fold_product_sums_past_the_kernel_bounds(self, field):
        """A shape past the kernel's bounds takes the whole-vector body."""
        values = [random_values(field, 8, 300 + j) for j in range(9)]
        tables = [FieldVector(field, v) for v in values]
        sums = [[(1, tuple(range(9)))]]
        got = FieldVector.fold_product_sums(tables, sums, 2)
        assert [[int(e) for e in row] for row in got] == _product_sums(
            values, sums, 2, None, field.prime
        )
        assert len(FieldVector.fold_product_sums(tables, sums * 5, 5)) == 5

    def test_fold_product_sums_reject_mismatches(self, field):
        a, b = FieldVector(field, 6), FieldVector(field, 8)
        with pytest.raises(ValueError, match="odd"):
            FieldVector.fold_product_sums([FieldVector(field, 5)], [[(1, (0,))]], 2)
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector.fold_product_sums([a, b], [[(1, (0, 1))]], 2)
        with pytest.raises(ValueError, match="weights"):
            FieldVector.fold_product_sums([a], [[(1, (0,))]], 2, a)
        with pytest.raises(ValueError, match="index"):
            FieldVector.fold_product_sums([a], [[(1, (1,))]], 2)
        with pytest.raises(ValueError, match="sign"):
            FieldVector.fold_product_sums([a], [[(2, (0,))]], 2)
        with pytest.raises(ValueError, match="multiple of 4"):
            FieldVector.fold_and_product_sums([a], 3, [[(1, (0,))]], 2)
        with pytest.raises(ValueError, match="weights"):
            FieldVector.fold_and_product_sums([b], 3, [[(1, (0,))]], 2, b)

    def test_fold_accepts_an_int_and_needs_an_even_length(self, field):
        vector = FieldVector(field, random_values(field, 8, 80))
        assert vector.fold(5) == vector.fold(field(5))
        with pytest.raises(ValueError, match="odd"):
            FieldVector(field, 5).fold(3)

    def test_fold_leaves_the_padding_canonical(self, field):
        folded = FieldVector(field, random_values(field, 18, 81)).fold(7)
        for index in range(len(folded), folded._allocated_n):
            value = sum(
                int(folded._planes[k][index]) << (LIMB_BITS * k)
                for k in range(field.limbs)
            )
            assert value < field.prime


class TestBlocksViewsAndIndexedSampling:
    """The three surfaces this implementation serves without a kernel of its
    own -- strided splits, views, and a sampling window -- each checked
    against the whole-vector form it is a piece of."""

    @pytest.mark.parametrize("block", [1, 2, 4, 8])
    def test_split_and_fold_match_the_index_arithmetic(self, field, block):
        values = random_values(field, 32, 60)
        vector = FieldVector(field, values)
        lo = [values[at + i] for at in range(0, 32, 2 * block) for i in range(block)]
        hi = [
            values[at + block + i]
            for at in range(0, 32, 2 * block)
            for i in range(block)
        ]
        left, right = vector.split_even_odd(block)
        assert [int(e) for e in left] == lo
        assert [int(e) for e in right] == hi
        r = field(values[0])
        folded = vector.fold(r, block)
        assert [int(e) for e in folded] == [
            (a + int(r) * (b - a)) % field.prime for a, b in zip(lo, hi, strict=True)
        ]

    def test_a_view_shares_the_parent_planes(self, field):
        values = random_values(field, 32, 61)
        vector = FieldVector(field, values)
        window = vector.view(8, 8)
        assert [int(e) for e in window] == values[8:16]
        window[0] = field.one
        assert int(vector[8]) == 1
        with pytest.raises(ValueError):
            vector.view(1, 8)

    @pytest.mark.parametrize(
        ("n", "period"),
        [
            (1, 1),
            (8, 1),
            (8, 8),
            (9, 3),
            (40, 8),
            (64, 16),
            (64, 64),
            (333, 333),
            (96, 12),
        ],
    )
    def test_lift_twisted_matches_the_pairwise_formula(self, field, n, period):
        """The kernel against the definition: tables that are a whole number
        of groups, the operands' own length, and ragged ones read lane by
        lane."""
        a = FieldVector(field, random_values(field, n, 63))
        b = FieldVector(field, random_values(field, n, 64))
        twist = FieldVector(field, random_values(field, period, 65))
        lifted = FieldVector.lift_twisted(a, b, twist)
        p = field.prime
        want = []
        for i in range(n):
            product = int(b[i]) * int(twist[i % period])
            want += [(int(a[i]) + product) % p, (int(a[i]) - product) % p]
        assert [int(e) for e in lifted] == want
        for e in lifted:
            assert_canonical(e, field)
        out = FieldVector(field, 2 * n)
        assert FieldVector.lift_twisted(a, b, twist, out=out) is out and out == lifted

    def test_lift_twisted_checks_its_operands(self, field):
        a, b = FieldVector(field, 8), FieldVector(field, 8)
        with pytest.raises(ValueError, match="does not divide"):
            FieldVector.lift_twisted(a, b, FieldVector(field, 3))
        with pytest.raises(ValueError, match="length mismatch"):
            FieldVector.lift_twisted(a, FieldVector(field, 4), FieldVector(field, 8))
        with pytest.raises(ValueError, match="holds 8 elements, not 16"):
            FieldVector.lift_twisted(
                a, b, FieldVector(field, 8), out=FieldVector(field, 8)
            )

    @pytest.mark.parametrize("n", [1, 4, 8, 9, 64, 333])
    def test_fold_twisted_matches_the_pairwise_formula(self, field, n):
        """The kernel against the definition, and with the table `1 / (2 t)`
        it inverts a lift by `t`."""
        word = FieldVector(field, random_values(field, 2 * n, 66))
        table = FieldVector(field, random_values(field, n, 67))
        r = field(random_values(field, 1, 68)[0])
        folded = word.fold_twisted(table, r)
        p = field.prime
        half = (p + 1) // 2
        assert [int(e) for e in folded] == [
            (
                (int(word[2 * i]) + int(word[2 * i + 1])) * half
                + (int(word[2 * i]) - int(word[2 * i + 1])) * int(table[i]) * int(r)
            )
            % p
            for i in range(n)
        ]
        a = FieldVector(field, random_values(field, n, 69))
        b = FieldVector(field, random_values(field, n, 70))
        lifted = FieldVector.lift_twisted(a, b, table)
        assert lifted.fold_twisted(table.scale(2).inverse(), r) == a + b * r
        out = FieldVector(field, n)
        assert word.fold_twisted(table, r, out=out) is out and out == folded

    def test_fold_twisted_checks_its_operands(self, field):
        word = FieldVector(field, 16)
        with pytest.raises(ValueError, match="not 8"):
            word.fold_twisted(FieldVector(field, 4), 1)
        table = FieldVector(field, 8)
        with pytest.raises(ValueError, match="cannot write into an operand"):
            word.fold_twisted(table, 1, out=table)

    def test_fold_twisted_blocks_is_fold_twisted_per_block(self, field):
        """Each block folded with its own element of `r`: the front's body,
        a `fold_twisted` per block."""
        m, count = 5, 3
        word = FieldVector(field, random_values(field, 2 * m * count, 71))
        table = FieldVector(field, random_values(field, m, 72))
        r = FieldVector(field, random_values(field, count, 73))
        folded = word.fold_twisted_blocks(table, r)
        expected = FieldVector.concat(
            [
                word.query(range(2 * m * b, 2 * m * (b + 1))).fold_twisted(table, r[b])
                for b in range(count)
            ]
        )
        assert folded == expected

    def test_a_view_of_a_view_starts_where_the_view_does(self, field):
        values = random_values(field, 64, 62)
        vector = FieldVector(field, values)
        nested = vector.view(32, 32).view(8, 8)
        assert [int(e) for e in nested] == values[40:48]
        nested[0] = field.one
        assert int(vector[40]) == 1

    def test_a_sampling_window_is_the_slice_of_the_whole_fill(self, field):
        whole = FieldVector(field, 32)
        whole.sample_random(b"pmf-window")
        window = FieldVector(field, 4)
        window.sample_random(b"pmf-window", 20)
        assert [int(e) for e in window] == [int(e) for e in whole][20:24]
        assert int(field.element_from_seed(b"pmf-window", 7)) == int(whole[7])


class TestDestinations:
    """The same surface here, where `fma` has no fused kernel.

    It still avoids the intermediate a caller would allocate -- the product
    goes straight into the destination and the addend folds in after -- so the
    API is the same shape even though one pass more runs.
    """

    def test_out_gets_the_result_and_is_returned(self, field):
        values = random_values(field, 13, 80)
        other = random_values(field, 13, 81)
        a, b = FieldVector(field, values), FieldVector(field, other)
        element = b[0]
        for got, want in (
            (lambda d: a.add(b, out=d), a + b),
            (lambda d: a.sub(b, out=d), a - b),
            (lambda d: a.rsub(element, out=d), element - a),
            (lambda d: a.mul(b, out=d), a * b),
            (lambda d: a.neg(out=d), -a),
            (lambda d: a.scale(element, out=d), a.scale(element)),
        ):
            dest = FieldVector(field, 13)
            assert got(dest) is dest
            assert [int(e) for e in dest] == [int(e) for e in want]

    def test_fma_is_the_expression_it_names(self, field):
        a = FieldVector(field, random_values(field, 9, 82))
        b = FieldVector(field, random_values(field, 9, 83))
        c = FieldVector(field, random_values(field, 9, 84))
        want = [int(e) for e in (a + b * c)]
        assert [int(e) for e in a.fma(b, c)] == want
        dest = FieldVector(field, 9)
        assert a.fma(b, c, out=dest) is dest
        assert [int(e) for e in dest] == want
        assert [int(e) for e in a.fma(b, c[0])] == [int(e) for e in (a + b.scale(c[0]))]

    def test_fma_accumulating_in_place(self, field):
        """``x.fma(b, c, out=x)`` -- the shape a caller accumulating in place
        writes, and the one an unfused implementation gets wrong by forming
        the product in the destination before adding what is still there."""
        a = FieldVector(field, random_values(field, 9, 87))
        b = FieldVector(field, random_values(field, 9, 88))
        c = FieldVector(field, random_values(field, 9, 89))
        want = [int(e) for e in (a + b * c)]
        assert [int(e) for e in a.fma(b, c, out=a)] == want
        assert [int(e) for e in a] == want  # and a is what was written

        # The broadcast form, and a destination aliasing the other operands.
        d = FieldVector(field, random_values(field, 9, 90))
        scalar = c[0]
        want = [int(e) for e in (d + b.scale(scalar))]
        assert [int(e) for e in d.fma(b, scalar, out=d)] == want

        e = FieldVector(field, random_values(field, 9, 91))
        want = [int(x) for x in (e + b * c)]
        assert [int(x) for x in e.fma(b, c, out=b)] == want

    def test_a_destination_is_checked(self, field):
        a = FieldVector(field, random_values(field, 8, 85))
        b = FieldVector(field, random_values(field, 8, 86))
        with pytest.raises(ValueError, match="holds"):
            a.add(b, out=FieldVector(field, 4))
        not_a_vector: Any = [0] * 8  # the check is a runtime one
        with pytest.raises(TypeError, match="out must be"):
            a.add(b, out=not_a_vector)


#: Primes at the edges of the two reductions the vector kernels choose
#: between: an aligned prime (n = 52 L) with the largest c below 2^44 takes the
#: shorter one at the limit of its bounds; aligned and shifted primes with the
#: largest c in budget take the general one at the limit of its.
_REDUCTION_EDGES = {
    "156-largest-c-below-2^44": (156, 17592185995263),
    "208-largest-c-below-2^44": (208, 17592185978879),
    "260-largest-c-below-2^44": (260, 17592185978879),
    "208-largest-c": (208, 4503599626891263),
    "193-largest-c": (193, 137438679039),
}


@pytest.fixture(params=list(_REDUCTION_EDGES), ids=list(_REDUCTION_EDGES))
def edge_field(request):
    bits, c = _REDUCTION_EDGES[request.param]
    return PseudoMersenneField((1 << bits) - c)


class TestReductionEdges:
    """Every arithmetic kernel at the edges of its reduction's bounds, on
    values that push them: 0, 1, p - 1, p - 2 and all-ones low limbs among
    uniform ones."""

    @staticmethod
    def edge_values(f, n, seed):
        p = f.prime
        low = (1 << (LIMB_BITS * (f.limbs - 1))) - 1
        edges = [0, 1, p - 1, p - 2, low, p - low, (p + 1) // 2]
        local = random.Random(seed)  # noqa: S311 - test data, not a key
        return [
            local.choice(edges) if local.random() < 0.4 else local.randrange(p)
            for _ in range(n)
        ]

    def test_elementwise(self, edge_field):
        f, p = edge_field, edge_field.prime
        xs, ys = self.edge_values(f, 333, 1), self.edge_values(f, 333, 2)
        a, b = FieldVector(f, xs), FieldVector(f, ys)
        s = self.edge_values(f, 1, 3)[0]
        assert [int(e) for e in a + b] == [
            (x + y) % p for x, y in zip(xs, ys, strict=True)
        ]
        assert [int(e) for e in a - b] == [
            (x - y) % p for x, y in zip(xs, ys, strict=True)
        ]
        assert [int(e) for e in a * b] == [
            x * y % p for x, y in zip(xs, ys, strict=True)
        ]
        assert [int(e) for e in a.scale(s)] == [x * s % p for x in xs]
        assert [int(e) for e in -a] == [(-x) % p for x in xs]
        for e in a * b:
            assert_canonical(e, f)

    def test_fold_inner_products(self, edge_field):
        """The lazy sums at their bound: pairs (p - 1, p - 2) against (1, 2)
        make the three products p - 1, p - 4 and p - 1, whose limbs are all
        close to 2^52, over more groups than a 64-bit limb sum holds without
        the reductions between batches; then the edge mix."""
        f, p = edge_field, edge_field.prime
        pairs = 8 * 4200 + 5
        for xs, ys in (
            ([p - 1, p - 2] * pairs, [1, 2] * pairs),
            (self.edge_values(f, 2002, 7), self.edge_values(f, 2002, 8)),
        ):
            n = len(xs)
            a, b = FieldVector(f, xs), FieldVector(f, ys)
            expected = [
                sum(
                    (xs[2 * i] + t * (xs[2 * i + 1] - xs[2 * i]))
                    * (ys[2 * i] + t * (ys[2 * i + 1] - ys[2 * i]))
                    for i in range(n // 2)
                )
                % p
                for t in (0, 1, 2)
            ]
            assert [int(e) for e in a.fold_inner_products(b)] == expected

    def test_fold_and_inner_products(self, edge_field):
        """As test_fold_inner_products, through the folds: at r = 0 a fold
        keeps the first member of each pair, so the folded vectors carry the
        pairs that hold the lazy sums at their bound."""
        f, p = edge_field, edge_field.prime
        pairs = 8 * 4200 + 5
        local = random.Random(9)  # noqa: S311 - test data, not a key
        xs, ys = [], []
        for _ in range(pairs):
            for kept_x, kept_y in ((p - 1, 1), (p - 2, 2)):
                xs += [kept_x, local.randrange(p)]
                ys += [kept_y, local.randrange(p)]
        a, b = FieldVector(f, xs), FieldVector(f, ys)
        _, _, sums = a.fold_and_inner_products(b, 0)
        assert [int(e) for e in sums] == [
            pairs * (p - 1) % p,
            pairs * (p - 4) % p,
            pairs * -9 % p,  # 2 (-4) - (-1) + 2 (-1) a pair
        ]

    def test_folds_and_lift(self, edge_field):
        f, p = edge_field, edge_field.prime
        xs = self.edge_values(f, 334, 4)
        ws = self.edge_values(f, 167, 5)
        r = self.edge_values(f, 1, 6)[0]
        word, table = FieldVector(f, xs), FieldVector(f, ws)
        assert [int(e) for e in word.fold(r)] == [
            (xs[2 * i] + r * (xs[2 * i + 1] - xs[2 * i])) % p for i in range(167)
        ]
        half = (p + 1) // 2
        assert [int(e) for e in word.fold_twisted(table, r)] == [
            (
                (xs[2 * i] + xs[2 * i + 1]) * half
                + (xs[2 * i] - xs[2 * i + 1]) * ws[i] * r
            )
            % p
            for i in range(167)
        ]
        a, b = FieldVector(f, xs[:167]), FieldVector(f, ws)
        lifted = [int(e) for e in FieldVector.lift_twisted(a, b, table)]
        assert lifted[0::2] == [(xs[i] + ws[i] * ws[i]) % p for i in range(167)]
        assert lifted[1::2] == [(xs[i] - ws[i] * ws[i]) % p for i in range(167)]


class TestUint64Transfer:
    """`set_uint64` and `to_uint64`: 64-bit integers in and out in one call."""

    @pytest.mark.parametrize("n", LENGTHS)
    def test_round_trip(self, field, n):
        rng = random.Random(n)  # noqa: S311 - test data, not a key
        values = array("Q", [rng.getrandbits(64) for _ in range(n)])
        vector = FieldVector(field, n)
        vector.set_uint64(values)
        assert [int(e) for e in vector] == list(values)
        assert vector.to_uint64() == values

    def test_a_bit_field_is_a_shift_and_a_mask(self, field):
        rng = random.Random(1)  # noqa: S311 - test data, not a key
        values = array("Q", [rng.getrandbits(64) for _ in range(13)])
        for shift, width in ((0, 11), (11, 11), (60, 4), (63, 1), (0, 64), (5, 64)):
            vector = FieldVector(field, len(values))
            vector.set_uint64(values, shift=shift, width=width)
            mask = (1 << width) - 1
            assert [int(e) for e in vector] == [(v >> shift) & mask for v in values]

    def test_a_bit_field_of_128_bit_values(self, field):
        rng = random.Random(2)  # noqa: S311 - test data, not a key
        wide = [rng.getrandbits(128) for _ in range(13)]
        low = array("Q", [v & ((1 << 64) - 1) for v in wide])
        high = array("Q", [v >> 64 for v in wide])
        for shift, width in ((0, 64), (60, 11), (64, 11), (100, 28), (127, 1)):
            vector = FieldVector(field, len(wide))
            vector.set_uint64(low, shift=shift, width=width, high=high)
            mask = (1 << width) - 1
            assert [int(e) for e in vector] == [(v >> shift) & mask for v in wide]
        with pytest.raises(ValueError, match="shift < 128"):
            vector.set_uint64(low, shift=128, high=high)
        with pytest.raises(ValueError, match="differ in length"):
            vector.set_uint64(low, high=high[:3])

    def test_a_range_from_start_leaves_the_rest(self, field):
        vector = FieldVector(field, [7] * 16)
        vector.set_uint64(array("Q", range(5)), start=3)
        assert [int(e) for e in vector] == [7] * 3 + list(range(5)) + [7] * 8

    def test_an_element_from_2_64_up_has_no_uint64(self, field):
        for big in (1 << 64, field.prime - 1):
            assert FieldVector(field, [1, big, 2]).to_uint64() is None
        top = (1 << 64) - 1
        assert FieldVector(field, [top]).to_uint64() == array("Q", [top])

    def test_the_preconditions_are_checked(self, field):
        vector = FieldVector(field, 4)
        with pytest.raises(ValueError, match="3 values from 2 into 4"):
            vector.set_uint64(array("Q", [1, 2, 3]), start=2)
        with pytest.raises(ValueError, match="shift"):
            vector.set_uint64(array("Q", [0] * 4), shift=64)
        with pytest.raises(ValueError, match="width"):
            vector.set_uint64(array("Q", [0] * 4), width=0)
        with pytest.raises(TypeError, match="64-bit"):
            vector.set_uint64(array("I", [0] * 4))
