# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The vector kernels give the same bits on one thread and on several.

The pseudo-Mersenne and extension-field vector kernels split a long vector
into chunks across the library's threads once it is a few grains long. Every
test here runs a kernel at the default limit of one thread and again at a
raised limit, on lengths past the threshold that are not a multiple of the
chunks, and asserts the results are identical -- vectors, sums and digests
alike. The kernels' values are checked against references in
`test_pmf_vector.py`, `test_field_vector.py` and `test_pmf_ntt.py`; what is
new on these paths is the split.
"""

from __future__ import annotations

import random
from array import array
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from vfhe.arith import (
    ExtensionField,
    FieldVector,
    PseudoMersenneField,
    PseudoMersenneNTT,
    occurrence_counts,
)
from vfhe.util import parallel

if TYPE_CHECKING:
    from collections.abc import Callable

    from vfhe.arith import PseudoMersenneVector

THREADS = 8

#: Past every arithmetic kernel's threshold (two grains of 2^15 elements),
#: and not a multiple of eight chunks.
N_ARITH = (1 << 17) + 40
#: Past the threshold of the data-movement kernels (two chunks of 4 MiB of
#: planes) at three limbs.
N_MOVE = (1 << 19) + 40


def at_one_and_many(compute: Callable[[], Any]) -> tuple[Any, Any]:
    """`compute()` with the library limited to one thread, then to THREADS."""
    try:
        parallel.set_num_threads(1)
        serial = compute()
        parallel.set_num_threads(THREADS)
        threaded = compute()
    finally:
        parallel.set_num_threads()
    return serial, threaded


def same_on_one_and_many(compute: Callable[[], Any]) -> None:
    serial, threaded = at_one_and_many(compute)
    assert threaded == serial


def sampled(field, n: int, seed: bytes) -> FieldVector:
    vector = FieldVector(field, n)
    vector.sample_random(seed)
    return vector


def words(n: int, seed: int) -> array:
    rng = random.Random(seed)  # noqa: S311 - test data, not a key
    return array("Q", rng.randbytes(8 * n))


@pytest.fixture(
    scope="module", params=[156, 256], ids=["3-limb-aligned", "5-limb-shifted"]
)
def pmf(request):
    return PseudoMersenneField.generate(request.param, two_adicity=8)


@pytest.fixture(scope="module")
def pmf_operands(pmf):
    return [sampled(pmf, N_ARITH, b"operand %d" % i) for i in range(4)]


#: The pseudo-Mersenne kernels, each on an object holding long operands a, b,
#: c, d and elements s, r.
PMF_KERNELS: dict[str, Callable[..., Any]] = {
    "add": lambda o: o.a + o.b,
    "sub": lambda o: o.a - o.b,
    "mul": lambda o: o.a * o.b,
    "neg": lambda o: -o.a,
    "add_scalar": lambda o: o.a + o.s,
    "sub_scalar": lambda o: o.a - o.s,
    "scalar_sub": lambda o: o.s - o.a,
    "scale": lambda o: o.a.scale(o.s),
    "sum": lambda o: o.a.sum(),
    "inverse": lambda o: o.a.inverse(),
    "fold": lambda o: o.a.fold(o.r),
    "fold_twisted": lambda o: o.a.fold_twisted(o.b.query(range(len(o.a) // 2)), o.r),
    "lift_twisted": lambda o: FieldVector.lift_twisted(o.a, o.b, o.c),
    # Tables read cyclically: a whole group long, and not a whole group.
    "lift_twisted_period_8": lambda o: FieldVector.lift_twisted(
        o.a, o.b, o.c.query(range(8))
    ),
    "lift_twisted_period_3": lambda o: FieldVector.lift_twisted(
        o.a, o.b, o.c.query(range(3))
    ),
    "fold_inner_products": lambda o: o.a.fold_inner_products(o.b),
    "fold_and_inner_products": lambda o: o.a.fold_and_inner_products(o.b, o.r),
    "fold_product_sums": lambda o: FieldVector.fold_product_sums(
        [o.a, o.b, o.c], SUMS, 3
    ),
    "fold_product_sums_weighted": lambda o: FieldVector.fold_product_sums(
        [o.a, o.b, o.c], SUMS, 3, o.d.query(range(len(o.a) // 2))
    ),
    "fold_and_product_sums": lambda o: FieldVector.fold_and_product_sums(
        [o.a, o.b, o.c], o.r, SUMS, 4
    ),
    "fold_and_product_sums_weighted": lambda o: FieldVector.fold_and_product_sums(
        [o.a, o.b, o.c], o.r, SUMS, 4, o.d.query(range(len(o.a) // 4))
    ),
}

#: Two signed sums of products, one with a repeated factor.
SUMS = [[(1, (0,)), (-1, (1, 2))], [(-1, (2, 0, 1)), (1, (1, 1))]]


@pytest.fixture(scope="module")
def long_pair(pmf):
    return sampled(pmf, N_MOVE, b"move a"), sampled(pmf, N_MOVE, b"move b")


class TestPseudoMersenne:
    @pytest.mark.parametrize("kernel", list(PMF_KERNELS))
    def test_kernel(self, pmf, pmf_operands, kernel):
        compute = PMF_KERNELS[kernel]
        a, b, c, d = pmf_operands
        o = SimpleNamespace(a=a, b=b, c=c, d=d, s=pmf(123456789), r=pmf(987654321))
        same_on_one_and_many(lambda: compute(o))

    def test_fold_twisted_over_its_own_word(self, pmf, pmf_operands):
        """A fold written over the low half of its word reads pairs that
        earlier groups' outputs overwrite, so it must run group after group
        on any number of threads."""
        # A view's length is a whole number of lane groups.
        half = len(pmf_operands[0]) // 2 // 8 * 8
        a = pmf_operands[0].query(range(2 * half))
        untwist = pmf_operands[1].query(range(half))
        r = pmf(987654321)
        expected = a.fold_twisted(untwist, r)

        def fold_in_place() -> FieldVector:
            word = a.copy()
            word.fold_twisted(untwist, r, out=word.view(0, half))
            return word.query(range(half))

        serial, threaded = at_one_and_many(fold_in_place)
        assert serial == expected
        assert threaded == expected

    @pytest.mark.parametrize(
        ("shift", "width", "with_high"),
        [(0, 64, False), (5, 13, False), (60, 30, True)],
    )
    def test_set_uint64(self, pmf, shift, width, with_high):
        low, high = words(N_MOVE, 1), words(N_MOVE, 2)

        def loaded() -> FieldVector:
            vector = FieldVector(pmf, N_MOVE + 3)
            vector.set_uint64(
                low, start=3, shift=shift, width=width, high=high if with_high else None
            )
            return vector

        same_on_one_and_many(loaded)

    def test_to_uint64(self, pmf, long_pair):
        small = FieldVector(pmf, N_MOVE)
        small.set_uint64(words(N_MOVE, 3))
        same_on_one_and_many(lambda: small.to_uint64())
        a = long_pair[0]
        assert at_one_and_many(lambda: a.to_uint64()) == (None, None)

    @pytest.mark.parametrize(
        "kernel", ["copy", "split_even_odd", "interleave", "concat", "query"]
    )
    def test_movement(self, long_pair, kernel):
        a, b = long_pair
        indices = array("Q", (i * 7919 % N_MOVE for i in range(N_MOVE)))
        compute = {
            "copy": lambda: a.copy(),
            "split_even_odd": lambda: a.split_even_odd(),
            "interleave": lambda: FieldVector.interleave(a, b),
            "concat": lambda: FieldVector.concat([a, b.query(range(77)), b]),
            "query": lambda: a.query(indices),
        }[kernel]
        same_on_one_and_many(compute)

    @pytest.mark.parametrize("threads", [1, THREADS])
    def test_query_out_of_range(self, long_pair, threads):
        a = long_pair[0]
        indices = array("Q", range(N_MOVE))
        indices[N_MOVE - 5] = N_MOVE
        try:
            parallel.set_num_threads(threads)
            with pytest.raises(IndexError):
                a.query(indices)
        finally:
            parallel.set_num_threads()

    @pytest.mark.parametrize(("group", "stride"), [(1, 1), (2, 2), (3, 1), (60, 60)])
    def test_hash_elements(self, pmf_operands, group, stride):
        """Batched windows, overlapping ones, and windows past a BLAKE3 chunk,
        which are hashed one at a time."""
        a = pmf_operands[0]
        same_on_one_and_many(lambda: bytes(a.hash_elements(group, stride)))

    @pytest.mark.parametrize(("n", "blocks"), [(64, 301), (4, 4101)])
    def test_ntt_blocks(self, pmf, n, blocks):
        """Blocks of whole groups, and blocks shorter than a group, which go
        through a staging group."""
        plan = PseudoMersenneNTT(pmf, n)
        a = cast("PseudoMersenneVector", sampled(pmf, n * blocks, b"blocks"))
        same_on_one_and_many(lambda: plan.forward(a, blocks=blocks))
        same_on_one_and_many(lambda: plan.inverse(a, blocks=blocks))

    @pytest.mark.parametrize(("k", "blocks"), [(8, 304), (8, 301), (2, 2051)])
    def test_ntt_encode_blocks(self, pmf, k, blocks):
        """Messages placed by 8 x 8 tiles, one at a time, and short."""
        plan = PseudoMersenneNTT(pmf, 64)
        message = cast("PseudoMersenneVector", sampled(pmf, k * blocks, b"messages"))
        same_on_one_and_many(lambda: plan.encode_blocks(message, blocks))


@pytest.fixture(scope="module")
def field():
    return PseudoMersenneField.generate(156, two_adicity=8)


class TestDigits:
    """`set_radix_decomposition` against the front's body, and on one thread and many."""

    @pytest.mark.parametrize(
        ("rows", "columns", "width", "digits", "with_high"),
        [
            (1, 13, 8, 3, False),
            (4, 9, 64, 3, True),
            (8, 3, 30, 6, True),
            (2, 17, 13, 12, True),
            (16, 8, 1, 9, False),
        ],
    )
    def test_matches_the_generic_body(
        self, field, rows, columns, width, digits, with_high
    ):
        """Digits astride bit 64 and bit 128, past bit 128 (zero) and past
        bit 64 without `high` (zero), shapes not a multiple of the eight
        columns a tile takes, and elements past the blocks left alone."""
        count = rows * columns
        low, high = words(count, rows), words(count, columns)
        size = digits * count + 5
        native = FieldVector(field, list(range(size)))
        generic = FieldVector(field, list(range(size)))
        kwargs = {"high": high if with_high else None, "columns": columns}
        native.set_radix_decomposition(low, width, digits, **kwargs)
        FieldVector.set_radix_decomposition(generic, low, width, digits, **kwargs)
        assert native == generic
        rest = [int(e) for e in native.query(range(digits * count, size))]
        assert rest == list(range(digits * count, size))

    def test_matches_a_bit_field_per_digit(self, field):
        """The transposed digits are the transposed values' bit fields."""
        rows, columns, width, digits = 4, 6, 20, 7
        count = rows * columns
        low, high = words(count, 7), words(count, 8)
        got = FieldVector(field, digits * count)
        got.set_radix_decomposition(low, width, digits, high=high, columns=columns)
        order = [i * columns + k for k in range(columns) for i in range(rows)]
        t_low = array("Q", (low[at] for at in order))
        t_high = array("Q", (high[at] for at in order))
        expected = FieldVector(field, digits * count)
        for j in range(digits):
            if width * j < 128:
                expected.set_uint64(
                    t_low, start=j * count, shift=width * j, width=width, high=t_high
                )
        assert got == expected

    def test_on_one_thread_and_many(self, field):
        rows, columns, digits = 8, 11003, 4
        low, high = words(rows * columns, 3), words(rows * columns, 4)

        def digits_of() -> FieldVector:
            vector = FieldVector(field, digits * rows * columns)
            vector.set_radix_decomposition(low, 40, digits, high=high, columns=columns)
            return vector

        same_on_one_and_many(digits_of)

    def test_checks_its_operands(self, field):
        vector = FieldVector(field, 12)
        values = array("Q", range(6))
        with pytest.raises(ValueError, match="columns"):
            vector.set_radix_decomposition(values, 8, 2, columns=4)
        with pytest.raises(ValueError, match="digits"):
            vector.set_radix_decomposition(values, 8, 3)
        with pytest.raises(ValueError, match="width"):
            vector.set_radix_decomposition(values, 65, 1)
        # No digits fit in any vector, however many values there are.
        before = vector.copy()
        vector.set_radix_decomposition(array("Q", range(20)), 8, 0)
        FieldVector.set_radix_decomposition(vector, array("Q", range(20)), 8, 0)
        assert vector == before


def reference_counts(values, bound: int) -> tuple[list[int], list[int]]:
    seen: dict[int, int] = {}
    prior, totals = [], [0] * bound
    for value in values:
        prior.append(seen.get(value, 0))
        seen[value] = prior[-1] + 1
        totals[value] += 1
    return prior, totals


class TestOccurrenceCounts:
    @pytest.mark.parametrize(("n", "bound"), [(0, 4), (1, 1), (37, 5), (1000, 1 << 16)])
    def test_matches_a_dictionary(self, n, bound):
        rng = random.Random(n)  # noqa: S311 - test data, not a key
        values = array("Q", (rng.randrange(bound) for _ in range(n)))
        counts = occurrence_counts(values, bound)
        assert counts is not None
        prior, totals = counts
        assert (list(prior), list(totals)) == reference_counts(values, bound)

    def test_a_value_out_of_range_gives_none(self):
        assert occurrence_counts(array("Q", [0, 3, 4, 1]), 4) is None
        assert occurrence_counts(array("Q", [2**64 - 1]), 4) is None

    def test_on_one_thread_and_many(self):
        rng = random.Random(5)  # noqa: S311 - test data, not a key
        n, bound = (1 << 18) + 5, 1 << 10
        values = array("Q", (rng.randrange(bound) for _ in range(n)))
        serial, threaded = at_one_and_many(lambda: occurrence_counts(values, bound))
        assert serial == threaded
        prior, totals = serial
        assert (list(prior), list(totals)) == reference_counts(values, bound)
        values[n - 3] = bound
        assert at_one_and_many(lambda: occurrence_counts(values, bound)) == (None, None)

    def test_rejects_other_buffers(self):
        with pytest.raises(TypeError):
            occurrence_counts(array("i", [1, 2]), 4)
        # Two rows of 64-bit words: counting them as one row would be wrong.
        rows = memoryview(array("Q", [0, 1, 2, 3])).cast("B").cast("Q", (2, 2))
        with pytest.raises(TypeError):
            occurrence_counts(rows, 4)


_FUSED_PRIME = 562949953421201
_GENERIC_PRIME = (1 << 61) - 1


@pytest.fixture(
    scope="module",
    params=[(_GENERIC_PRIME, 4), (_FUSED_PRIME, 4), (_FUSED_PRIME, 2)],
    ids=["generic-d4", "fused-d4", "fused-d2"],
)
def extension(request):
    prime, d = request.param
    return ExtensionField(prime, d, 3)


@pytest.fixture(scope="module")
def ext_operands(extension):
    return [sampled(extension, N_ARITH, b"operand %d" % i) for i in range(3)]


@pytest.fixture(scope="module")
def table(extension):
    """A table over the prime subfield, as long as the operands."""
    vector = FieldVector(ExtensionField(extension.prime, 1), N_ARITH)
    vector.sample_random(b"table")
    return vector


#: The extension-field kernels, each on an object holding long operands a, b,
#: c, a table t over the prime subfield, and elements s, r.
EXTENSION_KERNELS: dict[str, Callable[..., Any]] = {
    "add": lambda o: o.a + o.b,
    "sub": lambda o: o.a - o.b,
    "neg": lambda o: -o.a,
    "add_scalar": lambda o: o.a + o.s,
    "sub_scalar": lambda o: o.a - o.s,
    "scalar_sub": lambda o: o.s - o.a,
    "mul": lambda o: o.a * o.b,
    "scale": lambda o: o.a.scale(o.s),
    "fma": lambda o: o.a.fma(o.b, o.c),
    "fma_scalar": lambda o: o.a.fma(o.b, o.s),
    "mul_plane": lambda o: o.a.mul(o.t),
    "fma_plane": lambda o: o.a.fma(o.b, o.t),
    "fold_twisted": lambda o: o.a.fold_twisted(o.b.query(range(len(o.a) // 2)), o.r),
    "fold_twisted_plane": lambda o: o.a.fold_twisted(
        o.t.query(range(len(o.a) // 2)), o.r
    ),
    # 1003 * 61 positions: blocks that start and end inside lane groups.
    "fold_twisted_blocks": lambda o: o.a.query(
        range(2 * 1003 * 61)
    ).fold_twisted_blocks(o.b.query(range(1003)), o.c.query(range(61))),
    "lift_twisted": lambda o: FieldVector.lift_twisted(o.a, o.b, o.c),
    "lift_twisted_period_8": lambda o: FieldVector.lift_twisted(
        o.a, o.b, o.b.query(range(8))
    ),
    "lift_twisted_plane": lambda o: FieldVector.lift_twisted(o.a, o.b, o.t),
    "fma_interleave": lambda o: FieldVector.fma_interleave(o.a, o.b, o.c, -o.c),
    "fma_interleave_scalar": lambda o: FieldVector.fma_interleave(o.a, o.b, o.r, o.c),
}


class TestExtension:
    @pytest.mark.parametrize("kernel", list(EXTENSION_KERNELS))
    def test_kernel(self, extension, ext_operands, table, kernel):
        compute = EXTENSION_KERNELS[kernel]
        a, b, c = ext_operands
        s = extension([5, 6, 7, 8][: extension.d])
        r = extension([9, 10, 11, 12][: extension.d])
        o = SimpleNamespace(a=a, b=b, c=c, t=table, s=s, r=r)
        same_on_one_and_many(lambda: compute(o))

    def test_fold_twisted_over_its_own_word(self, extension, ext_operands):
        """A fold written over the low half of its word reads pairs that
        earlier windows' outputs overwrite, so it must run window after
        window on any number of threads."""
        # A view's length is a whole number of lane groups.
        half = len(ext_operands[0]) // 2 // 8 * 8
        a, b = ext_operands[0].query(range(2 * half)), ext_operands[1]
        untwist = b.query(range(half))
        r = extension([9, 10, 11, 12][: extension.d])
        expected = a.fold_twisted(untwist, r)

        def fold_in_place() -> FieldVector:
            word = a.copy()
            word.fold_twisted(untwist, r, out=word.view(0, half))
            return word.query(range(half))

        serial, threaded = at_one_and_many(fold_in_place)
        assert serial == expected
        assert threaded == expected

    @pytest.mark.parametrize(("group", "stride"), [(1, 1), (2, 2), (5, 1), (40, 40)])
    def test_hash_elements(self, ext_operands, group, stride):
        a = ext_operands[0]
        same_on_one_and_many(lambda: bytes(a.hash_elements(group, stride)))

    def test_hash_fibers(self, ext_operands):
        a = ext_operands[0].query(range(1 << 17))
        same_on_one_and_many(lambda: bytes(a.hash_fibers(4, 1 << 15)))
