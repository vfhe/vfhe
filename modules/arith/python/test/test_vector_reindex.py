# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`FieldVector.reindex`: a vector read through a map of index bits.

Each implementation's native kernel is checked against the definition:
element ``offset + sum_j ((i >> j) & 1) * 2^bits[j]``, gathered with
`query`. The maps cover each way the kernel copies: runs of consecutive
source words, one lookup per position (a dropped or permuted low bit),
fixed source bits, lengths below and around a vector's 8 lanes, and long
vectors split across threads. The generic default is checked against the
native kernel, and maps that read past the vector or read a bit twice are
refused.
"""

from __future__ import annotations

import pytest
from vfhe.arith import ExtensionField, FieldVector, PseudoMersenneField
from vfhe.util import parallel

PRIME = (1 << 61) - 1


@pytest.fixture(
    scope="module",
    params=["pmf-156", "pmf-256", "ext-d1", "ext-d2", "ext-d4"],
)
def field(request):
    kind, _, size = request.param.partition("-")
    if kind == "pmf":
        return PseudoMersenneField.generate(int(size), two_adicity=8)
    return ExtensionField(PRIME, int(size[1:]), 3)


def _sampled(field, n: int, seed: bytes = b"reindex") -> FieldVector:
    vector = FieldVector(field, n)
    vector.sample_random(seed)
    return vector


def _by_definition(vector: FieldVector, bits: list, offset: int) -> FieldVector:
    sources = [
        offset + sum(((i >> j) & 1) << b for j, b in enumerate(bits) if b is not None)
        for i in range(1 << len(bits))
    ]
    return vector.query(sources)


#: (source bits, map, offset): every shape of map the kernel distinguishes.
MAPS = [
    (5, [0, 1, 2, 3, 4], 0),  # the identity: one run
    (6, [0, 1, 2, 3, 4], 1 << 5),  # a run, the top bit fixed to 1
    (6, [1, 2, 3, 4, 5], 1),  # bit 0 fixed to 1: a stride of 2, no run
    (5, [None, 0, 1, 2, 3, 4], 0),  # a new low variable: every word twice
    (5, [0, 1, 2, None, 3, 4], 0),  # a new variable in the middle
    (5, [4, 3, 2, 1, 0], 0),  # bits reversed: every word looked up
    (6, [2, 0, 4], 0b100010),  # a restriction and a renaming at once
    (4, [], 0b1011),  # one element
    (3, [0, 1, None], 0),  # shorter than a vector's lanes
    (10, [3, 4, 5, 6, 7, 8, 9, 0, 1, 2], 0),  # a run of 2^7 words, wrapped
    (10, [0, 1, 2, None, None, 3, 4, 5, 6, 7, 8, 9], 0),  # a run of 8, repeated
]


@pytest.mark.parametrize(("n_bits", "bits", "offset"), MAPS)
def test_reindex_is_the_definition(field, n_bits, bits, offset):
    vector = _sampled(field, 1 << n_bits)
    assert vector.reindex(bits, offset) == _by_definition(vector, bits, offset)


@pytest.mark.parametrize(("n_bits", "bits", "offset"), MAPS)
def test_the_generic_default_is_the_kernel(field, n_bits, bits, offset):
    vector = _sampled(field, 1 << n_bits)
    assert FieldVector.reindex(vector, bits, offset) == vector.reindex(bits, offset)


def test_reindex_reads_a_vector_of_any_length(field):
    # Only the indices read have to exist: a table that is not a power of
    # two long, read on its low positions.
    vector = _sampled(field, 13)
    bits = [None, 0, 1, 3]
    assert vector.reindex(bits) == _by_definition(vector, bits, 0)


def test_reindex_writes_into_out(field):
    vector = _sampled(field, 16)
    out = _sampled(field, 8, b"out")
    assert vector.reindex([1, 2, 3], 1, out=out) is out
    assert out == _by_definition(vector, [1, 2, 3], 1)
    with pytest.raises(ValueError, match="elements"):
        vector.reindex([1, 2], out=out)
    with pytest.raises(ValueError, match="cannot write over"):
        vector.reindex([0, 1, 2, 3], out=vector)


@pytest.mark.parametrize(
    ("bits", "offset", "message"),
    [
        ([0, 0], 0, "read twice"),
        ([0, 1], 1, "offset sets"),
        ([0, 4], 0, "reads index 17"),
        ([0, 1], 16, "reads index 19"),
        ([0, -2], 0, "non-negative or None"),
        ([0], -1, "offset must be non-negative"),
        ([None] * 64, 0, "2\\^64"),
    ],
)
def test_reindex_refuses_what_is_not_a_map_onto_the_vector(
    field, bits, offset, message
):
    vector = _sampled(field, 16)
    with pytest.raises(ValueError, match=message):
        vector.reindex(bits, offset)
    with pytest.raises(ValueError, match=message):
        FieldVector.reindex(vector, bits, offset)


#: Long enough to be split across threads by every implementation's grain,
#: for each kernel shape: runs, lookups, and runs of a few words.
LONG_MAPS = [
    [*range(1, 19), 0],
    [None, *range(19)],
    [*range(4), None, *range(4, 19)],
]


@pytest.mark.parametrize("bits", LONG_MAPS, ids=["runs", "lookups", "short-runs"])
def test_reindex_on_threads_is_the_serial_one(field, bits):
    vector = _sampled(field, 1 << 19)
    try:
        parallel.set_num_threads(1)
        serial = vector.reindex(bits)
        parallel.set_num_threads(8)
        threaded = vector.reindex(bits)
    finally:
        parallel.set_num_threads()
    assert threaded == serial
    for i in range(0, 1 << len(bits), 4099):
        source = sum(((i >> j) & 1) << b for j, b in enumerate(bits) if b is not None)
        assert threaded[i] == vector[source]
