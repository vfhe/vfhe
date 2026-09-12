# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The spot-check sampler on its own, away from any scheme.

`test_basefold.py` covers the geometry basefold passes it; what is checked
here is that the sampler is a function of `(span, shift, rep)` and nothing
else -- the point of it being a free function rather than a method.
"""

from __future__ import annotations

import pytest
from vfhe.polycom import query_positions

SEED = bytes(range(32))


@pytest.mark.parametrize(
    ("span", "shift", "rep"),
    [
        (1 << 10, 0, 7),  # no walk: the positions themselves must differ
        (1 << 10, 4, 12),  # one fold per level, basefold's shape
        (1 << 16, 9, 40),  # several folds per published oracle
        (1 << 4, 4, 1),  # a single endpoint, and exactly one query
    ],
)
def test_positions_are_in_range_and_end_apart(span, shift, rep):
    positions = query_positions(SEED, span, shift, rep)
    assert len(positions) == rep
    assert all(0 <= q < span for q in positions)
    ends = [q >> shift for q in positions]
    assert len(set(ends)) == rep
    # Distinct endpoints imply distinct positions at every height above.
    for height in range(shift + 1):
        above = [q >> height for q in positions]
        assert len(set(above)) == rep


def test_is_a_pure_function_of_the_seed():
    assert query_positions(SEED, 1 << 12, 3, 8) == query_positions(SEED, 1 << 12, 3, 8)
    assert query_positions(bytes(32), 1 << 12, 3, 8) != query_positions(
        SEED, 1 << 12, 3, 8
    )


def test_the_prefix_grows_rather_than_changes():
    """Raising `rep` extends the sequence: a candidate is accepted or
    rejected on what came before it, never on how many were asked for."""
    many = query_positions(SEED, 1 << 12, 3, 16)
    assert query_positions(SEED, 1 << 12, 3, 4) == many[:4]


def test_a_larger_span_at_the_same_shift_is_not_the_same_draw():
    """The mask is part of the derivation, so the geometry is bound into the
    positions and two oracles of different lengths do not share them."""
    assert query_positions(SEED, 1 << 12, 3, 8) != query_positions(SEED, 1 << 13, 3, 8)


def test_rep_beyond_the_endpoints_is_refused_rather_than_looping():
    with pytest.raises(ValueError, match="end apart"):
        query_positions(SEED, 1 << 10, 4, 65)  # only 64 endpoints exist
    # ...and the boundary case is legal, since it terminates.
    assert len(query_positions(SEED, 1 << 10, 4, 64)) == 64


@pytest.mark.parametrize(
    ("span", "shift", "rep"),
    [
        (0, 0, 1),
        (-8, 0, 1),
        (100, 0, 1),
        (1 << 8, 9, 1),
        (1 << 8, -1, 1),
        (1 << 8, 0, 0),
    ],
)
def test_the_geometry_is_checked(span, shift, rep):
    with pytest.raises(ValueError):
        query_positions(SEED, span, shift, rep)
