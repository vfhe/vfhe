# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`Random`: the entropy stream, unpredictable by construction, so what can be
checked is its shape, its unbiased rejection loop, and that threads and forks
never meet; and the seeded stream, a function of its seed."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from vfhe.crypto.random import Random

entropy = Random()


def test_bytes_have_the_right_shape_and_differ():
    assert entropy.bytes(0) == b""
    first, second = entropy.bytes(64), entropy.bytes(64)
    assert len(first) == len(second) == 64
    assert first != second
    # larger than the 1 KiB pool: served from a fresh seed
    assert len(entropy.bytes(4096)) == 4096
    with pytest.raises(ValueError, match="non-negative"):
        entropy.bytes(-1)


def test_below_stays_in_range_and_spreads():
    assert entropy.below(1) == 0
    assert all(0 <= entropy.below(3) < 3 for _ in range(64))
    values = [entropy.below((1 << 61) - 1) for _ in range(64)]
    assert len(set(values)) == len(values)
    with pytest.raises(ValueError, match="bound must be in"):
        entropy.below(0)


def test_below_many_covers_its_range_evenly():
    assert entropy.below_many(0, 3) == []
    assert entropy.below_many(5, 1) == [0] * 5
    values = entropy.below_many(30000, 3)
    assert set(values) == {0, 1, 2}
    # Each count is 10000 +- 82 (one standard deviation).
    assert all(abs(values.count(v) - 10000) < 500 for v in range(3))
    with pytest.raises(ValueError, match="bound"):
        entropy.below_many(4, 0)
    with pytest.raises(ValueError, match="count"):
        entropy.below_many(-1, 3)


def test_bounds_at_64_bits():
    assert entropy.below((1 << 64) - 1) < (1 << 64) - 1
    assert all(v < (1 << 64) - 1 for v in entropy.below_many(4, (1 << 64) - 1))
    with pytest.raises(ValueError, match=r"\[1, 2\*\*64\)"):
        entropy.below_many(4, 1 << 64)


def test_bytes_of_every_size_around_the_pool():
    for size in (1, 7, 511, 512, 513, 1023, 1024, 1025, 4096):
        first, second = entropy.bytes(size), entropy.bytes(size)
        assert len(first) == len(second) == size
        if size >= 8:
            assert first != second


def test_several_threads_never_repeat():
    """Draws made on several threads at once -- cffi releases the GIL around
    each -- never meet, through the pool or past it."""

    def draw(_):
        return [entropy.bytes(16) for _ in range(256)] + [entropy.bytes(4096)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        draws = [chunk for chunks in pool.map(draw, range(8)) for chunk in chunks]
    words = [chunk[i : i + 8] for chunk in draws for i in range(0, len(chunk), 8)]
    assert len(set(words)) == len(words)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_does_not_repeat_its_parent():
    """The child inherits the parent's pool and must not serve it again."""
    entropy.bytes(16)
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child reports through the pipe
        try:
            os.write(write_end, entropy.bytes(32))
        finally:
            os._exit(0)
    os.close(write_end)
    parent = entropy.bytes(32)
    with os.fdopen(read_end, "rb") as child_out:
        child = child_out.read()
    os.waitpid(pid, 0)
    assert len(child) == 32
    assert child != parent


def test_a_seeded_source_replays_every_entry_point():
    def draws(source: Random) -> tuple:
        return (
            source.bytes(100),
            source.bytes(600),
            source.below_many(8, 1000),
            source.below(1 << 60),
        )

    seeded = draws(Random(b"seed"))
    assert draws(Random(b"seed")) == seeded
    assert draws(Random(b"other")) != seeded
    assert draws(entropy) != seeded
