# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The digest chain and Merkle trees over packed leaf digests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.crypto.hash import DIGEST_LEN, DigestChain
from vfhe.crypto.merkle import Merkle
from vfhe.crypto.random import seeded_below_many

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

#: The opened leaf: not the first, so the path is not all left siblings.
OPENED = 12345


def _random_bytes(amount: int) -> bytes:
    """Returns ``amount`` reproducible bytes."""
    return bytes(seeded_below_many(amount, 256, b"vfhe-bench", b"hash-input"))


def digest_chain(benchmark: BenchmarkFixture) -> None:
    """A `DigestChain` absorbing 4096 digests packed end to end, as one native call."""
    packed = _random_bytes(4096 * DIGEST_LEN)
    expected = DigestChain(
        packed[i : i + DIGEST_LEN] for i in range(0, len(packed), DIGEST_LEN)
    ).digest()
    assert benchmark(lambda: DigestChain().absorb_packed(packed).digest()) == expected


def merkle_commit(benchmark: BenchmarkFixture) -> None:
    """A Merkle tree built over 2^16 leaf digests packed end to end."""
    digests = _random_bytes((1 << 16) * DIGEST_LEN)
    leaf = digests[OPENED * DIGEST_LEN : (OPENED + 1) * DIGEST_LEN]
    tree = benchmark(Merkle, digests)
    assert Merkle.verify(tree.root, OPENED, tree.open(OPENED), leaf)


def merkle_open_verify(benchmark: BenchmarkFixture) -> None:
    """One leaf's opening, then its verification against the root, in a 2^16-leaf tree."""
    digests = _random_bytes((1 << 16) * DIGEST_LEN)
    leaf = digests[OPENED * DIGEST_LEN : (OPENED + 1) * DIGEST_LEN]
    tree = Merkle(digests)
    assert benchmark(lambda: Merkle.verify(tree.root, OPENED, tree.open(OPENED), leaf))
