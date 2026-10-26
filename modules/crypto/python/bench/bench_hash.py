# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""BLAKE3, the digest chain, and Merkle trees over packed leaf digests."""

from __future__ import annotations

from harness import Case, Shape, benchmark
from vfhe.crypto import (
    DIGEST_LEN,
    Blake3Stream,
    DigestChain,
    Merkle,
    chain_digest,
    seeded,
)

#: The opened leaf: not the first, so the path is not all left siblings.
OPENED = 12345


def random_bytes(amount: int) -> bytes:
    """Returns ``amount`` reproducible bytes."""
    return seeded.bytes(amount, b"vfhe-bench", b"hash-input")


@benchmark(shapes=[Shape("b1048576", amount=1 << 20)])
def blake3_stream(shape: Shape) -> Case:
    """BLAKE3 of one buffer through `Blake3Stream`: a new stream, one update, the digest."""
    data = random_bytes(shape["amount"])
    halves = Blake3Stream()
    halves.update(data[: len(data) // 2])
    halves.update(data[len(data) // 2 :])
    expected = halves.digest()

    def op() -> bytes:
        stream = Blake3Stream()
        stream.update(data)
        return stream.digest()

    def check(result: bytes) -> None:
        assert result == expected

    return Case(op, check=check)


@benchmark(shapes=[Shape("parts4096", parts=1 << 12)])
def digest_chain(shape: Shape) -> Case:
    """A `DigestChain` absorbing digests packed end to end, as one native call."""
    parts = shape["parts"]
    packed = random_bytes(parts * DIGEST_LEN)
    expected = chain_digest(
        packed[i : i + DIGEST_LEN] for i in range(0, len(packed), DIGEST_LEN)
    )

    def check(result: bytes) -> None:
        assert result == expected

    return Case(
        lambda: DigestChain().absorb_packed(packed).digest(), check=check, items=parts
    )


@benchmark(shapes=[Shape("leaves65536", log_leaves=16)], tier="headline")
def merkle_commit(shape: Shape) -> Case:
    """A Merkle tree built over 2^log_leaves leaf digests packed end to end."""
    leaves = 1 << shape["log_leaves"]
    digests = random_bytes(leaves * DIGEST_LEN)
    leaf = digests[OPENED * DIGEST_LEN : (OPENED + 1) * DIGEST_LEN]

    def check(tree: Merkle) -> None:
        path = tree.open(OPENED)
        assert Merkle.verify(tree.root, OPENED, path, leaf, hash=lambda d: d)

    return Case(lambda: Merkle.from_digests(digests), check=check, items=leaves)


@benchmark(shapes=[Shape("leaves65536", log_leaves=16)])
def merkle_open_verify(shape: Shape) -> Case:
    """One leaf's opening, then its verification against the root."""
    leaves = 1 << shape["log_leaves"]
    digests = random_bytes(leaves * DIGEST_LEN)
    leaf = digests[OPENED * DIGEST_LEN : (OPENED + 1) * DIGEST_LEN]
    tree = Merkle.from_digests(digests)

    def op() -> bool:
        path = tree.open(OPENED)
        return Merkle.verify(tree.root, OPENED, path, leaf, hash=lambda d: d)

    def check(result: bool) -> None:
        assert result

    return Case(op, check=check)
