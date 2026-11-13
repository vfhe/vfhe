# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The C-backed BLAKE3 Merkle tree over packed leaf digests, its openings, and `join_digests`."""

import pytest
from vfhe.crypto.hash import DIGEST_LEN, digest
from vfhe.crypto.merkle import Merkle, join_digests
from vfhe.crypto.random import Random


def random_digests(count: int) -> list[bytes]:
    return [digest(Random().bytes(48)) for _ in range(count)]


@pytest.mark.parametrize("size", [1, 2, 3, 8, 17, 64])
def test_open_verifies_every_leaf(size):
    digests = random_digests(size)
    tree = Merkle(b"".join(digests))
    assert len(tree) == size and len(tree.root) == DIGEST_LEN
    height = (size - 1).bit_length()
    for index, leaf in enumerate(digests):
        path = tree.open(index)
        assert len(path) == height * DIGEST_LEN
        assert Merkle.verify(tree.root, index, path, leaf)


def test_verify_rejects_tampering():
    digests = random_digests(16)
    tree = Merkle(b"".join(digests))
    index = 5
    path = tree.open(index)
    assert Merkle.verify(tree.root, index, path, digests[index])
    # wrong leaf, wrong position, wrong sibling, wrong root
    assert not Merkle.verify(tree.root, index, path, digests[index + 1])
    assert not Merkle.verify(tree.root, index + 1, path, digests[index])
    assert not Merkle.verify(
        tree.root, index, bytes([path[0] ^ 1]) + path[1:], digests[index]
    )
    assert not Merkle.verify(bytes(DIGEST_LEN), index, path, digests[index])


def test_verify_rejects_an_index_beyond_the_tree():
    # A path of h siblings addresses 2^h leaves; a larger index
    # must not verify by its low bits alone.
    digests = random_digests(4)
    tree = Merkle(b"".join(digests))
    path = tree.open(1)
    assert Merkle.verify(tree.root, 1, path, digests[1])
    for index in (5, 9, 1 + (4 << 40), 2**64 - 3):
        assert not Merkle.verify(tree.root, index, path, digests[1])


def test_root_is_deterministic_and_binding():
    digests = random_digests(8)
    assert Merkle(b"".join(digests)).root == Merkle(b"".join(digests)).root
    other = list(digests)
    other[3] = digest(other[3])
    assert Merkle(b"".join(other)).root != Merkle(b"".join(digests)).root
    swapped = [digests[1], digests[0], *digests[2:]]
    assert Merkle(b"".join(swapped)).root != Merkle(b"".join(digests)).root


def test_root_matches_hand_computed_tree():
    digests = random_digests(4)
    left = digest(digests[0] + digests[1])
    right = digest(digests[2] + digests[3])
    assert Merkle(b"".join(digests)).root == digest(left + right)


def test_non_power_of_two_pads_with_zero_digests():
    digests = random_digests(3)
    tree = Merkle(b"".join(digests))
    assert tree.root == Merkle(b"".join([*digests, bytes(DIGEST_LEN)])).root


def test_a_single_leaf_is_its_own_root():
    (digest,) = random_digests(1)
    tree = Merkle(digest)
    assert tree.root == digest and tree.open(0) == b""
    assert Merkle.verify(tree.root, 0, b"", digest)


def test_any_bytes_like_buffer_is_accepted_and_read_once():
    packed = b"".join(random_digests(8))
    roots = {
        Merkle(form).root for form in (packed, bytearray(packed), memoryview(packed))
    }
    assert len(roots) == 1
    buffer = bytearray(packed)
    tree = Merkle(buffer)
    buffer[0] ^= 0xFF
    assert tree.root == Merkle(packed).root  # the tree does not watch the buffer
    assert Merkle.verify(
        memoryview(tree.root), 2, bytearray(tree.open(2)), memoryview(packed)[64:96]
    )


def test_opening_past_the_last_leaf_is_refused():
    tree = Merkle(b"".join(random_digests(5)))
    with pytest.raises(IndexError):
        tree.open(5)
    with pytest.raises(IndexError):
        tree.open(-1)


def test_bad_sizes_are_refused():
    for bad in (b"", b"\x00" * 31, b"\x00" * 33):
        with pytest.raises(ValueError, match="whole digests"):
            Merkle(bad)
    tree = Merkle(b"".join(random_digests(4)))
    path = tree.open(1)
    for root, leaf, opening in (
        (b"short", bytes(DIGEST_LEN), path),
        (tree.root, b"short", path),
        (tree.root, bytes(DIGEST_LEN), path[:-1]),
    ):
        with pytest.raises(ValueError, match="path whole digests"):
            Merkle.verify(root, 1, opening, leaf)


def test_join_digests_hashes_each_index_across_the_arrays():
    parts = [bytes(range(i, i + 96)) for i in (0, 7, 50)]
    joined = join_digests(parts)
    assert len(joined) == 96
    for j in range(3):
        expected = digest(b"".join(part[32 * j : 32 * (j + 1)] for part in parts))
        assert joined[32 * j : 32 * (j + 1)] == expected
    # Read-only views are accepted, as a kernel's digest buffer is one.
    assert join_digests([memoryview(part) for part in parts]) == joined


def test_join_digests_rejects_unequal_arrays():
    with pytest.raises(ValueError, match="at least one"):
        join_digests([])
    with pytest.raises(ValueError, match="same number"):
        join_digests([bytes(DIGEST_LEN), bytes(2 * DIGEST_LEN)])
    with pytest.raises(ValueError, match="same number"):
        join_digests([bytes(DIGEST_LEN + 1)])
