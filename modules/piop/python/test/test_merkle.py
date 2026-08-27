# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Tests for vfhe.piop.merkle: the C-backed BLAKE3 Merkle tree, its openings,
and the leaf-hashing contract (`.hash()` or an explicit `hash=` callable).
"""

import random

import pytest
from vfhe.arith import Field, FieldElement, Polynomial, Ring
from vfhe.piop import Merkle, MerklePath
from vfhe.piop.merkle import DIGEST_LEN, hash_bytes, leaf_digest


class HashableLeaf:
    """The contract a leaf type must satisfy: a `.hash()` returning bytes."""

    def __init__(self, value: bytes):
        self.value = value

    def hash(self) -> bytes:
        return hash_bytes(self.value)


def random_leaves(count: int) -> list[HashableLeaf]:
    return [HashableLeaf(random.randbytes(48)) for _ in range(count)]


@pytest.mark.parametrize("size", [1, 2, 8, 64])
def test_open_verifies_every_leaf(size):
    leaves = random_leaves(size)
    tree = Merkle(leaves)
    assert len(tree) == size
    assert len(tree.root) == DIGEST_LEN
    for index, leaf in enumerate(leaves):
        path = tree.open(index)
        assert len(path) == max(0, (size - 1).bit_length())
        assert Merkle.verify(tree.root, index, path, leaf)


def test_verify_rejects_tampering():
    leaves = random_leaves(16)
    tree = Merkle(leaves)
    index = 5
    path = tree.open(index)
    assert Merkle.verify(tree.root, index, path, leaves[index])

    # wrong leaf, wrong position, wrong sibling, wrong root
    assert not Merkle.verify(tree.root, index, path, leaves[index + 1])
    assert not Merkle.verify(tree.root, index + 1, path, leaves[index])
    tampered = list(path.siblings)
    tampered[0] = bytes(b ^ 1 for b in tampered[0])
    assert not Merkle.verify(tree.root, index, MerklePath(tampered), leaves[index])
    assert not Merkle.verify(bytes(DIGEST_LEN), index, path, leaves[index])


def test_root_is_deterministic_and_binding():
    leaves = random_leaves(8)
    assert Merkle(leaves).root == Merkle(list(leaves)).root
    other = list(leaves)
    other[3] = HashableLeaf(other[3].value + b"!")
    assert Merkle(other).root != Merkle(leaves).root
    swapped = list(leaves)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert Merkle(swapped).root != Merkle(leaves).root


def test_root_matches_hand_computed_tree():
    leaves = random_leaves(4)
    digests = [leaf.hash() for leaf in leaves]
    left = hash_bytes(digests[0] + digests[1])
    right = hash_bytes(digests[2] + digests[3])
    assert Merkle(leaves).root == hash_bytes(left + right)


def test_non_power_of_two_pads_with_zero_digests():
    leaves = random_leaves(3)
    padded = [*leaves, HashableLeaf(b"")]
    # the pad leaf's own hash is irrelevant: only its digest reaches the tree
    tree = Merkle(leaves)
    reference = Merkle(
        padded,
        hash=lambda leaf: bytes(DIGEST_LEN) if leaf is padded[3] else leaf.hash(),
    )
    assert tree.root == reference.root
    for index, leaf in enumerate(leaves):
        assert Merkle.verify(tree.root, index, tree.open(index), leaf)


def test_recommit_reuses_the_tree():
    leaves = random_leaves(8)
    tree = Merkle(leaves)
    first = tree.root
    tree.leaves[2] = HashableLeaf(b"replaced")
    second = tree.commit()
    assert second != first
    assert tree.root == second
    assert Merkle.verify(second, 2, tree.open(2), tree.leaves[2])


def test_explicit_hash_callable_and_missing_hash_method():
    leaves = [random.randbytes(32) for _ in range(8)]
    with pytest.raises(TypeError, match=r"no \.hash\(\) method"):
        Merkle(leaves)
    tree = Merkle(leaves, hash=hash_bytes)
    assert Merkle.verify(tree.root, 3, tree.open(3), leaves[3], hash=hash_bytes)
    # the same leaf under a different hash must not verify
    assert not Merkle.verify(
        tree.root, 3, tree.open(3), leaves[3], hash=lambda x: hash_bytes(x + b"x")
    )


def test_library_leaf_types():
    field = Field((1 << 61) - 1, 4, 3)
    elements = [FieldElement(field, [i + 1, i + 2, i + 3, i + 4]) for i in range(8)]
    tree = Merkle(elements)  # FieldElement.hash() satisfies the contract
    assert Merkle.verify(tree.root, 4, tree.open(4), elements[4])

    ring = Ring(1024, prime_size=[49], split_degree=4)
    polys = [Polynomial(ring).from_array([i + 1, i + 2]) for i in range(8)]
    # Polynomial has no .hash(); its get_hash() returns four 64-bit words
    tree = Merkle(polys, hash=lambda p: p.get_hash())
    assert Merkle.verify(
        tree.root, 6, tree.open(6), polys[6], hash=lambda p: p.get_hash()
    )


def test_rejects_bad_hash_values_and_empty_leaves():
    with pytest.raises(ValueError, match="at least one leaf"):
        Merkle([])
    with pytest.raises(ValueError, match=f"{DIGEST_LEN} bytes"):
        Merkle([b"short"], hash=lambda leaf: leaf)
    with pytest.raises(TypeError, match="64-bit words"):
        Merkle([1], hash=lambda leaf: leaf)
    assert leaf_digest(b"", hash=hash_bytes) == hash_bytes(b"")
