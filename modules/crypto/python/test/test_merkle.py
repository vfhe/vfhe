# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for vfhe.crypto.merkle: the C-backed BLAKE3 Merkle tree, its openings,
and the leaf-hashing contract (`.hash()` or an explicit `hash=` callable).
"""

import pytest
from vfhe.arith import Field, FieldElement, Polynomial, Ring
from vfhe.crypto import Merkle, MerklePath, entropy, join_digests
from vfhe.crypto.merkle import DIGEST_LEN, hash_bytes, leaf_digest


class HashableLeaf:
    """The contract a leaf type must satisfy: a `.hash()` returning bytes."""

    def __init__(self, value: bytes):
        self.value = value

    def hash(self) -> bytes:
        return hash_bytes(self.value)


def random_leaves(count: int) -> list[HashableLeaf]:
    return [HashableLeaf(entropy.bytes(48)) for _ in range(count)]


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


@pytest.mark.parametrize("size", [1, 2, 8, 64])
def test_open_many_is_open_on_each_leaf_end_to_end(size):
    tree = Merkle(random_leaves(size))
    indices = [size - 1, 0, size // 2, 0]
    packed = tree.open_many(indices)
    assert packed == b"".join(tree.open(i).to_bytes() for i in indices)
    assert tree.open_many([]) == b""
    if size == 1:
        assert packed == b""


def test_verify_many_is_verify_on_each_leaf():
    leaves = random_leaves(32)
    tree = Merkle(leaves)
    indices = [3, 17, 3, 31, 0]
    paths = tree.open_many(indices)
    width = len(paths) // len(indices)
    digests = b"".join(leaf_digest(leaves[i]) for i in indices)
    assert Merkle.verify_many(tree.root, indices, paths, digests) is None
    assert Merkle.verify_many(tree.root, [], b"", b"") is None
    # The first leaf that does not authenticate, wherever it is.
    for bad in range(len(indices)):
        moved = list(indices)
        moved[bad] = (moved[bad] + 1) % 32
        assert Merkle.verify_many(tree.root, moved, paths, digests) == bad
        tampered = bytearray(paths)
        tampered[bad * width] ^= 1
        assert Merkle.verify_many(tree.root, indices, bytes(tampered), digests) == bad
        # One opening a sibling short: the buffer no longer splits into one
        # opening per index.
        short = paths[: bad * width] + paths[bad * width + DIGEST_LEN :]
        with pytest.raises(ValueError, match="do not split"):
            Merkle.verify_many(tree.root, indices, short, digests)
    with pytest.raises(ValueError, match="one 32-byte digest"):
        Merkle.verify_many(tree.root, indices, paths, digests[:-DIGEST_LEN])


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
    held = tree.leaves
    assert held is not None  # a tree built from leaves keeps them
    held[2] = HashableLeaf(b"replaced")
    second = tree.commit()
    assert second != first
    assert tree.root == second
    assert Merkle.verify(second, 2, tree.open(2), held[2])


def test_explicit_hash_callable_and_missing_hash_method():
    leaves = [entropy.bytes(32) for _ in range(8)]
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


class TestFromDigests:
    """A tree over leaf digests that were never Python objects.

    The point of the constructor is what it does *not* do, so the checks are
    that it agrees exactly with the tree built the ordinary way -- same root,
    same openings, same length -- for inputs where both are possible.
    """

    @staticmethod
    def digests(count: int) -> list[bytes]:
        return [hash_bytes(b"leaf-%d" % i) for i in range(count)]

    @pytest.mark.parametrize("count", [1, 2, 3, 8, 17])
    def test_agrees_with_the_object_path(self, count):
        leaves = self.digests(count)
        packed = Merkle.from_digests(b"".join(leaves))
        objects = Merkle(leaves, hash=lambda leaf: leaf)
        assert packed.root == objects.root
        assert len(packed) == len(objects) == count
        for index in range(count):
            assert packed.open(index).siblings == objects.open(index).siblings
            # the leaves are already digests, so verify needs the identity too
            assert Merkle.verify(
                packed.root, index, packed.open(index), leaves[index], hash=lambda x: x
            )

    def test_recommits_from_the_digests_it_is_given(self):
        """A tree over packed digests has no leaf list to walk, so it is
        recommitted with the new digests; the same digests give the same
        root, and other digests the root a fresh tree over them has."""
        leaves = self.digests(8)
        tree = Merkle.from_digests(b"".join(leaves))
        first = tree.root
        assert tree.commit(b"".join(leaves)) == first
        other = b"".join(self.digests(9)[1:])
        assert tree.commit(other) == Merkle.from_digests(other).root
        assert tree.root == Merkle.from_digests(other).root

    def test_recommitting_needs_digests_of_the_same_count(self):
        tree = Merkle.from_digests(b"".join(self.digests(8)))
        with pytest.raises(TypeError, match="new digests"):
            tree.commit()
        with pytest.raises(ValueError, match="8 digests"):
            tree.commit(b"".join(self.digests(7)))
        objects = Merkle(self.digests(8), hash=lambda leaf: leaf)
        with pytest.raises(TypeError, match="from its leaves"):
            objects.commit(b"".join(self.digests(8)))

    def test_any_bytes_like_buffer_is_accepted(self):
        """The producer hands over a read-only view of a C buffer, not
        `bytes`; a tree has to take that and anything else bytes-like, and
        give the same root for the same bytes."""
        leaves = self.digests(8)
        packed = b"".join(leaves)
        roots = {
            Merkle.from_digests(form).root
            for form in (packed, bytearray(packed), memoryview(packed))
        }
        assert len(roots) == 1

    def test_the_buffer_is_read_once_and_not_kept(self):
        """The C tree copies the digests while it is built, and the tree
        holds no reference to the buffer: changing the buffer afterwards
        changes neither the root nor an opening, and the buffer can be
        freed."""
        import gc
        import weakref

        leaves = self.digests(8)
        buffer = bytearray(b"".join(leaves))
        tree = Merkle.from_digests(buffer)
        first, path = tree.root, tree.open(3)
        buffer[0] ^= 0xFF
        assert tree.root == first
        assert tree.open(3).siblings == path.siblings

        class _Digests(bytearray):
            """A bytearray that can be weakly referenced."""

        held = _Digests(b"".join(leaves))
        alive = weakref.ref(held)
        tree = Merkle.from_digests(held)
        del held
        gc.collect()
        assert alive() is None
        assert tree.root == first

    def test_a_buffer_that_is_not_whole_digests_is_refused(self):
        for bad in (b"", b"\x00" * 31, b"\x00" * 33):
            with pytest.raises(ValueError, match="multiple of"):
                Merkle.from_digests(bad)


def test_join_digests_hashes_each_index_across_the_arrays():
    parts = [bytes(range(i, i + 96)) for i in (0, 7, 50)]
    joined = join_digests(parts)
    assert len(joined) == 96
    for j in range(3):
        expected = hash_bytes(b"".join(part[32 * j : 32 * (j + 1)] for part in parts))
        assert joined[32 * j : 32 * (j + 1)] == expected
    # Read-only views are accepted, as a kernel's digest buffer is one.
    assert join_digests([memoryview(part) for part in parts]) == joined
    with pytest.raises(ValueError):
        join_digests([parts[0], parts[1][:64]])
    with pytest.raises(ValueError):
        join_digests([])
