# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for vfhe.crypto.digest: the chain h_i = H(h_{i-1} | part_i) from the
seed H("vfhe.chain"), part by part and over packed runs.
"""

import pytest
from vfhe.crypto import DigestChain, chain_digest, entropy
from vfhe.crypto.merkle import hash_bytes


def reference_chain(parts) -> bytes:
    state = hash_bytes(b"vfhe.chain")
    for part in parts:
        state = hash_bytes(state + part)
    return state


# 32 is the last length whose link fits one BLAKE3 block, 33 the first that
# does not.
@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 64, 100, 2000])
def test_chain_matches_its_definition(length):
    parts = [entropy.bytes(length) for _ in range(5)]
    assert chain_digest(parts) == reference_chain(parts)
    if length:
        packed = DigestChain().absorb_packed(b"".join(parts), length)
        assert packed.digest() == reference_chain(parts)


def test_absorbing_a_run_is_absorbing_its_parts():
    parts = [entropy.bytes(32) for _ in range(300)]
    chain = DigestChain().absorb(b"tag").absorb_packed(bytearray(b"".join(parts)))
    assert chain.digest() == chain_digest([b"tag", *parts])


def test_any_bytes_like_part():
    part = entropy.bytes(40)
    expected = reference_chain([part])
    assert DigestChain().absorb(bytearray(part)).digest() == expected
    assert DigestChain().absorb(memoryview(part)).digest() == expected


def test_empty_chain_and_empty_run_are_the_seed():
    seed = hash_bytes(b"vfhe.chain")
    assert DigestChain().digest() == seed
    assert DigestChain().absorb_packed(b"").digest() == seed


def test_a_run_must_be_whole_parts():
    with pytest.raises(ValueError):
        DigestChain().absorb_packed(bytes(33))
