# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The digest chain: the chain h_i = H(h_{i-1} | part_i) from the
seed H("vfhe.chain"), part by part and over packed runs.
"""

import pytest
from vfhe.crypto.hash import DIGEST_LEN, DigestChain, digest
from vfhe.crypto.random import Random


def reference_chain(parts) -> bytes:
    state = digest(b"vfhe.chain")
    for part in parts:
        state = digest(state + part)
    return state


# 32 is the last length whose link fits one BLAKE3 block, 33 the first that
# does not.
@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 64, 100, 2000])
def test_chain_matches_its_definition(length):
    parts = [Random().bytes(length) for _ in range(5)]
    assert DigestChain(parts).digest() == reference_chain(parts)


def test_absorbing_packed_digests_is_absorbing_each():
    parts = [Random().bytes(DIGEST_LEN) for _ in range(300)]
    chain = DigestChain().absorb(b"tag").absorb_packed(bytearray(b"".join(parts)))
    assert chain.digest() == DigestChain([b"tag", *parts]).digest()


def test_any_bytes_like_part():
    part = Random().bytes(40)
    expected = reference_chain([part])
    assert DigestChain().absorb(bytearray(part)).digest() == expected
    assert DigestChain().absorb(memoryview(part)).digest() == expected


def test_empty_chain_and_no_digests_are_the_seed():
    seed = digest(b"vfhe.chain")
    assert DigestChain().digest() == seed
    assert DigestChain().absorb_packed(b"").digest() == seed


def test_packed_digests_must_be_whole():
    with pytest.raises(ValueError):
        DigestChain().absorb_packed(bytes(33))


def test_digest_matches_the_published_vectors():
    assert (
        digest(b"").hex()
        == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    assert (
        digest(b"abc").hex()
        == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    )
