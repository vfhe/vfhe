# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Serialization of a CKKS ciphertext and a relinearization key, at the CKKS shapes.

Each benchmark writes to or reads from an in-memory byte stream with the
codecs' default options: `CKKSCiphertextCodec` for the ciphertext, and a
list of `KeySwitchKeyCodec` over `SampleCodec` for the relinearization key,
which holds one key-switch key per level.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from harness import Case, Shape, benchmark
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Ciphertext, CKKS_Scheme
from vfhe.fhe.codec import CKKSCiphertextCodec, CKKSSchemeCodec
from vfhe.mlwe.codec import KeySwitchKeyCodec, SampleCodec
from vfhe.util.io import ListCodec

if TYPE_CHECKING:
    from vfhe.util.io import Codec

SHAPES = ["ckks-n12-3p", "ckks-n13-7p", "ckks-n14-7p"]


def scheme_and_key(shape: Shape) -> tuple[CKKS_Scheme, Any]:
    """Returns the shape's CKKS scheme and a secret key."""
    n = 1 << shape["log_n"]
    ring = Ring(
        n, prime_size=shape["prime_bits"] + shape["special_bits"], split_degree=1
    )
    scheme = CKKS_Scheme(
        ring,
        scaling_factor=2.0 ** shape["scale_bits"],
        special_primes=len(shape["special_bits"]),
    )
    return scheme, scheme.key_gen_sparse(n // 8, 3.2)


def ciphertext(shape: Shape) -> tuple[CKKS_Scheme, CKKS_Ciphertext]:
    """Returns the shape's scheme and an encryption of a constant vector."""
    scheme, key = scheme_and_key(shape)
    return scheme, scheme.encrypt(scheme.encode([0.5] * (scheme.N // 2)), key)


def same_ciphertext(a: CKKS_Ciphertext, b: CKKS_Ciphertext) -> bool:
    return a.get_b_poly() == b.get_b_poly() and all(
        a.get_a_poly(j) == b.get_a_poly(j) for j in range(a.r)
    )


def encode(codec: Codec[Any], obj: Any) -> bytes:
    """Writes ``obj`` with ``codec`` into a new in-memory stream; returns its bytes."""
    stream = io.BytesIO()
    codec.encode(obj, stream)
    return stream.getvalue()


def decode(codec: Codec[Any], data: bytes) -> Any:
    """Reads one object with ``codec`` from ``data``."""
    return codec.decode(io.BytesIO(data))


def ciphertext_codec(scheme: CKKS_Scheme) -> Codec[Any]:
    return CKKSCiphertextCodec(CKKSSchemeCodec(scheme))


def key_codec(scheme: CKKS_Scheme) -> Codec[Any]:
    return ListCodec(KeySwitchKeyCodec(SampleCodec(CKKSSchemeCodec(scheme))))


@benchmark(shapes=SHAPES)
def serialize_ciphertext(shape: Shape) -> Case:
    """Writing one fresh CKKS ciphertext at the top level."""
    scheme, original = ciphertext(shape)
    codec = ciphertext_codec(scheme)

    def check(data: bytes) -> None:
        assert same_ciphertext(decode(codec, data), original)

    return Case(lambda: encode(codec, original), check=check)


@benchmark(shapes=SHAPES)
def deserialize_ciphertext(shape: Shape) -> Case:
    """Reading one fresh CKKS ciphertext at the top level."""
    scheme, original = ciphertext(shape)
    codec = ciphertext_codec(scheme)
    data = encode(codec, original)

    def check(result: CKKS_Ciphertext) -> None:
        assert same_ciphertext(result, original)

    return Case(lambda: decode(codec, data), check=check)


@benchmark(shapes=SHAPES)
def serialize_relinearization_key(shape: Shape) -> Case:
    """Writing the relinearization key."""
    scheme, key = scheme_and_key(shape)
    rlk = scheme.gen_rlk(key, key)
    codec = key_codec(scheme)
    expected = encode(codec, rlk)

    def check(data: bytes) -> None:
        assert data == expected

    return Case(lambda: encode(codec, rlk), check=check)


@benchmark(shapes=SHAPES)
def deserialize_relinearization_key(shape: Shape) -> Case:
    """Reading the relinearization key."""
    scheme, key = scheme_and_key(shape)
    rlk = scheme.gen_rlk(key, key)
    codec = key_codec(scheme)
    data = encode(codec, rlk)

    def check(result: Any) -> None:
        assert encode(codec, result) == data

    return Case(lambda: decode(codec, data), check=check)
