# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Serialization of a CKKS ciphertext and a relinearization key, at the CKKS parameters.

Each benchmark writes to or reads from an in-memory byte stream with the
codecs' default options: `CKKSCiphertextCodec` for the ciphertext, and a
list of `KeySwitchKeyCodec` over `SampleCodec` for the relinearization key,
which holds one key-switch key per level.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme
from vfhe.fhe.codec import CKKSCiphertextCodec, CKKSSchemeCodec
from vfhe.mlwe.codec import KeySwitchKeyCodec, SampleCodec
from vfhe.util.io import ListCodec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from vfhe.fhe import CKKS_Ciphertext
    from vfhe.util.io import Codec

#: The CKKS chains arith's polynomial benchmarks run at: ring degree 2^log_n,
#: the chain's prime sizes from level 0 up, the primes key switching adds, and
#: the encoding scale.
CKKS = {
    "n12-3p": {
        "log_n": 12,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n13-7p": {
        "log_n": 13,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n14-7p": {
        "log_n": 14,
        "prime_bits": [60] + [40] * 6,
        "special_bits": [60],
        "scale_bits": 40,
    },
}
chains = pytest.mark.parametrize("p", list(CKKS.values()), ids=list(CKKS))


def _scheme_and_key(p: dict) -> tuple[CKKS_Scheme, Any]:
    ring = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    scheme = CKKS_Scheme(
        ring,
        scaling_factor=2.0 ** p["scale_bits"],
        special_primes=len(p["special_bits"]),
    )
    return scheme, scheme.key_gen_sparse(scheme.N // 8, 3.2)


def _ciphertext(p: dict) -> tuple[CKKS_Scheme, CKKS_Ciphertext]:
    """The scheme and an encryption of a constant vector."""
    scheme, key = _scheme_and_key(p)
    return scheme, scheme.encrypt(scheme.encode([0.5] * (scheme.N // 2)), key)


def _same(a: CKKS_Ciphertext, b: CKKS_Ciphertext) -> bool:
    return a.get_b_poly() == b.get_b_poly() and all(
        a.get_a_poly(j) == b.get_a_poly(j) for j in range(a.r)
    )


def _encode(codec: Codec[Any], obj: Any) -> bytes:
    stream = io.BytesIO()
    codec.encode(obj, stream)
    return stream.getvalue()


def _decode(codec: Codec[Any], data: bytes) -> Any:
    return codec.decode(io.BytesIO(data))


def _ciphertext_codec(scheme: CKKS_Scheme) -> Codec[Any]:
    return CKKSCiphertextCodec(CKKSSchemeCodec(scheme))


def _key_codec(scheme: CKKS_Scheme) -> Codec[Any]:
    return ListCodec(KeySwitchKeyCodec(SampleCodec(CKKSSchemeCodec(scheme))))


@chains
def serialize_ciphertext(benchmark: BenchmarkFixture, p: dict) -> None:
    """Writing one fresh CKKS ciphertext at the top level."""
    scheme, original = _ciphertext(p)
    codec = _ciphertext_codec(scheme)
    assert _same(_decode(codec, benchmark(_encode, codec, original)), original)


@chains
def deserialize_ciphertext(benchmark: BenchmarkFixture, p: dict) -> None:
    """Reading one fresh CKKS ciphertext at the top level."""
    scheme, original = _ciphertext(p)
    codec = _ciphertext_codec(scheme)
    data = _encode(codec, original)
    assert _same(benchmark(_decode, codec, data), original)


@chains
def serialize_relinearization_key(benchmark: BenchmarkFixture, p: dict) -> None:
    """Writing the relinearization key."""
    scheme, key = _scheme_and_key(p)
    rlk = scheme.gen_rlk(key, key)
    codec = _key_codec(scheme)
    assert benchmark(_encode, codec, rlk) == _encode(codec, rlk)


@chains
def deserialize_relinearization_key(benchmark: BenchmarkFixture, p: dict) -> None:
    """Reading the relinearization key."""
    scheme, key = _scheme_and_key(p)
    rlk = scheme.gen_rlk(key, key)
    codec = _key_codec(scheme)
    data = _encode(codec, rlk)
    assert _encode(codec, benchmark(_decode, codec, data)) == data
