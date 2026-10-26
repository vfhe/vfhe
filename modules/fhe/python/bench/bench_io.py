# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Serialization of a CKKS ciphertext and a relinearization key, at the CKKS shapes.

The codecs are this package's, run through `vfhe.util.io` with the default
profile.
"""

from __future__ import annotations

from typing import Any

from harness import Case, Shape, benchmark
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Ciphertext, CKKS_Scheme
from vfhe.util.io import Serializer

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


@benchmark(shapes=SHAPES)
def serialize_ciphertext(shape: Shape) -> Case:
    """``Serializer().dumps`` of one fresh CKKS ciphertext at the top level."""
    scheme, original = ciphertext(shape)
    serializer = Serializer()

    def check(blob: bytes) -> None:
        assert same_ciphertext(serializer.loads(blob, schemes=scheme), original)

    return Case(lambda: serializer.dumps(original), check=check)


@benchmark(shapes=SHAPES)
def deserialize_ciphertext(shape: Shape) -> Case:
    """``Serializer().loads`` of one fresh CKKS ciphertext at the top level."""
    scheme, original = ciphertext(shape)
    serializer = Serializer()
    blob = serializer.dumps(original)

    def check(result: CKKS_Ciphertext) -> None:
        assert same_ciphertext(result, original)

    return Case(lambda: serializer.loads(blob, schemes=scheme), check=check)


@benchmark(shapes=SHAPES)
def serialize_relinearization_key(shape: Shape) -> Case:
    """``Serializer().dumps`` of the relinearization key."""
    scheme, key = scheme_and_key(shape)
    rlk = scheme.gen_rlk(key, key)
    serializer = Serializer()
    expected = serializer.dumps(rlk)

    def check(blob: bytes) -> None:
        assert blob == expected

    return Case(lambda: serializer.dumps(rlk), check=check)


@benchmark(shapes=SHAPES)
def deserialize_relinearization_key(shape: Shape) -> Case:
    """``Serializer().loads`` of the relinearization key."""
    scheme, key = scheme_and_key(shape)
    rlk = scheme.gen_rlk(key, key)
    serializer = Serializer()
    blob = serializer.dumps(rlk)

    def check(result: Any) -> None:
        assert serializer.dumps(result) == blob

    return Case(lambda: serializer.loads(blob, schemes=scheme), check=check)
