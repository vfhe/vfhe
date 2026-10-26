# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""CKKS operations at the CKKS shapes: products, rotations, encoding, and SlotToCoeff.

Ciphertexts are fresh encryptions at the top level, and keys are generated for that
level only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness import Case, Shape, benchmark
from vfhe.arith import Ring, RNSPolynomial
from vfhe.fhe import CKKS_Ciphertext, CKKS_LinearTransform, CKKS_Scheme

if TYPE_CHECKING:
    from vfhe.mlwe import MLWE_Key

SHAPES = ["ckks-n12-3p", "ckks-n13-7p", "ckks-n14-7p"]

#: Chains with few primes for SlotToCoeff, whose rotation keys grow with the
#: number of primes.
SLOT_TO_COEFF_SHAPES = [
    Shape(
        "n10-3p", log_n=10, prime_bits=[60, 40, 40], special_bits=[60], scale_bits=40
    ),
    Shape(
        "n13-3p", log_n=13, prime_bits=[60, 40, 40], special_bits=[60], scale_bits=40
    ),
]

# How close a decrypted slot must be to its value.
TOLERANCE = 1e-3


def scheme_of(shape: Shape) -> CKKS_Scheme:
    """Returns the scheme over the shape's chain and special primes."""
    ring = Ring(
        1 << shape["log_n"],
        prime_size=shape["prime_bits"] + shape["special_bits"],
        split_degree=1,
    )
    return CKKS_Scheme(
        ring,
        scaling_factor=2.0 ** shape["scale_bits"],
        special_primes=len(shape["special_bits"]),
    )


def slot_values(count: int, offset: int = 0) -> list[complex]:
    """Returns ``count`` fixed complex values of magnitude below 1."""
    return [
        complex(((i * 7919 + offset) % 1000) / 1000 - 0.5, ((i * 104729) % 1000) / 2000)
        for i in range(count)
    ]


def decoded(scheme: CKKS_Scheme, c: CKKS_Ciphertext, key: MLWE_Key) -> list[complex]:
    """Returns the slots of ``c``."""
    return scheme.decode(scheme.decrypt(c, key), c.delta)


def close(got: list[complex], expected: list[complex]) -> bool:
    """Reports whether the first slots agree within `TOLERANCE`."""
    return all(
        abs(g - e) < TOLERANCE for g, e in zip(got[:8], expected[:8], strict=True)
    )


@benchmark(shapes=SHAPES, tier="headline")
def ckks_multiply_rescale(shape: Shape) -> Case:
    """Two ciphertexts multiplied, relinearized, then rescaled to the level below."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    slots = scheme.N // 2
    x, y = slot_values(slots), slot_values(slots, offset=3)
    a = scheme.encrypt(scheme.encode(x), key)
    b = scheme.encrypt(scheme.encode(y), key)

    def check(result: CKKS_Ciphertext) -> None:
        assert result.lvl == 1
        assert close(
            decoded(scheme, result, key), [u * v for u, v in zip(x, y, strict=True)]
        )

    return Case(lambda: scheme.rescale(scheme.multiply(a, b, rlk)), check=check)


@benchmark(shapes=SHAPES, tier="headline")
def ckks_rotate(shape: Shape) -> Case:
    """The slots of one ciphertext rotated by one position."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    ksk = scheme.gen_ksk_automorphism(key, key, scheme.rotation_generator(1), lvl=0)
    slots = scheme.N // 2
    x = slot_values(slots)
    c = scheme.encrypt(scheme.encode(x), key)

    def check(result: CKKS_Ciphertext) -> None:
        assert close(decoded(scheme, result, key), x[1:] + x[:1])

    return Case(lambda: scheme.rotate(c, 1, ksk), check=check)


@benchmark(shapes=SHAPES)
def ckks_multiply_batch(shape: Shape) -> Case:
    """Eight products in one call, each relinearized and rescaled to the level below."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    slots = scheme.N // 2
    xs = [slot_values(slots, offset=k) for k in range(8)]
    ys = [slot_values(slots, offset=k + 8) for k in range(8)]
    lhs = [scheme.encrypt(scheme.encode(x), key) for x in xs]
    rhs = [scheme.encrypt(scheme.encode(y), key) for y in ys]

    def check(results: list[CKKS_Ciphertext]) -> None:
        expected = [u * v for u, v in zip(xs[0], ys[0], strict=True)]
        assert close(decoded(scheme, results[0], key), expected)

    return Case(
        lambda: scheme.multiply_batch(lhs, rhs, rlk, lvl=1), check=check, items=8
    )


@benchmark(shapes=SHAPES)
def ckks_encode(shape: Shape) -> Case:
    """``N/2`` complex values, given as a Python list, encoded at the top level."""
    scheme = scheme_of(shape)
    x = slot_values(scheme.N // 2)

    def check(result: RNSPolynomial) -> None:
        assert close(scheme.decode(result), x)

    return Case(lambda: scheme.encode(x), check=check)


@benchmark(shapes=SHAPES)
def ckks_decode(shape: Shape) -> Case:
    """A decrypted plaintext decoded into its ``N/2`` complex values, as a list."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    x = slot_values(scheme.N // 2)
    plaintext = scheme.decrypt(scheme.encrypt(scheme.encode(x), key), key)

    def check(result: list[complex]) -> None:
        assert close(result, x)

    return Case(lambda: scheme.decode(plaintext), check=check)


@benchmark(shapes=SLOT_TO_COEFF_SHAPES)
def ckks_slot_to_coeff(shape: Shape) -> Case:
    """SlotToCoeff of all ``N/2`` slots by baby-step giant-step [HS18], not rescaled."""
    scheme = scheme_of(shape)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    transform = CKKS_LinearTransform.slot_to_coeff(scheme)
    keys = {
        k: scheme.gen_ksk_automorphism(key, key, scheme.rotation_generator(k), lvl=0)
        for k in transform.rotations
    }
    half = scheme.N // 2
    x = slot_values(half)
    c = scheme.encrypt(scheme.encode(x), key)

    def check(result: CKKS_Ciphertext) -> None:
        m = scheme.decrypt(result, key).get_polynomial(signed=True)
        got = [complex(m[t], m[t + half]) / result.delta for t in range(8)]
        assert close(got, x)

    return Case(lambda: transform.apply(c, keys), check=check)
