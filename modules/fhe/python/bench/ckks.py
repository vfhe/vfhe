# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""CKKS operations at the CKKS parameters: products, rotations, encoding, and SlotToCoeff.

Ciphertexts are fresh encryptions at the top level, and keys are generated for that
level only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Ciphertext, CKKS_LinearTransform, CKKS_Scheme

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from vfhe.mlwe import MLWE_Key

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
#: Chains with few primes for SlotToCoeff, whose rotation keys grow with the
#: number of primes.
SLOT_TO_COEFF = {
    "n10-3p": {
        "log_n": 10,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
    "n13-3p": {
        "log_n": 13,
        "prime_bits": [60, 40, 40],
        "special_bits": [60],
        "scale_bits": 40,
    },
}

# How close a decrypted slot must be to its value.
TOLERANCE = 1e-3


def _scheme_of(p: dict) -> CKKS_Scheme:
    """The scheme over the chain and its special primes."""
    ring = Ring(
        1 << p["log_n"], prime_size=p["prime_bits"] + p["special_bits"], split_degree=1
    )
    return CKKS_Scheme(
        ring,
        scaling_factor=2.0 ** p["scale_bits"],
        special_primes=len(p["special_bits"]),
    )


def _slot_values(count: int, offset: int = 0) -> list[complex]:
    """``count`` fixed complex values of magnitude below 1."""
    return [
        complex(((i * 7919 + offset) % 1000) / 1000 - 0.5, ((i * 104729) % 1000) / 2000)
        for i in range(count)
    ]


def _decoded(scheme: CKKS_Scheme, c: CKKS_Ciphertext, key: MLWE_Key) -> list[complex]:
    return scheme.decode(scheme.decrypt(c, key), c.delta)


def _close(got: list[complex], expected: list[complex]) -> bool:
    """Whether the first slots agree within `TOLERANCE`."""
    return all(
        abs(g - e) < TOLERANCE for g, e in zip(got[:8], expected[:8], strict=True)
    )


@chains
def ckks_multiply_rescale(benchmark: BenchmarkFixture, p: dict) -> None:
    """Two ciphertexts multiplied, relinearized, then rescaled to the level below."""
    scheme = _scheme_of(p)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    x, y = _slot_values(scheme.N // 2), _slot_values(scheme.N // 2, offset=3)
    a = scheme.encrypt(scheme.encode(x), key)
    b = scheme.encrypt(scheme.encode(y), key)
    result = benchmark(lambda: scheme.rescale(scheme.multiply(a, b, rlk)))
    assert result.lvl == 1 and _close(
        _decoded(scheme, result, key), [u * v for u, v in zip(x, y, strict=True)]
    )


@chains
def ckks_rotate(benchmark: BenchmarkFixture, p: dict) -> None:
    """The slots of one ciphertext rotated by one position."""
    scheme = _scheme_of(p)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    ksk = scheme.gen_ksk_automorphism(key, key, scheme.rotation_generator(1), lvl=0)
    x = _slot_values(scheme.N // 2)
    c = scheme.encrypt(scheme.encode(x), key)
    assert _close(
        _decoded(scheme, benchmark(scheme.rotate, c, 1, ksk), key), x[1:] + x[:1]
    )


@chains
def ckks_multiply_batch(benchmark: BenchmarkFixture, p: dict) -> None:
    """Eight products in one call, each relinearized and rescaled to the level below."""
    scheme = _scheme_of(p)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    rlk = scheme.gen_rlk(key, key, lvl=0)
    xs = [_slot_values(scheme.N // 2, offset=k) for k in range(8)]
    ys = [_slot_values(scheme.N // 2, offset=k + 8) for k in range(8)]
    lhs = [scheme.encrypt(scheme.encode(x), key) for x in xs]
    rhs = [scheme.encrypt(scheme.encode(y), key) for y in ys]
    products = benchmark(scheme.multiply_batch, lhs, rhs, rlk, lvl=1)
    assert _close(
        _decoded(scheme, products[0], key),
        [u * v for u, v in zip(xs[0], ys[0], strict=True)],
    )


@chains
def ckks_encode(benchmark: BenchmarkFixture, p: dict) -> None:
    """``N/2`` complex values, given as a Python list, encoded at the top level."""
    scheme = _scheme_of(p)
    x = _slot_values(scheme.N // 2)
    assert _close(scheme.decode(benchmark(scheme.encode, x)), x)


@chains
def ckks_decode(benchmark: BenchmarkFixture, p: dict) -> None:
    """A decrypted plaintext decoded into its ``N/2`` complex values, as a list."""
    scheme = _scheme_of(p)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    x = _slot_values(scheme.N // 2)
    plaintext = scheme.decrypt(scheme.encrypt(scheme.encode(x), key), key)
    assert _close(benchmark(scheme.decode, plaintext), x)


@pytest.mark.parametrize("p", list(SLOT_TO_COEFF.values()), ids=list(SLOT_TO_COEFF))
def ckks_slot_to_coeff(benchmark: BenchmarkFixture, p: dict) -> None:
    """SlotToCoeff of all ``N/2`` slots by baby-step giant-step [HS18], not rescaled."""
    scheme = _scheme_of(p)
    key = scheme.key_gen_sparse(scheme.N // 8, 3.2)
    transform = CKKS_LinearTransform.slot_to_coeff(scheme)
    keys = {
        k: scheme.gen_ksk_automorphism(key, key, scheme.rotation_generator(k), lvl=0)
        for k in transform.rotations
    }
    half = scheme.N // 2
    x = _slot_values(half)
    c = scheme.encrypt(scheme.encode(x), key)
    result = benchmark(transform.apply, c, keys)
    m = scheme.decrypt(result, key).get_polynomial(signed=True)
    assert _close([complex(m[t], m[t + half]) / result.delta for t in range(8)], x)
