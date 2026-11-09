# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`crt_lift_mod_2_128`: integers from their residues modulo several primes,
against the Python CRT."""

from __future__ import annotations

import random
from array import array

import pytest
from vfhe.arith import crt, crt_lift_mod_2_128

#: One prime; two narrow ones; four wide ones whose product passes 2^128.
PRIMES = [
    [1073741827],
    [1073741827, 1073741831],
    [562949948178433, 1152921504577486849],
    [2**61 - 1, 2**62 - 57, 2**62 - 87, 2**60 - 93],
]


@pytest.mark.parametrize("primes", PRIMES)
@pytest.mark.parametrize("count", [1, 7, 64, 1000])
def test_lift_is_the_crt_modulo_2_128(primes, count):
    rng = random.Random(count + len(primes))  # noqa: S311 - test data
    q = 1
    for p in primes:
        q *= p
    values = [rng.randrange(q) for _ in range(count)]
    values[0] = q - 1
    if count > 1:
        values[1] = 0
    residues = [array("Q", [v % p for v in values]) for p in primes]
    low, high = crt_lift_mod_2_128(residues, primes)
    assert len(low) == len(high) == count
    for i, value in enumerate(values):
        lifted = low[i] | high[i] << 64
        assert lifted == value % (1 << 128)
        assert crt([r[i] for r in residues], primes) == value


def test_residues_need_not_be_reduced():
    primes = [1073741827, 562949948178433]
    values = [12345678901234567, 5, 2**60 + 11]
    residues = [
        array("Q", [v % p + p * (i % 3) for i, v in enumerate(values)]) for p in primes
    ]
    low, high = crt_lift_mod_2_128(residues, primes)
    assert [low[i] | high[i] << 64 for i in range(3)] == values
    assert all(h == 0 for h in high)


def test_lift_checks_its_arguments():
    good = [array("Q", [1, 2]), array("Q", [3, 4])]
    with pytest.raises(ValueError, match="distinct"):
        crt_lift_mod_2_128(good, [1073741827, 1073741827])
    with pytest.raises(ValueError, match="buffers for"):
        crt_lift_mod_2_128(good[:1], [1073741827, 1073741831])
    with pytest.raises(ValueError, match="differ in length"):
        crt_lift_mod_2_128([good[0], array("Q", [1])], [1073741827, 1073741831])
    with pytest.raises(TypeError, match="64-bit"):
        crt_lift_mod_2_128([array("I", [1, 2]), good[1]], [1073741827, 1073741831])
    low, high = crt_lift_mod_2_128([array("Q"), array("Q")], [1073741827, 1073741831])
    assert len(low) == len(high) == 0
