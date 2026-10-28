# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The use case the tests of this module read: ``(a * b) * (a * b)`` under
a CKKS scheme with or without special primes, recorded on the tape."""

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme
from vfhe.util import trace

N = 256


@pytest.fixture(autouse=True)
def a_clean_tape():
    yield
    trace.enable().clear()


def _use_case(special_primes):
    primes = [45, 45, 45, 50] if special_primes else [45, 45, 45]
    scheme = CKKS_Scheme(
        Ring(N, prime_size=primes, split_degree=1),
        scaling_factor=2**40,
        special_primes=special_primes,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])
    slots = N // 2
    a = scheme.encrypt(scheme.encode([1.5, -2.0] + [0.0] * (slots - 2)), key)
    b = scheme.encrypt(scheme.encode([0.5, 1.0] + [0.0] * (slots - 2)), key)
    with trace.recording() as t:
        c = a * b
        d = c * c
    return scheme, a, b, c, d, t.operations()


@pytest.fixture
def use_case():
    """The scheme, the operands, the products and the trace's operations."""
    return _use_case
