# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The execution tape over CKKS: a product's tensor slots, digits and
rescales are on it, and what it recorded last is what the ciphertext holds."""

import pytest
from vfhe import arith
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme
from vfhe.util import trace

N = 256


@pytest.fixture(autouse=True)
def a_clean_tape():
    yield
    arith.set_generic_only(False)
    trace.enable().clear()


def test_two_products_leave_two_d2_and_their_rescales():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])
    slots = N // 2
    a = scheme.encrypt(scheme.encode([1.5, -2.0, 0.5, 3.0] + [0.0] * (slots - 4)), key)
    b = scheme.encrypt(scheme.encode([0.5, 1.0, -1.5, 2.0] + [0.0] * (slots - 4)), key)
    with trace.recording() as t:
        c = a * b
        d = c * c
    events = t.events()
    # d2 is the first slot of each tensor product: the first product of a
    # multiplication, whose operands are the two mask components.
    products = [e for e in events if e.kind == "mul"]
    assert products[0].value == a.get_a_poly(0) * b.get_a_poly(0)
    squares = [e for e in products if e.inputs[0] == e.inputs[1]]
    assert squares[0].value == c.get_a_poly(0) * c.get_a_poly(0)
    rescales = [e for e in events if e.kind == "round_division"]
    assert len(rescales) == 4
    assert [e.value for e in rescales[-2:]] == [d.get_a_poly(0), d.get_b_poly()]
    assert rescales[-1].ring == d.ring
    # The same products through the kernels.
    arith.set_generic_only(True)
    generic = (a * b) * (a * b)
    arith.set_generic_only(False)
    assert [generic.get_a_poly(0), generic.get_b_poly()] == [
        d.get_a_poly(0),
        d.get_b_poly(),
    ]
