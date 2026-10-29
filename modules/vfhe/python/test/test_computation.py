# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The computation read from the tape of the use case: two multiplications
with their operands, slots, digits, key components and rescales, equal to
what the ciphertexts hold."""

import pytest
from vfhe.arith import Ring
from vfhe.mlwe import MLWE_Scheme
from vfhe.util import trace
from vfhe.vfhe.computation import Computation

N = 256


@pytest.mark.parametrize("special_primes", [0, 1])
def test_the_use_case_reads_as_two_multiplications(use_case, special_primes):
    scheme, a, b, c, d, operations = use_case(special_primes)
    computation = Computation(operations)
    first, second = computation.multiplications
    assert first.operands == [
        [a.get_a_poly(0), a.get_b_poly()],
        [b.get_a_poly(0), b.get_b_poly()],
    ]
    assert second.operands == [[c.get_a_poly(0), c.get_b_poly()]] * 2
    assert first.quadratic == a.get_a_poly(0) * b.get_a_poly(0)
    assert first.constant == a.get_b_poly() * b.get_b_poly()
    cross = a.get_a_poly(0) * b.get_b_poly() + a.get_b_poly() * b.get_a_poly(0)
    assert first.linear == cross
    switch = first.key_switch
    assert switch.primes == scheme.ring.primes
    assert [len(k) for k in switch.key] == [2, 2, 2]
    assert len(switch.mod_down) == 2 * bool(special_primes)
    assert all(not division.exact for division in switch.mod_down)
    primes = scheme.ring.primes
    assert [r.primes for r in first.rescales] == [(primes[-1],)] * 2
    assert [r.primes for r in second.rescales] == [(primes[-2],)] * 2
    assert [r.quotient for r in second.rescales] == [d.get_a_poly(0), d.get_b_poly()]
    assert all(r.exact and r.modulus == r.primes[0] for r in second.rescales)
    assert len(computation.rescales) == 4


def test_a_product_without_a_key_is_left_out_and_rank_two_refused():
    ring = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    scheme = MLWE_Scheme(ring, special_primes=0, module_rank=2)
    key = scheme.key_gen(3.2)
    rlk = scheme.gen_rlk(key, scheme.quadratic_key_polys(key))
    ct = scheme.sample(scheme.ring.random_element(), key)
    with trace.recording() as t:
        scheme.multiply(ct, ct, None)
    assert Computation(t.operations()).multiplications == []
    with trace.recording() as t:
        scheme.multiply(ct, ct, rlk)
    with pytest.raises(ValueError, match="rank 1"):
        Computation(t.operations())


def test_a_later_piece_of_the_tape_needs_the_earlier_values(use_case):
    _, a, b, _, _, operations = use_case(0)
    earlier = Computation(operations)
    with trace.recording() as t:
        _ = a * b  # the key's components were recorded in the first piece
    later = t.operations()
    with pytest.raises(ValueError, match="pass the earlier values as known"):
        Computation(later)
    [product] = Computation(later, known=earlier.values).multiplications
    assert product.operands == earlier.multiplications[0].operands
    assert product.key_switch.key == earlier.multiplications[0].key_switch.key
