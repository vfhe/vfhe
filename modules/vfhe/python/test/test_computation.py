# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The computation read from the tape of the use case: two multiplications
with their operands, slots, digits, key components and rescales, equal to
what the ciphertexts hold; and the plaintext-linear steps, layers and
levels of a computation with products by plaintexts."""

from typing import cast

import pytest
from vfhe.arith import Ring
from vfhe.fhe import CKKS_Ciphertext, CKKS_Scheme
from vfhe.mlwe import MLWE_Scheme
from vfhe.util import trace
from vfhe.vfhe.computation import Computation, Division

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


def _components(ct):
    return [ct.get_a_poly(0), ct.get_b_poly()]


def test_plaintext_linear_steps_are_read_through_additions():
    ring = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    scheme = CKKS_Scheme(ring, scaling_factor=2**40)
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])
    rotation_key = scheme.gen_rotation_key(key, 1)
    a, b = (scheme.encrypt(scheme.encode([v]), key) for v in (1.0, 2.0))
    p, q = scheme.encode([0.5]), scheme.encode([0.25])
    r = scheme.encode([0.5], ring=scheme.rings[1])
    with trace.recording() as t:
        x = scheme.multiply_plain(a, p)
        [y] = scheme.linear_combinations([a, b], [[q, p]])
        z = cast("CKKS_Ciphertext", x + y - b)
        square = scheme.rescale(scheme.multiply(z, z, scheme.rlk))
        scheme.linear_combinations([square], [[r]])
        _ = z + p
        _ = scheme.rotate(a, 1, rotation_key)
    computation = Computation(t.operations())
    first, second = computation.linears  # x and y are only added into z
    assert [term.sign for term in first.terms] == [1, 1, 1, -1]
    assert [term.plaintext for term in first.terms] == [p, q, p, None]
    assert [term.ciphertext for term in first.terms] == [
        _components(a),
        _components(a),
        _components(b),
        _components(b),
    ]
    assert first.output == _components(z)
    for step in (first, second):
        for i in range(2):
            total = None
            for term in step.terms:
                value = term.ciphertext[i]
                if term.plaintext is not None:
                    value = term.plaintext * value
                value = value if term.sign > 0 else -value
                total = value if total is None else total + value
            assert total == step.output[i]
    [multiplication] = computation.multiplications
    [rescale_mask, rescale_body] = computation.rescales
    assert computation.written_by(multiplication.versions[0][0]) is first
    [term] = second.terms
    assert isinstance(computation.written_by(term.versions[0]), Division)
    assert computation.written_by(first.terms[0].versions[0]) is None
    assert set(first.terms[0].versions) <= computation.inputs
    assert [first.layer, multiplication.layer, rescale_mask.layer, second.layer] == [
        0,
        0,
        1,
        1,
    ]
    assert [first.ell, multiplication.ell, rescale_body.ell, second.ell] == [3, 3, 3, 2]
    assert computation.unread[0].rsplit("/", 1)[-1] == "MLWE.add_poly"
    assert any("automorphism" in path for path in computation.unread)
