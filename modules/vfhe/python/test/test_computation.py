# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The computation read from the tape of the use case: two multiplications
with their operands, slots, digits, key components and rescales, equal to
what the ciphertexts hold; and the plaintext-linear steps, layers and
levels of a computation with products by plaintexts."""

import math
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


def test_a_product_without_a_key_is_a_tensor_and_rank_two_refused():
    ring = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    scheme = MLWE_Scheme(ring, special_primes=0, module_rank=1)
    key = scheme.key_gen(3.2)
    ct = scheme.sample(scheme.ring.random_element(), key)
    with trace.recording() as t:
        extended = scheme.multiply(ct, ct, None)
    computation = Computation(t.operations())
    assert computation.multiplications == []
    [tensor] = computation.tensors
    assert tensor.quadratic == extended.get_a_poly(0)
    assert tensor.constant == extended.get_b_poly()
    assert computation.written_by(_version(t, extended.obj_a_i(1))) is tensor
    scheme = MLWE_Scheme(ring, special_primes=0, module_rank=2)
    key = scheme.key_gen(3.2)
    rlk = scheme.gen_rlk(key, scheme.quadratic_key_polys(key))
    ct = scheme.sample(scheme.ring.random_element(), key)
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


def _version(t, handle) -> int:
    version = t.version_of(handle)
    assert version is not None
    return version


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


def _disjoint_chain(special_primes=0):
    """A CKKS scheme whose two levels share no prime, as for a rational
    rescale: two 48-bit primes, then one 44-bit prime."""
    sizes = [48, 48, 44] + [50] * special_primes
    Rq = Ring(N, split_degree=1, prime_size=sizes)

    def subring(*indices):
        return Rq.quotient_ring(mask=sum(1 << Rq.prime_indices[i] for i in indices))

    special = (3,) * special_primes
    scheme = CKKS_Scheme(
        [subring(0, 1), subring(2)],
        scaling_factor=2**24,
        special_primes=special_primes,
        special_rings=[subring(0, 1, *special), subring(2, *special)]
        if special_primes
        else None,
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    rlk = scheme.gen_rlk(key, [-(secret * secret)])
    return scheme, key, rlk


def _centered(value, modulus):
    return [((v + modulus // 2) % modulus) - modulus // 2 for v in value]


def test_a_remainder_is_exact_when_the_divisor_exceeds_the_quotient_ring():
    # Dividing by both 48-bit primes leaves a 44-bit ring: the remainder,
    # up to 2^95, is given over the ring of x, where it is exact.
    scheme, key, rlk = _disjoint_chain()
    x = scheme.encrypt(scheme.encode([0.5]), key)
    with trace.recording() as t:
        rescaled = scheme.rational_rescale(scheme.multiply(x, x, rlk))
    computation = Computation(t.operations())
    for rescale in computation.rescales:
        P = rescale.modulus
        assert P.bit_length() > 90
        assert rescale.remainder.ring == rescale.x.ring
        r = rescale.remainder.get_polynomial(signed=True)
        assert r == _centered(rescale.x.get_polynomial(signed=True), P)
        assert max(abs(v) for v in r) <= (P - 1) // 2
    assert [r.quotient for r in computation.rescales] == [
        rescaled.get_a_poly(0),
        rescaled.get_b_poly(),
    ]


def test_a_rational_rescale_is_read_from_the_component_before_its_lift():
    scheme, key, rlk = _disjoint_chain()
    x = scheme.encrypt(scheme.encode([0.5]), key)
    with trace.recording() as t:
        product = scheme.multiply(x, x, rlk)
        scheme.rational_rescale(product)
    computation = Computation(t.operations())
    [multiplication] = computation.multiplications
    assert multiplication.rescales == computation.rescales
    added = scheme.rings[1].primes
    for rescale, component in zip(
        computation.rescales, _components(product), strict=True
    ):
        assert rescale.source == component
        assert rescale.lift == math.prod(added)
        assert rescale.primes == tuple(scheme.rings[0].primes)
        assert rescale.x == component.scaled_lift(rescale.x.ring)
        assert computation.written_by(rescale.source_version) is multiplication
    assert computation.unread == []


def test_lazy_relinearization_reads_tensors_a_linear_step_and_a_key_switch():
    # Squares left extended, a linear combination of the extended products,
    # one relinearization, a bias, a rational rescale.
    scheme, key, rlk = _disjoint_chain()
    xs = [scheme.encrypt(scheme.encode([v]), key) for v in (0.5, 0.25)]
    weights = [[scheme.encode([0.25]), scheme.encode([-0.5])]]
    with trace.recording() as t:
        squares = [scheme.multiply(x, x, None) for x in xs]
        [combined] = scheme.linear_combinations(squares, weights)
        relinearized = scheme.relinearize(combined, rlk)
        scheme.rational_rescale(relinearized)
    computation = Computation(t.operations())
    assert computation.multiplications == []
    assert len(computation.tensors) == 2
    [linear] = computation.linears
    assert [len(term.ciphertext) for term in linear.terms] == [3, 3]
    assert [term.ciphertext for term in linear.terms] == [
        [s.get_a_poly(0), s.get_a_poly(1), s.get_b_poly()] for s in squares
    ]
    assert linear.output == [
        combined.get_a_poly(0),
        combined.get_a_poly(1),
        combined.get_b_poly(),
    ]
    assert [computation.written_by(t.versions[0]) for t in linear.terms] == (
        computation.tensors
    )
    [switch] = computation.key_switches
    assert switch.decomposed == combined.get_a_poly(0)
    assert switch.slots == [combined.get_a_poly(1), combined.get_b_poly()]
    assert switch.output == _components(relinearized)
    assert computation.written_by(switch.versions[0]) is linear
    assert [computation.written_by(r.source_version) for r in computation.rescales] == [
        switch,
        switch,
    ]
    assert [linear.layer, switch.layer, computation.rescales[0].layer] == [1, 1, 1]
    assert computation.unread == []


def test_a_relinearized_extended_product_is_a_multiplication():
    scheme, key, rlk = _disjoint_chain(special_primes=1)
    x = scheme.encrypt(scheme.encode([0.5]), key)
    with trace.recording() as t:
        product = scheme.relinearize(scheme.multiply(x, x, None), rlk)
    computation = Computation(t.operations())
    [multiplication] = computation.multiplications
    assert computation.tensors == [] and computation.key_switches == []
    assert multiplication.key_switch.output == _components(product)
    assert len(multiplication.key_switch.mod_down) == 2


def test_a_key_switch_of_a_ciphertext_adds_to_the_body_only():
    ring = Ring(N, prime_size=[45, 45, 50], split_degree=1)
    scheme = MLWE_Scheme(ring, special_primes=1)
    key, other = scheme.key_gen(3.2), scheme.key_gen(3.2)
    ksk = scheme.gen_ksk(other, key)
    ct = scheme.sample(scheme.ring.random_element(), key)
    with trace.recording() as t:
        switched = scheme.keyswitch(ct, ksk)
    computation = Computation(t.operations())
    [switch] = computation.key_switches
    assert switch.decomposed == ct.get_a_poly(0)
    assert switch.slots == [None, ct.get_b_poly()]
    assert switch.output == _components(switched)
    assert computation.written_by(_version(t, switched.obj_b())) is switch


def test_batched_calls_read_as_their_single_versions():
    scheme, key, rlk = _disjoint_chain()
    xs = [scheme.encrypt(scheme.encode([v]), key) for v in (0.5, 0.25)]
    weights = [[scheme.encode([0.25]), scheme.encode([-0.5])]] * 2

    def counts(batched):
        with trace.recording() as t:
            if batched:
                products = scheme.multiply_batch(xs, xs[::-1], rlk, n_threads=4)
                extended = [scheme.multiply(x, x, None) for x in xs]
                combined = scheme.linear_combinations(extended, weights, n_threads=4)
                relinearized = scheme.relinearize_batch(combined, rlk, n_threads=4)
                scheme.rational_rescale_batch(products + relinearized, n_threads=4)
            else:
                products = [
                    scheme.multiply(a, b, rlk)
                    for a, b in zip(xs, xs[::-1], strict=True)
                ]
                extended = [scheme.multiply(x, x, None) for x in xs]
                combined = [
                    scheme.linear_combinations(extended, [w])[0] for w in weights
                ]
                relinearized = [scheme.relinearize(c, rlk) for c in combined]
                for c in products + relinearized:
                    scheme.rational_rescale(c)
        computation = Computation(t.operations(), known=known)
        known.update(computation.values)  # the weights, first seen in one run
        return [
            len(computation.multiplications),
            len(computation.key_switches),
            len(computation.tensors),
            len(computation.linears),
            len(computation.rescales),
            computation.unread,
        ]

    with trace.recording() as t:  # the inputs and keys, seen once
        for x in xs:
            scheme.multiply(x, x, rlk)
    known = Computation(t.operations()).values
    assert counts(batched=False) == counts(batched=True) == [2, 2, 2, 2, 8, []]


def test_a_copy_of_an_earlier_value_asks_for_the_earlier_values():
    scheme, key, rlk = _disjoint_chain()
    x = scheme.encrypt(scheme.encode([0.5]), key)
    with trace.recording() as t:
        scheme.multiply(x, x, rlk)
    t.operations()
    with trace.recording() as t:
        x.copy()
    with pytest.raises(ValueError, match="pass the earlier values as known"):
        Computation(t.operations())


def test_a_batch_holding_one_product_twice_reads_two_multiplications():
    scheme, key, rlk = _disjoint_chain()
    x = scheme.encrypt(scheme.encode([0.5]), key)
    with trace.recording() as t:
        products = scheme.multiply_batch([x, x], [x, x], rlk)
    computation = Computation(t.operations())
    assert [m.key_switch.output for m in computation.multiplications] == [
        _components(p) for p in products
    ]
    slots = [(m.linear, m.constant) for m in computation.multiplications]
    assert slots[0] == slots[1] and computation.tensors == []
