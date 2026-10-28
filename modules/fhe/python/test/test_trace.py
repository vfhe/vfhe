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


def _scheme_and_pair():
    scheme = CKKS_Scheme(
        Ring(N, 300, split_degree=1), scaling_factor=2**49, special_primes=0
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])
    slots = N // 2
    a = scheme.encrypt(scheme.encode([1.5, -2.0, 0.5, 3.0] + [0.0] * (slots - 4)), key)
    b = scheme.encrypt(scheme.encode([0.5, 1.0, -1.5, 2.0] + [0.0] * (slots - 4)), key)
    return scheme, a, b


def test_the_use_case_reads_as_operations_with_their_versions():
    scheme, a, b = _scheme_and_pair()
    with trace.recording() as t:
        c = a * b
        d = c * c
    [first, second] = t.operations()
    assert first.name == second.name == "CKKS_Ciphertext.__mul__"
    [product] = first.find("MLWE_Scheme.multiply")
    assert product.path.split("/") == [
        "CKKS_Ciphertext.__mul__",
        "MLWE.__mul__",
        "CKKS_Scheme.multiply",
        "MLWE_Scheme.multiply",
    ]
    kinds = [e.kind for e in product.all_events() if e.kind != "input"]
    assert kinds[:4] == ["mul", "mul", "mul_addto", "mul"]
    # The relinearization key's components are inputs, seen by the first
    # product; the result's components were written by the last rescale.
    inputs = {e.sequence for e in first.all_events() if e.kind == "input"}
    assert isinstance(scheme.rlk, list)
    samples = next(s for key in scheme.rlk for s in key.mlwe if s is not None)
    assert all(t.version_of(s.obj_b()) in inputs for s in samples)
    [rescale] = second.find("CKKS_Scheme.rescale")
    divisions = [e for e in rescale.all_events() if e.kind == "round_division"]
    assert len(divisions) == 2
    assert t.version_of(d.obj_a_i(0)) == divisions[0].sequence
    assert t.version_of(d.obj_b()) == divisions[1].sequence
    earlier = t.version_of(c.obj_b())
    assert earlier is not None and earlier < divisions[0].sequence


def test_a_rational_rescale_has_its_divisions_under_its_name():
    Rq = Ring(N, split_degree=1, prime_size=[50, 50, 50, 30, 50])

    def subring(*indices):
        return Rq.quotient_ring(mask=sum(1 << Rq.prime_indices[i] for i in indices))

    scheme = CKKS_Scheme(
        [subring(0, 1, 2), subring(0, 1, 3)],
        scaling_factor=2**50,
        special_primes=1,
        special_rings=[subring(0, 1, 2, 4), subring(0, 1, 3, 4)],
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    ct = scheme.encrypt(scheme.encode([1.0, 2.0] + [0.0] * (N // 2 - 2)), key)
    with trace.recording() as t:
        rescaled = scheme.rational_rescale(ct)
    [operation] = t.operations()
    assert operation.name == "CKKS_Scheme.rational_rescale"
    events = operation.all_events()
    divisions = [e for e in events if e.kind == "round_division"]
    assert [e.value for e in divisions] == [
        rescaled.get_a_poly(0),
        rescaled.get_b_poly(),
    ]
    assert all(e.ring == rescaled.ring for e in divisions)
    # The lifted operands reach the tape as inputs: the lift has no slot.
    inputs = {e.sequence for e in events if e.kind == "input"}
    assert all(len(e.inputs) == 1 and e.inputs[0] in inputs for e in divisions)
