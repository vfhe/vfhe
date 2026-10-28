# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The execution tape over sample operations: every recorded operation
replays in Python on the recorded operands to the recorded output."""

import pytest
from vfhe.arith import Polynomial, Ring
from vfhe.mlwe import MLWE_Scheme
from vfhe.util import trace

N = 256


@pytest.fixture
def scheme():
    Rq = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    return MLWE_Scheme(Rq, special_primes=0, module_rank=1)


@pytest.fixture(autouse=True)
def a_clean_tape():
    yield
    trace.enable().clear()


def _message(scheme, coefficients):
    return Polynomial(scheme.ring).from_array(coefficients)


def _replay(event, values):
    """The event's operation, computed in Python from its operands."""
    x = [values[i] for i in event.inputs]
    if event.kind == "add":
        return x[0] + x[1]
    if event.kind == "sub":
        return x[0] - x[1]
    if event.kind == "mul":
        return x[0] * x[1]
    if event.kind == "mul_addto":
        return x[0] + x[1] * x[2]
    raise AssertionError(f"no replay for {event.kind}")


def _check_replay(events):
    values = {e.sequence: e.value for e in events}
    replayed = 0
    for e in events:
        if e.kind in ("add", "sub", "mul", "mul_addto"):
            assert _replay(e, values) == e.value, e
            replayed += 1
    return replayed


def test_sample_addition_records_operands_then_sums_and_replays(scheme):
    key = scheme.key_gen(3.2)
    c1 = scheme.sample(_message(scheme, [1, 2, 3]), key)
    c2 = scheme.sample(_message(scheme, [4, 5, 6]), key)
    with trace.recording() as t:
        c3 = c1 + c2
    c3 += c1  # a later change must not reach the recorded values
    events = t.events()
    assert [e.kind for e in events] == ["input", "input", "add", "input", "input", "add"]
    assert [e.sequence for e in events] == [1, 2, 3, 4, 5, 6]
    assert events[2].inputs == (1, 2) and events[5].inputs == (4, 5)
    assert all(e.ring == c1.ring for e in events)
    assert _check_replay(events) == 2
    assert events[2].value == c1.get_a_poly(0) + c2.get_a_poly(0)
    assert events[5].value == c1.get_b_poly() + c2.get_b_poly()


def test_plaintext_product_and_decryption_replay(scheme):
    key = scheme.key_gen(3.2)
    c = scheme.sample(_message(scheme, [1, 2, 3]), key)
    p = scheme.ring.random_element()
    with trace.recording() as t:
        d = c * p
        scheme.linear_decrypt(d, key)
    events = t.events()
    kinds = [e.kind for e in events]
    assert kinds.count("mul") >= 3 and "sub" in kinds
    assert all(e.value is not None for e in events if e.kind != "mark")
    assert _check_replay(events) >= 4


def test_sampler_outputs_are_secret(scheme):
    key = scheme.key_gen(3.2)
    with trace.recording() as t:
        scheme.sample(_message(scheme, [1, 2, 3]), key)
    events = t.events()
    secret = [e for e in events if e.secret]
    assert secret
    assert all(e.kind.startswith("sample") for e in secret)
    assert all(not e.secret for e in events if not e.kind.startswith("sample"))
