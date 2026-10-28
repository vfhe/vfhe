# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The execution tape over sample operations. A traced run takes each
operation's generic path, so the tape is the dispatch's alone: this checks
that the generic path computes what the kernel computes, bit for bit, and
that every recorded operation replays in Python on its recorded operands."""

import pytest
from vfhe import arith
from vfhe.arith import Polynomial, Ring
from vfhe.mlwe import MGSW_Scheme, MLWE_Scheme
from vfhe.util import trace

N = 256
REPLAYED = (
    "add",
    "sub",
    "mul",
    "mul_addto",
    "mul_subto",
    "scale_by",
    "permute",
    "round_division",
    "round_division_approximate",
    "decompose_digit",
)


@pytest.fixture
def bv():
    Rq = Ring(N, prime_size=[45, 45, 45], split_degree=1)
    return MLWE_Scheme(Rq, special_primes=0, module_rank=1)


@pytest.fixture
def ghs():
    Rq = Ring(N, prime_size=[45, 45, 45, 50], split_degree=1)
    return MLWE_Scheme(Rq, special_primes=1, module_rank=1)


@pytest.fixture(autouse=True)
def a_clean_tape():
    yield
    arith.set_generic_only(False)
    trace.enable().clear()


def _message(scheme, coefficients):
    return Polynomial(scheme.ring).from_array(coefficients)


def _samples(scheme, key, count):
    return [
        scheme.sample(_message(scheme, [i + 1, i + 2, i + 3]), key)
        for i in range(count)
    ]


def _rlk(scheme, key):
    s_0 = key.poly[0]
    return scheme.gen_rlk(key, [-(s_0 * s_0)])


def _components(sample):
    return [sample.get_a_poly(j) for j in range(sample.r)] + [sample.get_b_poly()]


def _values(events):
    """The value each event wrote, zeros and copies materialized."""
    values = {}
    for e in events:
        if e.kind == "zero":
            values[e.sequence] = Polynomial(e.ring).from_array([0])
        elif e.kind == "copy":
            values[e.sequence] = values[e.inputs[0]]
        elif e.value is not None:
            values[e.sequence] = e.value
    return values


def _replay(e, values):
    """The event's operation, computed in Python from its operands."""
    x = [values[i] for i in e.inputs]
    if e.kind == "add":
        return x[0] + x[1]
    if e.kind == "sub":
        return x[0] - x[1]
    if e.kind == "mul":
        return x[0] * x[1]
    if e.kind == "mul_addto":
        return x[0] + x[1] * x[2]
    if e.kind == "mul_subto":
        return x[0] - x[1] * x[2]
    if e.kind == "scale_by":
        return x[0] * [e.scalar[i] for i in e.ring.prime_indices]
    if e.kind == "permute":
        return x[0].automorphism(e.param)
    if e.kind == "round_division":
        return x[0].copy().round_division(e.ring)
    if e.kind == "round_division_approximate":
        return x[0].copy().round_division_approximate(e.ring)
    if e.kind == "decompose_digit":
        return x[0].decompose_digit(
            e.ring, list(e.primes), e.log_base, e.level, e.dropped, e.centered
        )
    raise AssertionError(f"no replay for {e.kind}")


def _kinds(events):
    return [e.kind for e in events if e.kind not in ("input", "mark")]


def _check(events):
    """Replays every event of a replayable kind, and returns how many; the
    other kinds are zeros and copies, which carry no computation."""
    kinds = _kinds(events)
    assert set(kinds) <= set(REPLAYED) | {"zero", "copy"}, set(kinds)
    values = _values(events)
    replayed = 0
    for e in events:
        if e.kind in REPLAYED:
            assert _replay(e, values) == e.value, (e.sequence, e.kind)
            replayed += 1
    assert replayed == sum(k in REPLAYED for k in kinds)
    return replayed


def _same(generic, kernel):
    """Whether two operations' results agree component by component."""
    if isinstance(generic, list):
        return all(_same(g, k) for g, k in zip(generic, kernel, strict=True))
    return _components(generic) == _components(kernel)


def _generic_and_kernel(operation):
    """The operation's result through the generic interface, and through
    the kernels."""
    arith.set_generic_only(True)
    generic = operation()
    arith.set_generic_only(False)
    return generic, operation()


def test_sample_addition_records_operands_then_sums_and_replays(bv):
    key = bv.key_gen(3.2)
    c1, c2 = _samples(bv, key, 2)
    with trace.recording() as t:
        c3 = c1 + c2
    c3 += c1  # a later change must not reach the recorded values
    events = [e for e in t.events() if e.kind != "mark"]
    assert [e.kind for e in events] == [
        "input",
        "input",
        "add",
        "input",
        "input",
        "add",
    ]
    sequences = [e.sequence for e in events]
    assert events[2].inputs == (sequences[0], sequences[1])
    assert events[5].inputs == (sequences[3], sequences[4])
    assert all(e.ring == c1.ring for e in events)
    assert _check(events) == 2
    assert events[2].value == c1.get_a_poly(0) + c2.get_a_poly(0)


def test_plaintext_product_and_decryption_replay(bv):
    key = bv.key_gen(3.2)
    (c,) = _samples(bv, key, 1)
    p = bv.ring.random_element()
    with trace.recording() as t:
        d = c * p
        bv.linear_decrypt(d, key)
    events = t.events()
    assert _kinds(events).count("mul") >= 3 and "sub" in _kinds(events)
    assert _check(events) >= 4


def test_sampler_outputs_are_secret(bv):
    key = bv.key_gen(3.2)
    with trace.recording() as t:
        bv.sample(_message(bv, [1, 2, 3]), key)
    events = t.events()
    assert [e.kind for e in events if e.secret]
    assert all(e.secret == e.kind.startswith("sample") for e in events)


@pytest.mark.parametrize("setup", ["bv", "ghs"])
def test_multiply_generic_path_matches_the_kernel_and_replays(request, setup):
    scheme = request.getfixturevalue(setup)
    key = scheme.key_gen(3.2)
    rlk = _rlk(scheme, key)
    c1, c2 = _samples(scheme, key, 2)
    generic, kernel = _generic_and_kernel(lambda: scheme.multiply(c1, c2, rlk))
    assert _same(generic, kernel)
    with trace.recording() as t:
        traced = scheme.multiply(c1, c2, rlk)
    assert _same(traced, kernel)
    events = t.events()
    kinds = _kinds(events)
    # The tensor product's three slots, then the digits of d2 and their
    # products with the key, then the division by the special primes (GHS)
    # or the pass-through additions (BV).
    assert kinds[:4] == ["mul", "mul", "mul_addto", "mul"]
    digits = [e for e in events if e.kind == "decompose_digit"]
    assert len(digits) == 3 and all(e.log_base == 0 and e.centered for e in digits)
    assert [e.primes for e in digits] == [(p,) for p in scheme.ring.primes]
    assert kinds.count("mul_subto") == 6  # three digits, two key components
    assert kinds.count("round_division_approximate") == (2 if setup == "ghs" else 0)
    assert _check(events) == sum(k in REPLAYED for k in kinds)
    final = [e for e in events if e.kind not in ("input", "mark")][-2:]
    assert [e.value for e in final] == _components(traced)


def test_rank_two_products_match_and_replay():
    Rq = Ring(N, prime_size=[45, 45, 45, 50], split_degree=1)
    scheme = MLWE_Scheme(Rq, special_primes=1, module_rank=2)
    key = scheme.key_gen(3.2)
    rlk = scheme.gen_rlk(key, scheme.quadratic_key_polys(key))
    c1, c2 = _samples(scheme, key, 2)
    generic, kernel = _generic_and_kernel(lambda: scheme.multiply(c1, c2, rlk))
    assert _same(generic, kernel)
    with trace.recording() as t:
        scheme.multiply(c1, c2, rlk)
    events = t.events()
    assert _kinds(events).count("decompose_digit") == 9  # three quadratic slots
    assert _check(events) > 0


def test_batch_multiply_divided_to_a_level_matches_and_replays(ghs):
    key = ghs.key_gen(3.2)
    rlk = _rlk(ghs, key)
    cts = _samples(ghs, key, 4)
    lhs, rhs = [cts[0], cts[2]], [cts[1], cts[3]]
    generic, kernel = _generic_and_kernel(
        lambda: ghs.multiply_batch(lhs, rhs, rlk, 0, lvl=1)
    )
    assert _same(generic, kernel)
    with trace.recording() as t:
        traced = ghs.multiply_batch(lhs, rhs, rlk, 0, lvl=1)
    assert _same(traced, kernel)
    events = t.events()
    assert _kinds(events).count("round_division") == 4
    assert _check(events) > 0
    assert all(f.ring == ghs.rings[1] for f in traced)


def test_automorphism_forms_match_and_replay(ghs):
    key = ghs.key_gen(3.2)
    keys = [ghs.gen_ksk_automorphism(key, key, gen)[0] for gen in (5, 25)]
    c1, c2 = _samples(ghs, key, 2)

    def forms():
        return [
            ghs.automorphism(c1, 5, keys[0]),
            *ghs.automorphism_batch([c1, c2], [5, 25], keys),
            *ghs.automorphisms(c1, [5, 25], keys),
        ]

    generic, kernel = _generic_and_kernel(forms)
    assert _same(generic, kernel)
    with trace.recording() as t:
        traced = forms()
    assert _same(traced, kernel)
    events = t.events()
    kinds = _kinds(events)
    assert kinds.count("permute") >= 2 + 4 + 2
    assert kinds.count("decompose_digit") == 3 * (1 + 2 + 1)  # hoisted: once
    assert _check(events) > 0


def test_automorphism_sum_matches_and_replays(ghs):
    key = ghs.key_gen(3.2)
    gens = [5, 1, 25]
    keys = [
        ghs.gen_ksk_automorphism(key, key, gen)[0] if gen != 1 else None for gen in gens
    ]
    cts = _samples(ghs, key, 3)
    generic, kernel = _generic_and_kernel(
        lambda: ghs.automorphism_sum(cts, gens, keys, 0)
    )
    assert _same(generic, kernel)
    with trace.recording() as t:
        out = ghs.automorphism_sum(cts, gens, keys, 0)
    assert _same(out, kernel)
    events = t.events()
    assert _kinds(events).count("round_division_approximate") == 2
    assert _check(events) > 0


def test_mgsw_external_product_matches_and_replays(bv):
    key = bv.key_gen(3.2)
    selector = MGSW_Scheme(bv).encrypt(Polynomial(bv.ring).from_array([1]), key)
    (c,) = _samples(bv, key, 1)
    generic, kernel = _generic_and_kernel(lambda: selector.external_product(c))
    assert _same(generic, kernel)
    with trace.recording() as t:
        out = selector.external_product(c)
    assert _same(out, kernel)
    events = t.events()
    assert _kinds(events).count("decompose_digit") == 6  # two components, three primes
    assert _check(events) > 0


def test_linear_combinations_replay(bv):
    key = bv.key_gen(3.2)
    cts = _samples(bv, key, 3)
    rows = [[bv.ring.random_element() for _ in cts] for _ in range(2)]
    with trace.recording() as t:
        outs = bv.linear_combinations(cts, rows, 0)
    events = t.events()
    assert _kinds(events).count("mul") == 4 and _kinds(events).count("mul_addto") == 8
    assert _check(events) == 12
    assert [e.value for e in events if e.kind == "mul_addto"][-2:] == _components(
        outs[1]
    )
