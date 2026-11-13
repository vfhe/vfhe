# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for `FieldFoldableRS.residue_code`: the code a foldable code over
`R_q` induces on the residue field at one of its primes.

The oracle is the ring code itself: the CRT components (`crt_components`)
of a ring codeword must be the residue code's codewords of the message's
components, and folding must commute with taking components -- for the
general code with a transform base and with a Vandermonde base, with its
twists in `F_p` and in the residue fields, and for the ``"rs"``
instantiation, over narrow and wide primes. The residue code must also
serve as a basefold code on its own.

A ring code with twists in the residue fields is defined through its
residue codes, so it gets ring-level checks too: decoding inverts encoding,
folding a codeword is encoding the folded message, the pair and fiber folds
agree with the full fold, and basefold runs over it.
"""

import pytest
from vfhe.arith import MLE, FieldVector, MLE_Variable, Ring
from vfhe.piop import IOP, Relation_Eval, Statement
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS, FoldableRS

# (instantiation, k0, c, d, twist_field) over N = 64 and split degree 4, so
# k = 16 slots: a base of n0 = 8 <= 16 has a transform, n0 = 32 a
# Vandermonde base, and the "rs" code needs n_d = 16 <= 16.
CODES = [
    ("general", 2, 4, 2, "prime"),
    ("general", 8, 4, 2, "prime"),
    ("rs", 2, 2, 2, "prime"),
    ("general", 2, 4, 2, "field"),
    ("general", 8, 4, 2, "field"),
]


def _ring_code(instantiation, k0, c, d, twist_field) -> FoldableRS:
    ring = Ring(64, prime_size=[28, 50], split_degree=4)
    return FoldableRS(
        ring, k0=k0, c=c, d=d, instantiation=instantiation, twist_field=twist_field
    )


@pytest.fixture(params=CODES, ids=lambda p: f"{p[0]}-k0{p[1]}-{p[4]}")
def code(request):
    return _ring_code(*request.param)


@pytest.fixture(params=[p for p in CODES if p[4] == "field"], ids=lambda p: f"k0{p[1]}")
def field_twisted(request):
    return _ring_code(*request.param)


def _component(vector: FieldVector, s: int, length: int) -> FieldVector:
    """Component s of each of `length` elements, from a `crt_components`
    vector of them."""
    return vector.query(range(s * length, (s + 1) * length))


def test_components_of_a_ring_codeword_are_residue_codewords(code):
    ring = code.ring
    slots = ring.N // ring.split_degree
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    for index in range(ring.ell):
        residue = FieldFoldableRS.residue_code(code, index)
        assert residue.field is ring.residue_field(index)
        assert residue.n_d == code.n_d and residue.instantiation == code.instantiation
        m = ring.crt_components(message, index)
        w = ring.crt_components(word, index)
        for s in range(slots):
            component = _component(w, s, code.n_d)
            assert residue.encode(_component(m, s, code.k_d)) == component
            ok, decoded = residue.decode(component)
            assert ok and decoded == _component(m, s, code.k_d)


def test_folding_commutes_with_taking_components(code):
    ring = code.ring
    slots = ring.N // ring.split_degree
    word = code.encode([ring.random_element() for _ in range(code.k_d)])
    r = ring.random_exceptional()
    for level in range(code.d, 0, -1):
        folded = code.fold(word, r, level)
        for index in range(ring.ell):
            residue = FieldFoldableRS.residue_code(code, index)
            rs = ring.crt_components([r], index)
            w = ring.crt_components(word, index)
            f = ring.crt_components(folded, index)
            n = len(word)
            for s in range(slots):
                assert residue.fold(_component(w, s, n), rs[s], level) == (
                    _component(f, s, n // 2)
                )
        word = folded


def test_tables_live_where_the_twist_field_says(code):
    ring = code.ring
    for index in range(ring.ell):
        residue = FieldFoldableRS.residue_code(code, index)
        if code.twist_field == "field" and code.instantiation == "general":
            # The ring code is defined by its residue codes, whose twists are
            # over the whole residue field.
            assert residue is code.residue_codes[index] and code.twists == []
            assert residue.twists_domain is residue.field
        else:
            assert residue.twists_domain.order == ring.primes[index]
            assert [[int(t) for t in row] for row in residue.twists] == [
                [row[index] for row in table] for table in code.twists
            ]
        # The bound runs on the field the twists come from; the ring code's
        # on its smallest prime.
        assert residue.relative_distance() >= code.relative_distance()


def test_field_twists_raise_the_distance_bound(field_twisted):
    code = field_twisted
    prime = FoldableRS(code.ring, k0=code.k0, c=code.c, d=code.d)
    assert code.relative_distance() > prime.relative_distance()


def test_a_field_twisted_ring_code_decodes_and_folds(field_twisted):
    code = field_twisted
    ring = code.ring
    message = [ring.random_element() for _ in range(code.k_d)]
    word = code.encode(message)
    ok, decoded = code.decode(word)
    assert ok and all(a == b for a, b in zip(decoded, message, strict=True))
    tampered = list(word)
    tampered[3] = tampered[3] + ring.random_element()
    assert not code.decode(tampered)[0]
    r = ring.random_exceptional()
    folded = code.fold(word, r, code.d)
    half = [message[2 * i] + r * message[2 * i + 1] for i in range(code.k_d // 2)]
    assert all(a == b for a, b in zip(folded, code.encode(half), strict=True))
    # The verifier's forms: one pair, many pairs, and fibers of four.
    assert code.fold_at(word, r, code.d, 5) == folded[5]
    indices = [0, 3, 6]
    pairs = code.fold_pairs(
        [word[2 * i] for i in indices],
        [word[2 * i + 1] for i in indices],
        r,
        code.d,
        indices,
    )
    assert all(a == folded[i] for a, i in zip(pairs, indices, strict=True))
    r2 = ring.random_exceptional()
    twice = code.fold(folded, r2, code.d - 1)
    fibers = code.join_fibers([code.fiber_at(word, j, 4) for j in indices], 4)
    refolded = code.fold_fibers(fibers, [r, r2], code.d, indices)
    assert all(a == twice[j] for a, j in zip(refolded, indices, strict=True))


def test_basefold_runs_over_a_field_twisted_ring_code(field_twisted):
    code = field_twisted
    ring = code.ring
    scheme = Basefold(code)
    variables = [MLE_Variable(f"x{i}") for i in range(code.k_d.bit_length() - 1)]
    table = [ring.random_element() for _ in range(code.k_d)]
    f = MLE(ring=ring, variables=variables, evaluations=table)
    oracle, opening = scheme.commit(f)
    point = {v: ring.random_exceptional() for v in variables}
    value = f.evaluate(point, in_place=False).constant()

    def iop(fiat_shamir: bool) -> IOP:
        out = IOP(domain=ring, fiat_shamir=fiat_shamir)
        out.register(Relation_Eval, BasefoldEval(scheme, rep=3))
        return out

    def statement(v) -> Statement:
        return Statement(Relation_Eval(), oracles=(oracle,), point=point, value=v)

    prover = iop(False)
    prover.prover.witnesses[oracle.commitment] = opening
    assert prover.run(statement(value))
    assert not iop(False).run(statement(value + 1))
    proof = iop(True).prove(statement(value))
    assert iop(True).verify(statement(value), proof)


def test_a_residue_code_is_a_basefold_code(code):
    residue = FieldFoldableRS.residue_code(code, 0)
    field = residue.field
    scheme = Basefold(residue)
    variables = [
        MLE_Variable(f"x{i}") for i in range(code.d + code.k0.bit_length() - 1)
    ]
    table = FieldVector(field, 1 << len(variables))
    table.sample_random(b"residue-code")
    f = MLE(field=field, variables=variables, evaluations=table)
    oracle, opening = scheme.commit(f)
    point = {v: field.random_element(f"z{i}".encode()) for i, v in enumerate(variables)}
    value = f.evaluate(point, in_place=False).constant()
    iop = IOP(domain=field)
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=3))
    iop.prover.witnesses[oracle.commitment] = opening
    statement = Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)
    assert iop.run(statement)
