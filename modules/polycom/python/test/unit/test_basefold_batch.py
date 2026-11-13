# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for basefold's batched opening [CCCFGS26, §7, Figs. 3-4]: several
evaluation claims, on one or more polynomials committed under one scheme,
opened by one sumcheck to a common point and one basefold run on a random
linear combination. Run over R_q and over an extension field.

Honest bundles accept, interactively and as a non-interactive argument; the
common-point sumcheck is skipped when the points coincide and the
combination when there is one polynomial; a false value anywhere in the
bundle is rejected; every message the batching adds is tamper-checked by
replaying an honest transcript against a verifier-only run; and the parking
rule gathers into one bundle the claims a pending statement produces later.
"""

from typing import Any

import pytest
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, Ring
from vfhe.piop import (
    IOP,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    exceptional_set_size,
)
from vfhe.piop.protocols import Sumcheck
from vfhe.piop.relations import Relation_Sum
from vfhe.polycom import Basefold, BasefoldEval, FieldFoldableRS, FoldableRS


class _Claims(Relation):
    """A bundle of evaluation claims, `claims` = ((oracle, point, value), ...);
    `_FanOut` turns it into one `Relation_Eval` per claim."""

    name = "claims"
    fields = ("claims",)

    def check(self, statement):
        return False


class _FanOut(Protocol):
    reduces_from = _Claims
    reduces_to = (Relation_Eval,)

    def _out(self, statements):
        return [
            self.reduce(statements, Relation_Eval, oracles=(o,), point=p, value=v)
            for s in statements
            for o, p, v in s.claims
        ]

    async def prove(self, prover, statements):
        return self._out(statements)

    async def verify(self, verifier, statements):
        return self._out(statements)


class _Domain:
    """A domain with a scheme and two committed polynomials f and g on four
    variables each (under different variable names)."""

    domain: Any

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "ring":
            self.domain = Ring(256, prime_size=[40, 30], split_degree=4)
            self.scheme = Basefold(FoldableRS(self.domain, k0=4, c=4, d=2))
        else:
            self.domain = Field(562949948178433, 2, 5)
            self.scheme = Basefold(FieldFoldableRS(self.domain, k0=4, c=4, d=2))
        self._count = 0
        self.f, self.g = self._polynomial("f"), self._polynomial("g")
        self.of, _ = self.scheme.commit(self.f)
        self.og, _ = self.scheme.commit(self.g)

    def _polynomial(self, name: str) -> MLE:
        variables = [MLE_Variable(f"{name}{i}") for i in range(4)]
        if self.kind == "ring":
            table = [self.domain.random_element() for _ in range(16)]
            return MLE(ring=self.domain, variables=variables, evaluations=table)
        vector = FieldVector(self.domain, 16)
        vector.sample_random(name.encode())
        return MLE(field=self.domain, variables=variables, evaluations=vector)

    def element(self):
        self._count += 1
        if self.kind == "ring":
            return self.domain.random_exceptional()
        return self.domain.random_element(f"point-{self._count}".encode())

    def point(self, f: MLE) -> dict:
        return {v: self.element() for v in f.variables}

    def claim(self, which: str, point: dict | None = None) -> tuple:
        oracle, f = (self.of, self.f) if which == "f" else (self.og, self.g)
        point = self.point(f) if point is None else point
        return (oracle, point, f.evaluate(point, in_place=False).constant())

    def same_point(self, claim: tuple, which: str) -> dict:
        """`claim`'s point, read on the variables of polynomial `which`."""
        f = self.f if which == "f" else self.g
        return dict(zip(f.variables, claim[1].values(), strict=True))

    def iop(self, fiat_shamir: bool = False) -> IOP:
        iop = IOP(domain=self.domain, fiat_shamir=fiat_shamir)
        iop.register(_Claims, _FanOut())
        iop.register(Relation_Eval, BasefoldEval(self.scheme, rep=3))
        return iop


@pytest.fixture(params=["ring", "field"])
def dom(request):
    return _Domain(request.param)


def _bundles(dom: _Domain) -> dict:
    first = dom.claim("f")
    return {
        "two polynomials, three points": [first, dom.claim("f"), dom.claim("g")],
        "one polynomial, two points": [first, dom.claim("f")],
        "two polynomials, one point": [
            first,
            dom.claim("g", dom.same_point(first, "g")),
        ],
    }


def _statement(claims) -> Statement:
    return Statement(_Claims(), claims=tuple(claims))


def _labels(iop: IOP) -> list[str]:
    return [label for label in iop.transcript.order if label.startswith("basefold")]


def test_bundles_accept(dom):
    for name, claims in _bundles(dom).items():
        iop = dom.iop()
        assert iop.run(_statement(claims)), name
        # One basefold run for the whole bundle.
        assert [x for x in _labels(iop) if x.endswith("/answers")] == [
            "basefold/0/answers"
        ], name


def test_bundles_accept_as_an_argument(dom):
    for name, claims in _bundles(dom).items():
        proof = dom.iop(fiat_shamir=True).prove(_statement(claims))
        assert dom.iop(fiat_shamir=True).verify(_statement(claims), proof), name


def test_what_the_bundle_shape_skips(dom):
    bundles = _bundles(dom)
    # Several points: the common-point sumcheck runs.
    iop = dom.iop()
    assert iop.run(_statement(bundles["two polynomials, three points"]))
    assert "basefold/0/point/q" in _labels(iop) and "basefold/0/alpha" in _labels(iop)
    # One point: no sumcheck, only the combination.
    iop = dom.iop()
    assert iop.run(_statement(bundles["two polynomials, one point"]))
    assert not any("/point/" in x for x in _labels(iop))
    assert "basefold/0/alpha" in _labels(iop)
    # One polynomial: the sumcheck, and no combination.
    iop = dom.iop()
    assert iop.run(_statement(bundles["one polynomial, two points"]))
    assert "basefold/0/point/q" in _labels(iop)
    assert "basefold/0/alpha" not in _labels(iop)


def test_a_false_value_anywhere_is_rejected(dom):
    for name, claims in _bundles(dom).items():
        for m in range(len(claims)):
            oracle, point, value = claims[m]
            wrong = list(claims)
            wrong[m] = (oracle, point, value + 1)
            assert not dom.iop().run(_statement(wrong)), (name, m)


def test_two_values_for_one_polynomial_at_one_point_are_rejected(dom):
    oracle, point, value = dom.claim("f")
    claims = [(oracle, point, value), (oracle, point, value + 1)]
    assert not dom.iop().run(_statement(claims))


def _replay(dom: _Domain, honest: IOP, claims, tamper) -> bool:
    """A verifier-only run over the honest transcript with `tamper` applied
    to each label's value: recorded challenges and query positions are
    returned as they were, so this replays the interaction exactly with the
    tampered messages."""
    iop = dom.iop()
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    return iop.loop.run_until_complete(iop.verifier.verify(_statement(claims)))


def test_tampered_batching_messages_are_rejected(dom):
    claims = _bundles(dom)["two polynomials, three points"]
    honest = dom.iop()
    assert honest.run(_statement(claims))
    assert _replay(dom, honest, claims, lambda _label, value: value)

    def shift_round(label, value):
        if label == "basefold/0/point/g1":
            return (value[0] + 1, value[1] - 1, value[2])  # g(0) + g(1) kept
        return value

    def shift_q(label, value):
        if label == "basefold/0/point/q":
            return (value[0] + 1, *value[1:])
        return value

    def drop_q(label, value):
        if label == "basefold/0/point/q":
            return value[:-1]
        return value

    def swap_first_layer(label, value):
        # The second commitment's leaf answered with the first's, path and
        # all: authenticated against the wrong root.
        if label == "basefold/0/answers":
            first = list(value[0])
            first[0] = (first[0][0], first[0][0])
            return (tuple(first), *value[1:])
        return value

    def truncate_first_layer(label, value):
        if label == "basefold/0/answers":
            first = list(value[0])
            first[0] = first[0][:1]
            return (tuple(first), *value[1:])
        return value

    for tamper in (
        shift_round,
        shift_q,
        drop_q,
        swap_first_layer,
        truncate_first_layer,
    ):
        assert not _replay(dom, honest, claims, tamper), tamper.__name__


def test_parking_gathers_a_claim_produced_later(dom):
    """A Sum on g is pending when the Eval claim on f reaches the frontier:
    the claim on f waits for the sumcheck's claim on g, an oracle under the
    same scheme, and both are opened in one bundle."""
    oracle, point, value = dom.claim("f")
    total = dom.g.table[0]
    for b in range(1, 16):
        total = total + dom.g.table[b]

    class _Split(Protocol):
        reduces_from = _Claims
        reduces_to = (Relation_Eval, Relation_Sum)

        def _out(self, statements):
            return [
                self.reduce(
                    statements,
                    Relation_Eval,
                    oracles=(oracle,),
                    point=point,
                    value=value,
                ),
                self.reduce(statements, Relation_Sum, oracles=[dom.og], value=total),
            ]

        async def prove(self, prover, statements):
            return self._out(statements)

        async def verify(self, verifier, statements):
            return self._out(statements)

    iop = IOP(domain=dom.domain)
    iop.prover.witnesses[dom.og] = dom.g
    iop.register(_Claims, _Split())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, BasefoldEval(dom.scheme, rep=3))
    assert iop.run(_statement([]))
    answers = [x for x in _labels(iop) if x.endswith("/answers")]
    assert len(answers) == 1
    assert "basefold/0/alpha" in _labels(iop)


def test_soundness_error_adds_the_batching_terms(dom):
    protocol = BasefoldEval(dom.scheme, rep=3)
    statement = Statement(Relation_Eval(), oracles=(dom.of,), point={}, value=0)
    size = exceptional_set_size(dom.domain)
    single = protocol.soundness_error(statement, dom.domain, 0.01, 0.5)
    assert size is not None and single is not None
    d = dom.scheme.code.d
    assert single == 2 * d / (0.01**3 * size) + (1 - 0.5 + 0.01 * d) ** 3
    batched = protocol.soundness_error(
        statement, dom.domain, 0.01, 0.5, claims=3, polynomials=2
    )
    assert batched == pytest.approx(single + (2 * 4 + 3 - 1) / size + 1 / size)
