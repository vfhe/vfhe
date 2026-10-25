# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchFold [CCCFGS26, §7]: evaluation claims on polynomials over
R_q proved by one basefold run per prime over the residue code, against the
ring commitments. Run with the ring code's twists in `F_p` and in the
residue fields.

Honest bundles accept, interactively and as a non-interactive argument; the
common-point sumcheck is skipped when every component of every point is one
point; a false value anywhere in the bundle is rejected; every message the
protocol adds is tamper-checked by replaying an honest transcript against a
verifier-only run; and one commitment can be opened by either `BasefoldEval`
or `BatchFoldEval`.
"""

import pytest
from vfhe.arith import MLE, Field, MLE_Variable, Polynomial, Ring
from vfhe.piop import (
    IOP,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    exceptional_set_size,
)
from vfhe.polycom import (
    Basefold,
    BasefoldEval,
    BatchFoldEval,
    FieldFoldableRS,
    FoldableRS,
)


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


class _Setting:
    """A ring basefold scheme over N = 64, split degree 4 (k = 16 components
    per prime) and primes of 28 and 50 bits, with two committed polynomials
    f and g on three variables each (under different variable names)."""

    def __init__(self, twist_field: str):
        self.ring = Ring(64, prime_size=[28, 50], split_degree=4)
        code = FoldableRS(self.ring, k0=2, c=4, d=2, twist_field=twist_field)
        self.scheme = Basefold(code)
        self.f, self.g = self._polynomial("f"), self._polynomial("g")
        self.of, _ = self.scheme.commit(self.f)
        self.og, _ = self.scheme.commit(self.g)

    def _polynomial(self, name: str) -> MLE:
        variables = [MLE_Variable(f"{name}{i}") for i in range(3)]
        table = [self.ring.random_element() for _ in range(8)]
        return MLE(ring=self.ring, variables=variables, evaluations=table)

    def point(self, f: MLE, integer: bool = False) -> dict:
        if integer:
            return {
                v: Polynomial(self.ring).from_array([3 + i])
                for i, v in enumerate(f.variables)
            }
        return {v: self.ring.random_exceptional() for v in f.variables}

    def claim(self, which: str, point: dict | None = None) -> tuple:
        oracle, f = (self.of, self.f) if which == "f" else (self.og, self.g)
        point = self.point(f) if point is None else point
        return (oracle, point, f.evaluate(point, in_place=False).constant())

    def same_point(self, claim: tuple, which: str) -> dict:
        """`claim`'s point, read on the variables of polynomial `which`."""
        f = self.f if which == "f" else self.g
        return dict(zip(f.variables, claim[1].values(), strict=True))

    def iop(self, fiat_shamir: bool = False, protocol: type = BatchFoldEval) -> IOP:
        iop = IOP(domain=self.ring, fiat_shamir=fiat_shamir)
        iop.register(_Claims, _FanOut())
        iop.register(Relation_Eval, protocol(self.scheme, rep=3))
        return iop


@pytest.fixture(scope="module", params=["prime", "field"])
def setting(request):
    return _Setting(request.param)


def _bundles(setting: _Setting) -> dict:
    first = setting.claim("f")
    integer = setting.claim("f", setting.point(setting.f, integer=True))
    return {
        "one claim": [first],
        "two polynomials, three points": [
            first,
            setting.claim("f"),
            setting.claim("g"),
        ],
        "one polynomial, two points": [first, setting.claim("f")],
        "two polynomials, one point": [
            first,
            setting.claim("g", setting.same_point(first, "g")),
        ],
        "two polynomials, one integer point": [
            integer,
            setting.claim("g", setting.same_point(integer, "g")),
        ],
    }


def _statement(claims) -> Statement:
    return Statement(_Claims(), claims=tuple(claims))


def _labels(iop: IOP) -> list[str]:
    return [label for label in iop.transcript.order if label.startswith("batchfold")]


def test_bundles_accept(setting):
    for name, claims in _bundles(setting).items():
        iop = setting.iop()
        assert iop.run(_statement(claims)), name
        labels = _labels(iop)
        # One set of query positions and one answer for every prime.
        assert [x for x in labels if x.endswith("/queries")] == [
            "batchfold/0/queries"
        ], name
        assert [x for x in labels if x.endswith("/answers")] == [
            "batchfold/0/answers"
        ], name
        # One basefold run per prime.
        assert [x for x in labels if x.endswith("/h0")] == [
            f"batchfold/0/p{i}/h0" for i in range(setting.ring.ell)
        ], name


def test_bundles_accept_as_an_argument(setting):
    for name, claims in _bundles(setting).items():
        proof = setting.iop(fiat_shamir=True).prove(_statement(claims))
        assert setting.iop(fiat_shamir=True).verify(_statement(claims), proof), name


def test_the_common_point_sumcheck_is_skipped_for_one_point(setting):
    bundles = _bundles(setting)
    # One ring point whose components differ: the sumcheck runs.
    iop = setting.iop()
    assert iop.run(_statement(bundles["two polynomials, one point"]))
    assert "batchfold/0/p0/point/q" in _labels(iop)
    # One integer point, the same in every component: no sumcheck.
    iop = setting.iop()
    assert iop.run(_statement(bundles["two polynomials, one integer point"]))
    labels = _labels(iop)
    assert not any("/point/" in x for x in labels)
    # The combination's challenge is the first round, then the folds start.
    assert labels[:2] == ["batchfold/0/round0", "batchfold/0/p0/g0"]


def test_the_primes_run_in_lockstep(setting):
    """Between two rounds' shared seeds, every prime writes its messages,
    prime 0 first: one challenge per round serves all primes."""
    iop = setting.iop()
    assert iop.run(_statement(_bundles(setting)["two polynomials, three points"]))
    segments: list[list[int]] = [[]]
    for label in _labels(iop):
        parts = label.split("/")
        if parts[2].startswith("round"):
            segments.append([])
        elif parts[2][0] == "p" and parts[2][1:].isdigit():
            segments[-1].append(int(parts[2][1:]))
    primes = list(range(setting.ring.ell))
    # The rounds of the sumcheck, the combination and the folds; the query
    # positions and the answers come after the last one.
    rounds = 1 + 3 + 1 + setting.scheme.code.d
    assert len(segments) == rounds + 1 and segments[0] == []
    for segment in segments[1:]:
        assert sorted(set(segment)) == primes and segment == sorted(segment)


def test_each_later_layer_has_one_tree_for_every_prime(setting):
    """A later fold layer is committed once, by one root for all primes,
    written after the fold that ends the previous layer; each query answers
    it with every prime's fiber and one path."""
    iop = setting.iop()
    assert iop.run(_statement(_bundles(setting)["two polynomials, three points"]))
    labels = _labels(iop)
    d = setting.scheme.code.d
    assert [x for x in labels if "/pi" in x] == [
        f"batchfold/0/pi{s}" for s in range(1, d)
    ]
    # Prime 0's fold message of the next round follows the root.
    root = labels.index("batchfold/0/pi1")
    assert labels[root + 1] == "batchfold/0/p0/g1"
    answers = iop.transcript.entries["batchfold/0/answers"].result()
    for _ring_top, later in answers:
        assert len(later) == d - 1
        for fibers, _path in later:
            assert len(fibers) == setting.ring.ell


def test_a_false_value_anywhere_is_rejected(setting):
    for name, claims in _bundles(setting).items():
        for m in range(len(claims)):
            oracle, point, value = claims[m]
            wrong = list(claims)
            wrong[m] = (oracle, point, value + 1)
            assert not setting.iop().run(_statement(wrong)), (name, m)


def test_two_values_for_one_polynomial_at_one_point_are_rejected(setting):
    oracle, point, value = setting.claim("f", setting.point(setting.f, integer=True))
    claims = [(oracle, point, value), (oracle, point, value + 1)]
    assert not setting.iop().run(_statement(claims))


def _replay(setting: _Setting, honest: IOP, claims, tamper) -> bool:
    """A verifier-only run over the honest transcript with `tamper` applied
    to each label's value: recorded challenges and query positions are
    returned as they were, so this replays the interaction exactly with the
    tampered messages."""
    iop = setting.iop()
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    return iop.loop.run_until_complete(iop.verifier.verify(_statement(claims)))


def test_tampered_messages_are_rejected(setting):
    claims = _bundles(setting)["two polynomials, three points"]
    honest = setting.iop()
    assert honest.run(_statement(claims))
    assert _replay(setting, honest, claims, lambda _label, value: value)

    def shift_round(label, value):
        if label == "batchfold/0/p1/point/g1":
            return (value[0] + 1, value[1] - 1, value[2])  # g(0) + g(1) kept
        return value

    def shift_q(label, value):
        if label == "batchfold/0/p0/point/q":
            value = value.copy()
            value[5] = value[5] + 1
        return value

    def shorten_q(label, value):
        if label == "batchfold/0/p0/point/q":
            return value.query(range(len(value) - 1))
        return value

    def scale_h0(label, value):
        if label == "batchfold/0/p1/h0":
            return value.scale(2)
        return value

    def swap_ring_leaf(label, value):
        # The second commitment's leaf answered with the first's, path and
        # all: authenticated against the wrong root.
        if label == "batchfold/0/answers":
            ring_top, layers = value[0]
            return ((ring_top[0], ring_top[0]), layers), *value[1:]
        return value

    def shift_prime_leaf(label, value):
        # The second layer's leaf at prime 1, for the first query.
        if label == "batchfold/0/answers":
            ring_top, later = value[0]
            (fibers, path), *rest = later
            fiber = fibers[1]
            fibers = (fibers[0], (fiber[0] + 1, *fiber[1:]), *fibers[2:])
            return ((ring_top, ((fibers, path), *rest)), *value[1:])
        return value

    def drop_prime(label, value):
        # The second layer answered without prime 1's fiber.
        if label == "batchfold/0/answers":
            ring_top, later = value[0]
            (fibers, path), *rest = later
            return ((ring_top, ((fibers[:1], path), *rest)), *value[1:])
        return value

    def swap_layer_path(label, value):
        # The second layer's path of another query.
        if label == "batchfold/0/answers":
            (ring_top, later), (_, other), *rest = value
            (fibers, _path), *tail = later
            changed = ((fibers, other[0][1]), *tail)
            return ((ring_top, changed), (_, other), *rest)
        return value

    def shift_root(label, value):
        if label == "batchfold/0/pi1":
            return bytes([value[0] ^ 1]) + value[1:]
        return value

    for tamper in (
        shift_round,
        shift_q,
        shorten_q,
        scale_h0,
        swap_ring_leaf,
        shift_prime_leaf,
        drop_prime,
        swap_layer_path,
        shift_root,
    ):
        assert not _replay(setting, honest, claims, tamper), tamper.__name__


def test_one_commitment_opens_under_either_protocol(setting):
    claims = _bundles(setting)["two polynomials, three points"]
    for protocol in (BasefoldEval, BatchFoldEval):
        assert setting.iop(protocol=protocol).run(_statement(claims)), protocol


def test_soundness_error_is_one_primes_error(setting):
    protocol = BatchFoldEval(setting.scheme, rep=3)
    statement = Statement(Relation_Eval(), oracles=(setting.of,), point={}, value=0)
    fields = [setting.ring.residue_field(i) for i in range(setting.ring.ell)]
    size = min(exceptional_set_size(field) or 0 for field in fields)
    assert size == min(p**4 for p in setting.ring.primes)
    error = protocol.soundness_error(
        statement, setting.ring, 0.01, 0.5, claims=3, polynomials=2
    )
    d, k, num_vars = 2, 16, 3
    # The largest of the primes' errors, all over a challenge set of size
    # min p_i^4: no factor for the two primes.
    expected = (
        2 * d / (0.01**3 * size)
        + (2 * num_vars + 3 * k - 1) / size
        + (2 * k - 1) / size
        + (1 - 0.5 + 0.01 * d) ** 3
    )
    assert error == pytest.approx(expected)


def test_a_scheme_over_a_field_is_refused():
    field = Field(562949948178433, 2, 5)
    scheme = Basefold(FieldFoldableRS(field, k0=4, c=4, d=2))
    with pytest.raises(TypeError):
        BatchFoldEval(scheme, rep=3)
