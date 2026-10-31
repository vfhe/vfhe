# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for `basefold_parameters`.

It reproduces the parameter estimator of [CCCFGS26] (k0 = 2^8, n0 = 2^10 and
its repetition counts, for 11 to 24 variables over 180- and 260-bit fields
and over R_q with full-field twists); what it returns is what the library
reports on the code and the protocol built from it -- the code's
`relative_distance`, `BasefoldEval.soundness_error` at most 2^-lambda, and
one query fewer above it -- and satisfies the soundness theorem's
conditions; batching costs; a field too small for the target raises; and
`proof_bits` counts the messages of a real opening."""

import math
from collections.abc import Callable

import pytest
from vfhe.arith import (
    MLE,
    Field,
    FieldElement,
    FieldVector,
    MLE_Variable,
    Polynomial,
    PseudoMersenneField,
    Ring,
    RNSRing,
)
from vfhe.crypto import DIGEST_LEN, MerklePath
from vfhe.piop import IOP, Protocol, Relation, Relation_Eval, Statement
from vfhe.polycom import (
    Basefold,
    BasefoldEval,
    FieldFoldableRS,
    FoldableRS,
    basefold_parameters,
)
from vfhe.polycom.basefold import basefold_soundness_error
from vfhe.polycom.parameters import _proof_bits

# The estimator's repetitions for 11 to 24 variables, at lambda = 128 and
# log2|F| = 180 and 260; it picks k0 = 2^8 and n0 = 2^10 for all of them.
_ESTIMATOR = {
    180: [316, 319, 321, 324, 326, 329, 332, 334, 337, 340, 343, 346, 349, 352],
    260: [314, 315, 317, 319, 320, 322, 324, 326, 327, 329, 331, 333, 335, 336],
}


def _ring_180() -> RNSRing:
    """R_q with one 45-bit prime and split degree 4: residue fields of 180
    bits."""
    return Ring(64, prime_size=[45], split_degree=4)


def _field_180() -> Field:
    """`F_(p^4)` for a 45-bit p, a residue field of `_ring_180`."""
    return _ring_180().residue_field(0)


def _pmf_260() -> Field:
    return PseudoMersenneField.generate(260, two_adicity=10)


_DOMAINS: dict[str, Callable[[], Field | RNSRing]] = {
    "f180": _field_180,
    "pmf260": _pmf_260,
    "ring": _ring_180,
}

#: What `soundness_error` is asked about; it reads none of it.
_STATEMENT = Statement(Relation_Eval(), point={}, value=0)


@pytest.mark.parametrize(
    ("name", "bits", "twist_field"),
    [("f180", 180, None), ("pmf260", 260, None), ("ring", 180, "field")],
)
def test_the_estimator_of_cccfgs26_is_reproduced(name, bits, twist_field):
    domain = _DOMAINS[name]()
    for num_vars, rep in zip(range(11, 25), _ESTIMATOR[bits], strict=True):
        p = basefold_parameters(domain, num_vars, twist_field=twist_field)
        assert (p.k0, p.n0, p.d, p.rep) == (256, 1024, num_vars - 8, rep)
        assert p.num_vars == num_vars and p.schedule == (1,) * p.d


# (domain, twist_field): fields with whole-field and prime-subfield twists,
# and R_q with each, the ring's default being the prime.
_CASES = [
    ("f180", None),
    ("f180", "prime"),
    ("pmf260", None),
    ("ring", None),
    ("ring", "field"),
]


@pytest.mark.parametrize(("name", "twist_field"), _CASES)
def test_the_choice_meets_the_library_bounds(name, twist_field):
    domain = _DOMAINS[name]()
    security = 128
    p = basefold_parameters(domain, 10, security, twist_field=twist_field)
    options = {} if twist_field is None else {"twist_field": twist_field}
    if isinstance(domain, Field):
        code = FieldFoldableRS(domain, p.k0, p.c, p.d, **options)
    else:
        code = FoldableRS(domain, p.k0, p.c, p.d, **options)
    assert code.relative_distance(security) == p.distance
    protocol = BasefoldEval(Basefold(code, list(p.schedule)), rep=p.rep)
    error = protocol.soundness_error(_STATEMENT, protocol.domain, p.gamma, p.delta)
    assert error == p.soundness_error <= 2.0**-security
    fewer = basefold_soundness_error(
        domain.exceptional_set_size, p.d, 10, p.rep - 1, p.gamma, p.delta
    )
    assert fewer > 2.0**-security
    # The theorem's conditions, with delta at their supremum.
    johnson = lambda x: 1 - math.sqrt(1 - x * (1 - p.gamma))  # noqa: E731
    assert p.delta <= johnson(johnson(p.distance))
    assert 3 * p.delta - p.d * p.gamma <= p.distance * (1 + 1e-12)
    assert p.rep < p.n0


def test_prime_subfield_twists_cost_queries():
    field = _field_180()
    whole = basefold_parameters(field, 16)
    prime = basefold_parameters(field, 16, twist_field="prime")
    assert prime.distance < whole.distance and prime.rep > whole.rep


def test_batching_terms_are_charged():
    field = _field_180()
    single = basefold_parameters(field, 14)
    bundle = basefold_parameters(field, 14, claims=4, polynomials=2)
    assert bundle.soundness_error <= 2.0**-128
    assert bundle.proof_bits > single.proof_bits
    code = FieldFoldableRS(field, bundle.k0, bundle.c, bundle.d)
    protocol = BasefoldEval(Basefold(code), rep=bundle.rep)
    error = protocol.soundness_error(
        _STATEMENT, field, bundle.gamma, bundle.delta, claims=4, polynomials=2
    )
    assert error == bundle.soundness_error
    with pytest.raises(ValueError, match="polynomial"):
        basefold_parameters(field, 14, claims=1, polynomials=2)


def test_a_field_too_small_for_the_target_raises():
    field = Field(562949948178433, 2, 5)  # 100 bits
    with pytest.raises(ValueError, match="no basefold parameters"):
        basefold_parameters(field, 12, 128)
    p = basefold_parameters(field, 12, 64)
    assert p.soundness_error <= 2.0**-64 and p.rep < p.n0
    with pytest.raises(ValueError, match="num_vars"):
        basefold_parameters(field, 0)
    with pytest.raises(ValueError, match="c must be a power of two"):
        basefold_parameters(field, 12, 64, inverse_rates=(2, 3))
    with pytest.raises(ValueError, match="twist_field"):
        basefold_parameters(field, 12, 64, twist_field="extension")


def _count(message) -> tuple[int, int]:
    """(elements, digests) in one prover message."""
    if isinstance(message, (FieldElement, Polynomial)):
        return 1, 0
    if isinstance(message, bytes):
        assert len(message) == DIGEST_LEN
        return 0, 1
    if isinstance(message, MerklePath):
        return 0, len(message.siblings)
    if isinstance(message, MLE):
        return len(message.table), 0
    elements = digests = 0
    for part in message:
        e, h = _count(part)
        elements, digests = elements + e, digests + h
    return elements, digests


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


@pytest.mark.parametrize("kind", ["field", "ring"])
@pytest.mark.parametrize("bundle", [False, True])
def test_proof_bits_counts_the_messages_of_an_opening(kind, bundle):
    """One claim, or three claims at distinct points on two polynomials: the
    elements and digests of the argument string, against the count."""
    k0, c, d, rep, num_vars = 4, 4, 3, 5, 5
    variables = [MLE_Variable(f"x{i}") for i in range(num_vars)]
    names = (b"f", b"g")
    domain: Field | RNSRing
    if kind == "field":
        domain = field = _field_180()
        scheme = Basefold(FieldFoldableRS(field, k0, c, d))
        element_bits = math.log2(field.order)
        tables = [FieldVector(field, 1 << num_vars) for _ in names]
        for table, name in zip(tables, names, strict=True):
            table.sample_random(name)
        fs = [MLE(field=field, variables=variables, evaluations=t) for t in tables]
    else:
        domain = ring = Ring(64, prime_size=[30, 28], split_degree=4)
        scheme = Basefold(FoldableRS(ring, k0, c, d))
        element_bits = ring.N * sum(math.log2(p) for p in ring.primes)
        fs = [
            MLE(
                ring=ring,
                variables=variables,
                evaluations=[ring.random_element() for _ in range(1 << num_vars)],
            )
            for _ in names
        ]

    def point_at(seed: bytes) -> dict:
        if isinstance(domain, Field):
            return {
                v: domain.random_element(seed + bytes([i]))
                for i, v in enumerate(variables)
            }
        return {v: domain.random_exceptional() for v in variables}

    oracles = [scheme.commit(f)[0] for f in fs]

    def claim(which: int, seed: bytes) -> tuple:
        point = point_at(seed)
        value = fs[which].evaluate(point, in_place=False).constant()
        return (oracles[which], point, value)

    if bundle:
        claims = [claim(0, b"z"), claim(1, b"y"), claim(0, b"x")]
    else:
        claims = [claim(0, b"z")]
    iop = IOP(domain=domain, fiat_shamir=True)
    iop.register(_Claims, _FanOut())
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=rep))
    proof = iop.prove(Statement(_Claims(), claims=tuple(claims)))
    elements, digests = _count([message for _, message in proof.messages])
    polynomials = 2 if bundle else 1
    expected = _proof_bits(
        element_bits,
        8 * DIGEST_LEN,
        k0,
        c * k0,
        d,
        rep,
        num_vars,
        len(claims),
        polynomials,
    )
    assert elements * element_bits + digests * 8 * DIGEST_LEN == pytest.approx(expected)
