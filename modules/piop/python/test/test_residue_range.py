# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The residue range relation and its decomposition protocol: the optimized
range PIOP of [CCCFGS26] on an integer table held as one oracle per prime.
The relation's decider, the protocol end to end and as an argument string,
the statements it reduces to, the digit table against a per-entry
reference, the CRT case of a bound above every prime, the ways a prover can
cheat, and agreement with `Relation_Range` on the unpacked table of a ring
oracle."""

import math
import random
from array import array

import pytest
from vfhe.arith import (
    MLE,
    ExtensionFieldElement,
    Field,
    FieldVector,
    MLE_Variable,
    Polynomial,
    PseudoMersenneField,
    Ring,
    crt,
)
from vfhe.piop import IOP, Protocol, Rejection, Relation_Eval, Statement, TableOracle
from vfhe.piop.protocols import RangeDecomposition, ResidueRangeDecomposition
from vfhe.piop.protocols.range import base_digits, digit_alphabets
from vfhe.piop.relations import (
    Relation_Lookup,
    Relation_Range,
    Relation_ResidueRange,
)
from vfhe.piop.relations.range import residue_primes

# n = 5, gamma = 1 -> ell* = 6, so mu' is a 64-entry table.
BOUND = 1 << 6
CHUNKS = 2
N_VARS = 5
M = 2
ALPHABET = 1 << 8
KAPPA = 2

#: A 50-bit prime with x^2 - 5 irreducible, and a 60-bit prime.
P50 = 562949948178433
P60 = 1152921504577486849


@pytest.fixture
def field():
    return PseudoMersenneField.generate(260, two_adicity=8)


@pytest.fixture
def fields():
    """Two residue fields: one of degree 2, one prime field."""
    return [Field(P50, 2, 5), Field(P60)]


@pytest.fixture
def ring():
    """Two 30-bit primes: a bound of 2^40 is above each of them."""
    return Ring(32, prime_size=[30, 30], split_degree=1)


def _values(seed=0, bound=BOUND, n=N_VARS, tweak=None):
    """`2^n` integers below `bound`; `tweak(values)` may edit them in place."""
    rng = random.Random(seed)
    values = [rng.randrange(bound) for _ in range(1 << n)]
    if tweak is not None:
        tweak(values)
    return values


def _residue_table(field, values, n=N_VARS):
    """The residues of `values` modulo the field's prime, as a table over
    `field` on fresh variables."""
    vec = FieldVector(field, 1 << n)
    vec.set_uint64(array("Q", [x % field.prime for x in values]))
    variables = [MLE_Variable(f"u{i}") for i in range(n)]
    return MLE(field=field, variables=variables, evaluations=vec)


def _oracles(fields, values, n=N_VARS):
    return [TableOracle(_residue_table(f, values, n)) for f in fields]


def _statement(oracles, bound=BOUND):
    return Statement(Relation_ResidueRange(bound), oracles=oracles)


def _protocol(field, **kw):
    kw.setdefault("m", M)
    kw.setdefault("alphabet", ALPHABET)
    kw.setdefault("kappa", KAPPA)
    kw.setdefault("chunks", CHUNKS)
    return ResidueRangeDecomposition(field, **kw)


def _iop(field, protocol=None, domain=None, chunks=CHUNKS, **kw):
    iop = IOP(domain=field if domain is None else domain, **kw)
    iop.register(Relation_ResidueRange, protocol or _protocol(field, chunks=chunks))
    return iop


def _most_acceptances(runs, error, false_failure=1e-6):
    """The most of `runs` independent cheating runs a protocol of soundness
    error `error` accepts, except with probability below `false_failure`
    (the binomial tail)."""
    tail = 1.0
    for t in range(runs + 1):
        tail -= math.comb(runs, t) * error**t * (1 - error) ** (runs - t)
        if tail < false_failure:
            return t
    return runs


def _assert_rarely_accepted(make_iop, statement, error, runs=64):
    """Run the false `statement` on `runs` fresh IOPs from `make_iop` and
    hold the acceptances to what a soundness error of `error` allows."""
    accepted = sum(bool(make_iop().run(statement)) for _ in range(runs))
    assert accepted <= _most_acceptances(runs, error)


class _Record(Protocol):
    """Discharges a relation by recording its statements unread."""

    def __init__(self, relation, seen):
        self.reduces_from = relation
        self.seen = seen

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, statements):
        self.seen.extend(statements)
        return []


# --- the relation's ideal decider ------------------------------------------


def test_relation_residue_range_check(fields):
    """Honest residues are accepted; an entry at the bound, a tuple whose
    residues are of no integer below `B`, and an entry with a nonzero `x`
    coefficient (outside the prime subfield of `F_(p^2)`) are not."""
    assert _statement(_oracles(fields, _values())).check()

    over = _values(tweak=lambda v: v.__setitem__(3, BOUND))
    assert not _statement(_oracles(fields, over)).check()

    # One prime's residue moved by one: the integer is still x modulo the
    # other prime but not equal to x, so it is at least that prime.
    oracles = _oracles(fields, _values())
    table = oracles[0].polynomial.table
    table[5] = table[5] + fields[0](1)
    assert not _statement(oracles).check()

    oracles = _oracles(fields, _values())
    oracles[0].polynomial.table[2] = ExtensionFieldElement(fields[0], [0, 1])
    assert not _statement(oracles).check()


def test_residue_primes(fields):
    oracles = _oracles(fields, _values())
    assert residue_primes(oracles) == [P50, P60]
    with pytest.raises(ValueError, match="distinct primes"):
        residue_primes(_oracles([fields[0], Field(P50, 2, 5)], _values()))
    shorter = _residue_table(fields[1], _values(n=N_VARS - 1), n=N_VARS - 1)
    with pytest.raises(ValueError, match="one number of variables"):
        residue_primes([oracles[0], TableOracle(shorter)])
    plain = TableOracle(
        MLE(
            variables=[MLE_Variable(f"u{i}") for i in range(N_VARS)],
            evaluations=_values(),
        )
    )
    with pytest.raises(TypeError, match="characteristic"):
        residue_primes([oracles[0], plain])


# --- the protocol, end to end ----------------------------------------------


def test_decomposition_accepts_an_honest_witness(field, fields):
    iop = _iop(field)
    assert iop.run(_statement(_oracles(fields, _values(seed=1))))
    assert iop.transcript.order[:4] == [
        "residue_range/mu",
        "residue_range/r0",
        "residue_range/r1",
        "residue_range/psi0",
    ]
    order = iop.transcript.order
    assert order.index("residue_range/w0") == order.index("residue_range/psi0") + 1
    # The spot-check point is drawn only after every answer is in.
    assert order.index("residue_range/s0") > order.index(f"residue_range/w{KAPPA - 1}")


@pytest.mark.parametrize(
    ("bound", "chunks", "lookups"),
    [(BOUND, CHUNKS, 1), (1 << 7, 2, 2)],
    ids=["divisible", "short-top-digit"],
)
def test_decomposition_emits_the_expected_bundle(field, fields, bound, chunks, lookups):
    """kappa evaluation claims on mu', kappa * U on the residue oracles at
    the integer points read in each oracle's field, and one lookup claim
    (two with a short top digit); every one of them true."""
    oracles = _oracles(fields, _values(seed=2, bound=bound))
    evals, looks = [], []
    iop = _iop(field, chunks=chunks)
    iop.register(Relation_Eval, _Record(Relation_Eval, evals))
    iop.register(Relation_Lookup, _Record(Relation_Lookup, looks))
    assert iop.run(_statement(oracles, bound))

    assert len(evals) == KAPPA + KAPPA * len(oracles)
    on_mu = [s for s in evals if s.oracles[0] not in oracles]
    assert len(on_mu) == KAPPA
    assert all(s.oracles[0].domain is field for s in on_mu)
    for oracle in oracles:
        on_u = [s for s in evals if s.oracles[0] is oracle]
        assert len(on_u) == KAPPA
        for s in on_u:
            assert all(c.field is oracle.domain for c in s.point.values())
            assert s.value == oracle.query(s.point)
    assert len(looks) == lookups
    beta, top_beta = digit_alphabets(bound, chunks)
    assert [s.relation.beta for s in looks] == [beta, top_beta][:lookups]
    assert all(s.check() for s in evals + looks)


@pytest.mark.parametrize("m", [0, 1, N_VARS])
@pytest.mark.parametrize("kappa", [1, 3])
@pytest.mark.parametrize(
    ("bound", "chunks"),
    [(1 << 6, 2), (1 << 8, 4), (1 << 8, 1), (1 << 7, 2)],
    ids=["default", "gamma-2", "c-1", "short-top"],
)
def test_decomposition_across_the_parameter_knobs(
    field, fields, bound, chunks, m, kappa
):
    """m, kappa and c move the block boundaries of the digit table; each
    setting has to survive an honest run and reject an entry above the
    bound, up to the soundness error (about 2% with one repetition)."""
    good = _statement(_oracles(fields, _values(seed=20 + m, bound=bound)), bound)
    assert _iop(field, _protocol(field, m=m, kappa=kappa, chunks=chunks)).run(good)

    bad = _values(seed=21, bound=bound, tweak=lambda v: v.__setitem__(1, bound + 1))
    cheating = _statement(_oracles(fields, bad), bound)
    assert not cheating.check()
    _assert_rarely_accepted(
        lambda: _iop(field, _protocol(field, m=m, kappa=kappa, chunks=chunks)),
        cheating,
        _protocol(field, m=m, kappa=kappa, chunks=chunks).soundness_error(cheating),
    )


def test_a_bound_above_every_prime(field, ring):
    """Two 30-bit primes and a 40-bit bound: entries above each prime are
    held only by their residue pair, and the digits are those of the CRT
    lift. An honest table accepts; an entry equal to `B` is rejected up to
    the soundness error."""
    bound, chunks = 1 << 40, 4
    residue_fields = [ring.residue_field(i) for i in range(len(ring.primes))]

    def tweak(v):
        v[0] = max(ring.primes) + 1  # above both primes

    honest = _values(seed=30, bound=bound, tweak=tweak)
    assert sum(x > max(ring.primes) for x in honest) > len(honest) // 2
    statement = _statement(_oracles(residue_fields, honest), bound)
    assert statement.check()
    assert _iop(field, chunks=chunks).run(statement)

    bad = _values(seed=30, bound=bound, tweak=lambda v: v.__setitem__(4, bound))
    cheating = _statement(_oracles(residue_fields, bad), bound)
    assert not cheating.check()
    _assert_rarely_accepted(
        lambda: _iop(field, chunks=chunks),
        cheating,
        _protocol(field, chunks=chunks).soundness_error(cheating),
    )


def _reference_digits(values, primes, n, bound, chunks):
    """The digit table of `mu'` computed entry by entry: digit `j` of the
    integer with the residues of entry `i` at `i + 2^n j`."""
    width = digit_alphabets(bound, chunks)[0].bit_length() - 1
    table = [0] * ((1 << n) * chunks)
    for i, x in enumerate(values):
        value = crt([x % p for p in primes], primes)
        for j, digit in enumerate(base_digits(value, width, chunks)):
            table[i + (1 << n) * j] = digit
    return table


@pytest.mark.parametrize(
    ("bound", "chunks", "large", "fits"),
    [
        (1 << 6, 2, False, True),
        (1 << 8, 4, False, True),
        (1 << 8, 1, False, True),
        (1 << 7, 2, False, True),
        (1 << 40, 4, False, True),
        (1 << 128, 2, True, True),
        (1 << 70, 1, True, False),
    ],
    ids=["default", "gamma-2", "c-1", "short-top", "crt", "64-bit-digits", "per-entry"],
)
def test_digit_table_is_the_per_entry_one(
    field, fields, ring, bound, chunks, large, fits
):
    """The digit table, built in whole-array steps from the CRT lift modulo
    2^128 or entry by entry when a digit is wider than 64 bits, holds the
    digits of the integers the residues define, LSB-first."""
    if large:
        residue_fields = fields
        q = P50 * P60
    else:
        residue_fields = [ring.residue_field(i) for i in range(len(ring.primes))]
        q = math.prod(ring.primes)
    values = _values(seed=40, bound=min(bound, q))
    statement = _statement(_oracles(residue_fields, values), bound)
    protocol = _protocol(field, chunks=chunks)
    shape = protocol._shape(statement)
    assert protocol._digits_fit(shape) is fits
    mu = protocol._digit_mle(statement, IOP(domain=field).prover, shape)
    primes = [f.prime for f in residue_fields]
    expected = _reference_digits(values, primes, N_VARS, bound, chunks)
    assert [int(e) for e in mu.table] == expected


# --- cheating ---------------------------------------------------------------


class _Answering(ResidueRangeDecomposition):
    """A prover that rewrites its answers at the integer points, and a
    verifier that records why it rejected."""

    def __init__(self, *args, rewrite, **kw):
        super().__init__(*args, **kw)
        self._rewrite = rewrite
        self.rejections = []

    def _query(self, statement, prover, shape, r):
        return self._rewrite(super()._query(statement, prover, shape, r))

    def _check_value(self, *args, **kw):
        try:
            super()._check_value(*args, **kw)
        except Rejection as rejection:
            self.rejections.append(str(rejection))
            raise


def test_out_of_range_entry_is_rejected(field, fields):
    """An entry at `B + 5`: its digits truncate, so they do not recompose,
    up to the soundness error."""
    bad = _values(tweak=lambda v: v.__setitem__(3, BOUND + 5))
    statement = _statement(_oracles(fields, bad))
    assert not statement.check()
    _assert_rarely_accepted(
        lambda: _iop(field), statement, _protocol(field).soundness_error(statement)
    )


@pytest.mark.parametrize("index", [0, 1])
def test_a_wrong_answer_is_rejected(field, fields, index):
    """One residue answer moved by one: the recomposition modulo that prime
    fails at every point, so every run rejects."""

    def rewrite(answers):
        answers = list(answers)
        answers[index] = answers[index] + fields[index](1)
        return tuple(answers)

    protocol = _Answering(field, M, ALPHABET, KAPPA, CHUNKS, rewrite=rewrite)
    assert not _iop(field, protocol).run(_statement(_oracles(fields, _values(seed=3))))
    assert protocol.rejections
    assert "does not recompose" in protocol.rejections[0]


def test_an_answer_outside_the_prime_subfield_is_rejected(field, fields):
    """An answer of the degree-2 oracle with a nonzero `x` coefficient is the
    residue of no integer; an answer in another field is not an element of
    the oracle's field at all. Both reject."""

    def lift(answers):
        return (answers[0] + ExtensionFieldElement(fields[0], [0, 1]), *answers[1:])

    protocol = _Answering(field, M, ALPHABET, KAPPA, CHUNKS, rewrite=lift)
    assert not _iop(field, protocol).run(_statement(_oracles(fields, _values(seed=4))))
    assert "does not recompose" in protocol.rejections[0]

    def swap(answers):
        return (fields[1](int(answers[0])), *answers[1:])

    protocol = _Answering(field, M, ALPHABET, KAPPA, CHUNKS, rewrite=swap)
    assert not _iop(field, protocol).run(_statement(_oracles(fields, _values(seed=4))))
    assert "outside its oracle's field" in protocol.rejections[0]


# --- Fiat-Shamir ------------------------------------------------------------


def test_fiat_shamir_round_trip(field, fields):
    statement = _statement(_oracles(fields, _values(seed=8)))
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert _iop(field, fiat_shamir=True).verify(statement, proof)
    # Prover messages only: the challenges are re-derived, never carried.
    assert not any(label.startswith("residue_range/r") for label in proof.labels)
    assert not any(label.startswith("residue_range/s") for label in proof.labels)


def test_fiat_shamir_is_deterministic(field, fields):
    statement = _statement(_oracles(fields, _values(seed=9)))
    proofs = [_iop(field, fiat_shamir=True).prove(statement) for _ in range(2)]
    assert proofs[0].messages == proofs[1].messages
    assert proofs[0].digest() == proofs[1].digest()


@pytest.mark.parametrize("target", ["residue_range/w0", "residue_range/psi0"])
@pytest.mark.parametrize("entry", [0, 1])
def test_fiat_shamir_rejects_a_tampered_proof(field, fields, target, entry):
    statement = _statement(_oracles(fields, _values(seed=10)))
    proof = _iop(field, fiat_shamir=True).prove(statement)
    messages = list(proof.messages)
    (at,) = [i for i, (label, _) in enumerate(messages) if label == target]
    label, value = messages[at]
    one = value[entry].field(1)
    messages[at] = (
        label,
        tuple(e + one if i == entry else e for i, e in enumerate(value)),
    )
    assert not _iop(field, fiat_shamir=True).verify(statement, type(proof)(messages))


# --- the IOP's domain, and parameter conditions -----------------------------


def test_the_decomposition_draws_in_its_field_inside_another_iop(field, fields):
    """In an IOP over a residue field the decomposition still draws its spot
    check in `F_p`, so the run accepts and its claims are true."""
    seen = []
    iop = _iop(field, domain=Field(P50, 2, 5))
    iop.register(Relation_Eval, _Record(Relation_Eval, seen))
    iop.register(Relation_Lookup, _Record(Relation_Lookup, seen))
    assert iop.run(_statement(_oracles(fields, _values(seed=5))))
    assert len(seen) == KAPPA * (1 + len(fields)) + 1
    assert all(s.check() for s in seen)


def test_parameters_are_validated(field, ring):
    residue_fields = [ring.residue_field(i) for i in range(len(ring.primes))]
    statement = _statement(_oracles(residue_fields, _values()))
    # sigma above the smallest prime leaves the exceptional set of a field.
    with pytest.raises(ValueError, match="smallest RNS prime"):
        _protocol(field, m=0, alphabet=1 << 31)._shape(statement)
    with pytest.raises(ValueError, match="must not exceed the variables"):
        _protocol(field, m=N_VARS + 1)._shape(statement)


# --- agreement with the ring relation ---------------------------------------

ELL = 2
NU = 5


def _ring_witness(ring, seed=0, tweak=None):
    """An `ELL`-variate oracle over `ring` with coefficients below `BOUND`;
    `tweak(coefficients)` may edit the first evaluation's coefficients."""
    rng = random.Random(seed)
    table = []
    for i in range(1 << ELL):
        coefficients = [rng.randrange(BOUND) for _ in range(ring.N)]
        if i == 0 and tweak is not None:
            tweak(coefficients)
        table.append(Polynomial(ring).from_array(coefficients))
    variables = [MLE_Variable(f"x{i}") for i in range(ELL)]
    return MLE(ring=ring, variables=variables, evaluations=table)


def _unpacked(ring, v):
    """The residue oracles of `unpack(v)`: entry `b + 2^ELL k` of table `i`
    is coefficient `k` of `v(b)` modulo prime `i`."""
    coefficients = [list(e.get_polynomial()) for e in v.table]
    values = [coefficients[b][k] for k in range(ring.N) for b in range(1 << ELL)]
    n = ELL + NU
    return [
        TableOracle(_residue_table(ring.residue_field(i), values, n))
        for i in range(len(ring.primes))
    ]


@pytest.mark.parametrize("tampered", [False, True])
def test_agrees_with_the_range_relation_on_the_unpacked_table(field, ring, tampered):
    """`Relation_ResidueRange` on the residues of `unpack(v)` decides as
    `Relation_Range` on `v`, and the residue protocol builds the digit table
    the ring protocol does, entry for entry."""
    assert ring.N == 1 << NU
    tweak = (lambda c: c.__setitem__(3, BOUND + 5)) if tampered else None
    v = _ring_witness(ring, seed=50, tweak=tweak)
    ring_statement = Statement(Relation_Range(BOUND), oracles=[TableOracle(v)])
    residue_statement = _statement(_unpacked(ring, v))
    assert residue_statement.check() == ring_statement.check() == (not tampered)

    residue_protocol = _protocol(field)
    ring_protocol = RangeDecomposition(field, M, ALPHABET, KAPPA, ring, CHUNKS)
    prover = IOP(domain=field).prover
    ours = residue_protocol._digit_mle(
        residue_statement, prover, residue_protocol._shape(residue_statement)
    )
    theirs = ring_protocol._digit_mle(
        ring_statement, prover, ring_protocol._shape(ring_statement)
    )
    assert len(ours.table) == len(theirs.table) == 1 << (ELL + NU + 1)
    assert all(a == b for a, b in zip(ours.table, theirs.table, strict=True))

    if not tampered:
        assert _iop(field).run(residue_statement)
