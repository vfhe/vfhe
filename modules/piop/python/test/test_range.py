# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The range relation and its decomposition protocol (Figure 13 of
[CCCFGS26]): the recomposition identity on its own, the protocol end to end,
the statements it reduces to, and the three ways a prover can cheat -- each
rejected by the branch that is supposed to catch it."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Polynomial, PseudoMersenneField, Ring
from vfhe.piop import (
    IOP,
    Protocol,
    RangeDecomposition,
    Relation_Eval,
    Relation_Lookup,
    Relation_Range,
    Statement,
    element_digest,
)
from vfhe.piop.range import balanced, base_digits, eq_int, uEval

# ell = 2, nu = 3, gamma = 1 -> ell* = 6, so mu' is a 64-entry table: small
# enough for the pure-Python degree-3 rounds the lookup argument will add.
BOUND = 1 << 6
CHUNKS = 2
ELL = 2
M = 2
ALPHABET = 1 << 8
KAPPA = 2


@pytest.fixture
def ring():
    return Ring(8, prime_size=[30, 30], split_degree=1)


@pytest.fixture
def field():
    # 260 bits is one of the two roomiest pseudo-Mersenne layouts, and the
    # same instance test_field_domain.py uses.
    return PseudoMersenneField.generate(260, two_adicity=8)


def _witness(ring, seed=0, bound=BOUND, tweak=None):
    """An `ELL`-variate oracle over `ring` with coefficients below `bound`.

    `tweak(coefficients)` may edit the first evaluation's coefficients in
    place, which is how the out-of-range case is built.
    """
    rng = random.Random(seed)
    table = []
    for i in range(1 << ELL):
        coefficients = [rng.randrange(bound) for _ in range(ring.N)]
        if i == 0 and tweak is not None:
            tweak(coefficients)
        table.append(Polynomial(ring).from_array(coefficients))
    variables = [MLE_Variable(f"x{i}") for i in range(ELL)]
    return MLE(ring=ring, variables=variables, evaluations=table)


def _protocol(field, **kw):
    params = {"m": M, "alphabet": ALPHABET, "kappa": KAPPA} | kw
    return RangeDecomposition(field, **params)


def _iop(field, protocol=None, **kw):
    iop = IOP(domain=field, **kw)
    iop.register(Relation_Range, protocol or _protocol(field))
    return iop


def _statement(v):
    return Statement(Relation_Range(BOUND, CHUNKS), oracles=[v])


# --- the relations' ideal deciders -----------------------------------------


def test_relation_range_check(ring):
    assert _statement(_witness(ring)).check()
    over = _witness(ring, tweak=lambda c: c.__setitem__(3, BOUND))
    assert not _statement(over).check()


def test_relation_range_rejects_bad_parameters():
    with pytest.raises(ValueError, match="power of two"):
        Relation_Range(100, 2)
    with pytest.raises(ValueError, match="must divide"):
        Relation_Range(1 << 5, 2)  # log2(B) = 5 is not a multiple of c = 2


def test_relation_lookup_check(field):
    variables = [MLE_Variable(f"y{i}") for i in range(3)]
    inside = MLE(
        field=field, variables=variables, evaluations=[field(i % 8) for i in range(8)]
    )
    assert Statement(Relation_Lookup(8), oracles=[inside]).check()
    outside = MLE(
        field=field,
        variables=variables,
        evaluations=[field(i % 8) if i else field(8) for i in range(8)],
    )
    assert not Statement(Relation_Lookup(8), oracles=[outside]).check()
    # A "negative" digit is a large canonical representative, not a small one.
    negative = MLE(
        field=field,
        variables=variables,
        evaluations=[field(i % 8) if i else -field(1) for i in range(8)],
    )
    assert not Statement(Relation_Lookup(8), oracles=[negative]).check()


# --- the recomposition identity, on its own --------------------------------


def _exact_uEval(v, r, ell, nu):
    """`unpack(v)` at `r`, computed over Z from v's integer coefficients.

    The honest witness has coefficients well below q, so this is the value
    `uEval` would return if nothing were ever reduced -- the reference the
    protocol's mod-q check is a shadow of.
    """
    tables = [list(p.get_polynomial()) for p in v.table]
    for i in range(ell):  # bind the X block, LSB-first
        t = r[i]
        tables = [
            [a + t * (b - a) for a, b in zip(lo, hi, strict=True)]
            for lo, hi in zip(tables[0::2], tables[1::2], strict=True)
        ]
    coefficients = tables[0]
    return sum(
        c * eq_int([(k >> j) & 1 for j in range(nu)], r[ell:])
        for k, c in enumerate(coefficients)
    )


def test_recomposition_identity(ring, field):
    """Figure 13's step-2 check, computed directly from an honest witness.

    Three separate facts, in increasing strength:

    1. every entry of `psi` lifts through `iota` to something well inside
       `I_p` -- the Claim of §I.5, `|mu(r^<=m, delta, j)| < 2^m beta sigma^m`,
       which is what makes the lift meaningful at all;
    2. the digits rebuild the coefficients **exactly over Z**, against v's
       integer coefficients;
    3. and hence congruently mod q against `uEval` of the ring element the
       verifier actually queries -- the form the protocol checks, and the
       only form available once v(x) is read out of R_q.
    """
    v = _witness(ring, seed=3)
    protocol = _protocol(field)
    statement = _statement(v)
    shape = protocol._shape(statement)
    digits = protocol._digit_table(v, shape)
    mu = MLE(
        field=field, variables=shape.variables, evaluations=[field(d) for d in digits]
    )
    claim_bound = (1 << M) * shape.beta * ALPHABET**M
    assert 2 * claim_bound < field.prime  # equation (38)

    rng = random.Random(11)
    for _ in range(4):
        r = [rng.randrange(ALPHABET) for _ in range(shape.ell + shape.nu)]
        psi = mu.evaluate(protocol._bind_prefix(shape, r), in_place=False)
        width = shape.ell + shape.nu - M
        left = 0
        for index, value in enumerate(psi.table):
            lifted = balanced(value, field.prime)
            assert abs(lifted) < claim_bound  # (1)
            delta = [(index >> t) & 1 for t in range(width)]
            left += lifted * eq_int(delta, r[M:]) * shape.beta ** (index >> width)

        assert left == _exact_uEval(v, r, shape.ell, shape.nu)  # (2)

        w = v.evaluate(
            {var: r[i] for i, var in enumerate(v.variables)}, in_place=False
        ).constant()
        assert (left - uEval(w, r[shape.ell :])) % ring.q_l == 0  # (3)


def test_digits_round_trip():
    width = (BOUND.bit_length() - 1) // CHUNKS
    for value in (0, 1, 7, 8, 63):
        digits = base_digits(value, width, CHUNKS)
        assert all(0 <= d < (1 << width) for d in digits)
        assert sum(d << (width * j) for j, d in enumerate(digits)) == value
    # Above the bound the high bits are dropped -- what a cheating prover
    # would have to submit, and what the recomposition check then catches.
    assert base_digits(BOUND, width, CHUNKS) == [0, 0]


# --- the protocol, end to end ----------------------------------------------


def test_decomposition_accepts_an_honest_witness(ring, field):
    v = _witness(ring, seed=1)
    iop = _iop(field)
    assert iop.run(_statement(v))
    assert iop.transcript.order[:4] == [
        "range/mu",
        "range/r0",
        "range/r1",
        "range/psi0",
    ]
    # The spot-check point is drawn only after every partial evaluation is in.
    assert iop.transcript.order.index("range/s0") > iop.transcript.order.index(
        f"range/psi{KAPPA - 1}"
    )


@pytest.mark.parametrize(
    ("bound", "chunks", "m", "kappa"),
    [
        (BOUND, CHUNKS, 0, 1),  # m = 0: psi is all of mu', and one repetition
        (BOUND, CHUNKS, 1, 3),  # a prefix strictly inside the X block
        (BOUND, CHUNKS, ELL, 2),  # m = ell: the whole X block bound
        (1 << 8, 4, 2, 2),  # gamma = 2: two digit variables, beta = 4
        (1 << 8, 1, 2, 2),  # c = 1: no digit block at all, beta = B
    ],
)
def test_decomposition_across_the_parameter_knobs(ring, field, bound, chunks, m, kappa):
    """m, kappa and c move the block boundaries of the digit table; each
    setting has to survive an honest run and reject a bad coefficient."""
    good = _witness(ring, seed=20 + m + chunks, bound=bound)
    statement = Statement(Relation_Range(bound, chunks), oracles=[good])
    protocol = RangeDecomposition(field, m=m, alphabet=ALPHABET, kappa=kappa)

    iop = IOP(domain=field)
    iop.register(Relation_Range, protocol)
    assert iop.run(statement)

    bad = _witness(ring, bound=bound, tweak=lambda c: c.__setitem__(1, bound + 1))
    iop = IOP(domain=field)
    iop.register(
        Relation_Range, RangeDecomposition(field, m=m, alphabet=ALPHABET, kappa=kappa)
    )
    assert not iop.run(Statement(Relation_Range(bound, chunks), oracles=[bad]))


def test_decomposition_emits_the_expected_bundle(ring, field):
    """kappa evaluation claims on mu', kappa on v, and one lookup claim."""
    v = _witness(ring, seed=2)
    seen = []

    class _Record(Protocol):
        reduces_from = Relation_Eval

        async def prove(self, _prover, _statements):
            return []

        async def verify(self, _verifier, statements):
            seen.extend(statements)
            return []

    iop = _iop(field)
    iop.register(Relation_Eval, _Record())
    assert iop.run(_statement(v))

    assert len(seen) == 2 * KAPPA
    on_mu = [s for s in seen if s.oracles[0].field is field]
    on_v = [s for s in seen if s.oracles[0].ring is not None]
    assert len(on_mu) == KAPPA and len(on_v) == KAPPA
    # Every one of them is a true claim about the honest witness.
    assert all(s.check() for s in seen)
    # The kappa claims on mu' share the spot-check block and differ in the
    # prefix, so no two of them are the same query.
    assert on_mu[0].point != on_mu[1].point


def test_lookup_claim_is_terminal_and_true(ring, field):
    v = _witness(ring, seed=4)
    seen = []

    class _Record(Protocol):
        reduces_from = Relation_Lookup

        async def prove(self, _prover, _statements):
            return []

        async def verify(self, _verifier, statements):
            seen.extend(statements)
            return []

    iop = _iop(field)
    iop.register(Relation_Lookup, _Record())
    assert iop.run(_statement(v))
    (claim,) = seen
    assert claim.relation.beta == Relation_Range(BOUND, CHUNKS).beta
    assert claim.check()


# --- the three ways to cheat -----------------------------------------------


class _Tampered(RangeDecomposition):
    """A prover that submits a digit table of its own choosing."""

    def __init__(self, *args, corrupt, **kw):
        super().__init__(*args, **kw)
        self._corrupt = corrupt

    def _digit_table(self, v, shape):
        table = super()._digit_table(v, shape)
        self._corrupt(table, shape)
        return table


def _cheat(field, corrupt):
    return _Tampered(field, m=M, alphabet=ALPHABET, kappa=KAPPA, corrupt=corrupt)


class _Swallow(Protocol):
    """Discharges a relation by accepting it, so a test can pin which of the
    two branches actually rejected."""

    def __init__(self, relation):
        self.reduces_from = relation

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, _statements):
        return []


def test_out_of_range_coefficient_is_rejected(ring, field):
    """Case 1: a coefficient >= B. Its digits truncate, so they no longer
    recompose, and the mod-q check of step 2 fails."""
    v = _witness(ring, tweak=lambda c: c.__setitem__(3, BOUND + 5))
    assert not _statement(v).check()  # the claim really is false
    assert not _iop(field).run(_statement(v))

    # The digits themselves stay inside t_beta here -- truncation keeps them
    # small -- so the lookup branch has nothing to say, and step 2 is the
    # only thing standing between this witness and an accepting run.
    iop = _iop(field)
    iop.register(Relation_Lookup, _Swallow(Relation_Lookup))
    assert not iop.run(_statement(v))


def test_perturbed_digit_is_rejected(ring, field):
    """Case 2: an honest witness, but the prover alters one digit. The digits
    stop recomposing, so again the mod-q check fails."""
    v = _witness(ring, seed=5)

    def corrupt(table, shape):
        table[0] = (table[0] + 1) % shape.beta

    assert not _iop(field, _cheat(field, corrupt)).run(_statement(v))


def test_borrowed_digit_passes_recomposition_but_fails_the_lookup(ring, field):
    """Case 3: the prover moves a unit between digits -- beta from the high
    digit into the low one. The value is unchanged, so step 2 accepts; the
    low digit is now >= beta, so the lookup claim is what rejects."""
    v = _witness(ring, seed=6)
    moved = {}

    def corrupt(table, shape):
        stride = 1 << (shape.ell + shape.nu)  # one step in the digit block
        for i in range(stride):
            if table[i + stride]:  # a high digit we can borrow from
                table[i] += shape.beta
                table[i + stride] -= 1
                moved["at"] = i
                return
        raise AssertionError("no high digit to borrow from")

    protocol = _cheat(field, corrupt)
    assert not _iop(field, protocol).run(_statement(v))
    assert "at" in moved

    # Pin the claim: it is the lookup branch that rejects, not step 2. With
    # the lookup claim swallowed, the same tampered run is accepted.
    iop = _iop(field, _cheat(field, corrupt))
    iop.register(Relation_Lookup, _Swallow(Relation_Lookup))
    assert iop.run(_statement(v))


def _replay_with_tamper(honest, field, statement, tamper):
    """A verifier-only run over an honest transcript with one message
    rewritten. The recorded challenges are replayed as they were, so this is
    the honest interaction with exactly one dishonest prover message."""
    iop = _iop(field)
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    return iop.loop.run_until_complete(iop.verifier.verify(statement._fork()))


def test_replay_harness_accepts_an_untampered_transcript(ring, field):
    v = _witness(ring, seed=12)
    statement = _statement(v)
    honest = _iop(field)
    assert honest.run(statement)
    assert _replay_with_tamper(honest, field, statement, lambda _label, value: value)


def test_every_prover_message_is_load_bearing(ring, field):
    """Rewriting any single prover message makes the run reject.

    `mu` and `psi` are tables of field elements, `w` a ring element; the
    tamper adds one to the first entry of whichever it is. Nothing the prover
    sends is decorative: `mu` and `w` feed the recomposition check, and `psi`
    feeds both that and the evaluation claim that pins it to `mu`.
    """
    v = _witness(ring, seed=13)
    statement = _statement(v)
    honest = _iop(field)
    assert honest.run(statement)

    messages = [
        label
        for label in honest.transcript.order
        if label not in honest.transcript.derived
    ]
    assert messages == [
        "range/mu",
        *(f"range/{k}{eta}" for eta in range(KAPPA) for k in ("psi", "w")),
    ]

    for target in messages:

        def tamper(label, value, target=target):
            if label != target:
                return value
            if isinstance(value, tuple):
                return (value[0] + field.one, *value[1:])
            return value + Polynomial(value.ring).from_array([1])

        assert not _replay_with_tamper(honest, field, statement, tamper), target


# --- Fiat-Shamir ------------------------------------------------------------


def test_fiat_shamir_round_trip(ring, field):
    v = _witness(ring, seed=8)
    statement = _statement(v)
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert _iop(field, fiat_shamir=True).verify(statement, proof)
    # Prover messages only: the challenges are re-derived, never carried.
    assert not any(label.startswith("range/r") for label in proof.labels)
    assert not any(label.startswith("range/s") for label in proof.labels)


def test_fiat_shamir_is_deterministic(ring, field):
    v = _witness(ring, seed=9)
    statement = _statement(v)
    digests = []
    for _ in range(2):
        iop = _iop(field, fiat_shamir=True)
        assert iop.run(statement)
        digests.append(
            [
                (label, element_digest(iop.transcript.entries[label].result()))
                for label in iop.transcript.order
            ]
        )
    assert digests[0] == digests[1]


def test_fiat_shamir_rejects_a_tampered_proof(ring, field):
    v = _witness(ring, seed=10)
    statement = _statement(v)
    proof = _iop(field, fiat_shamir=True).prove(statement)
    messages = list(proof.messages)
    label, table = messages[0]
    assert label == "range/mu"
    messages[0] = (label, (table[0] + field.one, *table[1:]))
    assert not _iop(field, fiat_shamir=True).verify(statement, type(proof)(messages))


# --- parameter conditions ---------------------------------------------------


def test_parameters_are_validated(ring, field):
    v = _witness(ring)
    statement = _statement(v)

    with pytest.raises(ValueError, match="power of two"):
        RangeDecomposition(field, m=M, alphabet=100, kappa=KAPPA)
    with pytest.raises(ValueError, match="kappa"):
        RangeDecomposition(field, m=M, alphabet=ALPHABET, kappa=0)

    # These three are conditions on the claim as well as on the constructor,
    # so they surface on first use rather than at construction.
    with pytest.raises(ValueError, match="must not exceed the number of variables"):
        RangeDecomposition(field, m=ELL + 1, alphabet=ALPHABET, kappa=KAPPA)._shape(
            statement
        )
    # sigma so large that 2*beta*(2*sigma)^m overruns a 260-bit prime.
    with pytest.raises(ValueError, match="equation 38"):
        RangeDecomposition(field, m=M, alphabet=1 << 128, kappa=KAPPA)._shape(statement)
    # sigma above the smallest RNS prime leaves the exceptional set of Z_q.
    too_wide = 1 << (min(ring.primes).bit_length() + 1)
    with pytest.raises(ValueError, match="smallest RNS prime"):
        RangeDecomposition(field, m=0, alphabet=too_wide, kappa=KAPPA)._shape(statement)


def test_soundness_error_is_the_theorem_bound(ring, field):
    v = _witness(ring)
    protocol = _protocol(field)
    reported = protocol.soundness_error(_statement(v))
    nu = ring.N.bit_length() - 1
    assert reported == max(
        ((ELL + nu) / ALPHABET) ** KAPPA, (ELL + nu + 1 - M) / field.prime
    )
