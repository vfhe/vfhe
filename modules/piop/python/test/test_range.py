# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The range relation and its decomposition protocol (the optimized range
PIOP `Pi*_range` of [CCCFGS26]): the recomposition identity on its own, the
protocol end to end, the statements it reduces to, and the three ways a prover can cheat -- each
rejected by the branch that is supposed to catch it."""

import math
import random

import pytest
from vfhe.arith import (
    MLE,
    FieldVector,
    MLE_Variable,
    Polynomial,
    PseudoMersenneField,
    Ring,
)
from vfhe.piop import (
    IOP,
    CommittedOracle,
    OracleKind,
    Protocol,
    Rejection,
    Relation_Eval,
    Statement,
    TableOracle,
    VirtualOracle,
    element_digest,
)
from vfhe.piop.protocols import Lookup, RangeDecomposition, Sumcheck, VirtualEval
from vfhe.piop.protocols.range import (
    base_digits,
    centered,
    digit_alphabets,
    digit_count,
    eq_int,
    unpacked_evaluation,
)
from vfhe.piop.relations import Relation_Lookup, Relation_Range, Relation_Sum
from vfhe.piop.virtual import resolve_table

# ell = 2, nu = 5, gamma = 1 -> ell* = 8, so mu' is a 256-entry table.
BOUND = 1 << 6
CHUNKS = 2
ELL = 2
NU = 5
M = 2
ALPHABET = 1 << 8
KAPPA = 2


def _bumped(value, one):
    """`value`, a table message, with `one` added to its first entry: a
    copy of a vector, or a new tuple."""
    if isinstance(value, FieldVector):
        out = value.copy()
        out[0] = out[0] + one
        return out
    return (value[0] + one, *value[1:])


@pytest.fixture
def ring():
    return Ring(1 << NU, prime_size=[30, 30], split_degree=1)


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


def _protocol(field, ring, **kw):
    kw.setdefault("m", M)
    kw.setdefault("alphabet", ALPHABET)
    kw.setdefault("kappa", KAPPA)
    kw.setdefault("chunks", CHUNKS)
    return RangeDecomposition(field, ring=ring, **kw)


def _iop(ring, field, protocol=None, chunks=CHUNKS, **kw):
    iop = IOP(domain=field, **kw)
    iop.register(Relation_Range, protocol or _protocol(field, ring, chunks=chunks))
    return iop


def _statement(v):
    """The range claim on the table `v`, named by a fresh oracle."""
    return Statement(Relation_Range(BOUND), oracles=[TableOracle(v)])


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


# --- the relations' ideal deciders -----------------------------------------


def test_relation_range_check(ring):
    assert _statement(_witness(ring)).check()
    over = _witness(ring, tweak=lambda c: c.__setitem__(3, BOUND))
    assert not _statement(over).check()


def test_relation_range_takes_any_positive_bound(ring):
    """The index is the bound alone, any positive integer: a bound that is
    not a power of two is still decided exactly."""
    assert Relation_Range(100).bound == 100
    for bad in (0, -4):
        with pytest.raises(ValueError, match="positive integer"):
            Relation_Range(bad)
    below = _witness(ring, seed=3, bound=100)
    assert Statement(Relation_Range(100), oracles=[TableOracle(below)]).check()
    at = _witness(ring, seed=3, bound=100, tweak=lambda c: c.__setitem__(3, 100))
    assert not Statement(Relation_Range(100), oracles=[TableOracle(at)]).check()


def test_digit_alphabets_rejects_bad_parameters():
    with pytest.raises(ValueError, match="B = 2\\^b"):
        digit_alphabets(100, 2)
    with pytest.raises(ValueError, match="B = 2\\^b"):
        digit_alphabets(1, 1)
    for chunks in (0, 3):
        with pytest.raises(ValueError, match="power of two"):
            digit_alphabets(1 << 6, chunks)
    # Four digits of two bits leave nothing for the top one of a 5-bit bound.
    with pytest.raises(ValueError, match="no bit"):
        digit_alphabets(1 << 5, 4)


def test_the_decomposition_needs_a_power_of_two_bound(ring, field):
    """The protocol checks `chunks` when built and the bound when it first
    sees a statement: a bound that is not a power of two has no digit
    decomposition."""
    with pytest.raises(ValueError, match="power of two"):
        _protocol(field, ring, chunks=3)
    statement = Statement(
        Relation_Range(100), oracles=[TableOracle(_witness(ring, bound=100))]
    )
    with pytest.raises(ValueError, match="B = 2\\^b"):
        _protocol(field, ring)._shape(statement)


@pytest.mark.parametrize(
    ("bits", "chunks", "beta", "top_beta"),
    [
        (6, 2, 8, 8),
        (8, 4, 4, 4),
        (8, 1, 256, 256),
        (5, 2, 8, 4),
        (49, 4, 1 << 13, 1 << 10),
    ],
)
def test_digit_alphabets(bits, chunks, beta, top_beta):
    """`c - 1` digits of `ceil(log2(B) / c)` bits and a top digit of what is
    left: the same alphabet when `c` divides `log2(B)`."""
    assert digit_alphabets(1 << bits, chunks) == (beta, top_beta)
    assert beta ** (chunks - 1) * top_beta == 1 << bits


@pytest.mark.parametrize(
    ("bits", "max_digit_bits", "chunks"),
    [
        (6, 6, 1),  # one digit already fits
        (6, 5, 2),
        (8, 2, 4),
        (5, 3, 2),  # a short top digit: 3 + 2 bits
        (44, 16, 4),  # 22-bit halves are too wide
        (55, 16, 4),
        (87, 16, 8),
        (109, 16, 8),
    ],
)
def test_digit_count_is_the_fewest_digits_that_fit(bits, max_digit_bits, chunks):
    assert digit_count(1 << bits, max_digit_bits) == chunks
    beta, _ = digit_alphabets(1 << bits, chunks)
    assert beta <= 1 << max_digit_bits
    if chunks > 1:
        assert digit_alphabets(1 << bits, chunks // 2)[0] > 1 << max_digit_bits


def test_digit_count_rejects_what_no_power_of_two_decomposes():
    with pytest.raises(ValueError, match="B = 2\\^b"):
        digit_count(100, 4)
    with pytest.raises(ValueError, match="at least 1"):
        digit_count(1 << 8, 0)
    # Nine bits in digits of at most three: four digits leave nothing for
    # the top one, and eight digits of two bits overshoot.
    with pytest.raises(ValueError, match="no power-of-two"):
        digit_count(1 << 9, 3)


def test_chunks_or_max_digit_bits(ring, field):
    for kw in ({"chunks": None}, {"max_digit_bits": 4}):
        with pytest.raises(ValueError, match="exactly one"):
            _protocol(field, ring, **kw)
    with pytest.raises(ValueError, match="at least 1"):
        _protocol(field, ring, chunks=None, max_digit_bits=0)


def test_max_digit_bits_sets_the_digit_count_per_bound(ring, field):
    """One protocol, two bounds: each claim's `c` follows from its bound,
    and both runs are honest ones that accept."""
    protocol = _protocol(field, ring, chunks=None, max_digit_bits=3)
    assert protocol.parameters()[5] == ("max_digit_bits", 3)
    for bits, chunks in ((6, 2), (8, 4)):
        bound = 1 << bits
        v = _witness(ring, seed=bits, bound=bound)
        statement = Statement(Relation_Range(bound), oracles=[TableOracle(v)])
        assert protocol._shape(statement).chunks == chunks
        iop = IOP(domain=field)
        iop.register(Relation_Range, protocol)
        assert iop.run(statement), bits


def test_decomposition_over_a_ring_listing_its_primes_out_of_base_order(ring, field):
    """The verifier reads the rows of `to_uint64` by the ring's prime order;
    a ring listing the base's primes the other way round proves as well."""
    reversed_ring = Ring(ring.N, primes=list(reversed(ring.primes)), split_degree=1)
    assert sorted(reversed_ring.prime_indices) != reversed_ring.prime_indices or (
        sorted(ring.prime_indices) != ring.prime_indices
    )
    for each in (ring, reversed_ring):
        v = _witness(each, seed=5)
        assert _iop(each, field).run(_statement(v))


def test_relation_lookup_check(field):
    variables = [MLE_Variable(f"y{i}") for i in range(3)]
    inside = MLE(
        field=field, variables=variables, evaluations=[field(i % 8) for i in range(8)]
    )
    assert Statement(Relation_Lookup(8), oracles=[TableOracle(inside)]).check()
    outside = MLE(
        field=field,
        variables=variables,
        evaluations=[field(i % 8) if i else field(8) for i in range(8)],
    )
    assert not Statement(Relation_Lookup(8), oracles=[TableOracle(outside)]).check()
    # A "negative" digit is a large canonical representative, not a small one.
    negative = MLE(
        field=field,
        variables=variables,
        evaluations=[field(i % 8) if i else -field(1) for i in range(8)],
    )
    assert not Statement(Relation_Lookup(8), oracles=[TableOracle(negative)]).check()


# --- the recomposition identity, on its own --------------------------------


def _exact_unpacked_evaluation(v, r, ell, nu):
    """`unpack(v)` at `r`, computed over Z from v's integer coefficients.

    The honest witness has coefficients well below q, so this is the value
    `unpacked_evaluation` would return if nothing were ever reduced -- the
    reference the protocol's mod-q check is a shadow of.
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


@pytest.mark.parametrize("m", [M, ELL + 1, ELL + NU])
def test_recomposition_identity(ring, field, m):
    """The recomposition check of the optimized range PIOP, computed directly
    from an honest witness, with the bound prefix inside the X block, reaching
    into the Z block, and covering all of X|Z.

    Three separate facts, in increasing strength:

    1. every entry of `psi` lifts through `iota` to something well inside
       `I_p` -- the no-overflow bound of the optimized range PIOP,
       `|mu(r^<=m, delta, j)| < 2^m beta sigma^m`, which is what makes the
       lift meaningful at all;
    2. the digits rebuild the coefficients **exactly over Z**, against v's
       integer coefficients;
    3. and hence congruently mod q against `unpacked_evaluation` of the ring
       element the verifier actually queries -- the form the protocol checks,
       and the only form available once v(x) is read out of R_q.
    """
    v = _witness(ring, seed=3)
    protocol = _protocol(field, ring, m=m)
    statement = _statement(v)
    shape = protocol._shape(statement)
    digits = protocol._digit_table(v, shape)
    mu = MLE(
        field=field, variables=shape.variables, evaluations=[field(d) for d in digits]
    )
    claim_bound = (1 << m) * shape.beta * ALPHABET**m
    assert 2 * claim_bound < field.prime  # p > 2*beta*(2*sigma)^m

    rng = random.Random(11)
    for _ in range(4):
        r = [rng.randrange(ALPHABET) for _ in range(shape.ell + shape.nu)]
        psi = mu.evaluate(protocol._bind_prefix(shape, r), in_place=False)
        width = shape.ell + shape.nu - m
        left = 0
        for index, value in enumerate(psi.table):
            lifted = centered(value, field.prime)
            assert abs(lifted) < claim_bound  # (1)
            delta = [(index >> t) & 1 for t in range(width)]
            left += lifted * eq_int(delta, r[m:]) * shape.beta ** (index >> width)

        assert left == _exact_unpacked_evaluation(v, r, shape.ell, shape.nu)  # (2)

        w = v.evaluate(
            {var: r[i] for i, var in enumerate(v.variables)}, in_place=False
        ).constant()
        assert (left - unpacked_evaluation(w, r[shape.ell :])) % ring.q_l == 0  # (3)


def test_digits_round_trip():
    width = (BOUND.bit_length() - 1) // CHUNKS
    for value in (0, 1, 7, 8, 63):
        digits = base_digits(value, width, CHUNKS)
        assert all(0 <= d < (1 << width) for d in digits)
        assert sum(d << (width * j) for j, d in enumerate(digits)) == value
    # Above the bound the high bits are dropped -- what a cheating prover
    # would have to submit, and what the recomposition check then catches.
    assert base_digits(BOUND, width, CHUNKS) == [0, 0]


@pytest.mark.parametrize(
    ("bound", "chunks"),
    [(BOUND, CHUNKS), (1 << 8, 4), (1 << 8, 1), (1 << 128, 2)],
    ids=["default", "gamma-2", "c-1", "64-bit-digits"],
)
def test_bulk_digit_table_is_the_per_coefficient_one(ring, field, bound, chunks):
    """The digit table built in whole-array steps holds exactly the digits
    of the per-coefficient table, including digits that start past bit 64."""
    if bound > math.prod(ring.primes):  # a q of at least B
        ring = Ring(1 << NU, prime_size=[45, 45, 45], split_degree=1)
    v = _witness(ring, seed=7, bound=min(bound, 1 << 20))
    statement = Statement(Relation_Range(bound), oracles=[TableOracle(v)])
    protocol = _protocol(field, ring, chunks=chunks)
    shape = protocol._shape(statement)
    bulk = protocol._digit_vector(v, shape)
    assert bulk is not None
    assert [int(e) for e in bulk] == protocol._digit_table(v, shape)


@pytest.mark.parametrize(
    "prime_size", [[30, 30], [45, 45, 45]], ids=["q-60-bits", "q-135-bits"]
)
def test_bulk_digits_of_coefficients_past_the_smallest_prime(field, prime_size):
    """A coefficient anywhere in [0, q) -- past the smallest prime, or past
    2^64 -- gets the digits of its integer value; the false claim it makes
    is still rejected."""
    ring = Ring(1 << NU, prime_size=prime_size, split_degree=1)

    def tweak(c):
        c[1] = min(ring.primes)
        c[2] = ring.q_l - 1
        c[3] = (1 << 70) + 5 if ring.q_l >> 70 else ring.q_l // 3

    v = _witness(ring, tweak=tweak)
    statement = _statement(v)
    protocol = _protocol(field, ring)
    shape = protocol._shape(statement)
    bulk = protocol._digit_vector(v, shape)
    assert bulk is not None
    assert [int(e) for e in bulk] == protocol._digit_table(v, shape)
    _assert_rarely_accepted(
        lambda: _iop(ring, field, _protocol(field, ring)),
        statement,
        protocol.soundness_error(statement),
    )


def test_digits_past_bit_128_of_a_q_above_2_128_are_digitized_one_by_one(field):
    """Digits that reach past bit 128 need the coefficients' bits there, which
    the lift modulo 2^128 does not have when q exceeds 2^128."""
    ring = Ring(1 << NU, prime_size=[45, 45, 45], split_degree=1)
    v = _witness(ring)
    statement = Statement(Relation_Range(1 << 132), oracles=[TableOracle(v)])
    protocol = _protocol(field, ring, chunks=4)
    assert protocol._digit_vector(v, protocol._shape(statement)) is None


# --- the protocol, end to end ----------------------------------------------


def test_decomposition_accepts_an_honest_witness(ring, field):
    v = _witness(ring, seed=1)
    iop = _iop(ring, field)
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
        (BOUND, CHUNKS, ELL + 1, 2),  # a prefix reaching into the Z block
        (BOUND, CHUNKS, ELL + NU, 2),  # all of X|Z bound: psi has c entries
        (1 << 8, 4, 2, 2),  # gamma = 2: two digit variables, beta = 4
        (1 << 8, 1, 2, 2),  # c = 1: no digit block at all, beta = B
    ],
)
def test_decomposition_across_the_parameter_knobs(ring, field, bound, chunks, m, kappa):
    """m, kappa and c move the block boundaries of the digit table; each
    setting has to survive an honest run and reject a bad coefficient.

    With one repetition the soundness error is ``(ell + nu) / sigma``, about
    2% here, so a single cheating run may pass: the cheating run is repeated
    and its acceptances are held to what that error allows."""
    good = _witness(ring, seed=20 + m + chunks, bound=bound)
    statement = Statement(Relation_Range(bound), oracles=[TableOracle(good)])
    protocol = RangeDecomposition(
        field, ring=ring, m=m, alphabet=ALPHABET, kappa=kappa, chunks=chunks
    )

    iop = IOP(domain=field)
    iop.register(Relation_Range, protocol)
    assert iop.run(statement)

    bad = _witness(ring, bound=bound, tweak=lambda c: c.__setitem__(1, bound + 1))
    cheating = Statement(Relation_Range(bound), oracles=[TableOracle(bad)])
    runs = 64
    accepted = 0
    for _ in range(runs):
        iop = IOP(domain=field)
        iop.register(
            Relation_Range,
            RangeDecomposition(
                field, ring=ring, m=m, alphabet=ALPHABET, kappa=kappa, chunks=chunks
            ),
        )
        accepted += bool(iop.run(cheating))
    assert accepted <= _most_acceptances(runs, protocol.soundness_error(cheating))


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

    iop = _iop(ring, field)
    iop.register(Relation_Eval, _Record())
    assert iop.run(_statement(v))

    assert len(seen) == 2 * KAPPA
    on_mu = [s for s in seen if s.oracles[0].polynomial.field is field]
    on_v = [s for s in seen if s.oracles[0].polynomial.ring is not None]
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

    iop = _iop(ring, field)
    iop.register(Relation_Lookup, _Record())
    assert iop.run(_statement(v))
    (claim,) = seen
    assert claim.relation.beta == digit_alphabets(BOUND, CHUNKS)[0]
    assert claim.check()


def test_the_decomposition_draws_in_its_field_inside_a_ring_iop(ring, field):
    """In an IOP over `R_q` the spot-check point is still drawn in `F_p`, so
    the claims the decomposition reduces to are true."""
    v = _witness(ring, seed=5)
    seen = []

    class _Record(Protocol):
        def __init__(self, relation):
            self.reduces_from = relation

        async def prove(self, _prover, _statements):
            return []

        async def verify(self, _verifier, statements):
            seen.extend(statements)
            return []

    iop = IOP(domain=ring)
    iop.register(Relation_Range, _protocol(field, ring))
    iop.register(Relation_Eval, _Record(Relation_Eval))
    iop.register(Relation_Lookup, _Record(Relation_Lookup))
    assert iop.run(_statement(v))
    assert len(seen) == 2 * KAPPA + 1
    assert all(s.check() for s in seen)


# --- a top digit shorter than the others -----------------------------------

#: log2(B) = 7 in c = 2 digits: the low one of 4 bits, the top one of 3.
SHORT = 1 << 7


def _short_statement(v, bound=SHORT):
    return Statement(Relation_Range(bound), oracles=[TableOracle(v)])


def _full_iop(ring, field, protocol=None, **kw):
    """The range proof with the lookup argument run, not left terminal."""
    iop = _iop(ring, field, protocol, **kw)
    iop.register(Relation_Lookup, Lookup(field))
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


#: The statement and argument-string digests of the range proof at bounds
#: whose bit count `c` divides, for `_witness(ring, seed=7, bound=B)`, with
#: the lookup claims left terminal; `_PINNED_WITH_LOOKUP` is the first one's
#: argument string with the lookup argument run. The statement is the bound
#: and the oracle, so the two `c` at `B = 2^8` share a statement digest.
_PINNED = {
    (6, 2): (
        "b1df74e7e7c1dc262130cf0cbbd8cd2e11e9a43b8517a92580ec6db2fc7d581a",
        "1999a82153eafd198748e489116a112193ade909502d0403bbfd91ac840e4b67",
    ),
    (8, 4): (
        "8f7897eea66213ced30d145daee1196e91b8d640629e8825ab5404df91ed2713",
        "4ce7d6b78be4ac2240375387a33cf59bc7561638719a7c1de7c368e699703106",
    ),
    (8, 1): (
        "8f7897eea66213ced30d145daee1196e91b8d640629e8825ab5404df91ed2713",
        "6761a090e800c1a598f88300beba52306669281d68f60d337f25d7a41e4e1a75",
    ),
}
_PINNED_WITH_LOOKUP = "71bee3c89188bd8fd961e232e04de46e7dbbe15ff93db182618c0815367b665a"


@pytest.mark.parametrize(("bits", "chunks"), list(_PINNED))
def test_divisible_bounds_are_pinned(ring, field, bits, chunks):
    v = _witness(ring, seed=7, bound=1 << bits)
    statement = _short_statement(v, 1 << bits)
    proof = _iop(ring, field, chunks=chunks, fiat_shamir=True).prove(statement)
    assert (statement.digest().hex(), proof.digest().hex()) == _PINNED[bits, chunks]
    if (bits, chunks) == (6, 2):
        proof = _full_iop(ring, field, chunks=chunks, fiat_shamir=True).prove(statement)
        assert proof.digest().hex() == _PINNED_WITH_LOOKUP


def _short_bundle(ring, field, v, bound=SHORT):
    """The lookup claims a run on `v` reduces to, recorded, not run."""
    seen = []

    class _Record(Protocol):
        reduces_from = Relation_Lookup

        async def prove(self, _prover, _statements):
            return []

        async def verify(self, _verifier, statements):
            seen.extend(statements)
            return []

    iop = _iop(ring, field)
    iop.register(Relation_Lookup, _Record())
    accepted = iop.run(_short_statement(v, bound))
    return accepted, seen


def test_a_short_top_digit_adds_a_lookup_on_the_top_digits(ring, field):
    """With `c` not dividing `log2(B)`, the bundle carries a second lookup,
    into `t_top_beta`, on the restriction of `mu'` to the top digit: its
    hypercube is exactly the top digits of every coefficient."""
    v = _witness(ring, seed=14, bound=SHORT)
    accepted, (low, top) = _short_bundle(ring, field, v)
    assert accepted
    assert digit_alphabets(SHORT, CHUNKS) == (16, 8)
    assert (low.relation.beta, top.relation.beta) == (16, 8)
    (mu,) = low.oracles
    (restricted,) = top.oracles
    assert isinstance(restricted, VirtualOracle) and restricted.is_restriction()
    assert restricted.constituents == [mu]
    assert restricted.num_vars == ELL + NU
    width = 4
    top_digits = [
        coefficient >> width
        for element in v.table
        for coefficient in element.get_polynomial()
    ]
    table = resolve_table(restricted)
    assert sorted(int(e) for e in table.table) == sorted(top_digits)
    assert low.check() and top.check()


@pytest.mark.parametrize(
    ("bits", "chunks", "m", "kappa"),
    [
        (7, 2, M, KAPPA),
        (7, 2, 0, 1),
        (7, 2, ELL + NU, 2),
        (11, 4, M, KAPPA),
        (9, 2, ELL + 1, 2),
    ],
)
def test_short_top_digit_across_the_parameter_knobs(
    ring, field, bits, chunks, m, kappa
):
    """Honest runs accept; a coefficient of `B` -- a top digit of exactly
    `top_beta`, inside `t_beta` -- is rejected. Its digits recompose and sit
    in `t_beta`, so the top digits' lookup is the only check that can, and
    with the lookups decided by their ideal deciders it does every time."""
    bound = 1 << bits
    statement = _short_statement(_witness(ring, seed=15, bound=bound), bound)
    protocol = RangeDecomposition(
        field, ring=ring, m=m, alphabet=ALPHABET, kappa=kappa, chunks=chunks
    )
    assert _iop(ring, field, protocol).run(statement)

    bad = _witness(ring, seed=15, bound=bound, tweak=lambda c: c.__setitem__(1, bound))
    cheating = _short_statement(bad, bound)
    assert not cheating.check()
    for _ in range(4):
        protocol = RangeDecomposition(
            field, ring=ring, m=m, alphabet=ALPHABET, kappa=kappa, chunks=chunks
        )
        assert not _iop(ring, field, protocol).run(cheating)


class _SwallowTop(Protocol):
    """Decides the lookup claims by their ideal deciders, except the one into
    `t_top_beta`, which it accepts unread."""

    reduces_from = Relation_Lookup

    def __init__(self, top_beta):
        self.top_beta = top_beta

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, statements):
        for statement in statements:
            if statement.relation.beta != self.top_beta and not statement.check():
                raise Rejection("lookup")
        return []


@pytest.mark.parametrize("excess", [0, 5, 15])
def test_a_digit_just_above_the_short_width_is_rejected(ring, field, excess):
    """A coefficient `B + excess`, below `beta^c`: the top digit is
    `top_beta`, one past the short alphabet. Rejected by the top digits'
    lookup -- with it accepted unread, the same run passes -- and by the
    lookup argument run in full (one run: its soundness error is about
    `2^ell* / p`, negligible at a 260-bit `p`)."""
    bad = _witness(
        ring, seed=16, bound=SHORT, tweak=lambda c: c.__setitem__(2, SHORT + excess)
    )
    statement = _short_statement(bad)
    assert not statement.check()
    assert not _iop(ring, field).run(statement)

    iop = _iop(ring, field)
    iop.register(Relation_Lookup, _SwallowTop(digit_alphabets(SHORT, CHUNKS)[1]))
    assert iop.run(statement)

    assert _full_iop(ring, field).run(
        _short_statement(_witness(ring, seed=16, bound=SHORT))
    )
    assert not _full_iop(ring, field).run(statement)


def test_short_top_digit_with_fiat_shamir(ring, field):
    v = _witness(ring, seed=17, bound=SHORT)
    statement = _short_statement(v)
    proof = _full_iop(ring, field, fiat_shamir=True).prove(statement)
    assert _full_iop(ring, field, fiat_shamir=True).verify(statement, proof)


def _borrow_one(table, shape):
    """Move one unit of digit 1 into digit 0 as `beta`: the value is kept,
    digit 0 leaves `t_beta`."""
    stride = 1 << (shape.ell + shape.nu)
    for i in range(stride):
        if table[i + stride]:
            table[i] += shape.beta
            table[i + stride] -= 1
            return
    raise AssertionError("no digit 1 to borrow from")


@pytest.mark.parametrize(
    ("bits", "chunks"),
    [(8, 4), (8, 8), (7, 4), (5, 2)],
    ids=["beta-4", "beta-2", "top-1-bit", "top-2-bits"],
)
def test_lookup_argument_over_tables_below_the_padding_unit(ring, field, bits, chunks):
    """The lookup argument run in full over tables of 2 and 4 entries, as the
    digit table and as the top digits' table: an honest witness is accepted;
    a borrowed digit (outside `t_beta`, value kept) and, with a short top
    digit, a coefficient of `B` (top digit `top_beta`) are rejected. Each
    rejection is the lookup argument's, of error about `2^ell* / p`."""
    bound = 1 << bits
    beta, top_beta = digit_alphabets(bound, chunks)
    assert min(beta, top_beta) < 8
    statement = _short_statement(_witness(ring, seed=18, bound=bound), bound)
    assert _full_iop(ring, field, chunks=chunks).run(statement)
    proof = _full_iop(ring, field, chunks=chunks, fiat_shamir=True).prove(statement)
    assert _full_iop(ring, field, chunks=chunks, fiat_shamir=True).verify(
        statement, proof
    )

    borrowed = _full_iop(ring, field, _cheat(ring, field, _borrow_one, chunks))
    assert not borrowed.run(statement)
    if top_beta != beta:
        bad = _witness(
            ring, seed=18, bound=bound, tweak=lambda c: c.__setitem__(1, bound)
        )
        assert not _full_iop(ring, field, chunks=chunks).run(
            _short_statement(bad, bound)
        )


# --- the three ways to cheat -----------------------------------------------


class _Tampered(RangeDecomposition):
    """A prover that submits a digit table of its own choosing."""

    def __init__(self, *args, corrupt, **kw):
        super().__init__(*args, **kw)
        self._corrupt = corrupt

    def _digit_vector(self, v, shape):
        return None  # build the integer table, which `_digit_table` corrupts

    def _digit_table(self, v, shape):
        table = super()._digit_table(v, shape)
        self._corrupt(table, shape)
        return table


def _cheat(ring, field, corrupt, chunks=CHUNKS):
    return _Tampered(
        field,
        m=M,
        alphabet=ALPHABET,
        kappa=KAPPA,
        ring=ring,
        chunks=chunks,
        corrupt=corrupt,
    )


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
    recompose, and the mod-q check of step 2 fails -- except with probability
    the soundness error, so the cheating run is repeated."""
    v = _witness(ring, tweak=lambda c: c.__setitem__(3, BOUND + 5))
    statement = _statement(v)
    assert not statement.check()  # the claim really is false
    error = _protocol(field, ring).soundness_error(statement)
    _assert_rarely_accepted(lambda: _iop(ring, field), statement, error)

    # The digits themselves stay inside t_beta here -- truncation keeps them
    # small -- so the lookup branch has nothing to say, and step 2 is the
    # only thing standing between this witness and an accepting run.
    def swallowing():
        iop = _iop(ring, field)
        iop.register(Relation_Lookup, _Swallow(Relation_Lookup))
        return iop

    _assert_rarely_accepted(swallowing, statement, error)


def test_perturbed_digit_is_rejected(ring, field):
    """Case 2: an honest witness, but the prover alters one digit. The digits
    stop recomposing, so again the mod-q check fails, up to the soundness
    error."""
    v = _witness(ring, seed=5)
    statement = _statement(v)

    def corrupt(table, shape):
        table[0] = (table[0] + 1) % shape.beta

    _assert_rarely_accepted(
        lambda: _iop(ring, field, _cheat(ring, field, corrupt)),
        statement,
        _protocol(field, ring).soundness_error(statement),
    )


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

    protocol = _cheat(ring, field, corrupt)
    assert not _iop(ring, field, protocol).run(_statement(v))
    assert "at" in moved

    # Pin the claim: it is the lookup branch that rejects, not step 2. With
    # the lookup claim swallowed, the same tampered run is accepted.
    iop = _iop(ring, field, _cheat(ring, field, corrupt))
    iop.register(Relation_Lookup, _Swallow(Relation_Lookup))
    assert iop.run(_statement(v))


def _replay_with_tamper(honest, ring, field, statement, tamper):
    """A verifier-only run over an honest transcript with one message
    rewritten. The recorded challenges are replayed as they were, so this is
    the honest interaction with exactly one dishonest prover message."""
    iop = _iop(ring, field)
    for label in honest.transcript.order:
        value = honest.transcript.entries[label].result()
        iop.transcript.write(label, tamper(label, value))
    return iop.loop.run_until_complete(iop.verifier.verify(statement._fork()))


def test_replay_harness_accepts_an_untampered_transcript(ring, field):
    v = _witness(ring, seed=12)
    statement = _statement(v)
    honest = _iop(ring, field)
    assert honest.run(statement)
    assert _replay_with_tamper(
        honest, ring, field, statement, lambda _label, value: value
    )


def test_every_prover_message_is_load_bearing(ring, field):
    """Rewriting any single prover message makes the run reject.

    `mu` and `psi` are tables of field elements, `w` a ring element; the
    tamper adds one to the first entry of whichever it is. Nothing the prover
    sends is decorative: `mu` and `w` feed the recomposition check, and `psi`
    feeds both that and the evaluation claim that pins it to `mu`.
    """
    v = _witness(ring, seed=13)
    statement = _statement(v)
    honest = _iop(ring, field)
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
            if isinstance(value, tuple | FieldVector):
                return _bumped(value, field.one)
            return value + Polynomial(value.ring).from_array([1])

        assert not _replay_with_tamper(honest, ring, field, statement, tamper), target


# --- Fiat-Shamir ------------------------------------------------------------


def test_fiat_shamir_round_trip(ring, field):
    v = _witness(ring, seed=8)
    statement = _statement(v)
    proof = _iop(ring, field, fiat_shamir=True).prove(statement)
    assert _iop(ring, field, fiat_shamir=True).verify(statement, proof)
    # Prover messages only: the challenges are re-derived, never carried.
    assert not any(label.startswith("range/r") for label in proof.labels)
    assert not any(label.startswith("range/s") for label in proof.labels)


def test_fiat_shamir_is_deterministic(ring, field):
    v = _witness(ring, seed=9)
    statement = _statement(v)
    digests = []
    for _ in range(2):
        iop = _iop(ring, field, fiat_shamir=True)
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
    proof = _iop(ring, field, fiat_shamir=True).prove(statement)
    messages = list(proof.messages)
    label, table = messages[0]
    assert label == "range/mu"
    messages[0] = (label, _bumped(table, field.one))
    assert not _iop(ring, field, fiat_shamir=True).verify(
        statement, type(proof)(messages)
    )


# --- parameter conditions ---------------------------------------------------


def test_parameters_are_validated(ring, field):
    v = _witness(ring)
    statement = _statement(v)

    with pytest.raises(ValueError, match="power of two"):
        _protocol(field, ring, alphabet=100)
    with pytest.raises(ValueError, match="kappa"):
        _protocol(field, ring, kappa=0)

    # These three are conditions on the claim as well as on the constructor,
    # so they surface on first use rather than at construction.
    with pytest.raises(ValueError, match="must not exceed the variables of the X"):
        _protocol(field, ring, m=ELL + NU + 1)._shape(statement)
    # sigma so large that 2*beta*(2*sigma)^m overruns a 260-bit prime.
    with pytest.raises(ValueError, match="2\\*beta"):
        _protocol(field, ring, alphabet=1 << 128)._shape(statement)
    # sigma above the smallest RNS prime leaves the exceptional set of Z_q.
    too_wide = 1 << (min(ring.primes).bit_length() + 1)
    with pytest.raises(ValueError, match="smallest RNS prime"):
        _protocol(field, ring, m=0, alphabet=too_wide)._shape(statement)


def test_a_bound_above_q_is_rejected(ring, field):
    q = math.prod(ring.primes)
    bound = 1 << q.bit_length()
    statement = Statement(Relation_Range(bound), oracles=[TableOracle(_witness(ring))])
    with pytest.raises(ValueError, match="modulus q"):
        _protocol(field, ring)._shape(statement)


def test_soundness_error_is_the_theorem_bound(ring, field):
    v = _witness(ring)
    protocol = _protocol(field, ring)
    reported = protocol.soundness_error(_statement(v))
    nu = ring.N.bit_length() - 1
    assert reported == max(
        ((ELL + nu) / ALPHABET) ** KAPPA, (ELL + nu + 1 - M) / field.prime
    )


@pytest.mark.parametrize("bits", [208, 156])
def test_decomposition_at_the_paper_prime_sizes(ring, bits):
    """The prime sizes the paper's parameter sets are stated for, with every
    variable of the X|Z block bound: an honest run accepts and a coefficient
    above the bound is rejected (to within the soundness error)."""
    field = PseudoMersenneField.generate(bits, two_adicity=8)
    protocol = _protocol(field, ring, m=ELL + NU)
    assert _iop(ring, field, protocol).run(_statement(_witness(ring, seed=5)))

    bad = _statement(_witness(ring, tweak=lambda c: c.__setitem__(1, BOUND + 1)))
    _assert_rarely_accepted(
        lambda: _iop(ring, field, _protocol(field, ring, m=ELL + NU)),
        bad,
        protocol.soundness_error(bad),
    )


@pytest.mark.parametrize(
    ("bits", "alphabet", "m", "kappa"),
    [(208, 1 << 11, 15, 20), (208, 1 << 7, 22, 51), (156, 1 << 7, 17, 51)],
)
def test_the_paper_parameter_sets_are_accepted(bits, alphabet, m, kappa):
    """The paper's parameter sets for its largest model (ell = 9, N = 2^13,
    c = 4, beta = 2^11) pass the side conditions and reach 128 bits. Only the
    shape is derived, so `v` is named by a committed oracle with no table."""
    field = PseudoMersenneField.generate(bits, two_adicity=8)
    ring = Ring(1 << 13, prime_size=[30, 30], split_degree=1)
    v = CommittedOracle([MLE_Variable(f"x{i}") for i in range(9)], None, None)
    statement = Statement(Relation_Range(1 << 44), oracles=[v])
    protocol = RangeDecomposition(field, m, alphabet, kappa, ring, 4)
    assert protocol._shape(statement).ell_star == 24
    assert protocol.soundness_error(statement) <= 2.0**-128


@pytest.mark.parametrize(
    ("N", "prime_size", "split_degree"),
    [(32, [30, 30], 1), (64, [50], 1), (256, [28, 60, 45], 4), (2, [40], 1)],
)
def test_the_verifier_evaluates_unpack_modulo_q(field, N, prime_size, split_degree):
    """`_unpacked_evaluation_mod_q` is `unpacked_evaluation` modulo `q`, on
    one prime or several, on a split ring, and at points whose coordinates
    exceed the primes."""
    ring = Ring(N, prime_size=prime_size, split_degree=split_degree)
    protocol = RangeDecomposition(field, 0, ALPHABET, KAPPA, ring, CHUNKS)
    rng = random.Random(N)
    nu = N.bit_length() - 1
    for trial in range(3):
        element = ring.random_element()
        point = [rng.randrange(ALPHABET) for _ in range(nu)]
        if trial == 2:
            point = [p + ring.primes[0] for p in point]
        expected = unpacked_evaluation(element, point) % ring.q_l
        assert protocol._unpacked_evaluation_mod_q(element, point) == expected
    with pytest.raises(ValueError, match="coefficients"):
        protocol._unpacked_evaluation_mod_q(ring.random_element(), [1] * (nu + 1))
