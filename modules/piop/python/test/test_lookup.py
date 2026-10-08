# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The lookup relation and its protocol (Figure 11 of [CCCFGS26]).

The headline property is the one the relation states: the verifier accepts
exactly when every entry of the oracle lies in `t_beta = {0, ..., beta - 1}`.
Around it: the two lemmas the protocol rests on tested on their own (the
memory-checking identity of Lemma I.3 and the grand-product layering of
Lemma I.4), the reduction's shape, and which branch rejects which kind of
lie."""

import random
from collections import Counter

import pytest
from vfhe.arith import MLE, MLE_Variable, PseudoMersenneField
from vfhe.piop import (
    IOP,
    Lookup,
    OracleKind,
    Protocol,
    Relation_Eval,
    Relation_Lookup,
    Relation_Sum,
    Statement,
    Sumcheck,
    VirtualEval,
    element_digest,
)
from vfhe.piop.lookup import FINGERPRINTS, counters, grand_product

# beta = 8 (b* = 3) and a 64-entry oracle (ell* = 6): the two sides of the
# protocol then have different widths, which is what catches an index that
# silently assumes they are equal.
BETA = 8
NUM_VARS = 6


@pytest.fixture
def field():
    return PseudoMersenneField.generate(260, two_adicity=8)


def _oracle(field, values):
    variables = [MLE_Variable(f"m{i}") for i in range(NUM_VARS)]
    return MLE(field=field, variables=variables, evaluations=[field(v) for v in values])


def _bounded(seed=0, beta=BETA):
    rng = random.Random(seed)
    return [rng.randrange(beta) for _ in range(1 << NUM_VARS)]


def _iop(field, protocol=None, **kw):
    iop = IOP(domain=field, **kw)
    iop.register(Relation_Lookup, protocol or Lookup(field))
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


def _statement(field, values, beta=BETA):
    return Statement(Relation_Lookup(beta), oracles=[_oracle(field, values)])


# --- Lemma I.3: the memory-checking identity --------------------------------


def test_counters_satisfy_the_multiset_identity():
    """`WS = RS u S` as multisets, which is what the four grand products
    later check in fingerprinted form."""
    values = _bounded(seed=1)
    read_ts, final_cts = counters(values, BETA)

    ws = Counter([(j, 0) for j in range(BETA)])
    ws += Counter([(v, read_ts[i] + 1) for i, v in enumerate(values)])
    rs = Counter([(v, read_ts[i]) for i, v in enumerate(values)])
    s = Counter([(j, final_cts[j]) for j in range(BETA)])
    assert ws == rs + s


def test_counters_are_the_occurrence_counts():
    values = [3, 0, 3, 3, 1]
    read_ts, final_cts = counters(values, 4)
    assert read_ts == [0, 0, 1, 2, 0]  # occurrences strictly before each index
    assert final_cts == [1, 1, 0, 3]  # totals per table entry


def test_counters_cannot_balance_an_out_of_range_value():
    """The identity fails for a value with no table slot to be counted into
    -- which is why no choice of counters rescues an unbounded oracle."""
    values = [*_bounded(seed=2)[:-1], BETA + 1]
    read_ts, final_cts = counters(values, BETA)

    ws = Counter([(j, 0) for j in range(BETA)])
    ws += Counter([(v, read_ts[i] + 1) for i, v in enumerate(values)])
    rs = Counter([(v, read_ts[i]) for i, v in enumerate(values)])
    s = Counter([(j, final_cts[j]) for j in range(BETA)])
    assert ws != rs + s


# --- Lemma I.4: the grand-product layering ----------------------------------


def test_grand_product_layering():
    """`f(X,1) = f(0,X) * f(1,X)` for every X, the product at `f[-2]`, and
    the last entry pinned so the identity holds at the all-ones point too."""
    for n in (1, 2, 3, 4):
        size = 1 << n
        leaves = [random.Random(n).randrange(2, 50) for _ in range(size)]
        f = grand_product(leaves, 0)

        assert len(f) == 2 * size
        assert f[:size] == leaves  # the bottom half is the leaves
        assert f[-1] == 0  # pinned, not computed
        for k in range(size):  # including k = size - 1, where both sides are 0
            assert f[size + k] == f[2 * k] * f[2 * k + 1]

        product = 1
        for value in leaves:
            product *= value
        assert f[-2] == product
        assert f[size + (size - 2)] == product  # i.e. f^(1) at (0, 1, ..., 1)


# --- the relation's ideal decider -------------------------------------------


def test_relation_lookup_check(field):
    assert _statement(field, _bounded()).check()
    assert not _statement(field, [*_bounded()[:-1], BETA]).check()


def test_relation_lookup_rejects_a_bad_table_size():
    with pytest.raises(ValueError, match="power of two"):
        Relation_Lookup(7)


# --- the headline: accept iff the entries are bounded -----------------------


@pytest.mark.parametrize(
    "values",
    [
        _bounded(seed=3),  # a generic bounded vector
        [0] * (1 << NUM_VARS),  # every entry the same
        [BETA - 1] * (1 << NUM_VARS),  # every entry the largest allowed
        [i % BETA for i in range(1 << NUM_VARS)],  # every table entry used evenly
    ],
    ids=["random", "all-zero", "all-max", "uniform"],
)
def test_accepts_a_bounded_vector(field, values):
    assert _iop(field).run(_statement(field, values))


@pytest.mark.parametrize(
    "bad",
    [BETA, BETA + 1, 2 * BETA, 1 << 40],
    ids=["just-over", "over", "double", "huge"],
)
@pytest.mark.parametrize(
    "where", [0, 17, (1 << NUM_VARS) - 1], ids=["first", "middle", "last"]
)
def test_rejects_an_unbounded_vector(field, bad, where):
    """One entry outside `t_beta`, anywhere, of any size, is rejected."""
    values = _bounded(seed=4)
    values[where] = bad
    assert not _statement(field, values).check()  # the claim really is false
    assert not _iop(field).run(_statement(field, values))


def test_rejects_a_negative_entry(field):
    """`-1` in the field is `p - 1`, a huge canonical representative -- it
    must not slip through as a small number."""
    values = _bounded(seed=5)
    mu = _oracle(field, values)
    mu.table = type(mu.table)(field, [*list(mu.table)[:-1], -field.one])
    statement = Statement(Relation_Lookup(BETA), oracles=[mu])
    assert not statement.check()
    assert not _iop(field).run(statement)


def test_rejects_when_every_entry_is_unbounded(field):
    values = [BETA + i for i in range(1 << NUM_VARS)]
    assert not _iop(field).run(_statement(field, values))


# --- the shape of the reduction ---------------------------------------------


def test_transcript_structure(field):
    iop = _iop(field)
    assert iop.run(_statement(field, _bounded(seed=6)))
    labels = iop.transcript.order
    assert labels[:4] == [
        "lookup/read_ts",
        "lookup/final_cts",
        "lookup/zeta",
        "lookup/tau",
    ]
    assert [x for x in labels if x.startswith("lookup/f1_")] == [
        f"lookup/f1_{a}" for a in FINGERPRINTS
    ]
    # Every challenge is drawn only after the message it answers.
    assert labels.index("lookup/zeta") > labels.index("lookup/final_cts")
    assert labels.index("lookup/chi") > labels.index("lookup/products")


def test_reduces_to_two_sums_and_four_evaluations(field):
    emitted = []

    class _Record(Protocol):
        def __init__(self, relation):
            self.reduces_from = relation

        async def prove(self, _prover, _statements):
            return []

        async def verify(self, _verifier, statements):
            emitted.extend(statements)
            return []

    iop = _iop(field)
    iop.register(Relation_Sum, _Record(Relation_Sum))
    iop.register(Relation_Eval, _Record(Relation_Eval))
    assert iop.run(_statement(field, _bounded(seed=7)))

    sums = [s for s in emitted if isinstance(s.relation, Relation_Sum)]
    evals = [s for s in emitted if isinstance(s.relation, Relation_Eval)]
    assert len(sums) == 2 and len(evals) == 4
    # Both zerochecks claim zero, over a degree-3 virtual oracle.
    assert all(int(s.value) == 0 for s in sums)
    assert [s.oracles[0].degree for s in sums] == [3, 3]
    # One sum per side of the cube: mu's variables, and the table's.
    assert sorted(s.oracles[0].num_vars for s in sums) == [
        BETA.bit_length() - 1,
        NUM_VARS,
    ]
    assert all(s.check() for s in sums)
    assert all(s.check() for s in evals)


def test_sumcheck_rounds_carry_four_nodes(field):
    """Degree 3 in each variable (eq~ times two factors of a product), so a
    round message is the polynomial at the nodes 0..3."""
    iop = _iop(field)
    assert iop.run(_statement(field, _bounded(seed=8)))
    rounds = [
        iop.transcript.entries[label].result()
        for label in iop.transcript.order
        if "/g" in label
    ]
    assert rounds, "the sumchecks must actually have run"
    assert {len(r) for r in rounds} == {4}
    # NUM_VARS rounds on mu's side, b* on the table's.
    assert len(rounds) == NUM_VARS + BETA.bit_length() - 1


# --- which branch rejects which lie -----------------------------------------


class _Swallow(Protocol):
    """Accepts a relation outright, so a test can pin which branch rejected."""

    def __init__(self, relation):
        self.reduces_from = relation

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, _statements):
        return []


def test_the_product_identity_is_what_catches_an_unbounded_entry(field):
    """An out-of-range entry breaks `WS = RS u S`, so the fingerprinted
    products stop matching -- and that check alone is enough to reject."""
    values = _bounded(seed=9)
    values[11] = BETA + 2

    iop = _iop(field)
    iop.register(Relation_Sum, _Swallow(Relation_Sum))
    iop.register(Relation_Eval, _Swallow(Relation_Eval))
    assert not iop.run(_statement(field, values))


def test_the_sumcheck_is_what_catches_a_mislayered_tree(field):
    """A prover that sends an internal node inconsistent with the leaves
    keeps the four products intact, so the product identity still passes --
    the zerocheck is the only thing that notices."""
    values = _bounded(seed=10)
    honest = _iop(field)
    assert honest.run(_statement(field, values))

    def tamper(label, value):
        # Index 0 is not the entry the evaluation claim reads (that is
        # 2^n - 2), so the four products survive this edit untouched.
        if label == "lookup/f1_WSmu":
            return (value[0] + field.one, *value[1:])
        return value

    assert not _replay_with_tamper(honest, field, values, tamper)

    # With the sum claims swallowed, the same tampered transcript is accepted:
    # nothing else in the protocol is looking at that entry.
    assert _replay_with_tamper(honest, field, values, tamper, swallow=(Relation_Sum,))


def _replay_with_tamper(honest, field, values, tamper, swallow=()):
    """A verifier-only run over an honest transcript with one message
    rewritten, the recorded challenges replayed as they were."""
    iop = _iop(field)
    for relation in swallow:
        iop.register(relation, _Swallow(relation))
    for label in honest.transcript.order:
        iop.transcript.write(
            label, tamper(label, honest.transcript.entries[label].result())
        )
    statement = _statement(field, values)
    return iop.loop.run_until_complete(iop.verifier.verify(statement._fork()))


def test_replay_harness_accepts_an_untampered_transcript(field):
    values = _bounded(seed=11)
    honest = _iop(field)
    assert honest.run(_statement(field, values))
    assert _replay_with_tamper(honest, field, values, lambda _label, value: value)


def test_every_prover_message_is_load_bearing(field):
    """Rewriting any single prover message, anywhere in the reduction, makes
    the run reject -- the lookup's own messages and those of the sumchecks
    and virtual-oracle openings it reduces to."""
    values = _bounded(seed=12)
    honest = _iop(field)
    assert honest.run(_statement(field, values))

    messages = [
        label
        for label in honest.transcript.order
        if label not in honest.transcript.derived
    ]
    assert [m for m in messages if m.startswith("lookup/")] == [
        "lookup/read_ts",
        "lookup/final_cts",
        *(f"lookup/f1_{a}" for a in FINGERPRINTS),
        "lookup/products",
    ]
    # And the reduction really did continue past them.
    assert any(m.startswith("sumcheck/") for m in messages)
    assert any(m.startswith("virtual/") for m in messages)

    for target in messages:

        def tamper(label, value, target=target):
            if label != target:
                return value
            return (value[0] + field.one, *value[1:])

        assert not _replay_with_tamper(honest, field, values, tamper), target


# --- Fiat-Shamir ------------------------------------------------------------


def test_fiat_shamir_round_trip(field):
    statement = _statement(field, _bounded(seed=13))
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert _iop(field, fiat_shamir=True).verify(statement, proof)
    for drawn in ("lookup/zeta", "lookup/tau", "lookup/chi"):
        assert drawn not in proof.labels  # challenges are re-derived


def test_fiat_shamir_rejects_an_unbounded_vector(field):
    values = _bounded(seed=14)
    values[3] = BETA
    statement = _statement(field, values)
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert not _iop(field, fiat_shamir=True).verify(statement, proof)


def test_fiat_shamir_is_deterministic(field):
    statement = _statement(field, _bounded(seed=15))
    runs = []
    for _ in range(2):
        iop = _iop(field, fiat_shamir=True)
        assert iop.run(statement)
        runs.append(
            [
                (label, element_digest(iop.transcript.entries[label].result()))
                for label in iop.transcript.order
            ]
        )
    assert runs[0] == runs[1]


# --- accounting -------------------------------------------------------------


def test_soundness_error_is_the_theorem_bound(field):
    statement = _statement(field, _bounded())
    reported = Lookup(field).soundness_error(statement)
    assert reported == ((1 << NUM_VARS) + BETA) / field.prime


# --- under the range protocol, which is what produces these claims ----------


def _range_iop(field):
    from vfhe.piop import RangeDecomposition, Relation_Range

    iop = _iop(field)
    iop.register(
        Relation_Range, RangeDecomposition(field, m=2, alphabet=1 << 8, kappa=2)
    )
    return iop


@pytest.mark.parametrize("bounded", [True, False], ids=["bounded", "unbounded"])
def test_range_proof_end_to_end(field, bounded):
    """The whole chain: a range claim over R_q reduces to a lookup over the
    digits, which reduces to two sumchecks, which reduce to evaluation claims
    on the real oracles. Accepted exactly when the coefficients are in range.
    """
    from vfhe.arith import Polynomial, Ring
    from vfhe.piop import Relation_Range

    ring = Ring(8, prime_size=[30, 30], split_degree=1)
    bound, chunks, ell = 1 << 6, 2, 2
    rng = random.Random(16)
    table = []
    for i in range(1 << ell):
        coefficients = [rng.randrange(bound) for _ in range(ring.N)]
        if i == 0 and not bounded:
            coefficients[2] = bound + 7
        table.append(Polynomial(ring).from_array(coefficients))
    v = MLE(
        ring=ring,
        variables=[MLE_Variable(f"x{i}") for i in range(ell)],
        evaluations=table,
    )

    statement = Statement(Relation_Range(bound, chunks), oracles=[v])
    assert statement.check() is bounded
    assert _range_iop(field).run(statement) is bounded
