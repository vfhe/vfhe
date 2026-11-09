# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The logarithmic-derivative lookup protocol `LogUpGKR` for
`Relation_Lookup`: the multiplicities sent per table, the two fractional
sums proven by GKR on trees of fractions.

The headline property is the one the relation states: the verifier accepts
exactly when every entry of the oracle lies in `t_beta = {0, ..., beta - 1}`,
whatever the sizes of the oracle and of the table (either tree may be the
deeper one, or of depth 0). Around it: the two helpers on their own
(`multiplicities` and `fraction_tree`), the shape of the transcript and of
the reduction, the grouping of several claims, Fiat-Shamir, and which check
rejects which lie."""

import random
from array import array

import pytest
from vfhe.arith import MLE, Field, FieldVector, MLE_Variable, PseudoMersenneField
from vfhe.piop import (
    IOP,
    OracleKind,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    TableOracle,
    element_digest,
)
from vfhe.piop.protocols import LogUpGKR, Lookup, Sumcheck, VirtualEval
from vfhe.piop.protocols.logup import fraction_tree, multiplicities
from vfhe.piop.relations import Relation_Lookup, Relation_Sum

# beta = 8 (a table tree of depth 3) and a 64-entry oracle (an oracle tree
# of depth 6): the two trees then have different depths, and the layers
# below the table's root run for the oracle's tree alone.
BETA = 8
NUM_VARS = 6


@pytest.fixture
def field():
    return PseudoMersenneField.generate(260, two_adicity=8)


def _oracle(field, values, prefix="m"):
    """A table oracle over `log2(len(values))` variables."""
    n = (len(values) - 1).bit_length()
    variables = [MLE_Variable(f"{prefix}{i}") for i in range(n)]
    return TableOracle(
        MLE(field=field, variables=variables, evaluations=[field(v) for v in values])
    )


def _bounded(seed=0, beta=BETA, num_vars=NUM_VARS):
    rng = random.Random(seed)
    return [rng.randrange(beta) for _ in range(1 << num_vars)]


def _iop(field, protocol=None, **kw):
    iop = IOP(domain=field, **kw)
    iop.register(Relation_Lookup, protocol or LogUpGKR(field))
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    return iop


def _statement(field, values, beta=BETA):
    return Statement(Relation_Lookup(beta), oracles=[_oracle(field, values)])


def _messages(iop):
    """The labels of the prover's messages, in transcript order."""
    return [x for x in iop.transcript.order if x not in iop.transcript.derived]


def _entries(iop, label):
    return iop.transcript.entries[label].result()


def _m_labels(iop):
    """The labels of the multiplicity tables, in transcript order."""
    return [
        x
        for x in iop.transcript.order
        if x.startswith("logup") and x.rsplit("/", 1)[-1][:1] == "m"
    ]


class _Record(Protocol):
    """Leaves the claims of `relation` terminal, keeping them in `seen`."""

    def __init__(self, relation, seen):
        self.reduces_from = relation
        self.seen = seen

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, statements):
        self.seen.extend(statements)
        return []


def _record_evals(iop, seen):
    """Leaves every evaluation claim terminal, on a virtual oracle too,
    keeping them in `seen`."""
    iop.register(Relation_Eval, _Record(Relation_Eval, seen))
    iop.register(Relation_Eval, _Record(Relation_Eval, seen), kind=OracleKind.virtual)


class _Swallow(Protocol):
    """Accepts a relation outright, so a test can pin which check rejected."""

    def __init__(self, relation):
        self.reduces_from = relation

    async def prove(self, _prover, _statements):
        return []

    async def verify(self, _verifier, _statements):
        return []


# --- the helpers --------------------------------------------------------------


def _projective_sum(left, right):
    (a, h), (a2, h2) = left, right
    return (a * h2 + a2 * h, h * h2)


@pytest.mark.parametrize("num_vars", [0, 1, 3, 5])
@pytest.mark.parametrize("ones", [False, True], ids=["numerators", "ones"])
def test_fraction_tree_root_is_the_sum_of_the_leaves(field, num_vars, ones):
    """The root is `(sum_i a_i prod_(j != i) h_j, prod_i h_i)`, the fraction
    `sum_i a_i / h_i`; with no numerators the leaves are `1 / h_i` and the
    leaf layer keeps None."""
    rng = random.Random(num_vars)
    size = 1 << num_vars
    h = [field(rng.randrange(1, 1 << 60)) for _ in range(size)]
    a = [field.one] * size if ones else [field(rng.randrange(1 << 60)) for _ in h]
    layers = fraction_tree(
        None if ones else FieldVector(field, a), FieldVector(field, h)
    )
    assert len(layers) == num_vars + 1
    assert (layers[-1][0] is None) is ones
    root_a, root_h = layers[0]
    assert len(root_h) == 1
    # Over one leaf the root is the leaf, whose numerators stay None.
    root = field.one if root_a is None else root_a[0]
    product = field.one
    for value in h:
        product = product * value
    total = field.zero
    for i in range(size):
        term = a[i]
        for j in range(size):
            if j != i:
                term = term * h[j]
        total = total + term
    assert root_h[0] == product
    assert root == total


@pytest.mark.parametrize("num_vars", [1, 2, 4])
@pytest.mark.parametrize("ones", [False, True], ids=["numerators", "ones"])
def test_fraction_tree_layers_follow_the_projective_rule(field, num_vars, ones):
    """Entry `i` of layer `k` is the projective sum of entries `i` and
    `i + 2^k` of layer `k + 1`, on every layer."""
    rng = random.Random(10 + num_vars)
    size = 1 << num_vars
    h = FieldVector(field, [field(rng.randrange(1, 1 << 60)) for _ in range(size)])
    a = (
        None
        if ones
        else FieldVector(field, [field(rng.randrange(1 << 60)) for _ in range(size)])
    )
    layers = fraction_tree(a, h)

    def pairs(layer, k):
        nums, dens = layer
        return [
            (field.one if nums is None else nums[i], dens[i]) for i in range(1 << k)
        ]

    for k in range(num_vars):
        assert len(layers[k][1]) == 1 << k
        above, below = pairs(layers[k], k), pairs(layers[k + 1], k + 1)
        for i in range(1 << k):
            assert above[i] == _projective_sum(below[i], below[i + (1 << k)])


def test_multiplicities_are_the_occurrence_counts():
    values = [3, 0, 3, 3, 1]
    assert list(multiplicities(values, 4)) == [1, 1, 0, 3]
    assert list(multiplicities(array("Q", values), 4)) == [1, 1, 0, 3]
    counts = multiplicities(values, 4)
    assert isinstance(counts, array) and counts.typecode == "Q"


@pytest.mark.parametrize("container", ["list", "array"])
def test_multiplicities_do_not_count_a_value_outside_the_table(container):
    """A value outside `t_beta` lands in no slot, so the counts sum to fewer
    than the values -- the same for a list and an ``array('Q')``."""
    values = [*_bounded(seed=1)[:-2], BETA, 1 << 40]
    counts = multiplicities(
        array("Q", values) if container == "array" else values, BETA
    )
    assert len(counts) == BETA
    assert sum(counts) == len(values) - 2
    assert list(counts) == [values.count(y) for y in range(BETA)]


def test_multiplicities_of_an_array_are_those_of_the_list():
    values = _bounded(seed=2)
    assert list(multiplicities(array("Q", values), BETA)) == list(
        multiplicities(values, BETA)
    )


# --- the headline: accept iff the entries are bounded -------------------------


@pytest.mark.parametrize(
    "num_vars, beta",
    [(6, 8), (3, 16), (4, 16), (10, 4), (0, 8), (4, 1), (0, 1)],
    ids=[
        "oracle-deeper",
        "table-deeper",
        "same-depth",
        "large-oracle",
        "oracle-depth-0",
        "table-depth-0",
        "both-depth-0",
    ],
)
def test_accepts_a_bounded_vector(field, num_vars, beta):
    """Either tree may be the deeper one, or have depth 0 (a single leaf
    sent as it is); accepted interactively and as an argument string."""
    statement = _statement(
        field, _bounded(seed=num_vars, beta=beta, num_vars=num_vars), beta
    )
    assert statement.check()
    assert _iop(field).run(statement)
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert _iop(field, fiat_shamir=True).verify(statement, proof)


@pytest.mark.parametrize(
    "values",
    [
        [0] * (1 << NUM_VARS),  # every entry the same
        [BETA - 1] * (1 << NUM_VARS),  # every entry the largest allowed
        [i % BETA for i in range(1 << NUM_VARS)],  # every table entry used evenly
    ],
    ids=["all-zero", "all-max", "uniform"],
)
def test_accepts_extreme_bounded_vectors(field, values):
    assert _iop(field).run(_statement(field, values))


@pytest.mark.parametrize(
    "bad",
    [BETA, BETA + 1, 2 * BETA, 1 << 40, -1],
    ids=["just-over", "over", "double", "huge", "negative"],
)
@pytest.mark.parametrize(
    "where", [0, 17, (1 << NUM_VARS) - 1], ids=["first", "middle", "last"]
)
def test_rejects_an_unbounded_vector(field, bad, where):
    """One entry outside `t_beta`, anywhere, of any size, is rejected; `-1`
    is `p - 1` in the field, not a small number."""
    values = _bounded(seed=4)
    values[where] = bad
    statement = _statement(field, values)
    assert not statement.check()  # the claim really is false
    assert not _iop(field).run(statement)


@pytest.mark.parametrize(
    "num_vars, beta",
    [(3, 16), (0, 8), (2, 1)],
    ids=["table-deeper", "depth-0", "beta-1"],
)
def test_rejects_an_unbounded_entry_for_other_shapes(field, num_vars, beta):
    values = _bounded(seed=5, beta=beta, num_vars=num_vars)
    values[-1] = beta
    statement = _statement(field, values, beta)
    assert not statement.check()
    assert not _iop(field).run(statement)


@pytest.mark.parametrize(
    "entry, accepted", [(0, True), (5, False)], ids=["inside", "outside"]
)
def test_all_depth_0_trees_check_the_lookup_identity(
    field: PseudoMersenneField, entry: int, accepted: bool
) -> None:
    """With every tree of depth 0 (one oracle entry, `t_1`) the verifier
    still checks the lookup identity: an entry outside the table rejects."""
    statement = _statement(field, [entry], beta=1)
    assert statement.check() is accepted
    assert bool(_iop(field).run(statement)) is accepted


def test_xi_inside_the_table_is_drawn_again(
    field: PseudoMersenneField, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first draw of `xi` inside `t_beta` is replaced by a second one,
    under its own label, and the run goes on with it."""
    iop = _iop(field)
    draw = iop.verifier._draw_challenge

    def in_table_first(label: str, domain: object = None) -> object:
        if label.endswith("/xi"):
            return field(BETA - 1)
        return draw(label, domain)

    monkeypatch.setattr(iop.verifier, "_draw_challenge", in_table_first)
    assert iop.run(_statement(field, _bounded(seed=6)))
    (first,) = [x for x in iop.transcript.order if x.endswith("/xi")]
    (second,) = [x for x in iop.transcript.order if x.endswith("/xi_1")]
    assert _entries(iop, first) == field(BETA - 1)
    assert int(_entries(iop, second)) >= BETA


@pytest.mark.parametrize("bad", [False, True], ids=["honest", "unbounded"])
def test_over_an_extension_field(bad: bool) -> None:
    """Over `F_(p^2)` an honest run is accepted and an entry outside the
    table is rejected."""
    field = Field(562949948178433, 2, 5)
    values = _bounded(seed=7)
    if bad:
        values[9] = BETA
    statement = _statement(field, values)
    assert statement.check() is not bad
    assert bool(_iop(field).run(statement)) is not bad


def test_a_characteristic_below_the_looked_up_entries_is_rejected() -> None:
    field = Field(61)
    with pytest.raises(ValueError, match="characteristic"):
        _iop(field).run(_statement(field, _bounded()))


def test_rejects_when_every_entry_is_unbounded(field):
    values = [BETA + i for i in range(1 << NUM_VARS)]
    assert not _iop(field).run(_statement(field, values))


def test_fiat_shamir_rejects_an_unbounded_vector(field):
    values = _bounded(seed=14)
    values[3] = BETA
    statement = _statement(field, values)
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert not _iop(field, fiat_shamir=True).verify(statement, proof)


class _WrongCounts(LogUpGKR):
    """A prover whose multiplicities are off: `move` takes one count from
    entry 0 to entry 1, otherwise one is added to entry 0."""

    def __init__(self, field, move=False):
        super().__init__(field)
        self.move = move

    def _multiplicities(self, vectors, beta):
        counts = super()._multiplicities(vectors, beta).to_list()
        counts[0] = counts[0] + (-self.field.one if self.move else self.field.one)
        if self.move:
            counts[1] = counts[1] + self.field.one
        return FieldVector(self.field, counts)


@pytest.mark.parametrize("move", [False, True], ids=["added", "moved"])
def test_rejects_wrong_multiplicities(field, move):
    """Multiplicities that are not the counts break the lookup identity on a
    bounded oracle, even when they keep the total."""
    values = [i % BETA for i in range(1 << NUM_VARS)]
    statement = _statement(field, values)
    assert statement.check()
    assert not _iop(field, _WrongCounts(field, move)).run(statement)


# --- the shape of the transcript and of the reduction -------------------------


def test_transcript_structure(field):
    """The only table the prover sends is `m`; then `xi`, the depth-0
    leaves (none here), the children of the roots, and per layer `k >= 1` a
    sumcheck of `k` rounds followed by the values `v{k}`."""
    iop = _iop(field)
    assert iop.run(_statement(field, _bounded(seed=6)))
    labels = iop.transcript.order
    assert not [x for x in labels if "read_ts" in x or "f1_" in x or "final_cts" in x]
    assert labels[:5] == [
        "logup/m3",
        "logup/xi",
        "logup/leaf",
        "logup/v0",
        "logup/psi0",
    ]
    assert _entries(iop, "logup/leaf") == ()
    assert len(_entries(iop, "logup/m3")) == BETA

    expected = ["logup/m3", "logup/leaf", "logup/v0"]
    for k in range(1, NUM_VARS):
        expected += ["sumcheck"] * k + [f"logup/v{k}"]
    messages = ["sumcheck" if x.startswith("sumcheck/") else x for x in _messages(iop)]
    assert messages == expected
    # Every challenge is drawn only after the message it answers.
    for k in range(1, NUM_VARS):
        assert labels.index(f"logup/psi{k - 1}") > labels.index(f"logup/v{k - 1}")
        assert labels.index(f"logup/lambda{k}") > labels.index(f"logup/psi{k - 1}")
    assert labels.index("logup/xi") > labels.index("logup/m3")
    # Two trees until the table's leaves, then the oracle's alone.
    assert labels.index("logup/alpha1_1") < labels.index("logup/v1")
    assert not [
        x for x in labels if x.startswith(f"logup/alpha{BETA.bit_length() - 1}_")
    ]


def test_depth_0_trees_send_their_leaf(field):
    """An oracle over no variables and the table of one entry are trees of
    depth 0: their leaves are the only values sent, and there is no layer."""
    iop = _iop(field)
    assert iop.run(_statement(field, [0], beta=1))
    assert _messages(iop) == ["logup/m0", "logup/leaf"]
    # The oracle side sends its denominator, the table side its numerator.
    assert len(_entries(iop, "logup/leaf")) == 2


def test_the_layer_sumchecks_carry_four_nodes(field):
    """Each layer's summand is of degree 3 in each variable; the run has
    `D (D - 1) / 2` rounds for a deepest tree of depth `D`."""
    iop = _iop(field)
    assert iop.run(_statement(field, _bounded(seed=8)))
    rounds = [
        _entries(iop, x)
        for x in iop.transcript.order
        if x.startswith("sumcheck/") and "/g" in x
    ]
    assert {len(r) for r in rounds} == {4}
    assert len(rounds) == NUM_VARS * (NUM_VARS - 1) // 2


class _Outputs(LogUpGKR):
    """Keeps what the verifier half returns."""

    def __init__(self, field, out):
        super().__init__(field)
        self.out = out

    async def verify(self, verifier, statements):
        result = await super().verify(verifier, statements)
        self.out.extend(result)
        return result


@pytest.mark.parametrize(
    "num_vars, beta", [(6, 8), (3, 16), (0, 8), (4, 1)], ids=lambda x: str(x)
)
def test_reduces_to_one_evaluation_per_tree(field, num_vars, beta):
    """One true evaluation claim on the oracle and one on `m`, at points
    over their own variables, and nothing else."""
    seen, out = [], []
    iop = _iop(field, _Outputs(field, out))
    _record_evals(iop, seen)
    statement = _statement(field, _bounded(seed=7, beta=beta, num_vars=num_vars), beta)
    assert iop.run(statement)
    assert len(out) == 2 and all(isinstance(s.relation, Relation_Eval) for s in out)
    assert [id(s) for s in seen] == [id(s) for s in out]
    assert all(s.check() for s in seen)
    (f,) = statement.oracles
    on_f = [s for s in seen if s.oracles[0] is f]
    assert len(on_f) == 1 and len(on_f[0].point) == num_vars
    (on_m,) = [s for s in seen if s.oracles[0] is not f]
    assert len(on_m.point) == beta.bit_length() - 1


# --- which check rejects which lie --------------------------------------------


def _replay_with_tamper(honest, make_iop, statement, tamper, swallow=()):
    """A verifier-only run over an honest transcript with one message
    rewritten, the recorded challenges replayed as they were."""
    iop = make_iop()
    for relation in swallow:
        iop.register(relation, _Swallow(relation))
    for label in honest.transcript.order:
        iop.transcript.write(label, tamper(label, _entries(honest, label)))
    return iop.loop.run_until_complete(iop.verifier.verify(statement._fork()))


def _bump(field, target):
    """A tamper adding one to the first entry of the message `target`."""

    def tamper(label, value):
        if label != target:
            return value
        return (value[0] + field.one, *value[1:])

    return tamper


def test_replay_harness_accepts_an_untampered_transcript(field):
    statement = _statement(field, _bounded(seed=11))
    honest = _iop(field)
    assert honest.run(statement)
    assert _replay_with_tamper(
        honest, lambda: _iop(field), statement, lambda _label, value: value
    )


def test_gkr_checks_catch_a_tampered_tree_value(field):
    """A wrong value on any layer of either tree is caught by GKR's own
    checks (the roots' comparison on layer 0, the layer sumcheck's final
    claim below): the evaluation claims are not needed for it."""
    statement = _statement(field, _bounded(seed=12))
    honest = _iop(field)
    assert honest.run(statement)
    for k in range(NUM_VARS):
        label = f"logup/v{k}"
        for index in range(len(_entries(honest, label))):

            def tamper(name, value, label=label, index=index):
                if name != label:
                    return value
                value = list(value)
                value[index] = value[index] + field.one
                return tuple(value)

            assert not _replay_with_tamper(
                honest,
                lambda: _iop(field),
                statement,
                tamper,
                swallow=(Relation_Eval,),
            ), (label, index)


def test_the_evaluation_claim_on_m_is_what_catches_a_wrong_m(field):
    """`m` enters the trees only through its leaves: a wrong `m` sent with
    the honest trees is caught by the evaluation claim on `m`, and accepted
    once that claim is swallowed."""
    statement = _statement(field, _bounded(seed=13))
    honest = _iop(field)
    assert honest.run(statement)
    tamper = _bump(field, "logup/m3")
    assert not _replay_with_tamper(honest, lambda: _iop(field), statement, tamper)
    assert _replay_with_tamper(
        honest, lambda: _iop(field), statement, tamper, swallow=(Relation_Eval,)
    )


# --- several claims run together ----------------------------------------------


class _AllInTable(Relation):
    """Every oracle's entries in `t_beta`, beta given per oracle."""

    name = "all_in_table"
    fields = ("oracles",)
    index_fields = ("betas",)

    def __init__(self, betas):
        super().__init__(index={"betas": tuple(betas)})
        self.betas = tuple(betas)

    def check(self, statement):
        return all(
            Statement(Relation_Lookup(beta), oracles=[o]).check()
            for beta, o in zip(self.betas, statement.oracles, strict=True)
        )


class _Each(Protocol):
    """`_AllInTable` -> one `Relation_Lookup` per oracle, all on the frontier
    at once."""

    name = "each"
    reduces_from = _AllInTable
    reduces_to = (Relation_Lookup,)

    def _outputs(self, statement):
        return [
            self.reduce([statement], Relation_Lookup(beta), oracles=[o])
            for beta, o in zip(statement.relation.betas, statement.oracles, strict=True)
        ]

    async def prove(self, _prover, statements):
        return self._outputs(statements[0])

    async def verify(self, _verifier, statements):
        return self._outputs(statements[0])


def _several(field, tables, betas):
    oracles = [
        _oracle(field, values, prefix=f"o{k}_") for k, values in enumerate(tables)
    ]
    return Statement(_AllInTable(betas), oracles=oracles)


def _several_iop(field, protocol=None, **kw):
    iop = _iop(field, protocol, **kw)
    iop.register(_AllInTable, _Each())
    return iop


def _v0(iop):
    (label,) = [x for x in iop.transcript.order if x.endswith("/v0")]
    return _entries(iop, label)


def _counts(field, tables, padding=0):
    """The multiplicities of `t_beta` over the tables and `padding` zeros."""
    counts = [0] * BETA
    for values in tables:
        for value in values:
            counts[value] += 1
    counts[0] += padding
    return [field(c) for c in counts]


@pytest.mark.parametrize("k", [2, 3, 4])
def test_claims_into_one_table_of_one_size_are_one_tree(field, k):
    """k claims with the same beta and size: one `m` of `beta` entries,
    counting the padding zero blocks, and one oracle-side tree over their
    concatenation (four values in `v0` per tree), ending in one evaluation
    claim on the stacked oracle and one on `m`. Accepted interactively and
    as an argument string."""
    tables = [_bounded(seed=10 + i) for i in range(k)]
    statement = _several(field, tables, [BETA] * k)
    seen, out = [], []
    iop = _several_iop(field, _Outputs(field, out))
    _record_evals(iop, seen)
    assert iop.run(statement)
    (label,) = _m_labels(iop)
    assert label.endswith("/m3")
    padding = ((1 << (k - 1).bit_length()) - k) << NUM_VARS
    assert list(_entries(iop, label)) == _counts(field, tables, padding)
    assert len(_v0(iop)) == 4 + 4
    assert len(out) == 2 and [id(s) for s in seen] == [id(s) for s in out]
    assert all(s.check() for s in seen)
    s = (k - 1).bit_length()
    assert sorted(len(c.point) for c in seen) == [BETA.bit_length() - 1, NUM_VARS + s]
    assert _several_iop(field).run(statement)
    proof = _several_iop(field, fiat_shamir=True).prove(statement)
    assert _several_iop(field, fiat_shamir=True).verify(statement, proof)


def test_the_padding_blocks_are_zeros_the_verifier_counts(field):
    """A group of three is padded with a block of zeros: a prover that counts
    the zeros is accepted even when no member holds a zero."""
    tables = [[1 + (i % (BETA - 1)) for i in range(1 << NUM_VARS)] for _ in range(3)]
    iop = _several_iop(field)
    assert iop.run(_several(field, tables, [BETA] * 3))
    (label,) = _m_labels(iop)
    assert _entries(iop, label)[0] == field(1 << NUM_VARS)


@pytest.mark.parametrize("where", [0, 1, 2])
def test_an_unbounded_entry_in_any_member_rejects(field, where):
    tables = [_bounded(seed=20 + i) for i in range(3)]
    tables[where][5] = BETA
    assert not _several_iop(field).run(_several(field, tables, [BETA] * 3))


def test_claims_of_other_sizes_share_m_in_separate_trees(field):
    """Claims into one table over different numbers of variables: one `m`
    counting all of them, one oracle-side tree per size."""
    tables = [
        _bounded(seed=30),
        _bounded(seed=31, num_vars=NUM_VARS - 1),
        _bounded(seed=32),
    ]
    iop = _several_iop(field)
    assert iop.run(_several(field, tables, [BETA] * 3))
    (label,) = _m_labels(iop)
    assert list(_entries(iop, label)) == _counts(field, tables)
    assert len(_v0(iop)) == 3 * 4  # {0, 2}, {1} and the table


def test_claims_into_other_tables_get_one_m_each(field):
    tables = [_bounded(seed=33), _bounded(seed=34, beta=16), _bounded(seed=35)]
    iop = _several_iop(field)
    assert iop.run(_several(field, tables, [BETA, 16, BETA]))
    assert [x.rsplit("/", 1)[-1] for x in _m_labels(iop)] == ["m3", "m4"]
    assert len(_v0(iop)) == 4 * 4  # {0, 2} and its table, {1} and its table


# Mixed tables and sizes: two of (8, 2^6), one of (8, 2^5), one depth-0 oracle
# into 8, one of (16, 2^3), a depth-0 oracle into the table of one entry and
# one of (1, 2^2).
_MIXED = [
    (BETA, _bounded(seed=40)),
    (BETA, _bounded(seed=41, num_vars=NUM_VARS - 1)),
    (16, _bounded(seed=42, beta=16, num_vars=3)),
    (BETA, [5]),
    (1, [0]),
    (BETA, _bounded(seed=43)),
    (1, [0] * 4),
]


def test_mixed_tables_and_sizes(field):
    """Tables of 8, 16 and 1 entries, oracles of several sizes including
    depth-0 ones: one `m` per table, one evaluation claim per oracle-side
    tree and per `m`, all true; accepted interactively and as an argument
    string."""
    betas = [b for b, _ in _MIXED]
    tables = [t for _, t in _MIXED]
    statement = _several(field, tables, betas)
    seen, out = [], []
    iop = _several_iop(field, _Outputs(field, out))
    _record_evals(iop, seen)
    assert iop.run(statement)
    assert [x.rsplit("/", 1)[-1] for x in _m_labels(iop)] == ["m3", "m4", "m0"]
    # Oracle-side trees: the pair (8, 2^6), (8, 2^5), (8, 1), (16, 2^3),
    # (1, 1) and (1, 2^2).
    assert len(out) == 6 + 3
    assert [id(s) for s in seen] == [id(s) for s in out]
    assert all(s.check() for s in seen)
    proof = _several_iop(field, fiat_shamir=True).prove(statement)
    assert _several_iop(field, fiat_shamir=True).verify(statement, proof)


@pytest.mark.parametrize("where", range(len(_MIXED)))
def test_mixed_rejects_an_unbounded_entry_in_any_member(field, where):
    betas = [b for b, _ in _MIXED]
    tables = [list(t) for _, t in _MIXED]
    tables[where][-1] = betas[where]
    assert not _several_iop(field).run(_several(field, tables, betas))


# --- every prover message is load-bearing -------------------------------------


def _single(field):
    return _statement(field, _bounded(seed=50)), lambda: _iop(field)


def _grouped(field):
    tables = [_bounded(seed=51), _bounded(seed=52), _bounded(seed=53, num_vars=3), [3]]
    statement = _several(field, tables, [BETA] * 4)
    return statement, lambda: _several_iop(field)


@pytest.mark.parametrize("case", [_single, _grouped], ids=["single", "grouped"])
def test_every_prover_message_is_load_bearing(field, case):
    """Rewriting any single prover message, anywhere in the reduction, makes
    the run reject -- the lookup's own messages and those of the sumchecks
    and virtual-oracle openings it reduces to. An empty message (the leaves
    when no tree has depth 0) has nothing to rewrite."""
    statement, make_iop = case(field)
    honest = make_iop()
    assert honest.run(statement)
    messages = [m for m in _messages(honest) if _entries(honest, m) != ()]
    assert any(m.startswith("sumcheck/") for m in messages)
    if case is _grouped:
        assert any(m.endswith("/leaf") for m in messages)
        assert any(m.startswith("virtual/") for m in messages)
    for target in messages:
        assert not _replay_with_tamper(
            honest, make_iop, statement, _bump(field, target)
        ), target


# --- Fiat-Shamir ----------------------------------------------------------------


def test_fiat_shamir_round_trip(field):
    statement = _statement(field, _bounded(seed=60))
    proof = _iop(field, fiat_shamir=True).prove(statement)
    assert _iop(field, fiat_shamir=True).verify(statement, proof)
    for drawn in ("logup/xi", "logup/psi0", "logup/lambda1", "logup/alpha1_1"):
        assert drawn not in proof.labels  # challenges are re-derived


def test_fiat_shamir_is_deterministic(field):
    statement = _statement(field, _bounded(seed=61))
    runs = []
    for _ in range(2):
        iop = _iop(field, fiat_shamir=True)
        assert iop.run(statement)
        runs.append(
            [
                (label, element_digest(_entries(iop, label)))
                for label in iop.transcript.order
            ]
        )
    assert runs[0] == runs[1]


def test_the_registry_binds_the_lookup_protocol(field):
    """A proof made under one lookup protocol does not verify under the
    other: their digests differ, and so do the registries."""
    statement = _statement(field, _bounded(seed=62))
    assert LogUpGKR(field).digest() != Lookup(field).digest()
    assert (
        _iop(field, Lookup(field)).registry_digest()
        != _iop(field, LogUpGKR(field)).registry_digest()
    )
    for made, checked in ((Lookup, LogUpGKR), (LogUpGKR, Lookup)):
        proof = _iop(field, made(field), fiat_shamir=True).prove(statement)
        assert _iop(field, made(field), fiat_shamir=True).verify(statement, proof)
        assert not _iop(field, checked(field), fiat_shamir=True).verify(
            statement, proof
        )


# --- accounting -----------------------------------------------------------------


@pytest.mark.parametrize("num_vars, beta", [(6, 8), (3, 16), (0, 1)])
def test_soundness_error_is_the_draw_of_xi(field, num_vars, beta):
    statement = _statement(field, _bounded(beta=beta, num_vars=num_vars), beta)
    reported = LogUpGKR(field).soundness_error(statement)
    assert reported == ((2 << num_vars) + beta) / (field.exceptional_set_size - beta)


# --- under the range protocol, which is what produces these claims ------------


def _range_iop(field, ring, chunks):
    from vfhe.piop.protocols import RangeDecomposition
    from vfhe.piop.relations import Relation_Range

    iop = _iop(field)
    iop.register(
        Relation_Range,
        RangeDecomposition(
            field, m=2, alphabet=1 << 8, kappa=2, ring=ring, chunks=chunks
        ),
    )
    return iop


@pytest.mark.parametrize(
    "bound, bad",
    [(1 << 6, None), (1 << 6, 7), (1 << 7, None), (1 << 7, 7), (1 << 7, 0)],
    ids=["bounded", "unbounded", "short-top", "short-top-unbounded", "short-top-at-B"],
)
def test_range_proof_end_to_end(field, bound, bad):
    """The whole chain: a range claim over R_q reduces to lookups on the
    digits, proven by this protocol, which reduce to evaluation claims on
    the real oracles. Accepted exactly when the coefficients are in range. A
    bound of 7 bits in two digits adds a lookup into the smaller table of
    the top digit (two tables in one run); a coefficient of `B` has digits
    in `t_beta` and is caught by that one."""
    from vfhe.arith import Polynomial, Ring
    from vfhe.piop.relations import Relation_Range

    ring = Ring(8, prime_size=[30, 30], split_degree=1)
    chunks, ell = 2, 2
    rng = random.Random(16)
    table = []
    for i in range(1 << ell):
        coefficients = [rng.randrange(bound) for _ in range(ring.N)]
        if i == 0 and bad is not None:
            coefficients[2] = bound + bad
        table.append(Polynomial(ring).from_array(coefficients))
    v = TableOracle(
        MLE(
            ring=ring,
            variables=[MLE_Variable(f"x{i}") for i in range(ell)],
            evaluations=table,
        )
    )

    statement = Statement(Relation_Range(bound), oracles=[v])
    bounded = bad is None
    assert statement.check() is bounded
    iop = _range_iop(field, ring, chunks)
    assert iop.run(statement) is bounded
    if bounded:
        tables = 1 if bound == 1 << 6 else 2
        assert len(_m_labels(iop)) == tables
