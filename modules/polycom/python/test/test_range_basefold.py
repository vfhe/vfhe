# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
"""The range proof compiled with basefold: `mu'`, the counters and the four
`f^(1)` tables are sent as commitments (`BasefoldMessages`), one field scheme
per table size, and `v` is committed with ring basefold. Every evaluation
claim then ends in a `BasefoldEval` bundle: one per scheme. Honest runs
accept, interactively and as an argument string; a tampered commitment and
an out-of-range coefficient are rejected. The residue form of the claim,
one oracle per prime committed with field basefold over its own field, is
compiled the same way. With the lookup proven by `LogUpGKR` instead, the
sent oracles are `mu'` and one multiplicity table per digit table."""

import math
from array import array

import pytest
from vfhe.arith import (
    MLE,
    Field,
    FieldVector,
    MLE_Variable,
    Polynomial,
    PseudoMersenneField,
    Ring,
)
from vfhe.crypto import DIGEST_LEN
from vfhe.piop import (
    IOP,
    CommittedOracle,
    OracleKind,
    Proof,
    Protocol,
    Relation,
    Relation_Eval,
    Statement,
    TableOracle,
)
from vfhe.piop.protocols import (
    LogUpGKR,
    Lookup,
    RangeDecomposition,
    ResidueRangeDecomposition,
    Sumcheck,
    VirtualEval,
)
from vfhe.piop.protocols.range import digit_alphabets
from vfhe.piop.relation import hypercube_sum
from vfhe.piop.relations import (
    Relation_Lookup,
    Relation_Range,
    Relation_ResidueRange,
    Relation_Sum,
)
from vfhe.polycom import (
    Basefold,
    BasefoldEval,
    BasefoldMessages,
    FieldFoldableRS,
    FoldableRS,
)

# ell = 2, nu = 3, gamma = 1 -> ell* = 6; beta = 8 -> b* = 3. Small, for the
# pure-Python degree-3 rounds of the lookup argument.
BOUND = 1 << 6
CHUNKS = 2
ELL = 2
NU = 3
ALPHABET = 1 << 8
KAPPA = 2
REP = 4

#: The oracles the range proof and the lookup send, in transcript order.
SENT = ["range/mu", "read_ts", "final_cts", "f1_WSmu", "f1_RS", "f1_WSt", "f1_S"]


@pytest.fixture(params=[208, 156])
def field(request):
    return PseudoMersenneField.generate(request.param, two_adicity=8)


@pytest.fixture
def ring():
    return Ring(1 << NU, prime_size=[30, 30], split_degree=1)


def _witness(ring, bound=BOUND, tweak=None):
    table = []
    for i in range(1 << ELL):
        coefficients = [(131 * i + 17 * k + 5) % bound for k in range(ring.N)]
        if i == 0 and tweak is not None:
            tweak(coefficients)
        table.append(Polynomial(ring).from_array(coefficients))
    variables = [MLE_Variable(f"x{i}") for i in range(ELL)]
    return MLE(ring=ring, variables=variables, evaluations=table)


class _Setup:
    """The schemes, the committed `v` and the range claim on it. A bound
    whose bit count `CHUNKS` does not divide adds the table sizes of the
    lookup on the top digits: `ell + nu` variables on its oracle's side,
    `log2(top_beta)` on the table's. `lookup` is the protocol registered for
    `Relation_Lookup`."""

    def __init__(
        self,
        field,
        ring,
        v,
        m,
        bound=BOUND,
        sent=(),
        lookup: type[Lookup | LogUpGKR] = Lookup,
    ):
        self.field, self.ring, self.m, self.sent = field, ring, m, sent
        self.lookup = lookup
        relation = Relation_Range(bound)
        beta, top_beta = digit_alphabets(bound, CHUNKS)
        ell_star = ELL + NU + (CHUNKS.bit_length() - 1)
        sizes = [ell_star, beta.bit_length() - 1]
        if top_beta != beta:
            sizes += [ELL + NU, top_beta.bit_length() - 1]
        self.field_schemes = [
            Basefold(FieldFoldableRS(field, k0=4, c=4, d=size - 2))
            for size in sizes
            if size not in sent
        ]
        self.ring_scheme = Basefold(FoldableRS(ring, k0=2, c=4, d=ELL - 1))
        self.v, self.v_opening = self.ring_scheme.commit(v)
        self.v_table = v
        self.statement = Statement(relation, oracles=[self.v])

    def iop(self, domain=None, **kw) -> IOP:
        """The compiled range proof's IOP, over `domain` (the field unless
        given)."""
        messages = BasefoldMessages(self.field_schemes, sent=self.sent)
        iop = IOP(domain=self.field if domain is None else domain, **kw)
        iop.register(
            Relation_Range,
            RangeDecomposition(
                self.field,
                self.m,
                ALPHABET,
                KAPPA,
                self.ring,
                CHUNKS,
                messages=messages,
            ),
        )
        iop.register(Relation_Lookup, self.lookup(self.field, messages=messages))
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
        for scheme in [*self.field_schemes, self.ring_scheme]:
            iop.register(Relation_Eval, BasefoldEval(scheme, rep=REP))
        iop.prover.witnesses[self.v] = self.v_table
        iop.prover.witnesses[self.v.commitment] = self.v_opening
        return iop


def _sent(iop, name):
    """The transcript entry of the oracle message `name`."""
    (label,) = [x for x in iop.transcript.order if x.endswith(name)]
    return iop.transcript.entries[label].result()


@pytest.mark.parametrize("m", [ELL, ELL + NU])
def test_compiled_range_proof_accepts(field, ring, m):
    setup = _Setup(field, ring, _witness(ring), m)
    iop = setup.iop()
    assert iop.run(setup.statement)
    # Each oracle message is a root, not the table.
    for name in SENT:
        root = _sent(iop, name)
        assert isinstance(root, bytes) and len(root) == DIGEST_LEN, name
    # One basefold run per scheme: the claims under a scheme form one bundle.
    runs = [
        x
        for x in iop.transcript.order
        if x.startswith("basefold") and x.endswith("/answers")
    ]
    assert len(runs) == 3


def test_schemes_reused_across_proofs_keep_no_opening_of_theirs(field, ring):
    """The oracles a proof sends are committed for its prover: two proofs
    under the same schemes leave no opening on the field schemes, and only
    the one of `v`, committed up front without a prover, on the ring's."""
    setup = _Setup(field, ring, _witness(ring), ELL + NU)
    for _ in range(2):
        assert setup.iop().run(setup.statement)
    assert all(not scheme.openings for scheme in setup.field_schemes)
    assert list(setup.ring_scheme.openings) == [setup.v.commitment]


def test_compiled_range_proof_as_an_argument_string(field, ring):
    setup = _Setup(field, ring, _witness(ring), ELL + NU)
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)


def test_a_tampered_commitment_is_rejected(field, ring):
    setup = _Setup(field, ring, _witness(ring), ELL + NU)
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    for target in SENT:
        messages = list(proof.messages)
        (index,) = [
            i for i, (label, _) in enumerate(messages) if label.endswith(target)
        ]
        label, root = messages[index]
        messages[index] = (label, bytes([root[0] ^ 1]) + root[1:])
        assert not setup.iop(fiat_shamir=True).verify(
            setup.statement, type(proof)(messages)
        ), target


def _most_acceptances(runs, error, false_failure=1e-6):
    """The most of `runs` independent cheating runs a protocol of soundness
    error `error` accepts, except with probability below `false_failure`."""
    tail = 1.0
    for t in range(runs + 1):
        tail -= math.comb(runs, t) * error**t * (1 - error) ** (runs - t)
        if tail < false_failure:
            return t
    return runs


def test_an_out_of_range_coefficient_is_rejected(field, ring):
    """A single cheating run may pass with the soundness error, so the runs
    are repeated and their acceptances bounded."""
    bad = _witness(ring, tweak=lambda c: c.__setitem__(1, BOUND + 1))
    setup = _Setup(field, ring, bad, ELL + NU)
    error = RangeDecomposition(
        field, ELL + NU, ALPHABET, KAPPA, ring, CHUNKS
    ).soundness_error(setup.statement)
    runs = 16
    accepted = sum(bool(setup.iop().run(setup.statement)) for _ in range(runs))
    assert accepted <= _most_acceptances(runs, error)


def test_compiled_range_proof_with_a_short_top_digit(field, ring):
    """log2(B) = 7 in two digits, the top one of 3 bits: the lookup on the
    top digits is compiled too, and ends in claims on the committed `mu'`.
    An honest run accepts, interactively and as an argument string; a
    coefficient of `B`, whose digits recompose and sit in `t_beta`, is
    rejected (the lookup's soundness error is negligible here)."""
    bound = 1 << 7
    setup = _Setup(field, ring, _witness(ring, bound=bound), ELL + NU, bound=bound)
    iop = setup.iop()
    assert iop.run(setup.statement)
    runs = [
        x
        for x in iop.transcript.order
        if x.startswith("basefold") and x.endswith("/answers")
    ]
    assert len(runs) == 5
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)

    bad = _witness(ring, bound=bound, tweak=lambda c: c.__setitem__(1, bound))
    setup = _Setup(field, ring, bad, ELL + NU, bound=bound)
    assert not setup.iop().run(setup.statement)


# --- the same proofs with the lookup proven by LogUpGKR ----------------------

#: The oracles the range proof and LogUpGKR send, in transcript order: `mu'`
#: and the multiplicities of the digit table, `m3`.
LOGUP_SENT = ["range/mu", "/m3"]


def _basefold_runs(iop):
    return [
        x
        for x in iop.transcript.order
        if x.startswith("basefold") and x.endswith("/answers")
    ]


@pytest.mark.parametrize("m", [ELL, ELL + NU])
def test_compiled_range_proof_accepts_under_logup(field, ring, m):
    """`mu'` and `m` are sent as roots, and the trees never are. One basefold
    run per scheme with a claim: `mu'` (ell* variables), `m` (b*) and `v`."""
    setup = _Setup(field, ring, _witness(ring), m, lookup=LogUpGKR)
    iop = setup.iop()
    assert iop.run(setup.statement)
    for name in LOGUP_SENT:
        root = _sent(iop, name)
        assert isinstance(root, bytes) and len(root) == DIGEST_LEN, name
    assert not [x for x in iop.transcript.order if "read_ts" in x or "/f1_" in x]
    assert len(_basefold_runs(iop)) == 3


def test_compiled_range_proof_as_an_argument_string_under_logup(field, ring):
    setup = _Setup(field, ring, _witness(ring), ELL + NU, lookup=LogUpGKR)
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)


def test_a_tampered_commitment_is_rejected_under_logup(field, ring):
    setup = _Setup(field, ring, _witness(ring), ELL + NU, lookup=LogUpGKR)
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    for target in LOGUP_SENT:
        messages = list(proof.messages)
        (index,) = [
            i for i, (label, _) in enumerate(messages) if label.endswith(target)
        ]
        label, root = messages[index]
        messages[index] = (label, bytes([root[0] ^ 1]) + root[1:])
        assert not setup.iop(fiat_shamir=True).verify(
            setup.statement, Proof(messages)
        ), target


def test_an_out_of_range_coefficient_is_rejected_under_logup(field, ring):
    """The runs are repeated and their acceptances bounded by the error of
    the range reduction plus that of the lookup on `mu'`."""
    bad = _witness(ring, tweak=lambda c: c.__setitem__(1, BOUND + 1))
    setup = _Setup(field, ring, bad, ELL + NU, lookup=LogUpGKR)
    error = RangeDecomposition(
        field, ELL + NU, ALPHABET, KAPPA, ring, CHUNKS
    ).soundness_error(setup.statement)
    beta = digit_alphabets(BOUND, CHUNKS)[0]
    ell_star = ELL + NU + (CHUNKS.bit_length() - 1)
    error += ((1 << ell_star) + beta) / (field.exceptional_set_size - beta)
    runs = 16
    accepted = sum(bool(setup.iop().run(setup.statement)) for _ in range(runs))
    assert accepted <= _most_acceptances(runs, error)


def test_compiled_range_proof_with_a_short_top_digit_under_logup(field, ring):
    """log2(B) = 7 in two digits, the top one of 3 bits: one `m` per digit
    table (`m4` and `m3`), each committed. The top digits' lookup reads a
    restriction of `mu'`, so its claim lands on `mu'`: the scheme of
    `ell + nu` variables gets no claim, and the runs are those of `mu'`,
    `m4`, `m3` and `v`. An honest run accepts, interactively and as an
    argument string; a coefficient of `B` is rejected."""
    bound = 1 << 7
    setup = _Setup(
        field,
        ring,
        _witness(ring, bound=bound),
        ELL + NU,
        bound=bound,
        lookup=LogUpGKR,
    )
    iop = setup.iop()
    assert iop.run(setup.statement)
    for name in ("range/mu", "/m4", "/m3"):
        root = _sent(iop, name)
        assert isinstance(root, bytes) and len(root) == DIGEST_LEN, name
    assert len(_basefold_runs(iop)) == 4
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)

    bad = _witness(ring, bound=bound, tweak=lambda c: c.__setitem__(1, bound))
    setup = _Setup(field, ring, bad, ELL + NU, bound=bound, lookup=LogUpGKR)
    assert not setup.iop().run(setup.statement)


def test_sizes_without_a_scheme_can_be_sent_as_their_entries(field, ring):
    """The digit table of 2^3 entries is sent as its entries and read as a
    table; the larger tables are still committed. The proof accepts,
    interactively and as an argument string, and a wrong digit table is
    rejected."""
    setup = _Setup(field, ring, _witness(ring), ELL + NU, sent=[3])
    iop = setup.iop()
    assert iop.run(setup.statement)
    sizes = {name: _sent(iop, name) for name in SENT}
    assert all(isinstance(sizes[name], bytes) for name in ("range/mu", "read_ts"))
    tables = [entry for entry in sizes.values() if isinstance(entry, FieldVector)]
    assert tables and all(len(entry) == 1 << 3 for entry in tables)
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)
    messages = list(proof.messages)
    at = next(
        i
        for i, (label, _) in enumerate(messages)
        if any(label.endswith(name) for name in SENT)
        and isinstance(messages[i][1], FieldVector)
    )
    label, table = messages[at]
    bumped = table.copy()
    bumped[0] = bumped[0] + 1
    messages[at] = (label, bumped)
    assert not setup.iop(fiat_shamir=True).verify(setup.statement, Proof(messages))


def test_a_size_is_either_sent_or_committed(field):
    scheme = Basefold(FieldFoldableRS(field, k0=4, c=4, d=1))
    with pytest.raises(ValueError, match="both sent and committed"):
        BasefoldMessages([scheme], sent=[3])


def test_messages_need_one_scheme_per_table_size(field):
    small = Basefold(FieldFoldableRS(field, k0=4, c=4, d=1))
    with pytest.raises(ValueError, match="two schemes"):
        BasefoldMessages([small, Basefold(FieldFoldableRS(field, k0=2, c=8, d=2))])
    table = MLE(
        field=field,
        variables=[MLE_Variable(f"y{i}") for i in range(4)],
        evaluations=[field(i) for i in range(16)],
    )
    with pytest.raises(ValueError, match="no scheme for tables of 2\\^4"):
        BasefoldMessages([small]).send(IOP(domain=field).prover, "t", table)


def test_messages_from_one_code_commit_as_codes_built_per_size(field):
    """`from_code` serves every size from one code's tables, and a table
    sent through it gets the root the code built for its size gives."""
    code = FieldFoldableRS(field, k0=4, c=4, d=4)
    messages = BasefoldMessages.from_code(code, [3, 6, 4])
    assert sorted(messages.schemes) == [3, 4, 6]
    for num_vars, scheme in messages.schemes.items():
        assert scheme.code.d == num_vars - 2 and scheme.schedule == [1] * (num_vars - 2)
        table = MLE(
            field=field,
            variables=[MLE_Variable(f"y{i}") for i in range(num_vars)],
            evaluations=[field(3 * i + 1) for i in range(1 << num_vars)],
        )
        oracle = messages.send(IOP(domain=field).prover, "t", table)
        assert isinstance(oracle, CommittedOracle)
        built = Basefold(FieldFoldableRS(field, k0=4, c=4, d=num_vars - 2))
        assert oracle.commitment.root == built.commit(table)[0].commitment.root
    for num_vars in (2, 7):
        with pytest.raises(ValueError, match="serves tables"):
            BasefoldMessages.from_code(code, [num_vars])


class _RangeAndSum(Relation):
    """A range claim and a sum claim on one oracle: every coefficient below
    `BOUND`, and the hypercube sum `value`."""

    name = "range_and_sum"
    fields = ("oracles", "value")

    def check(self, statement: Statement) -> bool:
        (v,) = statement.oracles
        in_range = Statement(Relation_Range(BOUND), oracles=[v]).check()
        return in_range and hypercube_sum(v) == statement.value


class _Split(Protocol):
    """`_RangeAndSum` -> `Relation_Range` x `Relation_Sum`, on the same oracle."""

    name = "split"
    reduces_from = _RangeAndSum
    reduces_to = (Relation_Range, Relation_Sum)

    def _outputs(self, statement: Statement) -> list[Statement]:
        (v,) = statement.oracles
        return [
            self.reduce([statement], Relation_Range(BOUND), oracles=[v]),
            self.reduce([statement], Relation_Sum, oracles=[v], value=statement.value),
        ]

    async def prove(self, _prover, statements):
        return self._outputs(statements[0])

    async def verify(self, _verifier, statements):
        return self._outputs(statements[0])


def test_a_range_proof_runs_inside_an_iop_over_the_ring(field, ring):
    """In an IOP over `R_q`, the range proof draws its challenges in `F_p` and
    a sumcheck on the committed `v` draws them in `R_q`, each in its oracle's
    domain. The range proof's claims on `v` and the sumcheck's then form one
    bundle under `v`'s scheme: one opening of `v`, one basefold run per
    scheme in all."""
    v = _witness(ring)
    setup = _Setup(field, ring, v, ELL + NU)
    statement = Statement(
        _RangeAndSum(), oracles=[setup.v], value=hypercube_sum(TableOracle(v))
    )

    def iop(**kw):
        out = setup.iop(domain=ring, **kw)
        out.register(_RangeAndSum, _Split())
        return out

    run = iop()
    assert run.run(statement)
    runs = [
        x
        for x in run.transcript.order
        if x.startswith("basefold") and x.endswith("/answers")
    ]
    assert len(runs) == 3
    proof = iop(fiat_shamir=True).prove(statement)
    assert iop(fiat_shamir=True).verify(statement, proof)


class _TwoRanges(Relation):
    """Two oracles, every coefficient of each below `BOUND`."""

    name = "two_ranges"
    fields = ("oracles",)

    def check(self, statement: Statement) -> bool:
        return all(
            Statement(Relation_Range(BOUND), oracles=[v]).check()
            for v in statement.oracles
        )


class _Both(Protocol):
    """`_TwoRanges` -> one `Relation_Range` per oracle."""

    name = "both"
    reduces_from = _TwoRanges
    reduces_to = (Relation_Range,)

    def _outputs(self, statement: Statement) -> list[Statement]:
        return [
            self.reduce([statement], Relation_Range(BOUND), oracles=[v])
            for v in statement.oracles
        ]

    async def prove(self, _prover, statements):
        return self._outputs(statements[0])

    async def verify(self, _verifier, statements):
        return self._outputs(statements[0])


def test_two_range_claims_share_one_lookup(field, ring):
    """Two range claims with one digit table and one `mu'` size: one lookup
    run over both digit tables, so the counters and the four f^(1) tables
    are committed once, at one variable more. Accepted interactively and as
    an argument string; a coefficient out of range in either claim
    rejects."""
    ell_star = ELL + NU + (CHUNKS.bit_length() - 1)
    beta_bits = digit_alphabets(BOUND, CHUNKS)[0].bit_length() - 1
    field_schemes = [
        Basefold(FieldFoldableRS(field, k0=4, c=4, d=size - 2))
        for size in (ell_star, ell_star + 1, beta_bits)
    ]
    ring_scheme = Basefold(FoldableRS(ring, k0=2, c=4, d=ELL - 1))

    def setup(tweak_second=None, fiat_shamir=False):
        """The two-range statement and a factory of IOPs proving it."""
        tables = [_witness(ring), _witness(ring, tweak=tweak_second)]
        committed = [ring_scheme.commit(t) for t in tables]

        def iop() -> IOP:
            messages = BasefoldMessages(field_schemes)
            iop = IOP(domain=field, fiat_shamir=fiat_shamir)
            iop.register(_TwoRanges, _Both())
            iop.register(
                Relation_Range,
                RangeDecomposition(
                    field, ELL + NU, ALPHABET, KAPPA, ring, CHUNKS, messages=messages
                ),
            )
            iop.register(Relation_Lookup, Lookup(field, messages=messages))
            iop.register(Relation_Sum, Sumcheck())
            iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
            for scheme in [*field_schemes, ring_scheme]:
                iop.register(Relation_Eval, BasefoldEval(scheme, rep=REP))
            for table, (oracle, opening) in zip(tables, committed, strict=True):
                iop.prover.witnesses[oracle] = table
                iop.prover.witnesses[oracle.commitment] = opening
            return iop

        return Statement(_TwoRanges(), oracles=[o for o, _ in committed]), iop

    statement, make = setup()
    iop = make()
    assert iop.run(statement)
    lookups = [x for x in iop.transcript.order if x.endswith("/read_ts")]
    assert len(lookups) == 1
    sent = [x for x in iop.transcript.order if "lookup" in x and "/f1_" in x]
    assert len(sent) == 4
    statement, make = setup(fiat_shamir=True)
    assert make().verify(statement, make().prove(statement))
    statement, make = setup(tweak_second=lambda c: c.__setitem__(1, BOUND))
    assert not make().run(statement)


# --- the residue range proof -----------------------------------------------


def _residue_tables(fields, tweak=None):
    """The residues of `2^(ELL + NU)` integers below `BOUND`, one table per
    field over its own variables; `tweak(values)` may edit the integers."""
    n = ELL + NU
    values = [(131 * i + 5) % BOUND for i in range(1 << n)]
    if tweak is not None:
        tweak(values)
    tables = []
    for field in fields:
        vec = FieldVector(field, 1 << n)
        vec.set_uint64(array("Q", [x % field.prime for x in values]))
        variables = [MLE_Variable(f"u{i}") for i in range(n)]
        tables.append(MLE(field=field, variables=variables, evaluations=vec))
    return tables


class _ResidueSetup:
    """The residue range claim with each residue table committed with field
    basefold over its own field, and `mu'` and the lookup's tables sent
    through `BasefoldMessages` over `F_p`, one scheme per table size."""

    def __init__(self, field, tables, m=ELL + NU):
        self.field, self.m, self.tables = field, m, tables
        n = ELL + NU
        relation = Relation_ResidueRange(BOUND)
        beta, top_beta = digit_alphabets(BOUND, CHUNKS)
        sizes = [n + (CHUNKS.bit_length() - 1), beta.bit_length() - 1]
        if top_beta != beta:
            sizes += [n, top_beta.bit_length() - 1]
        self.field_schemes = [
            Basefold(FieldFoldableRS(field, k0=4, c=4, d=size - 2)) for size in sizes
        ]
        self.residue_schemes = [
            Basefold(FieldFoldableRS(table.field, k0=4, c=4, d=n - 2))
            for table in tables
        ]
        self.committed = [
            scheme.commit(table)
            for scheme, table in zip(self.residue_schemes, tables, strict=True)
        ]
        self.statement = Statement(
            relation, oracles=[oracle for oracle, _ in self.committed]
        )

    def protocol(self, messages=None) -> ResidueRangeDecomposition:
        return ResidueRangeDecomposition(
            self.field, self.m, ALPHABET, KAPPA, CHUNKS, messages=messages
        )

    def iop(self, **kw) -> IOP:
        messages = BasefoldMessages(self.field_schemes)
        iop = IOP(domain=self.field, **kw)
        iop.register(Relation_ResidueRange, self.protocol(messages))
        iop.register(Relation_Lookup, Lookup(self.field, messages=messages))
        iop.register(Relation_Sum, Sumcheck())
        iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
        for scheme in [*self.field_schemes, *self.residue_schemes]:
            iop.register(Relation_Eval, BasefoldEval(scheme, rep=REP))
        for table, (oracle, opening) in zip(self.tables, self.committed, strict=True):
            iop.prover.witnesses[oracle] = table
            iop.prover.witnesses[oracle.commitment] = opening
        return iop


@pytest.fixture
def residue_fields():
    """A 50-bit prime field of degree 2 and a 60-bit prime field; both have
    the 2-adicity the codes need (`c k0 2^d = 128` points)."""
    return [Field(562949948178433, 2, 5), Field(1152921504577486849)]


def test_compiled_residue_range_proof(field, residue_fields):
    """The residue oracles committed per prime: an honest run accepts,
    interactively and as an argument string, with one basefold run per
    scheme -- the claims at the integer points end in one bundle under each
    residue oracle's scheme."""
    setup = _ResidueSetup(field, _residue_tables(residue_fields))
    iop = setup.iop()
    assert iop.run(setup.statement)
    runs = [
        x
        for x in iop.transcript.order
        if x.startswith("basefold") and x.endswith("/answers")
    ]
    assert len(runs) == len(setup.field_schemes) + len(setup.residue_schemes) == 4
    proof = setup.iop(fiat_shamir=True).prove(setup.statement)
    assert setup.iop(fiat_shamir=True).verify(setup.statement, proof)


def test_compiled_residue_range_proof_rejects_an_out_of_range_entry(
    field, residue_fields
):
    """An entry above the bound; a single run may pass with the soundness
    error, so the runs are repeated and their acceptances bounded."""
    tables = _residue_tables(
        residue_fields, tweak=lambda v: v.__setitem__(1, BOUND + 1)
    )
    claim = Relation_ResidueRange(BOUND)
    assert not Statement(claim, oracles=[TableOracle(t) for t in tables]).check()
    setup = _ResidueSetup(field, tables)
    error = setup.protocol().soundness_error(setup.statement)
    runs = 16
    accepted = sum(bool(setup.iop().run(setup.statement)) for _ in range(runs))
    assert accepted <= _most_acceptances(runs, error)
