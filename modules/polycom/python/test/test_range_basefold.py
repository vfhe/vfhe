# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The range proof compiled with basefold: `mu'`, the counters and the four
`f^(1)` tables are sent as commitments (`BasefoldMessages`), one field scheme
per table size, and `v` is committed with ring basefold. Every evaluation
claim then ends in a `BasefoldEval` bundle: one per scheme. Honest runs
accept, interactively and as an argument string; a tampered commitment and
an out-of-range coefficient are rejected."""

import math

import pytest
from vfhe.arith import MLE, MLE_Variable, Polynomial, PseudoMersenneField, Ring
from vfhe.crypto import DIGEST_LEN
from vfhe.piop import IOP, OracleKind, Relation_Eval, Statement
from vfhe.piop.protocols import Lookup, RangeDecomposition, Sumcheck, VirtualEval
from vfhe.piop.relations import Relation_Lookup, Relation_Range, Relation_Sum
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
    """The schemes, the committed `v` and the range claim on it."""

    def __init__(self, field, ring, v, m):
        self.field, self.ring, self.m = field, ring, m
        ell_star = ELL + NU + (CHUNKS.bit_length() - 1)
        b_star = (BOUND.bit_length() - 1) // CHUNKS
        self.field_schemes = [
            Basefold(FieldFoldableRS(field, k0=4, c=4, d=ell_star - 2)),
            Basefold(FieldFoldableRS(field, k0=4, c=4, d=b_star - 2)),
        ]
        self.ring_scheme = Basefold(FoldableRS(ring, k0=2, c=4, d=ELL - 1))
        self.v, self.v_opening = self.ring_scheme.commit(v)
        self.v_table = v
        self.statement = Statement(Relation_Range(BOUND, CHUNKS), oracles=[self.v])

    def iop(self, **kw) -> IOP:
        messages = BasefoldMessages(self.field_schemes)
        iop = IOP(domain=self.field, **kw)
        iop.register(
            Relation_Range,
            RangeDecomposition(
                self.field, self.m, ALPHABET, KAPPA, self.ring, messages=messages
            ),
        )
        iop.register(Relation_Lookup, Lookup(self.field, messages=messages))
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
    error = RangeDecomposition(field, ELL + NU, ALPHABET, KAPPA, ring).soundness_error(
        setup.statement
    )
    runs = 16
    accepted = sum(bool(setup.iop().run(setup.statement)) for _ in range(runs))
    assert accepted <= _most_acceptances(runs, error)


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
