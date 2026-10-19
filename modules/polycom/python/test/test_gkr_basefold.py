# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""A circuit over a committed input: GKR down to the input layer, whose two
claims the driver parks and hands to BasefoldEval as one bundle."""

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring, RNSRing
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    GKR,
    IOP,
    OracleKind,
    OutputTest,
    Relation_Circuit,
    Relation_Equal,
    Relation_Eval,
    Relation_Layer,
    Relation_Sum,
    Statement,
    Sumcheck,
    SumDefStart,
    VirtualEval,
    VirtualOracle,
)
from vfhe.polycom import Basefold, BasefoldEval, FoldableRS


def _circuit():
    return gkr.Circuit(
        num_inputs=4,
        layers=[
            gkr.Layer(gates=[add_gate(0, 1), mul_gate(2, 3)]),
            gkr.Layer(gates=[add_gate(0, 1)]),
        ],
    )


def _instance(ring: RNSRing, scheme: Basefold):
    """A random input, committed under `scheme`, and the circuit claim on
    it with a public output."""
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    inputs = [ring.random_element() for _ in range(4)]
    w_in = MLE(ring=ring, variables=w, evaluations=inputs)
    a, b, c, d = inputs
    out = MLE(
        ring=ring,
        variables=[MLE_Variable("o0")],
        evaluations=[a + b + c * d, 0],
        public=True,
    )
    scheme.commit(w_in)  # the scheme records commitment and opening
    return Statement(Relation_Circuit.indexed(_circuit(), ring), oracles=(out, w_in))


def _iop(ring: RNSRing, scheme: Basefold, bundles: list, fiat_shamir=False) -> IOP:
    class _Counting(BasefoldEval):
        async def prove(self, prover, statements):
            bundles.append(len(statements))
            return await super().prove(prover, statements)

    iop = IOP(domain=ring, fiat_shamir=fiat_shamir)
    iop.register(Relation_Circuit, GKR())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, _Counting(scheme, rep=4))
    return iop


def _setup():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    scheme = Basefold(FoldableRS(ring, k0=2, c=4, d=1))
    return ring, scheme


def test_gkr_over_committed_input():
    ring, scheme = _setup()
    stmt = _instance(ring, scheme)
    bundles: list = []
    iop = _iop(ring, scheme, bundles)
    assert iop.run(stmt)
    assert bundles == [2]  # both input claims opened in one bundle
    opened = {
        x.rsplit("/", 1)[0] for x in iop.transcript.order if x.startswith("basefold")
    }
    assert len(opened) == 2


def test_gkr_over_committed_input_fiat_shamir():
    ring, scheme = _setup()
    stmt = _instance(ring, scheme)
    proof = _iop(ring, scheme, [], fiat_shamir=True).prove(stmt)
    assert _iop(ring, scheme, [], fiat_shamir=True).verify(stmt, proof)
    # Another instance (other inputs, same circuit) does not accept this proof.
    other = _instance(ring, scheme)
    assert not _iop(ring, scheme, [], fiat_shamir=True).verify(other, proof)


def test_commit_tags_the_table_with_its_commitment():
    from vfhe.piop import element_digest, oracle_kind

    ring, scheme = _setup()
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    entries = [ring.random_element() for _ in range(4)]
    f = MLE(ring=ring, variables=w, evaluations=entries)
    assert oracle_kind(f) is OracleKind.plain
    commitment, _ = scheme.commit(f)
    assert f.kind is OracleKind.committed  # pyright: ignore[reportAttributeAccessIssue]
    assert f.commitment is commitment  # pyright: ignore[reportAttributeAccessIssue]
    assert oracle_kind(f) is OracleKind.committed
    # The digest is the commitment's: another tagged object with the same
    # commitment digests the same, whatever its entries.
    other = MLE(ring=ring, variables=w, evaluations=[ring.random_element()] * 4)
    other.kind = OracleKind.committed  # pyright: ignore[reportAttributeAccessIssue]
    other.commitment = commitment  # pyright: ignore[reportAttributeAccessIssue]
    assert element_digest(f) == element_digest(other)
    # An untagged copy digests by content, as any plain table.
    copy = f.copy()
    assert oracle_kind(copy) is OracleKind.plain
    assert element_digest(copy) == element_digest(
        MLE(ring=ring, variables=w, evaluations=entries)
    )
    assert element_digest(copy) != element_digest(f)
    # A statement carrying the tagged table digests through the tag.
    x = {w[0]: ring.random_exceptional(), w[1]: ring.random_exceptional()}
    tagged = Statement(Relation_Eval(), oracles=(f,), point=x, value=0)
    same = Statement(Relation_Eval(), oracles=(other,), point=x, value=0)
    assert element_digest(tagged) == element_digest(same)
    # A public table is not committed.
    public = MLE(ring=ring, variables=w, evaluations=entries, public=True)
    with pytest.raises(ValueError, match="public"):
        scheme.commit(public)


def test_committed_table_as_a_constituent_is_opened_by_basefold():
    # A sum over G = W(x) * E(x), E public and W committed: the claim
    # VirtualEval leaves on W carries the tagged table, is `committed` to
    # oracle_kind, and reaches BasefoldEval through the default registration.
    ring, scheme = _setup()
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    W = MLE(ring=ring, variables=w, evaluations=[ring.random_element() for _ in w * 2])
    E = MLE(ring=ring, variables=w, evaluations=[1, 0, 1, 1], public=True)
    scheme.commit(W)
    G = VirtualOracle.product([W, E])
    total = W.table[0] + W.table[2] + W.table[3]
    stmt = Statement(Relation_Sum(), oracles=[G], value=total)
    kinds: list = []

    class _Recording(BasefoldEval):
        async def prove(self, prover, statements):
            kinds.extend(Relation_Eval.kind(s) for s in statements)
            return await super().prove(prover, statements)

    iop = IOP(domain=ring)
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, _Recording(scheme, rep=4))
    assert iop.run(stmt)
    assert kinds == [OracleKind.committed]
    assert any(label.startswith("basefold") for label in iop.transcript.order)
    bad = Statement(Relation_Sum(), oracles=[G], value=total + 1)
    iop = IOP(domain=ring)
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, BasefoldEval(scheme, rep=4))
    assert not iop.run(bad)
