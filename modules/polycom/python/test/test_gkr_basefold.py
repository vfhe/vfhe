# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""A circuit over a committed input: GKR down to the input layer, whose two
claims the driver parks and hands to BasefoldEval as one bundle."""

from vfhe.arith import MLE, MLE_Variable, Ring, RNSRing
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    GKR,
    IOP,
    ImplicitEval,
    OracleKind,
    Relation_Circuit,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Sumcheck,
    VirtualEval,
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
    return Statement(Relation_Circuit(_circuit()), oracles=[w_in], output=out)


def _iop(ring: RNSRing, scheme: Basefold, bundles: list, fiat_shamir=False) -> IOP:
    class _Counting(BasefoldEval):
        async def prove(self, prover, statements):
            bundles.append(len(statements))
            return await super().prove(prover, statements)

    iop = IOP(domain=ring, fiat_shamir=fiat_shamir)
    iop.register(Relation_Circuit, GKR())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, ImplicitEval(), kind=OracleKind.implicit)
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
