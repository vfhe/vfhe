# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Defined oracles: virtual oracles in a sumcheck and their free reduction
(VirtualEval), implicit oracles and the bundled ImplicitEval, the parking
rule of the driver, and Fiat-Shamir over statements carrying them."""

import random

from vfhe.arith import Ring
from vfhe.piop import (
    IOP,
    MLE,
    ImplicitEval,
    ImplicitOracle,
    MLE_Variable,
    OracleKind,
    Protocol,
    Relation_Eval,
    Relation_Sum,
    Relation_Zero,
    Statement,
    Sumcheck,
    VirtualEval,
    VirtualOracle,
    oracle_kind,
)


class _IntDomain:
    """Deterministic integer challenges (any nonzero integer is exceptional
    for exact integer arithmetic)."""

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random_exceptional(self):
        return self._rng.randrange(2, 1 << 20)


def _vars(*names):
    return [MLE_Variable(n) for n in names]


def _iop(domain=None, **kw):
    iop = IOP(domain=domain or _IntDomain(), **kw)
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(Relation_Eval, ImplicitEval(), kind=OracleKind.implicit)
    return iop


def test_oracle_kinds():
    x = _vars("x0")
    f = MLE(variables=x, evaluations=[1, 2])
    g = VirtualOracle.product([f])
    assert oracle_kind(f) is OracleKind.plain
    assert (
        oracle_kind(MLE(variables=x, evaluations=[1, 2], public=True))
        is OracleKind.public
    )
    assert oracle_kind(f, commitment=object()) is OracleKind.committed
    assert oracle_kind(g) is OracleKind.virtual
    body = VirtualOracle.product([f])
    assert oracle_kind(ImplicitOracle([], body, x)) is OracleKind.implicit


def test_virtual_products_form_sumcheck():
    # G(x0, x1) = f(x0, x1) * g(x0, x1) + 3 * h(x1): degree 2, a term over
    # a subset of the variables. The sumcheck runs on the products form and
    # ends in an Eval claim on G that VirtualEval opens into claims on f,
    # g and h, decided by the verifier in the ideal model.
    x = _vars("x0", "x1")
    x1 = x[1]
    f = MLE(variables=x, evaluations=[1, 2, 3, 4])
    g = MLE(variables=x, evaluations=[5, 6, 7, 8])
    h = MLE(variables=[x1], evaluations=[9, 10])
    G = VirtualOracle(x, [f, g, h], terms=[(1, (0, 1)), (3, (2,))])
    assert G.degree == 2
    total = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8 + 3 * (9 + 9 + 10 + 10)
    stmt = Statement(Relation_Sum(), oracles=[G], value=total)
    assert stmt.check()
    iop = _iop()
    assert iop.run(stmt)
    assert iop.transcript.order[:2] == ["sumcheck/g0", "sumcheck/r0"]
    assert len(iop.transcript.entries["sumcheck/g0"].result()) == 3  # degree 2
    assert "virtual/0/vals" in iop.transcript.order
    assert not _iop().run(Statement(Relation_Sum(), oracles=[G], value=total + 1))


def test_virtual_same_oracle_under_two_maps():
    # G(x, y) = W(x) * W(y): one oracle, two constituents, two renamings;
    # sum over the cube is (sum W)^2, and the reduction ends in two claims
    # on W at two points.
    w = _vars("w0", "w1")
    xs, ys = _vars("x0", "x1"), _vars("y0", "y1")
    W = MLE(variables=w, evaluations=[1, 2, 3, 4])
    maps = [dict(zip(w, xs, strict=True)), dict(zip(w, ys, strict=True))]
    G = VirtualOracle(xs + ys, [W, W], terms=[(1, (0, 1))], maps=maps)
    stmt = Statement(Relation_Sum(), oracles=[G], value=100)
    assert stmt.check()
    terminal = []

    class _Record(Protocol):
        reduces_from = Relation_Eval

        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    iop = _iop()
    iop.register(Relation_Eval, _Record())  # plain oracles
    assert iop.run(stmt)
    assert len(terminal) == 2 and all(s.oracles[0] is W for s in terminal)
    assert terminal[0].point != terminal[1].point
    assert all(s.check() for s in terminal)


def test_virtual_claims_dedupe_and_skip_public():
    x = _vars("x0", "x1")
    W = MLE(variables=x, evaluations=[1, 2, 3, 4])
    E = MLE(variables=x, evaluations=[1, 0, 0, 1], public=True)
    # G = E * W + W: three constituents, one public, W twice at the same point.
    G = VirtualOracle(x, [E, W, W], terms=[(1, (0, 1)), (1, (2,))])
    claims = G.claims({x[0]: 5, x[1]: 7})
    assert len(claims) == 1
    c, pt, idxs = claims[0]
    assert c is W and pt == {x[0]: 5, x[1]: 7} and idxs == [1, 2]
    assert not G.public
    assert VirtualOracle.product([E]).public
    # Symbolic binding: constituents untouched, sources become constants.
    H = G.evaluate({x[0]: 5})
    assert H.variables == [x[1]] and G.variables == x
    assert H.maps[1][x[0]] == 5 and H.constituents[1] is W


def _eq_table(variables, ring=None):
    """eq(z, p) over variables [p..., z...] (same width each), as a public
    table in W's domain."""
    n = len(variables) // 2
    table = [int((b & ((1 << n) - 1)) == (b >> n)) for b in range(1 << len(variables))]
    return MLE(ring=ring, variables=variables, evaluations=table, public=True)


def _grand_product_tree(W, levels):
    """Implicit oracles L_{levels-1}, ..., L_0 with L_i(z) = sum_p eq(z, p)
    L_{i+1}(p, 0) L_{i+1}(p, 1), L_levels = W [Tha13, §5.3.1]."""
    below = W
    oracles = []
    for i in range(levels):
        n = below.num_vars - 1
        z = _vars(*[f"z{i}_{k}" for k in range(n)])
        p = _vars(*[f"p{i}_{k}" for k in range(n)])
        eq = _eq_table(p + z, ring=W.ring)
        lo = dict(zip(below.variables[:-1], p, strict=True)) | {below.variables[-1]: 0}
        hi = dict(zip(below.variables[:-1], p, strict=True)) | {below.variables[-1]: 1}
        body = VirtualOracle(
            z + p,
            [eq, below, below],
            [(1, (0, 1, 2))],
            [
                {v: v for v in eq.variables},
                lo,
                hi,
            ],
        )
        below = ImplicitOracle(z, body, p)
        oracles.append(below)
    return oracles


def test_implicit_grand_product_one_level():
    w = _vars("w0", "w1")
    W = MLE(variables=w, evaluations=[2, 3, 5, 7])
    (L1,) = _grand_product_tree(W, 1)  # L1(z) = W(z, 0) * W(z, 1)
    (z,) = L1.variables
    assert L1.evaluate({z: 0}).constant() == 2 * 5
    assert L1.evaluate({z: 1}).constant() == 3 * 7
    r = 11
    v = L1.evaluate({z: r}).constant()
    assert (
        v == (1 - r) * 10 + r * 21
    )  # the MLE of the products, not the product of MLEs
    stmt = Statement(Relation_Eval(), oracles=[L1], point={z: r}, value=v)
    assert stmt.check()
    iop = _iop()
    assert iop.run(stmt)
    # No challenge for a single claim; the sumcheck over p has degree 3.
    assert iop.transcript.order[0] == "sumcheck/0/g0"
    assert len(iop.transcript.entries["sumcheck/0/g0"].result()) == 4
    assert not _iop().run(
        Statement(Relation_Eval(), oracles=[L1], point={z: r}, value=v + 1)
    )


def test_implicit_two_levels_bundle_claims():
    # Two claims on L1 (from the sumcheck of L0's definition) meet in one
    # ImplicitEval bundle; W's claims from the two combined bodies dedupe.
    w = _vars("w0", "w1", "w2")
    W = MLE(variables=w, evaluations=[2, 3, 5, 7, 11, 13, 17, 19])
    L1, L0 = _grand_product_tree(W, 2)
    (z,) = L0.variables
    r = 6
    v = L0.evaluate({z: r}).constant()
    sizes = []

    class _Counting(ImplicitEval):
        async def _combine(self, party, statements):
            sizes.append(len(statements))
            return await super()._combine(party, statements)

    terminal = []

    class _Record(Protocol):
        async def prove(self, prover, statements):
            return []

        async def verify(self, verifier, statements):
            terminal.extend(statements)
            return []

    iop = _iop()
    iop.register(Relation_Eval, _Counting(), kind=OracleKind.implicit)
    iop.register(Relation_Eval, _Record())
    iop.prover.witnesses[L1] = L1.materialize()  # the prover's table for L1
    assert iop.run(Statement(Relation_Eval(), oracles=[L0], point={z: r}, value=v))
    assert sizes == [1, 2, 1, 2]  # per party: L0 alone, then both L1 claims
    assert "implicit/0/0/0/a0" in iop.transcript.order  # RLC challenges
    assert len(terminal) == 2 and all(s.oracles[0] is W for s in terminal)
    assert all(s.check() for s in terminal)


def test_parking_rule_waits_for_all_claims_on_an_oracle():
    # Zero -> two Sum claims on f (synthetic); each sumcheck emits Eval(f).
    # The first Eval(f) must wait for the second Sum to be reduced, so the
    # bundle protocol for plain oracles sees both claims at once.
    sizes = []

    class _Split(Protocol):
        reduces_from = Relation_Zero
        reduces_to = (Relation_Sum,)

        async def prove(self, prover, statements):
            return self._go(statements)

        async def verify(self, verifier, statements):
            return self._go(statements)

        def _go(self, statements):
            (s,) = statements
            (f,) = s.oracles
            total = sum(f.table)
            return [
                self.reduce(statements, value=total),
                self.reduce(statements, value=total),
            ]

    class _Bundle(Protocol):
        reduces_from = Relation_Eval
        batching = True

        async def prove(self, prover, statements):
            sizes.append(len(statements))
            return []

        async def verify(self, verifier, statements):
            sizes.append(len(statements))
            assert all(s.check() for s in statements)
            return []

    x = _vars("x0", "x1")
    f = MLE(variables=x, evaluations=[1, 2, 3, 4])
    iop = _iop()
    iop.register(Relation_Zero, _Split())
    iop.register(Relation_Eval, _Bundle())
    assert iop.run(Statement(Relation_Zero(), oracles=[f]))
    assert sizes == [2, 2]


def test_public_eval_is_terminal_even_with_a_default_protocol():
    calls = []

    class _Never(Protocol):
        batching = True

        async def prove(self, prover, statements):
            calls.append("prove")
            return []

        async def verify(self, verifier, statements):
            calls.append("verify")
            return []

    x = _vars("x0")
    e = MLE(variables=x, evaluations=[3, 4], public=True)
    iop = _iop()
    iop.register(Relation_Eval, _Never())
    assert iop.run(Statement(Relation_Eval(), oracles=[e], point={x[0]: 2}, value=5))
    assert calls == []


def test_fiat_shamir_over_defined_oracles():
    ring = Ring(1024, prime_size=[49], split_degree=4)
    w = _vars("w0", "w1")
    W = MLE(ring=ring, variables=w, evaluations=[2, 3, 5, 7])
    (L1,) = _grand_product_tree(W, 1)
    (z,) = L1.variables
    r = ring.random_exceptional()
    v = L1.evaluate({z: r}).constant()
    stmt = Statement(Relation_Eval(), oracles=[L1], point={z: r}, value=v)
    proofs = []
    for _ in range(2):
        iop = _iop(domain=ring, fiat_shamir=True)
        proofs.append(iop.prove(stmt))
    assert proofs[0].labels == proofs[1].labels
    assert proofs[0].digest() == proofs[1].digest()
    assert _iop(domain=ring, fiat_shamir=True).verify(stmt, proofs[0])
    bad = Statement(Relation_Eval(), oracles=[L1], point={z: r}, value=v + 1)
    assert not _iop(domain=ring, fiat_shamir=True).verify(bad, proofs[0])
