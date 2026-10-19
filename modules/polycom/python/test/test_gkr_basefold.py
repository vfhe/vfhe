# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""A circuit over a committed input: GKR down to the input layer, whose two
claims the driver parks and hands to BasefoldEval as one bundle."""

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring, RNSRing, SparseMLE
from vfhe.circuit import add_gate, gkr, mul_gate
from vfhe.piop import (
    GKR,
    IOP,
    GKR_Start,
    OracleKind,
    OutputTest,
    Protocol,
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
    compose,
    with_start,
)
from vfhe.piop.circuit import Relation_CircuitSparseIndex
from vfhe.piop.sparse import FamilyShape, Relation_SpEval, dense_layering
from vfhe.piop.virtual import _one_of
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
    return Statement(
        Relation_CircuitSparseIndex.indexed(_circuit(), ring), oracles=(out, w_in)
    )


def _iop(ring: RNSRing, scheme: Basefold, bundles: list, fiat_shamir=False) -> IOP:
    class _Counting(BasefoldEval):
        async def prove(self, prover, statements):
            bundles.append(len(statements))
            return await super().prove(prover, statements)

    iop = IOP(domain=ring, fiat_shamir=fiat_shamir)
    iop.register(Relation_CircuitSparseIndex, GKR())
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


# --- the wiring predicates committed through a dense index oracle -------------
#
# The basic realization of piop's test_sparse.py, copied (pytest's importlib
# mode keeps test modules apart): `P_dense` over `B || T || C`, column 0 the
# value of the b-th nonzero of slot i, column j + 1 bit j of its index, no
# `ind*`, no packing; the decoder scatters the rows back; the protocol runs the
# registered sumcheck once per claim on the master, over `m + 1` constituents
# that are `P_dense` under maps fixing the slot and column bits. Here `P_dense`
# is committed with the scheme, so the claims `VirtualEval` leaves on it are
# bundled for `BasefoldEval`; the input is public, since one code fixes `k_d`
# and the two tables differ in size.


def _bits(n: int) -> int:
    return (n - 1).bit_length() if n > 1 else 0


def _layout(dense, shape):
    c = shape.width.bit_length()
    return dense.num_vars - len(shape.selectors) - c, c


def _row(ell: int, t: int, b: int, i: int, col: int) -> int:
    return b | (i << ell) | (col << (ell + t))


def _encode(shape, tables: tuple, ring: RNSRing) -> MLE:
    """The family's tables, one per slot, into one dense table over the ring."""
    nnz = max(len(m.evaluations) for m in tables)
    ell, t, c = max(1, _bits(nnz)), len(shape.selectors), shape.width.bit_length()
    entries = [0] * (1 << (ell + t + c))
    for i, member in enumerate(tables):
        width = shape.slots[i][2]
        for b, k in enumerate(sorted(member.evaluations)):
            entries[_row(ell, t, b, i, 0)] = member.evaluations[k]
            for j in range(width):
                entries[_row(ell, t, b, i, j + 1)] = (k >> j) & 1
    variables = [MLE_Variable(f"d{i}") for i in range(ell + t + c)]
    return MLE(ring=ring, variables=variables, evaluations=entries)


@with_start
class _BasicSpEval(Relation_SpEval):
    name = "basic_speval"

    def compute_head(self, statement):
        dense, shape = self.index["dense"], self.index["shape"]
        ell, _ = _layout(dense, shape)
        t = len(shape.selectors)
        evaluations: dict = {}
        for i, (_, _, width) in enumerate(shape.slots):
            for b in range(1 << ell):
                value = dense.table[_row(ell, t, b, i, 0)]
                if value == 0:
                    continue
                k = sum(
                    1 << j
                    for j in range(width)
                    if dense.table[_row(ell, t, b, i, j + 1)] == 1
                )
                evaluations[shape.index_of(i, k)] = value
        return SparseMLE(
            variables=self.index["sparse"].variables,
            evaluations=evaluations,
            ring=dense.ring,
        )


class _BasicSpEvalProtocol(Protocol):
    name = "basic_speval"
    reduces_from = _BasicSpEval.Start
    reduces_to = (Relation_Sum,)

    async def prove(self, prover, statements):
        return self._sums(statements)

    async def verify(self, verifier, statements):
        return self._sums(statements)

    def _sums(self, statements):
        (s,) = statements
        dense, shape = s.index["dense"], s.index["shape"]
        ell, _ = _layout(dense, shape)
        t, m = len(shape.selectors), shape.width
        bs, ts, cs = (
            dense.variables[:ell],
            dense.variables[ell : ell + t],
            dense.variables[ell + t :],
        )
        one = _one_of([dense])
        out = []
        for point, value in zip(s.points, s.values, strict=True):
            i = sum(
                (1 if point[tv] == 1 else 0) << j
                for j, tv in enumerate(shape.selectors)
            )
            xs = [point[xv] for xv in shape.xs]
            xs = [one * x if isinstance(x, int) else x for x in xs]
            maps = [
                {
                    **{b: b for b in bs},
                    **{tv: (i >> j) & 1 for j, tv in enumerate(ts)},
                    **{cv: (col >> j) & 1 for j, cv in enumerate(cs)},
                }
                for col in range(m + 1)
            ]
            terms = []
            for mask in range(1 << m):
                coeff = one
                for j, x in enumerate(xs):
                    coeff = coeff * ((x + x - one) if (mask >> j) & 1 else (one - x))
                terms.append(
                    (coeff, (0, *(j + 1 for j in range(m) if (mask >> j) & 1)))
                )
            G = VirtualOracle(list(bs), [dense] * (m + 1), terms, maps)
            out.append(self.reduce([s], oracles=[G], value=value))
        return out


def _dense_setup():
    """The ring, a scheme whose code encodes the dense table (64 entries for
    the two-layer circuit), and the dense circuit relation over it."""
    ring = Ring(1024, prime_size=[49], split_degree=4)
    scheme = Basefold(FoldableRS(ring, k0=4, c=4, d=4))
    sp = Relation_CircuitSparseIndex.indexed(_circuit(), ring)
    shape = FamilyShape.of(sp.index, sp.sparse_index_fields)
    dense = _encode(shape, sp.index["predicates"], ring)
    scheme.commit(dense)  # tagged: kind committed, its commitment
    relation = Relation_Circuit(
        index={
            "circuit": _circuit(),
            "dense": dense,
            "shape": shape,
            "speval": _BasicSpEval,
        }
    )
    return ring, scheme, relation


def _dense_instance(ring: RNSRing, relation, tweak=0):
    """A public input and the circuit's public output, through `relation`."""
    w = [MLE_Variable(f"w{i}") for i in range(2)]
    inputs = [ring.random_element() for _ in range(4)]
    a, b, c, d = inputs
    w_in = MLE(ring=ring, variables=w, evaluations=inputs, public=True)
    out = MLE(
        ring=ring,
        variables=[MLE_Variable("o0")],
        evaluations=[a + b + c * d + tweak, 0],
        public=True,
    )
    return Statement(relation, oracles=(out, w_in))


def _dense_iop(ring, scheme, bundles: list, fiat_shamir=False) -> IOP:
    class _Counting(BasefoldEval):
        async def prove(self, prover, statements):
            bundles.append([Relation_Eval.kind(s) for s in statements])
            return await super().prove(prover, statements)

    iop = IOP(domain=ring, fiat_shamir=fiat_shamir)
    dense = compose(dense_layering, "dense")
    iop.register(Relation_Circuit, dense)
    iop.register(Relation_Circuit.Start, dense)
    iop.register(Relation_CircuitSparseIndex, GKR())
    iop.register(Relation_CircuitSparseIndex.Start, GKR_Start())
    iop.register(Relation_Equal, OutputTest())
    iop.register(Relation_Layer.Start, SumDefStart())
    iop.register(Relation_Sum, Sumcheck())
    iop.register(Relation_Eval, VirtualEval(), kind=OracleKind.virtual)
    iop.register(_BasicSpEval.Start, _BasicSpEvalProtocol())
    iop.register(Relation_Eval, _Counting(scheme, rep=4))
    return iop


def test_gkr_over_committed_predicates():
    ring, scheme, relation = _dense_setup()
    dense, shape = relation.index["dense"], relation.index["shape"]
    assert dense.kind is OracleKind.committed  # pyright: ignore[reportAttributeAccessIssue]
    stmt = _dense_instance(ring, relation)
    assert stmt.check()
    bundles: list = []
    iop = _dense_iop(ring, scheme, bundles)
    assert iop.run(stmt)
    # Every claim on the dense table -- (m + 1) per claim on the master, six
    # of those for the two layers -- in one bundle, each committed to
    # oracle_kind through the tag; the public input leaves none.
    (kinds,) = bundles
    assert len(kinds) == 6 * (shape.width + 1)
    assert all(k is OracleKind.committed for k in kinds)
    assert not _dense_iop(ring, scheme, []).run(
        _dense_instance(ring, relation, tweak=1)
    )


def test_gkr_over_committed_predicates_fiat_shamir():
    ring, scheme, relation = _dense_setup()
    stmt = _dense_instance(ring, relation)
    proof = _dense_iop(ring, scheme, [], fiat_shamir=True).prove(stmt)
    assert _dense_iop(ring, scheme, [], fiat_shamir=True).verify(stmt, proof)
    other = _dense_instance(ring, relation)
    assert not _dense_iop(ring, scheme, [], fiat_shamir=True).verify(other, proof)
    bad = _dense_instance(ring, relation, tweak=1)
    assert not _dense_iop(ring, scheme, [], fiat_shamir=True).run(bad)
