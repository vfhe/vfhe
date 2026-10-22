# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two-phase (Libra) round-message strategy against the dense one:
identical messages on every round, on the shapes it accepts, and the dense
fallback on the ones it does not."""

import random

import pytest
from vfhe.arith import MLE, MLE_Variable, Ring, SparseMLE
from vfhe.piop import ImplicitOracle, TableOracle, VirtualOracle
from vfhe.piop.protocols.prover_view import _LibraProver, _ProverVirtual, prover_view


def _vars(prefix, n):
    return [MLE_Variable(f"{prefix}{i}") for i in range(n)]


def _random_sparse(rng, variables, nnz, ring=None, values=(1,)):
    size = 1 << len(variables)
    entries = {}
    for _ in range(nnz):
        entries[rng.randrange(size)] = rng.choice(values)
    return TableOracle(
        SparseMLE(ring=ring, variables=variables, evaluations=entries), public=True
    )


def _random_table(rng, variables, ring=None):
    if ring is not None:
        vals = [ring.random_element() for _ in range(1 << len(variables))]
    else:
        vals = [rng.randrange(1, 50) for _ in range(1 << len(variables))]
    return TableOracle(MLE(ring=ring, variables=variables, evaluations=vals))


def _run_both(G, challenges, witnesses=None):
    """Both strategies through every round with the same challenges;
    returns the two message lists and the two final constants."""
    dense = prover_view(G, witnesses, strategy="dense")
    libra = prover_view(G, witnesses, strategy="libra")
    assert isinstance(dense, _ProverVirtual) and isinstance(libra, _LibraProver)
    msgs = ([], [])
    views = [dense, libra]
    for var, r in zip(list(G.variables), challenges, strict=True):
        for i, view in enumerate(views):
            msgs[i].append(view.round_evals(var))
        views = [view.evaluate({var: r}, in_place=False) for view in views]
    return msgs[0], msgs[1], views[0].constant(), views[1].constant()


def _layer_shape(rng, s_z, s_x, ring=None, same_oracle=True, fixed_z=True):
    """A layer body: lin(z,x,y) A(x) + xmult(z,x,y) A(x) B(y), z bound."""
    z, x, y = _vars("z", s_z), _vars("x", s_x), _vars("y", s_x)
    wa = _vars("a", s_x)
    A = _random_table(rng, wa, ring)
    B = A if same_oracle else _random_table(rng, _vars("b", s_x), ring)
    lin = _random_sparse(rng, y + x + z, 6, ring, values=(1, 2))
    xmult = _random_sparse(rng, y + x + z, 6, ring, values=(1,))
    maps = [
        {v: v for v in lin.variables},
        {v: v for v in xmult.variables},
        dict(zip(A.variables, x, strict=True)),
        dict(zip(B.variables, y, strict=True)),
    ]
    G = VirtualOracle(
        z + x + y, [lin, xmult, A, B], [(1, (0, 2)), (1, (1, 2, 3))], maps
    )
    if fixed_z:
        rz = [ring.random_exceptional() if ring else rng.randrange(2, 99) for _ in z]
        G = G.evaluate(dict(zip(z, rz, strict=True)))
    return G


def test_libra_matches_dense_on_a_layer_shape():
    rng = random.Random(1)
    G = _layer_shape(rng, s_z=2, s_x=3)
    assert G.degree == 2  # per-variable degree, not the longest product
    chall = [rng.randrange(2, 1 << 16) for _ in G.variables]
    d_msgs, l_msgs, d_const, l_const = _run_both(G, chall)
    assert d_msgs == l_msgs and d_const == l_const
    assert all(len(m) == 3 for m in d_msgs)  # degree 2 in every round


def test_libra_matches_dense_over_a_ring_with_two_oracles():
    rng = random.Random(2)
    ring = Ring(1024, prime_size=[49], split_degree=4)
    G = _layer_shape(rng, s_z=1, s_x=2, ring=ring, same_oracle=False)
    chall = [ring.random_exceptional() for _ in G.variables]
    d_msgs, l_msgs, d_const, l_const = _run_both(G, chall)
    assert all(
        a == b
        for da, la in zip(d_msgs, l_msgs, strict=True)
        for a, b in zip(da, la, strict=True)
    )
    assert d_const == l_const


def test_libra_in_place_rounds_and_repeated_constituents():
    # One oracle in two terms and on both sides; in-place folding after the
    # first (out-of-place) round, as the sumcheck loop does it.
    rng = random.Random(3)
    G = _layer_shape(rng, s_z=1, s_x=2)
    chall = [rng.randrange(2, 1 << 16) for _ in G.variables]
    dense = prover_view(G, strategy="dense")
    libra = prover_view(G, strategy="libra")
    for i, (var, r) in enumerate(zip(list(G.variables), chall, strict=True)):
        assert dense.round_evals(var) == libra.round_evals(var)
        dense = dense.evaluate({var: r}, in_place=i > 0)
        libra = libra.evaluate({var: r}, in_place=i > 0)
    assert dense.constant() == libra.constant()


def test_libra_handles_predicate_only_variables_and_plain_terms():
    # A term without a predicate (A(x) * B(y)), a predicate reading only
    # part of y, and a summed variable read by no dense factor.
    rng = random.Random(4)
    x, y, e = _vars("x", 2), _vars("y", 2), _vars("e", 1)
    A = _random_table(rng, _vars("a", 2))
    B = _random_table(rng, _vars("b", 2))
    pred = _random_sparse(rng, [y[0], *x, *e], 5, values=(1, 3))
    maps = [
        {v: v for v in pred.variables},
        dict(zip(A.variables, x, strict=True)),
        dict(zip(B.variables, y, strict=True)),
    ]
    G = VirtualOracle(
        x + e + y, [pred, A, B], [(2, (0, 1, 2)), (1, (1, 2)), (1, (0, 1))], maps
    )
    chall = [rng.randrange(2, 1 << 16) for _ in G.variables]
    d_msgs, l_msgs, d_const, l_const = _run_both(G, chall)
    assert d_msgs == l_msgs and d_const == l_const


def test_libra_falls_back_on_unsupported_shapes():
    rng = random.Random(5)
    x, y, w = _vars("x", 1), _vars("y", 1), _vars("w", 1)
    A, B, C = (_random_table(rng, _vars(n, 1)) for n in "abc")
    pred = _random_sparse(rng, x + y + w, 3)
    maps = [
        {v: v for v in pred.variables},
        dict(zip(A.variables, x, strict=True)),
        dict(zip(B.variables, y, strict=True)),
        dict(zip(C.variables, w, strict=True)),
    ]
    three = VirtualOracle(x + y + w, [pred, A, B, C], [(1, (0, 1, 2, 3))], maps)
    assert isinstance(prover_view(three), _ProverVirtual)  # three blocks: dense
    with pytest.raises(ValueError):
        prover_view(three, strategy="libra")
    # A stray variable in a map is a construction error, not a constant.
    with pytest.raises(ValueError):
        VirtualOracle(x + y, [pred, A, B], [(1, (0, 1, 2))], maps[:3])
    # With w fixed to a constant the two-block shape fits.
    two = VirtualOracle(x + y + w, [pred, A, B], [(1, (0, 1, 2))], maps[:3])
    assert isinstance(prover_view(two.evaluate({w[0]: 1})), _LibraProver)
    # No sparse constituent at all: dense, whatever the order.
    dense_pred = TableOracle(
        pred.polynomial.materialize().evaluate({w[0]: 0}, in_place=False)
    )
    swapped = VirtualOracle(
        y + x, [dense_pred, A, B], [(1, (0, 1, 2))], [{v: v for v in x + y}, *maps[1:3]]
    )
    assert isinstance(prover_view(swapped), _ProverVirtual)


def test_libra_matches_dense_over_a_field():
    from vfhe.arith import Field

    rng = random.Random(6)
    field = Field(562949948178433, 2, 5)
    x, y = _vars("x", 2), _vars("y", 2)
    A = TableOracle(
        MLE(
            field=field,
            variables=_vars("a", 2),
            evaluations=[rng.randrange(1, 99) for _ in range(4)],
        )
    )
    B = TableOracle(
        MLE(
            field=field,
            variables=_vars("b", 2),
            evaluations=[rng.randrange(1, 99) for _ in range(4)],
        )
    )
    pred = TableOracle(
        SparseMLE(field=field, variables=y + x, evaluations={3: 1, 9: 2, 14: 1}),
        public=True,
    )
    maps = [
        {v: v for v in pred.variables},
        dict(zip(A.variables, x, strict=True)),
        dict(zip(B.variables, y, strict=True)),
    ]
    G = VirtualOracle(x + y, [pred, A, B], [(1, (0, 1, 2)), (1, (0, 1))], maps)
    chall = [field.random_exceptional() for _ in G.variables]
    d_msgs, l_msgs, d_const, l_const = _run_both(G, chall)
    assert all(
        a == b
        for da, la in zip(d_msgs, l_msgs, strict=True)
        for a, b in zip(da, la, strict=True)
    )
    assert d_const == l_const


def _master_shape(rng, s_z, s_x, ring=None, implicit=False):
    """The layer body of `_layer_shape` twice: over its two predicates as
    sparse tables, and over restrictions of one master P over `T || X`, the
    predicates at the slots T = 0 and T = 1 (`implicit`: P an `ImplicitOracle`
    with the master as its witness). Returns (direct, through, witnesses)."""
    z, x, y = _vars("z", s_z), _vars("x", s_x), _vars("y", s_x)
    A = _random_table(rng, _vars("a", s_x), ring)
    B = _random_table(rng, _vars("b", s_x), ring)
    lin = _random_sparse(rng, y + x + z, 6, ring, values=(1, 2))
    xmult = _random_sparse(rng, y + x + z, 6, ring, values=(1,))
    width = 2 * s_x + s_z
    t = _vars("t", 1)
    master = SparseMLE(
        ring=ring,
        variables=t + _vars("p", width),
        evaluations={
            **{(k << 1): v for k, v in lin.polynomial.nonzeros()},
            **{(k << 1) | 1: v for k, v in xmult.polynomial.nonzeros()},
        },
    )
    P = ImplicitOracle(master.variables) if implicit else TableOracle(master)
    r_lin = VirtualOracle.restriction(P, _vars("l", width), {t[0]: 0})
    r_xmult = VirtualOracle.restriction(P, _vars("m", width), {t[0]: 1})
    terms = [(1, (0, 2)), (1, (1, 2, 3))]
    wa, wb = (
        dict(zip(A.variables, x, strict=True)),
        dict(zip(B.variables, y, strict=True)),
    )
    direct = VirtualOracle(
        z + x + y,
        [lin, xmult, A, B],
        terms,
        [{v: v for v in lin.variables}, {v: v for v in xmult.variables}, wa, wb],
    )
    through = VirtualOracle(
        z + x + y,
        [r_lin, r_xmult, A, B],
        terms,
        [
            dict(zip(r_lin.variables, y + x + z, strict=True)),
            dict(zip(r_xmult.variables, y + x + z, strict=True)),
            wa,
            wb,
        ],
    )
    rz = [ring.random_exceptional() if ring else rng.randrange(2, 99) for _ in z]
    bound = dict(zip(z, rz, strict=True))
    witnesses = {P: master} if implicit else {}
    return direct.evaluate(bound), through.evaluate(bound), witnesses


@pytest.mark.parametrize("implicit", [False, True])
def test_restrictions_of_a_master_send_the_messages_of_the_slices(implicit):
    rng = random.Random(8)
    direct, through, witnesses = _master_shape(rng, s_z=2, s_x=2, implicit=implicit)
    chall = [rng.randrange(2, 1 << 16) for _ in direct.variables]
    d_msgs, d_libra, d_const, _ = _run_both(direct, chall)
    t_msgs, t_libra, t_const, t_lconst = _run_both(through, chall, witnesses)
    assert d_msgs == t_msgs == t_libra == d_libra
    assert d_const == t_const == t_lconst
    # The restrictions were kept sparse by the two-phase prover.
    libra = prover_view(through, witnesses, strategy="libra")
    assert isinstance(libra, _LibraProver)
    assert all(
        term.predicate is not None and isinstance(term.predicate.sparse, SparseMLE)
        for term in libra.terms
    )


def test_restrictions_of_a_master_over_a_ring():
    rng = random.Random(9)
    ring = Ring(1024, prime_size=[49], split_degree=4)
    direct, through, _ = _master_shape(rng, s_z=1, s_x=2, ring=ring)
    chall = [ring.random_exceptional() for _ in direct.variables]
    d_msgs, _, d_const, _ = _run_both(direct, chall)
    t_msgs, t_libra, t_const, t_lconst = _run_both(through, chall)
    for a, b, c in zip(d_msgs, t_msgs, t_libra, strict=True):
        assert all(u == v == w for u, v, w in zip(a, b, c, strict=True))
    assert d_const == t_const == t_lconst
