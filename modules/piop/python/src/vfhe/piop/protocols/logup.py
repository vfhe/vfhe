# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The logarithmic-derivative lookup argument [Hab22], with its two
fractional sums proven by GKR on trees of fractions [PH23]: a protocol for
`Relation_Lookup` next to the memory-checking one of `lookup.py`. The prover
sends one oracle per table, the multiplicities of its entries; the trees
are never committed. See `modules/piop/range.md`.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vfhe.arith import MLE, FieldVector, MLE_Variable, histogram
from vfhe.arith.mle import vector_table
from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.messages import OracleMessages
from vfhe.piop.oracle import EqOracle, ImplicitOracle, TableOracle
from vfhe.piop.protocol import Protocol, Rejection, exceptional_set_size
from vfhe.piop.relation import value_of
from vfhe.piop.relations import Relation_Lookup, Relation_Sum
from vfhe.piop.virtual import VirtualOracle, resolve_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vfhe.piop.oracle import Oracle
    from vfhe.piop.party import Party, Prover, Verifier
    from vfhe.piop.relation import Relation, Statement


def multiplicities(values: Sequence[int], beta: int) -> array:
    """`m[x]`, how many of `values` equal `x`, for every `x < beta`, as an
    ``array('Q')``.

    A value outside the table is not counted anywhere: the two sides of the
    lookup identity then differ, so a prover holding such a value produces a
    proof the verifier rejects rather than an exception. An ``array('Q')``
    whose values are all in range is counted natively.
    """
    if isinstance(values, array) and values.typecode == "Q":
        totals = histogram(values, beta)
        if totals is not None:
            return totals
    totals = array("Q", bytes(8 * beta))
    for value in values:
        if 0 <= value < beta:
            totals[value] += 1
    return totals


def _halves(vector: FieldVector) -> tuple[FieldVector, FieldVector]:
    """The two halves of a vector of even length: views when the half is a
    whole number of padding units, copies otherwise."""
    half = len(vector) // 2
    if half % vector.padding_unit:
        return vector.query(range(half)), vector.query(range(half, 2 * half))
    return vector.view(0, half), vector.view(half)


def fraction_tree(
    numerators: FieldVector | None, denominators: FieldVector
) -> list[tuple[FieldVector | None, FieldVector]]:
    """The layers of the tree of fractions over `2^n` leaves, root first.

    Layer `n` is the leaves `(numerators[i], denominators[i])`, and entry
    `i` of layer `k < n` is the projective sum of entries `i` and `i + 2^k`
    of layer `k + 1`: `(a, h) + (a', h') = (a h' + a' h, h h')`. In the
    least-significant-bit-first order of a table, the children of a point
    `x` of `{0,1}^k` are `(x, 0)` and `(x, 1)`. The root, layer 0, is
    `(sum_i a_i prod_(j != i) h_j, prod_i h_i)`, which represents
    `sum_i a_i / h_i` when no denominator vanishes. `numerators` None stands
    for all ones, and stays None on the leaves.
    """
    layers: list[tuple[FieldVector | None, FieldVector]] = [(numerators, denominators)]
    a, h = numerators, denominators
    while len(h) > 1:
        h0, h1 = _halves(h)
        if a is None:
            a = h0 + h1
        else:
            a0, a1 = _halves(a)
            a = (a0 * h1).fma(a1, h0)
        h = h0 * h1
        layers.append((a, h))
    layers.reverse()
    return layers


@dataclass(eq=False)
class _Tree:
    """One fractional sum of the run.

    On the `"oracle"` side the leaves are `(1, xi - f)` over the oracle `f`
    of a group of claims; on the `"table"` side they are `(m, xi - t)` over
    the multiplicities `m` and the table `t = (0, ..., beta - 1)`. `oracle`
    is `f` or `m`; `layers` is the prover's `fraction_tree`, each layer
    dropped once its values have been sent; `claim` the pair `(y, y')` on
    the layer reached, at the point the run shares; `children` the prover's
    values of the next layer at the last sumcheck's point, as that sumcheck
    left them.
    """

    side: str
    beta: int
    depth: int
    oracle: Any
    layers: list | None = None
    claim: tuple | None = None
    children: tuple | None = None

    def sends(self, k: int) -> tuple[bool, bool]:
        """Whether the numerators and the denominators of layer `k` are
        sent: on the leaves the oracle side's numerators are ones and the
        table side's denominators are `xi - t`, which the verifier knows."""
        if k < self.depth:
            return True, True
        return self.side == "table", self.side == "oracle"


class LogUpGKR(Protocol):
    """`Relation_Lookup -> Relation_Eval`: the lookup argument of [Hab22]
    with its fractional sums proven by GKR [PH23].

    Over a field of characteristic above the number of looked-up entries,
    every entry of `f` (over `n` variables) lies in `t_beta` exactly when,
    for some `m`,

        sum_(x in {0,1}^n) 1 / (X - f(x)) = sum_(y < beta) m(y) / (X - y),

    `m(y)` then being the number of entries equal to `y` (the set-inclusion
    lemma of [Hab22]). The prover sends `m`, the verifier draws `xi`
    outside `t_beta`, and the two sides at `X = xi` are the roots of two
    trees of fractions (`fraction_tree`): `(1, xi - f)` over `2^n` leaves
    and `(m, xi - t)` over `beta`. The two trees have their own depths, and
    the verifier compares the roots as fractions, `a_f h_t == a_t h_f` with
    `h_f != 0`.

    Each root is proven by GKR from the root down [PH23, §3]: a claim
    `(y, y')` on layer `k` at the point `rho` is, by the layer rule,

        y + lambda y' = sum_(x in {0,1}^k) eq~(rho, x) [a(x,0) h(x,1)
                        + a(x,1) h(x,0) + lambda h(x,0) h(x,1)]

    over layer `k + 1`, a sum of degree 3 in each variable, proven by the
    registered `Relation_Sum` protocol. The prover then sends the four
    values `a(r,0), a(r,1), h(r,0), h(r,1)` at the sumcheck's point `r`,
    the verifier checks them against the sumcheck's final claim, draws
    `psi` and takes the line through them to the claim on layer `k + 1` at
    `(r, psi)`. On the leaves, that claim is one evaluation claim on `f`
    (`f = xi - h`) or on `m` (`m = a`); the verifier evaluates the table's
    `t~(y) = sum_k 2^k y_k` itself.

    The driver hands the protocol every frontier claim at once, and one run
    proves all of them. Claims into the same `t_beta` share one `m`, the
    sum of their counts, and one table-side tree; among them, claims on
    oracles with the same number of variables are concatenated into one
    oracle-side tree, as in `Lookup` (`s = ceil(log2 k)` selector variables
    after the data ones, zero blocks as padding, counted in `m`), and the
    verifier adds the roots of the oracle-side trees of one table before
    comparing. Every tree of the run, whatever its table and depth, shares
    `xi` and the points: layer `k` of all the trees deeper than `k` is one
    sumcheck, the trees' claims combined by independent challenges. The
    run has `max depth * (max depth - 1) / 2` sumcheck rounds.
    """

    name = "logup"
    reduces_from: type[Relation] = Relation_Lookup
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    batching = True

    def __init__(self, field: Any, messages: OracleMessages | None = None) -> None:
        self.field = field
        # How the multiplicities are sent: their entries (the ideal model) by
        # default.
        self.messages = OracleMessages() if messages is None else messages

    def parameters(self) -> tuple:
        return (self.field, self.messages)

    def soundness_error(self, statement: Statement, domain: Any = None) -> float | None:
        """`(2^(n + 1) + beta) / (|A| - beta)`, twice the entries of a claim
        on an oracle over `n` variables, with `|A|` the size of the
        exceptional set the challenges are drawn from.

        The draw of `xi` is where a false claim can pass: the difference of
        its two sides, a rational function, has a numerator of degree below
        the entries looked up plus `beta`, and `xi` is drawn among the
        `|A| - beta` elements outside the table. Claims into the same table
        count together: a group of `k` claims on oracles over `n` variables
        is one run over `2^(n + s)` entries, `s = ceil(log2 k)`, with error
        `(2^(n + s) + beta) / (|A| - beta)`. A statement does not determine
        its group, so each claim reports `2^(n + 1)`: since `2^s < 2k`, the
        errors of a group's claims add up to at least the group's. The rest
        of the run is below it: `3 / |A|` per sumcheck round, and `3 / |A|`
        for each step from a layer to the next (`psi`, `lambda` and the
        challenges combining the trees). The bound needs a field of
        characteristic above the number of looked-up entries of the table.
        """
        (f,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        size = exceptional_set_size(self.field)
        if size is None:
            return None
        return ((2 << f.num_vars) + beta) / (size - beta)

    # --- the two halves ----------------------------------------------------

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        return await self._run(prover, statements, True)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        return await self._run(verifier, statements, False)

    async def _run(
        self, party: Party, statements: list[Statement], proving: bool
    ) -> list[Statement]:
        """Both halves, which draw the same challenges in the same order and
        emit the same claims: `proving` decides only who computes a message
        and who reads it."""
        iop = party.iop
        field = self.field
        label = f"{self.name}{statements[0].path}"

        # Round 1: one multiplicity table per beta; then xi.
        trees: list[_Tree] = []
        leaves: dict[int, Any] = {}
        tables = self._groups(statements)
        for beta, groups in tables.items():
            looked_up = sum(
                1 << (n + (len(g) - 1).bit_length()) for n, g in groups.items()
            )
            if field.prime <= looked_up:
                raise ValueError(
                    f"the field's characteristic must exceed the {looked_up} "
                    f"entries looked up in t_{beta}, padding included"
                )
        for beta, groups in tables.items():
            b = beta.bit_length() - 1
            oracle_side = [
                _Tree(
                    "oracle", beta, n + (len(g) - 1).bit_length(), self._group_oracle(g)
                )
                for n, g in groups.items()
            ]
            variables = self._names("m", b)
            if proving:
                vectors = [self._group_entries(party, g) for g in groups.values()]
                m_entries = self._multiplicities(vectors, beta)
                m = self.messages.send(
                    cast("Prover", party),
                    f"{label}/m{b}",
                    MLE(field=field, variables=variables, evaluations=m_entries),
                )
                for tree, vector in zip(oracle_side, vectors, strict=True):
                    leaves[id(tree)] = vector
            else:
                m = await self.messages.receive(
                    cast("Verifier", party), f"{label}/m{b}", variables, field
                )
                m_entries = None
            table = _Tree("table", beta, b, m)
            leaves[id(table)] = m_entries
            trees += [*oracle_side, table]
        xi = await self._draw_xi(iop, label, max(tables))
        if proving:
            for tree in trees:
                if tree.side == "oracle":
                    tree.layers = fraction_tree(None, leaves[id(tree)].rsub(xi))
                else:
                    tree.layers = fraction_tree(
                        leaves[id(tree)], self._table_entries(tree.beta).rsub(xi)
                    )
        del leaves

        # A tree of depth 0 is its own leaf: its root is that leaf's pair.
        outputs: list[Statement] = []
        single_leaf = [t for t in trees if t.depth == 0]
        values = await self._child_values(
            iop, proving, f"{label}/leaf", single_leaf, 0, [], xi, leaf=True
        )
        for tree, (a, h) in zip(single_leaf, values, strict=True):
            tree.claim = (a, h)
            outputs.append(self._leaf_claim(statements, tree, [], xi))

        # Layer 0: the children of every other root; then layer by layer.
        active = [t for t in trees if t.depth > 0]
        if not active and not proving:
            # Every root is a leaf: the identity is checked here, not on layer 0.
            self._check_roots(label, trees)
        point: list = []  # the sumcheck's point r on layer k (none on layer 0)
        lam: Any = None
        combination: list = []
        eq: Any = None
        final: Any = None
        k = 0
        while active:
            values = await self._child_values(
                iop, proving, f"{label}/v{k}", active, k, point, xi
            )
            # The verifier's checks; the prover goes on whatever they say, so
            # that a false claim gives a proof the verifier rejects.
            if k == 0:
                for tree, (a0, a1, h0, h1) in zip(active, values, strict=True):
                    tree.claim = (a0 * h1 + a1 * h0, h0 * h1)
                if not proving:
                    self._check_roots(label, trees)
            elif not proving:
                expected = None
                for c, (a0, a1, h0, h1) in zip(combination, values, strict=True):
                    term = c * (a0 * h1 + a1 * h0 + lam * h0 * h1)
                    expected = term if expected is None else expected + term
                if not (eq * expected == final):
                    raise Rejection(
                        f"{label}/v{k}: the children's values do not give the "
                        "layer sumcheck's final claim"
                    )
            psi = await iop.verifier.challenge(f"{label}/psi{k}", field)
            point = [*point, psi]
            for tree, (a0, a1, h0, h1) in zip(active, values, strict=True):
                tree.claim = (a0 + psi * (a1 - a0), h0 + psi * (h1 - h0))
                if tree.depth == k + 1:
                    outputs.append(self._leaf_claim(statements, tree, point, xi))
            active = [t for t in active if t.depth > k + 1]
            if not active:
                break
            k += 1
            lam = await iop.verifier.challenge(f"{label}/lambda{k}", field)
            combination = [field.one] + [
                await iop.verifier.challenge(f"{label}/alpha{k}_{j}", field)
                for j in range(1, len(active))
            ]
            eq, final, point = await self._layer_sumcheck(
                party, proving, statements, k, point, active, lam, combination
            )
        return outputs

    # --- the steps ---------------------------------------------------------

    def _groups(self, statements: list[Statement]) -> dict[int, dict[int, list]]:
        """`beta -> number of variables -> claims`, both in order of first
        appearance and each group in the order given (both parties receive
        the bundle in path order)."""
        groups: dict[int, dict[int, list]] = {}
        for statement in statements:
            beta = cast("Relation_Lookup", statement.relation).beta
            n = statement.oracles[0].num_vars
            groups.setdefault(beta, {}).setdefault(n, []).append(statement)
        return groups

    def _names(self, prefix: str, count: int) -> list:
        return [MLE_Variable(f"{prefix}{i}") for i in range(count)]

    def _table_entries(self, beta: int) -> FieldVector:
        """The table `(0, ..., beta - 1)` as a field vector."""
        vector = FieldVector(self.field, beta)
        vector.set_uint64(array("Q", range(beta)))
        return vector

    def _group_oracle(self, group: list[Statement]) -> Oracle:
        """The oracle a group's tree reads: the claim's own for a group of
        one, otherwise `sum_i sel_i * f_i` over the data variables then
        `s = ceil(log2 k)` selector variables, `sel_i` the public `eq~`
        table of block `i`, so that the blocks past `k` are zero."""
        oracles = [s.oracles[0] for s in group]
        if len(oracles) == 1:
            return oracles[0]
        field = self.field
        n = oracles[0].num_vars
        s = (len(oracles) - 1).bit_length()
        W = self._names("w", n + s)
        constituents: list = []
        maps: list[dict] = []
        terms: list[tuple[Any, tuple[int, ...]]] = []
        for i, f in enumerate(oracles):
            selector = TableOracle(
                MLE.eq(
                    field,
                    [field.one if i >> j & 1 else field.zero for j in range(s)],
                    variables=self._names("z", s),
                ),
                public=True,
            )
            constituents += [selector, f]
            maps += [
                {v: W[n + j] for j, v in enumerate(selector.variables)},
                {v: W[j] for j, v in enumerate(f.variables)},
            ]
            terms.append((field.one, (2 * i, 2 * i + 1)))
        return VirtualOracle(W, constituents, terms, maps)

    def _group_entries(self, prover: Any, group: list[Statement]) -> FieldVector:
        """The prover's entries of `_group_oracle(group)`: the claims' tables end
        to end, padded with zero blocks to a power of two."""
        vectors = []
        for statement in group:
            table = resolve_table(statement.oracles[0], prover.witnesses)
            vectors.append(
                table.table
                if vector_table(table)
                else FieldVector(self.field, list(table.table))
            )
        if len(vectors) == 1:
            return vectors[0]
        blocks = 1 << (len(vectors) - 1).bit_length()
        if blocks > len(vectors):
            size = len(vectors[0])
            vectors.append(FieldVector(self.field, (blocks - len(vectors)) * size))
        return FieldVector.concat(vectors)

    def _multiplicities(self, vectors: list[FieldVector], beta: int) -> FieldVector:
        """The multiplicities of `t_beta` over all the vectors, as a field
        vector. `to_uint64` is None when an entry is `2^64` or more: such an
        entry is outside the table and is not counted."""
        total = None
        for vector in vectors:
            values = vector.to_uint64()
            if values is None:
                values = [int(e) for e in vector]
            counts = multiplicities(values, beta)
            if total is None:
                total = counts
            else:
                for y in range(beta):
                    total[y] += counts[y]
        out = FieldVector(self.field, beta)
        if total is not None:
            out.set_uint64(total)
        return out

    async def _draw_xi(self, iop: Any, label: str, beta: int) -> Any:
        """`xi`, uniform outside `t_beta` for the largest table: drawn again,
        under a new label, while it lands in the table. Over an extension
        field, an element outside the prime subfield is outside the table."""
        attempt = 0
        while True:
            suffix = f"_{attempt}" if attempt else ""
            xi = await iop.verifier.challenge(f"{label}/xi{suffix}", self.field)
            try:
                value = int(xi)
            except ValueError:  # not in the prime subfield
                return xi
            if not (value < beta and xi == self.field(value)):
                return xi
            attempt += 1

    async def _child_values(
        self,
        iop: Any,
        proving: bool,
        label: str,
        trees: list[_Tree],
        k: int,
        point: list,
        xi: Any,
        leaf: bool = False,
    ) -> list[tuple]:
        """The values of every tree's layer `k + 1` at `(point, 0)` and
        `(point, 1)`, as `(a0, a1, h0, h1)`, or with `leaf` the pair `(a,
        h)` of a depth-0 tree's single leaf. The prover sends those the
        verifier does not know (`_Tree.sends`); both fill in the rest."""
        layer = k if leaf else k + 1
        if proving:
            sent = []
            for tree in trees:
                assert tree.layers is not None
                a, h = tree.layers[layer]
                send_a, send_h = tree.sends(layer)
                if leaf:
                    sent += ([a[0]] if send_a else []) + ([h[0]] if send_h else [])
                elif tree.children is not None:
                    sent += tree.children
                    tree.children = None
                else:
                    sent += (self._pair_at(a, point) if send_a else ()) + (
                        self._pair_at(h, point) if send_h else ()
                    )
                tree.layers[layer] = None  # read once
            iop.transcript.write(label, tuple(sent))
        else:
            sent = list(await iop.transcript.read(label))
        expected = sum((1 if leaf else 2) * sum(tree.sends(layer)) for tree in trees)
        if len(sent) != expected:
            raise Rejection(f"{label}: expected {expected} values, got {len(sent)}")
        out = []
        for tree in trees:
            send_a, send_h = tree.sends(layer)
            count = 1 if leaf else 2
            if send_a:
                a, sent = tuple(sent[:count]), sent[count:]
            else:
                a = (self.field.one,) * count
            if send_h:
                h, sent = tuple(sent[:count]), sent[count:]
            else:
                h = tuple(xi - self._table_value(point, bit) for bit in range(count))
            out.append(a + h)
        return out

    def _table_value(self, point: list, bit: int) -> Any:
        """`t~(point, bit)` for the table `t~(y) = sum_j 2^j y_j`; `bit` is
        the last coordinate, absent for a depth-0 table."""
        field = self.field
        value = field.zero
        for j, r in enumerate(point):
            value = value + field(1 << j) * r
        return value + field(bit << len(point)) if bit else value

    def _pair_at(self, vector: FieldVector, point: list) -> tuple:
        """`(v(point, 0), v(point, 1))` for the table `v` over
        `len(point) + 1` variables: on layer 0, and after a `Relation_Sum`
        protocol that leaves no values of its constituents."""
        if not point:
            return (vector[0], vector[1])
        variables = self._names("e", len(point) + 1)
        table = MLE(field=self.field, variables=variables, evaluations=vector)
        folded = table.evaluate(
            dict(zip(variables[:-1], point, strict=True)), in_place=False
        )
        return (folded.table[0], folded.table[1])

    def _check_roots(self, label: str, trees: list[_Tree]) -> None:
        """For every table, the sum of its oracle-side roots equals its
        table-side root as a fraction, with a nonzero denominator."""
        for beta in dict.fromkeys(t.beta for t in trees):
            total = None
            table = None
            for tree in trees:
                if tree.beta != beta:
                    continue
                assert tree.claim is not None
                if tree.side == "table":
                    table = tree.claim
                elif total is None:
                    total = tree.claim
                else:
                    (a, h), (a2, h2) = total, tree.claim
                    total = (a * h2 + a2 * h, h * h2)
            assert total is not None and table is not None
            (a_f, h_f), (a_t, h_t) = total, table
            if h_f == self.field.zero:
                raise Rejection(f"{label}: an entry of the looked-up oracles equals xi")
            if not (a_f * h_t == a_t * h_f):
                raise Rejection(
                    f"{label}: the two sides of the lookup identity differ for "
                    f"t_{beta}, so an entry is outside the table"
                )

    def _leaf_claim(
        self, statements: list[Statement], tree: _Tree, point: list, xi: Any
    ) -> Statement:
        """The claim a tree's leaves leave: `f(point) = xi - h` on the oracle
        side, `m(point) = a` on the table side."""
        assert tree.claim is not None
        a, h = tree.claim
        oracle = tree.oracle
        value = xi - h if tree.side == "oracle" else a
        return self.reduce(
            statements,
            Relation_Eval,
            oracles=[oracle],
            point=dict(zip(oracle.variables, point, strict=True)),
            value=value,
        )

    async def _layer_sumcheck(
        self,
        party: Party,
        proving: bool,
        statements: list[Statement],
        k: int,
        point: list,
        trees: list[_Tree],
        lam: Any,
        combination: list,
    ) -> tuple[Any, Any, list]:
        """The sumcheck of layer `k` over the trees deeper than `k`: the
        claim `sum_j c_j (y_j + lambda y'_j)` on the sum over `{0,1}^k` of
        `eq~(point, x) sum_j c_j [a_j(x,0) h_j(x,1) + a_j(x,1) h_j(x,0) +
        lambda h_j(x,0) h_j(x,1)]`, a single virtual oracle whose
        constituents are the halves of every tree's layer `k + 1`, proven by
        the registered `Relation_Sum` protocol. Returns `eq~(point, r)`, the
        sumcheck's final claim and its point `r`."""
        iop = party.iop
        field = self.field
        X = self._names(f"x{k}_", k)
        eq = EqOracle(field, point, variables=X)
        constituents: list = [eq]
        terms: list[tuple[Any, tuple[int, ...]]] = []
        witnesses: list = []
        sent_indices: list[list[int]] = []  # per tree, the constituents it sends
        value = None
        for c, tree in zip(combination, trees, strict=True):
            assert tree.claim is not None
            y, y2 = tree.claim
            claim = c * (y + lam * y2)
            value = claim if value is None else value + claim
            send_a, send_h = tree.sends(k + 1)
            halves: dict[str, int] = {}
            names = ("a0", "a1", "h0", "h1") if send_a else ("h0", "h1")
            for name in names:
                oracle = ImplicitOracle(X, field)
                constituents.append(oracle)
                halves[name] = len(constituents) - 1
                witnesses.append(oracle)
            sent_indices.append(
                [halves[n] for n in names if (send_a if n[0] == "a" else send_h)]
            )
            if proving:
                assert tree.layers is not None
                a, h = tree.layers[k + 1]
                vectors = (*_halves(a), *_halves(h)) if send_a else _halves(h)
                prover = cast("Prover", party)
                for name, vector in zip(names, vectors, strict=True):
                    prover.witnesses[constituents[halves[name]]] = MLE(
                        field=field, variables=X, evaluations=vector
                    )
            h0, h1 = halves["h0"], halves["h1"]
            if send_a:
                terms += [(c, (0, halves["a0"], h1)), (c, (0, halves["a1"], h0))]
            else:
                terms += [(c, (0, h1)), (c, (0, h0))]
            terms.append((c * lam, (0, h0, h1)))
        maps = [{v: v for v in X} for _ in constituents]
        summand = VirtualOracle(X, constituents, terms, maps)
        statement = self.reduce(
            statements, Relation_Sum, oracles=[summand], value=value
        )
        protocol = iop.protocol_for(statement)
        if protocol is None:
            raise RuntimeError("logup needs a protocol registered for Relation_Sum")
        half = protocol.prove if proving else protocol.verify
        out = await half(party, [statement])  # pyright: ignore[reportArgumentType]
        if len(out) != 1 or not isinstance(out[0].relation, Relation_Eval):
            raise RuntimeError(
                "logup needs the Relation_Sum protocol to end on one evaluation claim"
            )
        (claim,) = out
        if proving:
            # The values the children are sent as: the constituents at r, as
            # the sumcheck prover's tables hold them after its last round.
            prover = cast("Prover", party)
            for oracle in witnesses:
                del prover.witnesses[oracle]
            final = prover.witnesses.pop(claim, {})
            for tree, indices in zip(trees, sent_indices, strict=True):
                if all(j in final for j in indices):
                    tree.children = tuple(final[j] for j in indices)
        r = [claim.point[v] for v in X]
        return eq.query(dict(claim.point)), value_of(claim.value), r
