# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The lookup argument: Figure 11 of [CCCFGS26], Spartan-style offline
memory checking [Set20] reduced to four grand products [SL20], each of
which becomes a zerocheck over a virtual oracle. See
`modules/piop/range.md`.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any, cast

from vfhe.arith import MLE, FieldVector, MLE_Variable, count_occurrences
from vfhe.arith.mle import vector_table
from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.messages import OracleMessages
from vfhe.piop.oracle import EqOracle, TableOracle
from vfhe.piop.protocol import Protocol, Rejection
from vfhe.piop.relations import Relation_Lookup, Relation_Sum
from vfhe.piop.virtual import VirtualOracle, resolve_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vfhe.piop.oracle import Oracle
    from vfhe.piop.party import Prover, Verifier
    from vfhe.piop.relation import Relation, Statement


#: The four multiset fingerprints of the lookup PIOP: the write set and read
#: set on the oracle's side, then the write set and final counts on the
#: table's side.
FINGERPRINTS = ("WSmu", "RS", "WSt", "S")

# Which side of the argument each fingerprint is built over.
_SIDE = {"WSmu": "mu", "RS": "mu", "WSt": "t", "S": "t"}


def counters(values: Sequence[int], beta: int) -> tuple[Sequence[int], Sequence[int]]:
    """The `read_ts` and `final_cts` of Lemma I.3, in one pass.

    `read_ts[i]` is how many times `values[i]` has already occurred strictly
    before position `i`, and `final_cts[x]` the total count of `x`. Together
    they make the multiset identity `WS = RS u S` of the lemma hold exactly
    when every value is in the table.

    Out-of-range values are counted too, but have no `final_cts` slot to be
    counted *into* -- so the identity necessarily fails for them, which is
    the point: a prover holding an out-of-range value cannot produce counters
    that balance, whatever it does. Tolerating them here (rather than
    raising) is what lets a false statement produce a well-formed proof that
    the verifier then rejects, instead of an exception from the prover.

    Values given as an ``array('Q')`` and all in range are counted natively
    (`count_occurrences`), and the counters come back as arrays.
    """
    if isinstance(values, array) and values.typecode == "Q":
        counts = count_occurrences(values, beta)
        if counts is not None:
            return counts
    seen: dict[int, int] = {}
    read_ts, final_cts = [], [0] * beta
    for value in values:
        count = seen.get(value, 0)
        read_ts.append(count)
        seen[value] = count + 1
        if 0 <= value < beta:
            final_cts[value] = count + 1
    return read_ts, final_cts


def product_tree(leaves: list, zero: Any) -> list:
    """The product tree of `2^n` leaves: the table of the layered polynomial
    `f` of Lemma I.4, `2^(n+1)` entries read LSB-first.

    `f[k] = leaves[k]` below `2^n`, `f[2^n + k] = f[2k] * f[2k+1]` above it,
    and the very last entry pinned to zero. The pin is not decoration: the
    recurrence at `k = 2^n - 1` would be self-referential, and the zerocheck
    `f(X,1) - f(0,X) * f(1,X)` is identically zero over the *whole* cube only
    because both sides vanish there. The grand product itself lands at
    `f[2^(n+1) - 2]` -- the point `(0, 1, ..., 1)`.
    """
    size = len(leaves)
    table = list(leaves) + [zero] * size
    for k in range(size - 1):
        table[size + k] = table[2 * k] * table[2 * k + 1]
    return table


def product_tree_vector(leaves: FieldVector) -> FieldVector:
    """`product_tree` over a field vector of `2^n` leaves, in `n`
    whole-vector passes: layer `t` is the elementwise product of the even and
    odd entries of layer `t - 1`, and the table is the leaves, then layers
    `1..n`, then the pinned zero."""
    layers, layer = [leaves], leaves
    while len(layer) > 1:
        even, odd = layer.split_even_odd()
        layer = even * odd
        layers.append(layer)
    layers.append(FieldVector(leaves.field, 1))
    return FieldVector.concat(layers)


class Lookup(Protocol):
    """`Relation_Lookup -> Relation_Sum^2 x Relation_Eval^4` (Figure 11).

    Round 1: the prover sends the two counter polynomials of Lemma I.3, and
    the verifier replies with the fingerprint challenges `sigma_L`, `tau`.
    (The paper calls the first one `sigma`; §I already spends that letter on
    the alphabet size of the decomposition, so it is `zeta` in this code.)

    Round 2: the prover sends the internal nodes `f^(1)_a` of the four grand
    products -- one per fingerprint -- together with the four products
    themselves, and the verifier replies with `chi` and the two zerocheck
    points `rho`, `xi`.

    It then checks `WSt * WSmu == RS * S`, which is the multiset identity of
    Lemma I.3 in fingerprinted form, and emits two sum claims: one per side
    of the cube, each asserting that its pair of grand products was built
    correctly. Each is a zerocheck over the function `g` of Figure 12,

        sum_X eq~(point, X) * [ (f^(1)_a(X) - f_a(0,X) * f_a(1,X))
                          + chi * (f^(1)_b(X) - f_b(0,X) * f_b(1,X)) ] == 0

    expressed as a single `VirtualOracle` of degree 3. The layered `f_a` is
    itself virtual -- `(1 - W_n) * f^(0)_a + W_n * f^(1)_a`, with `f^(0)_a`
    spelled out over the oracles the verifier already has -- so the sumcheck
    ends on claims about `mu`, `read_ts`, `final_cts` and the `f^(1)_a`, and
    never on anything the verifier has to take on trust.
    """

    name = "lookup"
    reduces_from: type[Relation] = Relation_Lookup
    reduces_to: tuple[type[Relation], ...] = (Relation_Sum, Relation_Eval)

    def __init__(self, field: Any, messages: OracleMessages | None = None) -> None:
        self.field = field
        # How the counters and the f^(1)_a are sent: their entries (the ideal
        # model) by default.
        self.messages = OracleMessages() if messages is None else messages

    def soundness_error(self, statement: Statement, domain: Any = None) -> float:
        """`(2^ell* + 2^b*) / p` (Theorem I.5).

        The degree of the polynomial `P(Z, Z')` whose vanishing the
        fingerprint challenges test, over the field they are drawn from.
        """
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        return ((1 << mu.num_vars) + beta) / self.field.prime

    # --- the scaffolding both halves build identically ---------------------

    def _names(self, prefix: str, count: int) -> list:
        return [MLE_Variable(f"{prefix}{i}") for i in range(count)]

    def _table(self, b: int) -> TableOracle:
        """`t_beta` as a public oracle: the verifier holds it, so an
        evaluation claim on it never leaves the verifier's own hands."""
        return TableOracle(
            MLE(
                field=self.field,
                variables=self._names("t", b),
                evaluations=[self.field(j) for j in range(1 << b)],
            ),
            public=True,
        )

    def _layered(self, W: list, f1: Oracle, parts: list) -> VirtualOracle:
        """`f_a = (1 - W_n) * sum(parts) + W_n * f^(1)_a`, as a virtual
        oracle over `W`.

        `parts` is the affine form of the leaves: `(coefficient, oracle)`
        pairs, with `oracle=None` for a constant. A constant needs the
        indicator anyway -- a term must index at least one constituent, and
        the constant is only there on the `W_n = 0` half -- so riding it on
        `1 - W_n` is the correct algebra.
        """
        last = W[-1]
        low = MLE.eq(self.field, [self.field.zero], variables=[last])  # 1 - W_n
        high = MLE.eq(self.field, [self.field.one], variables=[last])  # W_n
        constituents = [
            TableOracle(low, public=True),
            TableOracle(high, public=True),
            f1,
        ]
        maps: list[dict] = [
            {last: last},
            {last: last},
            {v: W[i] for i, v in enumerate(f1.variables)},
        ]
        terms: list[tuple[Any, tuple[int, ...]]] = [(self.field.one, (1, 2))]
        for coefficient, oracle in parts:
            if oracle is None:
                terms.append((coefficient, (0,)))
                continue
            constituents.append(oracle)
            maps.append({v: W[i] for i, v in enumerate(oracle.variables)})
            terms.append((coefficient, (0, len(constituents) - 1)))
        return VirtualOracle(W, constituents, terms, maps)

    def _summand(self, X: list, point: list, chi: Any, pairs: list) -> VirtualOracle:
        """The function `g` of Figure 12 for one pair of fingerprints.

        `pairs` is `[(f^(1)_a, f_a), (f^(1)_b, f_b)]`. The parent slot takes
        the real oracle `f^(1)_a` rather than a view of `f_a`, because
        `f_a(X, 1)` *is* `f^(1)_a(X)` -- and because `claims()` is structural,
        so routing it through `f_a` would emit claims on `mu` and `read_ts`
        that the `(1 - W_n)` factor has already multiplied by zero.
        """
        n = len(X)
        eq = EqOracle(self.field, point, variables=X)
        constituents: list = [eq]
        maps: list[dict] = [{v: v for v in X}]
        terms: list[tuple[Any, tuple[int, ...]]] = []
        for coefficient, (f1, f_alpha) in zip(
            (self.field.one, chi), pairs, strict=True
        ):
            constituents.append(f1)
            maps.append({v: X[i] for i, v in enumerate(f1.variables)})
            parent = len(constituents) - 1
            W = f_alpha.variables
            shifted = {W[i + 1]: X[i] for i in range(n)}
            constituents += [f_alpha, f_alpha]
            maps += [
                {W[0]: self.field.zero} | shifted,  # f_a(0, X)
                {W[0]: self.field.one} | shifted,  # f_a(1, X)
            ]
            child = len(constituents) - 2
            terms.append((coefficient, (0, parent)))
            terms.append((-coefficient, (0, child, child + 1)))
        return VirtualOracle(X, constituents, terms, maps)

    def _scaffold(
        self, statement: Statement, read_ts: Oracle, final_cts: Oracle
    ) -> dict:
        """Everything downstream of the round-1 messages that does not depend
        on a challenge: the public table, and the variable lists the two
        sides are built over. Both parties derive it the same way."""
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        n, b = mu.num_vars, beta.bit_length() - 1
        return {
            "mu": mu,
            "beta": beta,
            "sizes": {"WSmu": n, "RS": n, "WSt": b, "S": b},
            "table": self._table(b),
            "read_ts": read_ts,
            "final_cts": final_cts,
            "X": {"mu": self._names("x", n), "t": self._names("y", b)},
            "W": {"mu": self._names("w", n + 1), "t": self._names("u", b + 1)},
        }

    def _leaves(self, tables: dict, parts: dict, zeta: Any, tau: Any) -> dict:
        """The four fingerprint leaf tables, equations (41)-(43), from the
        prover's tables of `mu`, `read_ts` and `final_cts` and the public
        table."""
        mu, table = tables["mu"].table, parts["table"].polynomial.table
        read_ts, final_cts = tables["read_ts"].table, tables["final_cts"].table
        read = mu.fma(read_ts, zeta)  # mu + zeta * read_ts
        return {
            "WSmu": read + (zeta - tau),
            "RS": read - tau,
            "WSt": table - tau,
            "S": table.fma(final_cts, zeta) - tau,
        }

    def _counter_table(self, variables: list, counts: Sequence[int]) -> MLE:
        """`counts`, small non-negative integers, as a field table."""
        vector = FieldVector(self.field, len(counts))
        vector.set_uint64(counts if isinstance(counts, array) else array("Q", counts))
        return MLE(field=self.field, variables=variables, evaluations=vector)

    def _virtuals(self, parts: dict, f1: dict, zeta: Any, tau: Any) -> dict:
        """The four layered oracles, each over its side's `W`."""
        mu, table = parts["mu"], parts["table"]
        read_ts, final_cts = parts["read_ts"], parts["final_cts"]
        one = self.field.one
        affine = {
            "WSmu": [(one, mu), (zeta, read_ts), (zeta - tau, None)],
            "RS": [(one, mu), (zeta, read_ts), (-tau, None)],
            "WSt": [(one, table), (-tau, None)],
            "S": [(one, table), (zeta, final_cts), (-tau, None)],
        }
        return {
            a: self._layered(parts["W"][_SIDE[a]], f1[a], affine[a])
            for a in FINGERPRINTS
        }

    def _outputs(
        self,
        statement: Statement,
        parts: dict,
        f1: dict,
        virtual: dict,
        chi: Any,
        points: dict,
        products: tuple,
    ) -> list[Statement]:
        """Two zerochecks, then one evaluation claim per `f^(1)_a` at the
        grand product's own point. The order is part of the protocol: child
        paths come from a counter on the parent, so both halves must emit the
        same bundle in the same sequence."""
        out = [
            self.reduce(
                [statement],
                Relation_Sum,
                oracles=[
                    self._summand(
                        parts["X"][side],
                        points[side],
                        chi,
                        [(f1[a], virtual[a]) for a in pair],
                    )
                ],
                value=self.field.zero,
            )
            for side, pair in (("mu", ("WSmu", "RS")), ("t", ("WSt", "S")))
        ]
        for a, product in zip(FINGERPRINTS, products, strict=True):
            variables = f1[a].variables
            point = dict.fromkeys(variables, self.field.one) | {
                variables[0]: self.field.zero
            }
            out.append(
                self.reduce(
                    [statement],
                    Relation_Eval,
                    oracles=[f1[a]],
                    point=point,
                    value=product,
                )
            )
        return out

    # --- the two halves ----------------------------------------------------

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = prover.iop
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        label = f"{self.name}{statement.path}"
        field = self.field

        mu_table = resolve_table(mu, prover.witnesses)
        # `to_uint64` is None when an entry is 2^64 or more: such an entry is
        # outside the table, and `counters` still counts it.
        values = mu_table.table.to_uint64() if vector_table(mu_table) else None
        if values is None:
            values = [int(e) for e in mu_table.table]
        read_counts, final_counts = counters(values, beta)
        tables = {
            "mu": mu_table,
            "read_ts": self._counter_table(self._names("r", mu.num_vars), read_counts),
            "final_cts": self._counter_table(
                self._names("c", beta.bit_length() - 1), final_counts
            ),
        }
        read_ts = self.messages.send(prover, f"{label}/read_ts", tables["read_ts"])
        final_cts = self.messages.send(
            prover, f"{label}/final_cts", tables["final_cts"]
        )
        zeta = await iop.verifier.challenge(f"{label}/zeta", self.field)
        tau = await iop.verifier.challenge(f"{label}/tau", self.field)

        parts = self._scaffold(statement, read_ts, final_cts)
        trees = {
            a: product_tree_vector(leaves)
            for a, leaves in self._leaves(tables, parts, zeta, tau).items()
        }
        f1 = {}
        for a in FINGERPRINTS:
            half = 1 << parts["sizes"][a]
            tree = trees[a]
            # A view starts at a multiple of the padding unit; a tree smaller
            # than that has its internal nodes gathered instead.
            if half % tree.padding_unit:
                internal = tree.query(range(half, 2 * half))
            else:
                internal = tree.view(half).copy()
            f1[a] = self.messages.send(
                prover,
                f"{label}/f1_{a}",
                MLE(
                    field=field,
                    variables=self._names(f"f{a}", parts["sizes"][a]),
                    evaluations=internal,
                ),
            )
        products = tuple(trees[a][-2] for a in FINGERPRINTS)
        iop.transcript.write(f"{label}/products", products)

        chi, points = await self._draw(iop, label, parts)
        virtual = self._virtuals(parts, f1, zeta, tau)
        # The sumcheck prover works on tables; a virtual constituent has none
        # of its own, so the layered oracles are handed over as witnesses.
        for a in FINGERPRINTS:
            prover.witnesses[virtual[a]] = MLE(
                field=field, variables=parts["W"][_SIDE[a]], evaluations=trees[a]
            )
        return self._outputs(statement, parts, f1, virtual, chi, points, products)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        label = f"{self.name}{statement.path}"

        read_ts = await self.messages.receive(
            verifier, f"{label}/read_ts", self._names("r", mu.num_vars), self.field
        )
        final_cts = await self.messages.receive(
            verifier,
            f"{label}/final_cts",
            self._names("c", beta.bit_length() - 1),
            self.field,
        )
        zeta = await iop.verifier.challenge(f"{label}/zeta", self.field)
        tau = await iop.verifier.challenge(f"{label}/tau", self.field)

        parts = self._scaffold(statement, read_ts, final_cts)
        f1 = {
            a: await self.messages.receive(
                verifier,
                f"{label}/f1_{a}",
                self._names(f"f{a}", parts["sizes"][a]),
                self.field,
            )
            for a in FINGERPRINTS
        }
        products = await iop.transcript.read(f"{label}/products")
        if len(products) != len(FINGERPRINTS):
            raise Rejection(f"{label}/products: expected {len(FINGERPRINTS)} values")

        chi, points = await self._draw(iop, label, parts)

        # Lemma I.3 in fingerprinted form: WS = RS u S becomes one product
        # identity, and the two sum claims below are what pins each of the
        # four products to the oracle it was supposed to be built from.
        ws_mu, rs, ws_t, s = products  # in FINGERPRINTS order
        if not (ws_t * ws_mu == rs * s):
            raise Rejection(
                f"{label}: the grand products do not satisfy WSt * WSmu == RS * S, "
                "so the write and read multisets differ"
            )

        virtual = self._virtuals(parts, f1, zeta, tau)
        return self._outputs(statement, parts, f1, virtual, chi, points, products)

    # --- helpers shared by the halves --------------------------------------

    async def _draw(self, iop: Any, label: str, parts: dict) -> tuple:
        """`chi` and the two zerocheck points, in one place so the two halves
        cannot draw them in different orders."""
        chi = await iop.verifier.challenge(f"{label}/chi", self.field)
        points = {
            "mu": [
                await iop.verifier.challenge(f"{label}/rho{i}", self.field)
                for i in range(parts["sizes"]["WSmu"])
            ],
            "t": [
                await iop.verifier.challenge(f"{label}/xi{i}", self.field)
                for i in range(parts["sizes"]["WSt"])
            ],
        }
        return chi, points
