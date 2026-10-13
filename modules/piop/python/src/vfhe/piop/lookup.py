# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The lookup relation: every evaluation of an oracle lies in a small table.

`Relation_Lookup` is the toolbox member of `piop.md` §8 item 4 and the target
of the range proof's decomposition step (`range.py`): once the digits of a
bounded ring element are on the table, all that is left to prove is that each
digit is a member of `t_beta = {0, ..., beta - 1}`.

Left terminal, the claim is decided by the ideal decider below, which reads
every hypercube evaluation. `Lookup` is the succinct discharge: Figure 11 of
[CCCFGS26], Spartan-style offline memory checking [Set20] reduced to four
grand products [SL20], each of which becomes a zerocheck over a virtual
oracle. See `modules/piop/range.md`.
"""

from __future__ import annotations

from typing import Any, cast

from vfhe.arith import MLE, MLE_Variable
from vfhe.arith.mle import native_table, vector_table

from .piop import (
    Protocol,
    Prover,
    Rejection,
    Relation,
    Relation_Eval,
    Relation_Sum,
    Statement,
    Verifier,
    _constant,
    _hypercube,
)
from .virtual import VirtualOracle

#: The four multiset fingerprints of the lookup PIOP: the write set and read set on the oracle's side, then the
#: write set and final counts on the table's side.
FINGERPRINTS = ("WSmu", "RS", "WSt", "S")


class Relation_Lookup(Relation):
    """`f(b) in t_beta` for every `b` in `{0,1}^n`, with
    `t_beta = {0, ..., beta - 1}` the relation's index.

    The table is an index rather than an instance field because it is the
    large, reusable, preprocessable part of the claim: one `beta` serves every
    lookup of a given digit width, and it is public data both parties hold
    (`piop.md` §3).
    """

    name = "lookup"
    fields = ("oracles",)

    def __init__(self, beta: int) -> None:
        if beta <= 0 or beta & (beta - 1):
            raise ValueError(f"beta must be a positive power of two, got {beta}")
        super().__init__(index=beta)

    @property
    def beta(self) -> int:
        """The table size; `t_beta` is `{0, ..., beta - 1}`.

        `Relation.index` is deliberately untyped in the base class -- it holds
        whatever a relation needs -- so a subclass narrows it where it reads
        it, the way `Relation_Circuit.circuit` does (`circuit.py`). The
        constructor is what guarantees the cast is honest.
        """
        return cast("int", self.index)

    def check(self, statement: Statement) -> bool:
        """The ideal decider: every hypercube evaluation is in range.

        A dense table in the evaluation basis already *is* the list of those
        evaluations, so it is read directly; anything else (a coefficient
        basis, a defined oracle) is enumerated the slow way.
        """
        (f,) = statement.oracles
        if native_table(f) or vector_table(f):
            return all(self._in_range(e) for e in f.table)
        return all(
            self._in_range(_constant(f.evaluate(b, in_place=False)))
            for b in _hypercube(f.variables)
        )

    def _in_range(self, value) -> bool:
        """Whether one coefficient-domain value is a member of `t_beta`.

        `int()` on a field element is its canonical representative in
        `[0, p)`, so a digit that overflowed the table is a large integer
        here, not a small negative one.
        """
        return 0 <= int(value) < self.beta


def counters(values: list[int], beta: int) -> tuple[list[int], list[int]]:
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
    """
    seen: dict[int, int] = {}
    read_ts, final_cts = [], [0] * beta
    for value in values:
        count = seen.get(value, 0)
        read_ts.append(count)
        seen[value] = count + 1
        if 0 <= value < beta:
            final_cts[value] = count + 1
    return read_ts, final_cts


def grand_product(leaves: list, zero: Any) -> list:
    """The layered polynomial of Lemma I.4 over `2^n` leaves, as one flat
    table of `2^(n+1)` entries read LSB-first.

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

    def __init__(self, field: Any) -> None:
        self.field = field

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

    def _table(self, b: int) -> MLE:
        """`t_beta` as a public oracle: the verifier holds it, so an
        evaluation claim on it never leaves the verifier's own hands."""
        return MLE(
            field=self.field,
            variables=self._names("t", b),
            evaluations=[self.field(j) for j in range(1 << b)],
            public=True,
        )

    def _layered(self, W: list, f1: MLE, parts: list) -> VirtualOracle:
        """`f_a = (1 - W_n) * sum(parts) + W_n * f^(1)_a`, as a virtual
        oracle over `W`.

        `parts` is the affine form of the leaves: `(coefficient, oracle)`
        pairs, with `oracle=None` for a constant. A constant needs the
        indicator anyway -- a term must index at least one constituent, and
        the constant is only there on the `W_n = 0` half -- so riding it on
        `1 - W_n` is both the fix and the correct algebra.
        """
        last = W[-1]
        low = MLE.eq(self.field, [self.field.zero], variables=[last])  # 1 - W_n
        high = MLE.eq(self.field, [self.field.one], variables=[last])  # W_n
        constituents = [low, high, f1]
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
        eq = MLE.eq(self.field, point, variables=X)
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

    def _scaffold(self, statement: Statement, read_ts: MLE, final_cts: MLE) -> dict:
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

    def _leaves(self, parts: dict, zeta: Any, tau: Any) -> dict:
        """The four fingerprint leaf tables, equations (41)-(43)."""
        mu, table = parts["mu"], parts["table"]
        read_ts, final_cts = parts["read_ts"], parts["final_cts"]
        return {
            "WSmu": [
                mu.table[i] + zeta * (read_ts.table[i] + self.field.one) - tau
                for i in range(1 << parts["sizes"]["WSmu"])
            ],
            "RS": [
                mu.table[i] + zeta * read_ts.table[i] - tau
                for i in range(1 << parts["sizes"]["RS"])
            ],
            "WSt": [table.table[j] - tau for j in range(1 << parts["sizes"]["WSt"])],
            "S": [
                table.table[j] + zeta * final_cts.table[j] - tau
                for j in range(1 << parts["sizes"]["S"])
            ],
        }

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
        side = {"WSmu": "mu", "RS": "mu", "WSt": "t", "S": "t"}
        return {
            a: self._layered(parts["W"][side[a]], f1[a], affine[a])
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
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        label = f"{self.name}{statement.path}"
        field = self.field

        read_counts, final_counts = counters([int(e) for e in mu.table], beta)
        read_ts = MLE(
            field=field,
            variables=self._names("r", mu.num_vars),
            evaluations=[field(c) for c in read_counts],
        )
        final_cts = MLE(
            field=field,
            variables=self._names("c", beta.bit_length() - 1),
            evaluations=[field(c) for c in final_counts],
        )
        iop.transcript.write(f"{label}/read_ts", tuple(read_ts.table))
        iop.transcript.write(f"{label}/final_cts", tuple(final_cts.table))
        zeta = iop.verifier.challenge(f"{label}/zeta")
        tau = iop.verifier.challenge(f"{label}/tau")

        parts = self._scaffold(statement, read_ts, final_cts)
        trees = {
            a: grand_product(leaves, field.zero)
            for a, leaves in self._leaves(parts, zeta, tau).items()
        }
        side = {"WSmu": "mu", "RS": "mu", "WSt": "t", "S": "t"}
        f1 = {}
        for a in FINGERPRINTS:
            half = 1 << parts["sizes"][a]
            f1[a] = MLE(
                field=field,
                variables=self._names(f"f{a}", parts["sizes"][a]),
                evaluations=trees[a][half:],
            )
            iop.transcript.write(f"{label}/f1_{a}", tuple(f1[a].table))
        products = tuple(trees[a][-2] for a in FINGERPRINTS)
        iop.transcript.write(f"{label}/products", products)

        chi, points = self._draw(iop, label, parts)
        virtual = self._virtuals(parts, f1, zeta, tau)
        # The sumcheck prover works on tables; a virtual constituent has none
        # of its own, so the layered oracles are handed over as witnesses.
        for a in FINGERPRINTS:
            prover.witnesses[virtual[a]] = MLE(
                field=field, variables=parts["W"][side[a]], evaluations=trees[a]
            )
        return self._outputs(statement, parts, f1, virtual, chi, points, products)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        if iop is None:
            raise RuntimeError("this party is not bound to an IOP")
        (mu,) = statement.oracles
        beta = cast("Relation_Lookup", statement.relation).beta
        label = f"{self.name}{statement.path}"

        read_ts = await self._read_table(
            iop, f"{label}/read_ts", self._names("r", mu.num_vars)
        )
        final_cts = await self._read_table(
            iop, f"{label}/final_cts", self._names("c", beta.bit_length() - 1)
        )
        zeta = iop.verifier.challenge(f"{label}/zeta")
        tau = iop.verifier.challenge(f"{label}/tau")

        parts = self._scaffold(statement, read_ts, final_cts)
        f1 = {
            a: await self._read_table(
                iop, f"{label}/f1_{a}", self._names(f"f{a}", parts["sizes"][a])
            )
            for a in FINGERPRINTS
        }
        products = await iop.transcript.read(f"{label}/products")
        if len(products) != len(FINGERPRINTS):
            raise Rejection(f"{label}/products: expected {len(FINGERPRINTS)} values")

        chi, points = self._draw(iop, label, parts)

        # Lemma I.3 in fingerprinted form: WS = RS u S becomes one product
        # identity, and the two sum claims below are what pins each of the
        # four products to the oracle it was supposed to be built from.
        ws_mu, rs, ws_t, s = (products[FINGERPRINTS.index(a)] for a in FINGERPRINTS)
        if not (ws_t * ws_mu == rs * s):
            raise Rejection(
                f"{label}: the grand products do not satisfy WSt * WSmu == RS * S, "
                "so the write and read multisets differ"
            )

        virtual = self._virtuals(parts, f1, zeta, tau)
        return self._outputs(statement, parts, f1, virtual, chi, points, products)

    # --- helpers shared by the halves --------------------------------------

    def _draw(self, iop: Any, label: str, parts: dict) -> tuple:
        """`chi` and the two zerocheck points, in one place so the two halves
        cannot draw them in different orders."""
        chi = iop.verifier.challenge(f"{label}/chi")
        points = {
            "mu": [
                iop.verifier.challenge(f"{label}/rho{i}")
                for i in range(parts["sizes"]["WSmu"])
            ],
            "t": [
                iop.verifier.challenge(f"{label}/xi{i}")
                for i in range(parts["sizes"]["WSt"])
            ],
        }
        return chi, points

    async def _read_table(self, iop: Any, label: str, variables: list) -> MLE:
        """One oracle message read back as a table over `variables`."""
        entries = await iop.transcript.read(label)
        if len(entries) != 1 << len(variables):
            raise Rejection(
                f"{label}: expected {1 << len(variables)} entries, got {len(entries)}"
            )
        return MLE(field=self.field, variables=variables, evaluations=list(entries))
