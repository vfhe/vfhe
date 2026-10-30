# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sumcheck protocols [LFKN92]: Relation_Sum and Relation_SumProd both
reduce to Relation_Eval claims.

Both protocols share the round machinery (`_SumcheckRounds`) and differ
only in their round message and closing step. The round messages are the
kernels of `rounds.py` (where the native/pure-Python decision is made, per
call), the prover's working form of a defined oracle comes from
`prover_view.py`, and the verifier's claim update is `interpolate_evals`.
See piop.md §4-§6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vfhe.arith import Field, Ring
from vfhe.piop.evaluation import Relation_Eval
from vfhe.piop.protocol import Protocol, Rejection, exceptional_set_size
from vfhe.piop.relation import Relation, Statement, value_of
from vfhe.piop.relations import Relation_Sum, Relation_SumProd

from .prover_view import prover_view
from .rounds import (
    interpolate_evals,
    prod_round_evals,
    prod_update_tables,
    round_evals,
)

if TYPE_CHECKING:
    from vfhe.piop.iop import IOP
    from vfhe.piop.party import Prover, Verifier


class _SumcheckRounds(Protocol):
    """The round machinery shared by the sumcheck family.

    Per round i, over the first remaining variable: the prover writes the
    round polynomial's evaluations (`_round_message`) and updates every
    bookkeeping table with the challenge (`_update_tables`; out of place on
    round 0 to keep the shared oracles intact, in place afterwards); the
    verifier checks g_i(0) + g_i(1) against the running claim and
    interpolates g_i(r_i) as the next one. One verifier serves the native
    and pure-Python provers — the wire format is shared, and its per-round
    work is O(1) coefficient operations.

    Subclasses provide `name` (the transcript label prefix), the round
    message, and the closing step of each half (`_prove_tail` /
    `_verify_tail`), and may override the table update.
    """

    name: str = "abstract"
    # Which round-message strategy the prover builds for a defined oracle
    # (`prover_view`): "auto", "dense" or "libra". A class attribute so
    # tests can pin it; both strategies send identical messages.
    strategy: str = "auto"

    def _label(self, statement: Statement) -> str:
        return f"{self.name}{statement.path}"

    def _round_message(self, factors: list, var) -> tuple:
        """The evaluations of this round's polynomial at 0..deg."""
        raise NotImplementedError

    def _update_tables(
        self, factors: list, var, r, in_place: list
    ) -> tuple[list, tuple | None]:
        """The factors' bookkeeping tables after the round over `var`, with
        `var` bound to the challenge `r` [XZZPS19, Alg. 1, step 5], each in
        place where `in_place` says; and the next round's message, or None.

        An update that also forms the next round's message returns it, and
        `prove` sends it instead of calling `_round_message`, so it must be
        the message `_round_message` gives for the updated tables. This one
        updates each factor with `MLE.evaluate` and returns None.
        """
        tables = [
            f.evaluate({var: r}, in_place=own)
            for f, own in zip(factors, in_place, strict=True)
        ]
        return tables, None

    def _prove_tail(
        self, iop: IOP, statement: Statement, label: str, factors: list, point: dict
    ) -> list[Statement]:
        """Close the prover half over the fully-folded tables; returns the
        statements the claim reduces to."""
        raise NotImplementedError

    async def _verify_tail(
        self, iop: IOP, statement: Statement, label: str, point: dict, claim
    ) -> list[Statement]:
        """Close the verifier half against the final claim g_n(r_n);
        returns the statements the claim reduces to."""
        raise NotImplementedError

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = prover.iop
        factors = [
            prover_view(f, prover.witnesses, self.strategy) for f in statement.oracles
        ]
        originals = tuple(factors)
        label = self._label(statement)
        point = {}
        message = None
        for i, var in enumerate(list(factors[0].variables)):
            if message is None:
                message = self._round_message(factors, var)
            iop.transcript.write(f"{label}/g{i}", message)
            r = await iop.verifier.challenge(f"{label}/r{i}")
            point[var] = r
            # Out of place while a factor is still the table of the
            # statement's own oracle; a prover view (`owned`) brings fresh
            # tables of its own.
            in_place = [
                f is not orig or getattr(f, "owned", False)
                for f, orig in zip(factors, originals, strict=True)
            ]
            factors, message = self._update_tables(factors, var, r, in_place)
        return self._prove_tail(iop, statement, label, factors, point)

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        (statement,) = statements
        iop = verifier.iop
        label = self._label(statement)
        claim = value_of(statement.value)
        point = {}
        for i, var in enumerate(list(statement.oracles[0].variables)):
            evals = await iop.transcript.read(f"{label}/g{i}")
            if not (evals[0] + evals[1] == claim):
                raise Rejection(f"{label} round {i}: g(0) + g(1) != claim")
            r = await verifier.challenge(f"{label}/r{i}")
            point[var] = r
            claim = interpolate_evals(evals, r)
        return await self._verify_tail(iop, statement, label, point, claim)


class Sumcheck(_SumcheckRounds):
    """Reduces `sum_{b in {0,1}^n} f(b) == v` to `f(r) == v_n` in n rounds.

    Round i: the prover sends (g_i(0), g_i(1)) — the oracle is a single
    multilinear polynomial, so deg g_i = 1; the verifier checks
    g_i(0) + g_i(1) against the running claim, the challenge r_i is drawn
    from the domain's exceptional set, and the claim becomes g_i(r_i).

    The oracle may also be a virtual oracle in products form: the round
    message is then g_i at 0..degree, computed by the prover view's
    `round_evals` (`prover_view.py`), and the final claim is an evaluation
    claim on the virtual oracle, which `VirtualEval` reduces to its
    constituents.
    """

    name = "sumcheck"
    reduces_from: type[Relation] = Relation_Sum
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring, Field)

    def soundness_error(
        self, statement: Statement, domain, degree: int | None = None
    ) -> float | None:
        """degree * num_vars / |A| (piop.md §6); None if |A| is unknown. The
        degree defaults to the oracle's (`degree` attribute of a defined
        oracle in products form), else 1 (a multilinear table)."""
        size = exceptional_set_size(domain)
        if size is None:
            return None
        if degree is None:
            degree = int(getattr(statement.oracles[0], "degree", 1))
        return degree * statement.num_vars / size

    def _round_message(self, factors: list, var) -> tuple:
        (f,) = factors
        return round_evals(f, var)

    def _update_tables(
        self, factors: list, var, r, in_place: list
    ) -> tuple[list, tuple | None]:
        """A prover view of a virtual oracle updates its own tables
        (`update_tables`), and may form the next round's message in the
        same pass; a table is updated by `MLE.evaluate`."""
        (f,) = factors
        update: Any = getattr(f, "update_tables", None)
        if callable(update):
            f, message = cast("tuple", update(var, r))
            return [f], message
        return super()._update_tables(factors, var, r, in_place)

    def _prove_tail(
        self, _iop: IOP, statement: Statement, _label: str, factors: list, point: dict
    ) -> list[Statement]:
        (cur,) = factors
        return [self.reduce([statement], point=point, value=cur.constant())]

    async def _verify_tail(
        self, _iop: IOP, statement: Statement, _label: str, point: dict, claim
    ) -> list[Statement]:
        return [self.reduce([statement], point=point, value=claim)]


class SumcheckProd(_SumcheckRounds):
    """Libra-style product sumcheck [XZZPS19, Alg. 3]: reduces one
    `sum_{b in {0,1}^n} prod_j f_j(b) == v` claim to k = #factors
    `Relation_Eval` claims, all at the same random point r.

    Round i: each factor is linear in x_i, so the round polynomial
    g_i(X) = sum_b prod_j f_j(b, X) has degree k, sent as its evaluations at
    0..k (values above 1 extrapolated division-free from the table halves as
    lo + t·(hi - lo)). The verifier checks g_i(0) + g_i(1) against the
    running claim and interpolates g_i(r_i) as the next claim. After the
    last round the prover sends the per-factor values v_j = f_j(r); the
    verifier checks prod_j v_j against the final claim and each (f_j, r,
    v_j) becomes an output Relation_Eval statement.

    The native (C) path covers exactly two native tables over a Ring —
    the reference Libra shape; other arities or representations use the
    pure-Python path. Both paths produce identical transcripts.
    """

    name = "sumcheckprod"
    reduces_from: type[Relation] = Relation_SumProd
    reduces_to: tuple[type[Relation], ...] = (Relation_Eval,)
    supported_domains: tuple[type, ...] = (Ring, Field)

    def soundness_error(self, statement: Statement, domain) -> float | None:
        """k * num_vars / |A| (piop.md §6); None if |A| is unknown."""
        size = exceptional_set_size(domain)
        if size is None:
            return None
        return len(statement.oracles) * statement.num_vars / size

    def _round_message(self, factors: list, var) -> tuple:
        return prod_round_evals(factors, var)

    def _update_tables(
        self, factors: list, var, r, in_place: list
    ) -> tuple[list, tuple | None]:
        """`prod_update_tables`: two field-backed tables also form the next
        round's message, in the same pass."""
        return prod_update_tables(factors, var, r, in_place)

    def _outputs(self, statement: Statement, point: dict, values: tuple):
        """One Relation_Eval claim per factor, all at the same point."""
        return [
            self.reduce([statement], oracles=[f], point=point, value=v)
            for f, v in zip(statement.oracles, values, strict=True)
        ]

    def _prove_tail(
        self, iop: IOP, statement: Statement, label: str, factors: list, point: dict
    ) -> list[Statement]:
        values = tuple(f.constant() for f in factors)
        iop.transcript.write(f"{label}/vals", values)
        return self._outputs(statement, point, values)

    async def _verify_tail(
        self, iop: IOP, statement: Statement, label: str, point: dict, claim
    ) -> list[Statement]:
        values = await iop.transcript.read(f"{label}/vals")
        prod = None
        for v in values:
            prod = v if prod is None else prod * v
        if not (prod == claim):
            raise Rejection(f"{label}: prod of claimed factor values != claim")
        return self._outputs(statement, point, values)
