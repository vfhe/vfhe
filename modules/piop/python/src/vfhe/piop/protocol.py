# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The protocol: a reduction between products of relations (piop.md
section 5), as a pair of prove / verify coroutines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vfhe.arith import ArithParent

from .relation import Relation, Statement, child_statement

if TYPE_CHECKING:
    from .party import Prover, Verifier


def exceptional_set_size(domain) -> int | None:
    """|A| for the domain's exceptional set; None if the domain has no size.

    Each arithmetic domain computes its own: the size depends on the
    representation, and a domain that is not an `ArithParent` (an int, a
    Fraction) has no exceptional set to report.
    """
    if isinstance(domain, ArithParent):
        return domain.exceptional_set_size
    return None


class Rejection(Exception):
    """Raised by a protocol's verify half when a round check fails."""


class Protocol:
    """A reduction between products of relations (piop.md §5).

    The two halves are coroutines over the same bundle of statements:
    `prove` sends the oracle messages, `verify` checks round consistency,
    and both draw challenges via `iop.verifier.challenge` and return the
    list of statements the bundle reduces to. Messages flow only through
    the IOP transcript (`iop.transcript.write` / `await iop.transcript.read`);
    statements are never sent — each party derives its own DAG from the
    common input and transcript.

    `batching = True` declares a bundle protocol, a many-to-fewer reduction
    [KP23, Def. 4]: the driver hands it a bundle of statements in one
    invocation instead of one at a time — every frontier statement of the
    relation for most relations; for `Relation_Eval`, every claim on the
    oracles with one `bundle_key` (by default, one oracle), *after* every
    other frontier statement that could still emit a claim on such an oracle
    has been discharged (the parking rule, piop.md §5).

    `supported_domains` lists the coefficient-domain types (e.g. Ring) for
    which native (C) kernels exist — declarative metadata, not a dispatch
    switch. The native/pure-Python decision is made at the data level,
    inside the round-message helpers (they inspect the oracle
    representation, e.g. `mle.native_table`), so protocol bodies are
    written once and never branch on it; both paths produce identical
    transcripts. An empty tuple means pure Python always.

    A protocol reports its own soundness error as
    `soundness_error(statement, domain)` (piop.md §6): the error of this
    one step for a claim of `statement`'s size, with the challenges drawn
    from `domain`'s exceptional set (`exceptional_set_size`),
    `None` when that set has no known size. The errors of the steps a
    reduction hands on to are those steps' own.

    `scheme` is the commitment scheme, with its public parameters, whose
    commitments an evaluation protocol proves against; `None` for any other
    protocol. The registry keys a protocol by it: a claim on a
    `CommittedOracle` goes to the protocol of the scheme the oracle was
    committed under (`IOP.protocol_for`).
    """

    reduces_from: type[Relation] | None = None
    reduces_to: tuple[type[Relation], ...] = ()
    batching: bool = False
    supported_domains: tuple[type, ...] = ()
    scheme: Any = None

    def bundle_key(self, oracle) -> Any:
        """What a bundle protocol groups claims on `oracle` by: the driver
        hands it, together, the frontier claims whose oracles have equal
        keys, and parks a claim while another frontier statement can still
        produce a claim on an oracle with its key. The default is the
        oracle's identity, a bundle per oracle; an evaluation protocol whose
        opening batches several polynomials returns its scheme instead, for
        one bundle per scheme. The key must be a function of public data, so
        that both parties form the same bundles."""
        return id(oracle)

    def reduce(
        self,
        statements: list[Statement],
        relation: Relation | type[Relation] | None = None,
        **fields,
    ) -> Statement:
        """One output statement of this reduction (default: reduces_to[0])."""
        return child_statement(
            tuple(statements), relation or self.reduces_to[0], **fields
        )

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        raise NotImplementedError

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        raise NotImplementedError
