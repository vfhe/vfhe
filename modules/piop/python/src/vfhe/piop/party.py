# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two parties (piop.md section 5): the worklist driver both run, the
parking rule, the prover's witnesses and the verifier's two samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.crypto import entropy

from .oracle import oracle_closure
from .protocol import Rejection
from .relation import Statement, statement_oracles

if TYPE_CHECKING:
    from .iop import IOP


class Party:
    """Common state of the two IOP roles."""

    def __init__(self, iop: IOP):
        self.iop = iop
        self.state = {}

    async def _drive(self, half: str, statement: Statement) -> list[Statement]:
        """Worklist driver: follow the registered reductions from the root
        statement until every branch of the DAG reaches a terminal relation
        (one with no registered protocol); returns the terminal statements.

        Both parties run this same deterministic traversal, so their DAGs,
        paths, and transcript labels agree.
        """
        worklist = [statement]
        terminal = []
        while worklist:
            stmt = self._next(worklist)
            worklist.remove(stmt)
            protocol = self.iop.protocol_for(stmt)
            if protocol is None:
                terminal.append(stmt)
                continue
            bundle = [stmt]
            if protocol.batching:
                key = self._bundle_key(stmt)
                same = [s for s in worklist if self._bundle_key(s) == key]
                for s in same:
                    worklist.remove(s)
                bundle += same
                bundle.sort(key=lambda s: s.path)
            worklist.extend(await getattr(protocol, half)(self, bundle))
        return terminal

    def _bundle_key(self, stmt: Statement) -> tuple:
        """Statements a bundle protocol receives together: same relation
        type and, for claims on an oracle (`Relation.subject`), the same
        protocol and the same `Protocol.bundle_key` of the oracle -- by
        default, the same oracle."""
        subject = stmt.relation.subject(stmt)
        if subject is not None:
            protocol = self.iop.protocol_for(stmt)
            if protocol is None:
                return (type(stmt.relation), id(subject))
            return (type(stmt.relation), id(protocol), protocol.bundle_key(subject))
        return (type(stmt.relation),)

    def _parked(self, stmt: Statement, worklist: list[Statement]) -> bool:
        """The parking rule (piop.md §5): a claim on an oracle headed for a
        bundle protocol waits while any other frontier statement, outside
        its own bundle, can still produce a claim that would join the bundle
        — i.e. has in its oracle closure an oracle with the same
        `Protocol.bundle_key` (by default, the same oracle). Definitions form
        a DAG and statements with no subject never park, so some statement
        is always ready. Claims on implicit oracles never reach the driver
        (the Composer consumes them), so this concerns committed and plain
        oracles.
        """
        subject = stmt.relation.subject(stmt)
        if subject is None:
            return False
        protocol = self.iop.protocol_for(stmt)
        if protocol is None or not protocol.batching:
            return False
        key = self._bundle_key(stmt)
        subject_key = protocol.bundle_key(subject)
        for other in worklist:
            if other is stmt or self._bundle_key(other) == key:
                continue
            if any(
                protocol.bundle_key(o) == subject_key
                for o in oracle_closure(statement_oracles(other))
            ):
                return True
        return False

    def _next(self, worklist: list[Statement]) -> Statement:
        """The first frontier statement not parked, in worklist order."""
        for stmt in worklist:
            if not self._parked(stmt, worklist):
                return stmt
        raise RuntimeError("every frontier statement is parked (cyclic definitions?)")


class Prover(Party):
    """Computes and sends the oracles; holds the witnesses.

    `witnesses` maps an instance handle to its private data — the witness is
    never a field of the statement itself, so statements stay safe to share.
    Keys are whatever public object the witness belongs to: a statement, an
    oracle (`witnesses[oracle] = table` for a committed or an implicit
    oracle, where `resolve_table` finds the table the prover works on), or
    a polynomial commitment (`witnesses[commitment] = opening`, the scheme's
    own prover data for its evaluation protocol). Oracles and commitments
    survive statement forking and reduction, so they are stable keys.
    """

    def __init__(self, iop: IOP):
        super().__init__(iop)
        self.witnesses = {}

    async def prove(self, statement: Statement) -> list[Statement]:
        """Follow the registered reductions; returns the terminal statements
        (the prover makes no decision)."""
        return await self._drive("prove", statement)


class Verifier(Party):
    """Sends challenges and is restricted to oracle queries (by convention
    in-process; a commitment compiler can enforce it cryptographically)."""

    def challenge(self, label: str):
        """The challenge `label`: on first call, sample from the domain's
        exceptional set and write it to the transcript; later calls return
        the recorded value.

        Either party may call this (`iop.verifier.challenge(label)`) — under
        Fiat-Shamir the challenge is a deterministic function of the
        transcript so far, so both sides compute the same value; a FS
        verifier is a subtype overriding only this method, and the IOP
        machinery stays unaware of FS. Public-coin honesty is the caller's
        responsibility: draw a challenge only after writing the round
        message it answers, so the transcript order stays canonical.
        """
        entry = self.iop.transcript.entries.get(label)
        if entry is not None and entry.done():
            return entry.result()
        value = self._draw_challenge(label)
        self.iop.transcript.write(label, value, derived=True)
        return value

    def challenge_bits(self, label: str, bits: int = 256) -> bytes:
        """The bit-string challenge `label`: on first call, sample
        ceil(bits / 8) uniform bytes and write them to the transcript; later
        calls return the recorded value.

        The second kind of verifier randomness, beside `challenge`, and
        deliberately *shapeless*: the verifier hands out raw coins, and how
        they become protocol randomness — query positions, a permutation, a
        subset — is the protocol's business, expanded on its side (e.g. by a
        hash in counter mode, as `BasefoldEval.query_positions` does for its
        spot-check positions). It is published because derived randomness
        must be reproducible by the prover — once an oracle is a commitment,
        the prover has to learn the query positions to answer them with
        authentication paths. It is not a domain challenge: no
        exceptional-set structure, so `soundness_error` accounting never
        sees it.

        Same contract as `challenge` otherwise — compute-if-absent, either
        party may call it, and a Fiat-Shamir verifier overrides both
        samplers to derive from the transcript instead.
        """
        entry = self.iop.transcript.entries.get(label)
        if entry is not None and entry.done():
            return entry.result()
        value = self._draw_bits(label, (bits + 7) // 8)
        self.iop.transcript.write(label, value, derived=True)
        return value

    def _draw_challenge(self, label: str):
        """Where a domain challenge's value comes from: fresh randomness
        here; the transcript chain in the Fiat-Shamir subtype (fs.py)."""
        domain = self.iop.domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        return domain.random_exceptional()

    def _draw_bits(self, label: str, nbytes: int) -> bytes:
        """Where a bit-string challenge's value comes from (see above)."""
        return entropy.bytes(nbytes)

    async def verify(self, statement: Statement) -> bool:
        """Follow the registered reductions, then decide every terminal
        statement with its relation's own decider — for Relation_Eval that
        is exactly one oracle query per claim. Never call `check()` on a
        non-terminal statement here: that decider enumerates the hypercube.
        """
        try:
            terminal = await self._drive("verify", statement)
            for stmt in terminal:
                if not (await stmt.resolved()).check():
                    return False
            return True
        except Rejection:
            return False
