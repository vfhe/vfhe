# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The two parties (piop.md section 5): the worklist driver both run, the
parking rule, the prover's witnesses and the verifier's two samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vfhe.crypto import entropy

from .and_composition import current_subprotocol
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
        worklist = _Frontier(self)
        worklist.extend([statement])
        terminal = []
        while worklist:
            stmt = self._next(worklist)
            key = worklist.bundle_key(stmt)
            worklist.remove(stmt)
            protocol = self.iop.protocol_for(stmt)
            if protocol is None:
                terminal.append(stmt)
                continue
            bundle = [stmt]
            if protocol.batching:
                same = [s for s in worklist if worklist.bundle_key(s) == key]
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

    def _parked(self, stmt: Statement, worklist: _Frontier) -> bool:
        """The parking rule (piop.md §5): a claim on an oracle headed for a
        bundle protocol waits while any other frontier statement, outside
        its own bundle, can still produce a claim that would join the bundle
        — i.e. has in its oracle closure an oracle with the same
        `Protocol.bundle_key` (by default, the same oracle). Definitions form
        a DAG and statements with no subject never park, so some statement
        is always ready. Claims on implicit oracles never reach the driver
        (the Composer consumes them), so this concerns committed and plain
        oracles.

        Decided by a lookup in the frontier's index (`_Frontier.reaching`).
        """
        bundle_protocol = worklist.bundle_protocol(stmt)
        if bundle_protocol is None:
            return False
        protocol, subject_key = bundle_protocol
        key = worklist.bundle_key(stmt)
        reaching = worklist.reaching(protocol, subject_key)
        return any(count and other != key for other, count in reaching.items())

    def _next(self, worklist: _Frontier) -> Statement:
        """The first frontier statement not parked, in worklist order."""
        for stmt in worklist:
            if not self._parked(stmt, worklist):
                return stmt
        raise RuntimeError("every frontier statement is parked (cyclic definitions?)")


class _Frontier:
    """The driver's worklist: the statements waiting to be discharged, in
    the order they were added.

    Each statement's bundle key (`Party._bundle_key`), oracle closure and,
    for a claim headed for a bundle protocol, that protocol and its
    subject's `Protocol.bundle_key` are computed once, when it is added.
    Statements and oracle definitions do not change while they wait, so
    neither do these.

    For each bundle protocol the parking rule has asked about, the frontier
    also keeps an index for `reaching`. It is built on the first question
    and updated as statements are added and removed.
    """

    def __init__(self, party: Party):
        self._party = party
        self._order: dict[int, Statement] = {}  # id -> statement, in order
        self._keys: dict[int, tuple] = {}
        self._closures: dict[int, list] = {}
        self._bundled: dict[int, tuple | None] = {}
        # id(protocol) -> (protocol, {subject key: {bundle key: count}})
        self._indices: dict[int, tuple[object, dict]] = {}

    def __bool__(self) -> bool:
        return bool(self._order)

    def __iter__(self):
        return iter(self._order.values())

    def bundle_key(self, stmt: Statement) -> tuple:
        """The bundle `stmt` belongs to (`Party._bundle_key`)."""
        return self._keys[id(stmt)]

    def bundle_protocol(self, stmt: Statement) -> tuple | None:
        """`(protocol, protocol.bundle_key(subject))` for a claim on an
        oracle headed for a bundle protocol; None for any other statement."""
        return self._bundled[id(stmt)]

    def extend(self, statements) -> None:
        """Add `statements` at the end of the worklist."""
        for stmt in statements:
            sid = id(stmt)
            if sid in self._order:
                raise ValueError("a statement already on the frontier was added again")
            self._order[sid] = stmt
            self._keys[sid] = self._party._bundle_key(stmt)
            self._closures[sid] = oracle_closure(statement_oracles(stmt))
            subject = stmt.relation.subject(stmt)
            protocol = None if subject is None else self._party.iop.protocol_for(stmt)
            self._bundled[sid] = (
                (protocol, protocol.bundle_key(subject))
                if protocol is not None and protocol.batching
                else None
            )
            for protocol, index in self._indices.values():
                self._update_index(protocol, index, sid, 1)

    def remove(self, stmt: Statement) -> None:
        """Take `stmt` off the worklist."""
        sid = id(stmt)
        for protocol, index in self._indices.values():
            self._update_index(protocol, index, sid, -1)
        del self._order[sid], self._keys[sid], self._closures[sid], self._bundled[sid]

    def reaching(self, protocol, subject_key) -> dict:
        """Count, by bundle key, the frontier statements that have in their
        oracle closure an oracle `o` with `protocol.bundle_key(o)` equal to
        `subject_key`. Returns a dict from bundle key to that count; a count
        can be 0 once the statements it counted are gone."""
        entry = self._indices.get(id(protocol))
        if entry is None:
            entry = self._indices[id(protocol)] = (protocol, {})
            for sid in self._order:
                self._update_index(protocol, entry[1], sid, 1)
        return entry[1].get(subject_key, {})

    def _update_index(self, protocol, index: dict, sid: int, delta: int) -> None:
        """Add `delta` to the count of statement `sid`'s bundle key under
        each `protocol.bundle_key` of an oracle in its closure."""
        key = self._keys[sid]
        for reached in {protocol.bundle_key(o) for o in self._closures[sid]}:
            counts = index.setdefault(reached, {})
            counts[key] = counts.get(key, 0) + delta


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

    async def challenge(self, label: str, domain=None):
        """The challenge `label`: on first call, sample from the exceptional
        set of `domain` (the IOP's domain by default) and write it to the
        transcript; later calls return the recorded value.

        A protocol passes `domain` when its challenge lives in another ring or
        field than the IOP's, such as a residue field of the IOP's ring.

        Either party may call this (`await iop.verifier.challenge(label)`) —
        under Fiat-Shamir the challenge is a deterministic function of the
        transcript so far, so both sides compute the same value; a FS
        verifier is a subtype overriding only the draw, and the IOP machinery
        stays unaware of FS. Public-coin honesty is the caller's
        responsibility: draw a challenge only after writing the round message
        it answers, so the transcript order stays canonical.

        It is awaited because, inside an AND-composition
        (`IOP.and_composition`), a challenge is shared with the other
        sub-protocols and waits until every one of them has written its
        messages for the round; outside one it returns at once.
        """
        bound = current_subprotocol(self.iop)
        if bound is not None:
            composition, index = bound
            return await composition.challenge(
                index, label, self._challenge_domain(domain)
            )
        return self._recorded_challenge(label, domain)

    async def challenge_bits(self, label: str, bits: int = 256) -> bytes:
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
        party may call it, a Fiat-Shamir verifier derives it from the
        transcript instead, and inside an AND-composition it is a round
        shared with the other sub-protocols.
        """
        bound = current_subprotocol(self.iop)
        if bound is not None:
            composition, index = bound
            return await composition.challenge_bits(index, label, (bits + 7) // 8)
        return self._recorded_bits(label, bits)

    def _recorded_challenge(self, label: str, domain=None):
        """`challenge` outside an AND-composition: the recorded value,
        or a fresh draw written to the transcript."""
        entry = self.iop.transcript.entries.get(label)
        if entry is not None and entry.done():
            return entry.result()
        value = self._draw_challenge(label, domain)
        self.iop.transcript.write(label, value, derived=True)
        return value

    def _recorded_bits(self, label: str, bits: int) -> bytes:
        """`challenge_bits` outside an AND-composition, as
        `_recorded_challenge`; also how an AND-composition draws its round
        seeds."""
        entry = self.iop.transcript.entries.get(label)
        if entry is not None and entry.done():
            return entry.result()
        value = self._draw_bits(label, (bits + 7) // 8)
        self.iop.transcript.write(label, value, derived=True)
        return value

    def _challenge_domain(self, domain):
        """The domain a challenge is drawn in: `domain`, or the IOP's when
        None."""
        domain = self.iop.domain if domain is None else domain
        if domain is None:
            raise RuntimeError("this IOP has no domain")
        return domain

    def _draw_challenge(self, label: str, domain=None):
        """Where a domain challenge's value comes from: fresh randomness
        here; the transcript chain in the Fiat-Shamir subtype (fs.py).
        `domain` is the IOP's when None."""
        return self._challenge_domain(domain).random_exceptional()

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
