# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""One protocol execution (piop.md section 5): the registry of
protocols, the parties, and the three runners."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .digest import element_digest
from .fs import FS_Verifier
from .oracle import CommittedOracle, ImplicitOracle, OracleKind, oracle_closure
from .party import Prover, Verifier
from .transcript import Proof, Transcript, Variable

if TYPE_CHECKING:
    from .protocol import Protocol
    from .relation import Relation, Statement


class IOP:
    """One protocol execution: event loop, coefficient domain, transcript,
    parties, and the relation -> protocol registry.

    `domain` is the Ring or Field the oracle coefficients live in; it defines
    the exceptional set challenges are drawn from (piop.md §6). The
    constructor instantiates the two parties — pass `prover` / `verifier`
    subclasses to customize them, or set `fiat_shamir=True` to get the
    non-interactive verifier (`fs.FS_Verifier`: challenges derived from the
    transcript chain instead of sampled; the runners seed the chain with the
    root statement's digest). An explicit `verifier=` wins over the flag. The
    transcript is shared by both parties: the sender writes, the receiver
    awaits — so rounds that do not depend on a pending message can be
    scheduled without blocking.

    An IOP object is **single-use**: one run consumes its transcript. There
    are three runners, all consuming it:

    - `run(statement)` — both parties at once, the interactive execution;
    - `prove(statement) -> Proof` — the prover alone (Fiat-Shamir only),
      producing the argument string;
    - `verify(statement, proof)` — the verifier alone against that string.

    So a non-interactive proof is produced and checked by two separate IOP
    objects over the same registry, which is the point: the checking side
    never needs the prover, its witnesses, or the oracles.
    """

    def __init__(
        self,
        domain=None,
        fiat_shamir: bool = False,
        prover: type[Prover] | None = None,
        verifier: type[Verifier] | None = None,
    ) -> None:
        self.loop = asyncio.new_event_loop()
        self.domain = domain
        self.fiat_shamir = fiat_shamir
        self.solo = False  # set while one party runs without a counterparty
        self.transcript = Transcript(self)
        # (type[Relation], OracleKind | None, scheme | None) -> Protocol
        self.protocols = {}
        self.prover = (prover or Prover)(self)
        if verifier is None:
            verifier = FS_Verifier if fiat_shamir else Verifier
        self.verifier = verifier(self)

    def new_variable(self, name: str) -> Variable:
        return Variable(name, self)

    def register(
        self,
        relation_type: type[Relation],
        protocol: Protocol,
        kind: OracleKind | None = None,
    ) -> Protocol:
        """Choose the protocol that discharges statements of `relation_type`.

        Relations without a registered protocol are terminal: the verifier
        decides them directly with the relation's own `check()`.

        A relation whose statements are claims on one oracle
        (`Relation.subject`, as `Relation_Eval`'s are) has several
        discharges, keyed by the kind of that oracle (`OracleKind`) and by
        the protocol's `Protocol.scheme`: register each commitment scheme's
        evaluation protocol with no kind, and `VirtualEval` for
        `OracleKind.virtual`. A claim on a committed oracle goes to the
        entry of the scheme it was committed under; a claim without an
        entry of its own falls back to the entry of its kind, then to the
        default (no kind, no scheme). Claims on public oracles never reach
        a protocol; claims on plain oracles are refused once a scheme is
        registered for the relation, since a plain oracle exists only in
        the ideal model; claims on implicit oracles never reach the driver
        (the Composer consumes them).
        """
        if kind is OracleKind.implicit:
            raise ValueError("no protocol discharges claims on implicit oracles")
        self.protocols[(relation_type, kind, protocol.scheme)] = protocol
        return protocol

    def protocol_for(self, statement: Statement) -> Protocol | None:
        """The protocol discharging `statement`, None if it is terminal."""
        relation = statement.relation
        rtype = type(relation)
        subject = relation.subject(statement)
        if subject is None:
            return self.protocols.get((rtype, None, None))
        kind = subject.kind
        if kind is OracleKind.implicit:
            raise RuntimeError(
                "an evaluation claim on an implicit oracle reached the driver; "
                "claims on a head are consumed by the Composer of the "
                "reduction that heads it"
            )
        if kind is OracleKind.public:
            return None
        if kind is OracleKind.plain and any(
            r is rtype and s is not None for r, _, s in self.protocols
        ):
            raise TypeError(
                f"a {rtype.__name__} claim on a plain oracle while a commitment "
                "scheme is registered for it: the claim must be on a committed "
                "oracle (the one the scheme's commit returned)"
            )
        scheme = subject.scheme if isinstance(subject, CommittedOracle) else None
        for key in (
            (rtype, kind, scheme),
            (rtype, None, scheme),
            (rtype, kind, None),
            (rtype, None, None),
        ):
            protocol = self.protocols.get(key)
            if protocol is not None:
                return protocol
        return None

    def _bind(self, statement: Statement) -> None:
        """Seed the Fiat-Shamir chain with the claim being proven, so every
        derived challenge depends on it (weak-FS, [DMWG23])."""
        if self.fiat_shamir and not self.transcript.seed:
            self.transcript.bind(element_digest(statement))

    @staticmethod
    def _root(statement: Statement) -> Statement:
        """`statement` checked as the root of a run: the index of a root is
        fixed before any reduction, so its oracles (through their
        definitions) are tables, never implicit, with one exception, the
        head of a started root, which a relation with an index head keeps in
        its index."""
        for o in oracle_closure(statement.relation.index_oracles()):
            if isinstance(o, ImplicitOracle) and o is not statement.head:
                raise TypeError(
                    f"the index of a root statement holds the implicit oracle "
                    f"{o!r}, which is not its head; a root's index oracles are "
                    "tables"
                )
        return statement

    def _prover_view(self, statement: Statement) -> Statement:
        """The prover's fork of the root, carrying its witness along."""
        fork = statement._fork()
        if statement in self.prover.witnesses:
            self.prover.witnesses[fork] = self.prover.witnesses[statement]
        return fork

    def prove(self, statement: Statement) -> Proof:
        """Run the prover alone and return the argument string.

        Fiat-Shamir only: the prover derives each challenge itself from the
        transcript chain, so no verifier has to be present — which is what
        makes the run non-interactive. (Interactively the challenges would
        be fresh randomness that a proof cannot carry and a verifier could
        not recompute, so this refuses rather than emit an unverifiable
        object.)
        """
        if not self.fiat_shamir:
            raise RuntimeError(
                "prove() produces a non-interactive proof; construct the IOP "
                "with fiat_shamir=True (or use run() for an interactive check)"
            )
        self._bind(self._root(statement))
        self.solo = True
        try:
            self.loop.run_until_complete(
                self.prover.prove(self._prover_view(statement))
            )
        finally:
            self.solo = False
        return Proof.of(self.transcript)

    def verify(self, statement: Statement, proof: Proof) -> bool:
        """Check `proof` against `statement`, with no prover present.

        The verifier drives exactly as it would against a live prover; each
        message it reads is taken from the proof in order, and each
        challenge is re-derived from the chain over everything read so far.
        A proof whose messages arrive out of turn, run out, or are left over
        at the end is rejected, as is one whose challenges no longer fit its
        messages — that is the Fiat-Shamir binding doing its work.
        """
        if not self.fiat_shamir:
            raise RuntimeError(
                "verify() checks a non-interactive proof; construct the IOP "
                "with fiat_shamir=True (or use run() for an interactive check)"
            )
        self._bind(self._root(statement))
        self.transcript.replay(proof)
        self.solo = True
        try:
            verdict = self.loop.run_until_complete(
                self.verifier.verify(statement._fork())
            )
        finally:
            self.solo = False
        # Trailing messages the verifier never asked for make it malformed.
        return verdict and self.transcript.fully_read

    def run(self, statement: Statement) -> bool:
        """Run both parties concurrently on the statement; the verdict is
        the verifier's output.

        Each party drives its own fork of the root statement (fresh
        child-path counters), so both derive the same DAG paths — and the
        same transcript labels — independently.
        """
        self._bind(self._root(statement))
        prover_statement = self._prover_view(statement)

        async def _run():
            prover_task = asyncio.ensure_future(self.prover.prove(prover_statement))
            verifier_task = asyncio.ensure_future(
                self.verifier.verify(statement._fork())
            )
            try:
                done, _ = await asyncio.wait(
                    {prover_task, verifier_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if prover_task in done:
                    # Re-raise a prover crash instead of leaving the
                    # verifier deadlocked on a message that never comes.
                    prover_task.result()
                verdict = await verifier_task
                if verdict:
                    # Accepting run: every challenge is resolved, so the
                    # prover finishes; a rejected run may leave it blocked
                    # on a never-drawn challenge, hence the cancel below.
                    await prover_task
                return verdict
            finally:
                prover_task.cancel()
                verifier_task.cancel()
                await asyncio.gather(prover_task, verifier_task, return_exceptions=True)

        return self.loop.run_until_complete(_run())
