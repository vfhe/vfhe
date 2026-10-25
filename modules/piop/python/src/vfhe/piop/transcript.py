# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The transcript of one run and the proof it leaves (piop.md section 5):
the labeled record of exchanged messages, its Fiat-Shamir chain, the
argument string of a non-interactive run, and the futures protocol values
are exchanged through."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from vfhe.crypto import chain_digest, hash_bytes

from .digest import element_digest
from .protocol import Rejection

if TYPE_CHECKING:
    from .iop import IOP


def _loop_of(iop: IOP | None):
    """The event loop an IOP future attaches to: the IOP's, else the current one."""
    if iop is not None:
        return iop.loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


class Transcript:
    """The ordered, labeled record of the messages exchanged in one run.

    `write` appends an entry and fixes its position in the canonical order;
    `read` awaits the entry with that label, so a receiver can be scheduled
    before its sender — underneath, entries are futures. Labels follow
    Merlin-style transcripts: they are part of the record (domain
    separation), not debug metadata, and a label can be written only once.
    The write order is the canonical order a Fiat-Shamir transformation
    hashes (piop.md §5); FS never touches this interface — a derandomized
    verifier subtype interacts with the same transcript.
    """

    def __init__(self, iop: IOP | None = None):
        self.iop = iop
        self.entries = {}  # label -> Variable
        self.order = []  # labels in write order — the canonical order
        self.derived = set()  # labels the verifier derived, not prover messages
        self.seed = b""  # sigma_0 pre-image: the root statement digest
        self.digests = []  # per-entry digests, extended lazily by state()
        self._states = []  # chain values: _states[i] hashes entries[: i+1]
        self._proof = None  # replay source: the prover messages being read back
        self._read = 0  # how many of them the verifier has consumed

    def _entry(self, label: str) -> Variable:
        var = self.entries.get(label)
        if var is None:
            var = self.entries[label] = Variable(label, self.iop)
        return var

    def write(self, label: str, value, derived: bool = False):
        """Append `value` under `label`; rewriting a label is an error.

        `derived=True` marks a verifier-computed entry (a challenge): part
        of the chain like any other, but not a prover message, so it is
        recomputed rather than stored in the proof.
        """
        self._entry(label).set_result(value)
        self.order.append(label)
        if derived:
            self.derived.add(label)
        return value

    async def read(self, label: str):
        """The value written under `label`, awaiting it if not written yet.

        Reading is the verifier's side of the wire. Against a live prover
        the await blocks until the message is written; against a stored
        proof (`replay`) it consumes that proof's next message, which must
        be the one expected — a single forward pass, no random access.
        """
        entry = self._entry(label)
        if self._proof is not None and not entry.done():
            self._consume(label)
        elif self._proof is None and self.iop is not None and self.iop.solo:
            raise RuntimeError(
                f"read of {label!r} with no counterparty: a prover-only run "
                "cannot receive messages (only verify halves read)"
            )
        # Shielded: a reader cancelled while waiting (a sub-protocol of an
        # AND-composition that another one's failure stops) must not cancel
        # the entry itself, which the writer still has to set.
        return await asyncio.shield(entry)

    def _consume(self, label: str) -> None:
        """Take the next message of the replayed proof, which must be the
        one the verifier is asking for."""
        assert self._proof is not None
        if self._read >= len(self._proof):
            raise Rejection(f"proof exhausted; verifier expected {label!r}")
        expected, value = self._proof.messages[self._read]
        if expected != label:
            raise Rejection(
                f"proof out of order: verifier expected {label!r}, "
                f"proof continues with {expected!r}"
            )
        self._read += 1
        self.write(label, value)

    def replay(self, proof: Proof) -> None:
        """Feed `proof`'s messages to this transcript as the verifier reads
        them, in place of a live prover."""
        if self.order:
            raise RuntimeError("transcript already used; replay into a fresh IOP")
        self._proof = proof

    @property
    def fully_read(self) -> bool:
        """Whether a replayed proof was consumed to its end.

        The end-of-input check, and not a tidiness one: bytes a verifier
        never reads make the proof **malleable** — an adversary appends or
        alters them to obtain a second, distinct accepting proof of the same
        statement, costing strong simulation-extractability [CFRG-FS, §6.2;
        spongefish's `check_eof`]. Unread messages must reject.
        """
        return self._proof is None or self._read == len(self._proof)

    def messages(self) -> tuple:
        """The prover messages, in write order — everything the verifier
        cannot recompute for itself. This is the proof (`Proof.of`)."""
        return tuple(
            (label, self.entries[label].result())
            for label in self.order
            if label not in self.derived
        )

    def bind(self, seed: bytes) -> None:
        """Set the chain's seed — the digest of the root statement, so every
        derived challenge binds the claim being proven (omitting it is the
        "weak Fiat-Shamir" bug [DMWG23]). Must happen before the first
        `state()` computation; `IOP.run` does it for Fiat-Shamir runs."""
        if self._states:
            raise RuntimeError("transcript chain already started; bind earlier")
        self.seed = seed

    def state(self, upto: int | None = None) -> bytes:
        """The chained hash of the first `upto` written entries (default:
        all written so far): h_i = H(h_{i-1} | H(label_i) | digest(value_i)),
        with h_{-1} = H(seed). Recursive rather than one concatenation — the
        BCS chain sigma_i = H(rt_i | sigma_{i-1}) [BCS16] — so per-entry
        digests and chain values are cached (`digests` / `_states`): a new
        entry costs one link, any earlier prefix is a lookup.
        """
        if upto is None:
            upto = len(self.order)
        if upto > len(self.order):
            raise IndexError(f"only {len(self.order)} entries written")
        while len(self._states) < upto:
            i = len(self._states)
            label = self.order[i]
            self.digests.append(element_digest(self.entries[label].result()))
            previous = self._states[i - 1] if i else hash_bytes(b"seed" + self.seed)
            self._states.append(
                hash_bytes(previous + hash_bytes(label.encode()) + self.digests[i])
            )
        return self._states[upto - 1] if upto else hash_bytes(b"seed" + self.seed)


class Proof:
    """The *argument string* of one Fiat-Shamir run: its prover messages,
    in the order they were written.

    "Argument string" is the term of art [CY24, §4.1; CO25, §2.1] and "NARG
    string" its implementation spelling [CFRG-FS, §2]; this class keeps the
    readable name. Deliberately not called a transcript — that word is the
    interactive record, prover *and* verifier messages both [CFRG-FS, §2].

    Only the prover's messages: the challenges are a deterministic function
    of what precedes them, so carrying them would store what the verifier
    recomputes anyway — and invite a verifier to trust them. ("The argument
    string pi contains a salt tau and all IP prover messages (and none of
    the IP verifier messages)" [CO25, §4.3].) The statement is not part of
    it either; public input travels separately and is bound into the chain
    seed.

    Labels ride along so a replay can detect a message arriving out of turn.
    They are redundant in principle (the verifier knows what it expects
    next) and a byte-level encoding would drop them; keeping them here buys
    a clear error instead of a silent misparse.
    """

    __slots__ = ("messages",)

    def __init__(self, messages):
        self.messages = tuple((str(label), value) for label, value in messages)

    @classmethod
    def of(cls, transcript: Transcript) -> Proof:
        """The proof recorded by a completed prover run."""
        return cls(transcript.messages())

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def digest(self) -> bytes:
        """A digest of the whole proof (`element_digest` hook)."""
        return chain_digest([b"proof"] + [element_digest(m) for m in self.messages])

    def __repr__(self) -> str:
        return f"Proof({len(self)} messages: {', '.join(self.labels)})"


class Value(asyncio.Future):
    """An anonymous protocol value that may not be computed yet (e.g. the
    evaluation of an MLE at a point containing unresolved challenges)."""

    def __init__(self, iop: IOP | None = None, value=None):
        self.iop = iop
        super().__init__(loop=_loop_of(iop))
        if value is not None:
            self.set_result(value)


class Variable(asyncio.Future):
    """A named protocol variable, typically a verifier challenge r_i."""

    def __init__(self, name: str, iop: IOP | None = None, value=None):
        self.name = name
        self.iop = iop
        super().__init__(loop=_loop_of(iop))
        if value is not None:
            self.set_result(value)
