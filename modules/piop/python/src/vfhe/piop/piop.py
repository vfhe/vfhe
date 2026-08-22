# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""IOP scaffolding: relations, statements, parties, and async plumbing.

The design and vocabulary (relation / language / statement / index / oracle)
are derived from the PIOP literature; see ``modules/piop/piop.md`` for the
full derivation and bibliography.
"""

from __future__ import annotations

import asyncio
import secrets

from .merkle import hash_bytes


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


def _pending(x) -> bool:
    """True for a future whose value has not been set yet."""
    return isinstance(x, asyncio.Future) and not x.done()


def _value_of(x):
    """Unwrap a resolved future to its value; anything else passes through."""
    if isinstance(x, asyncio.Future):
        return x.result()
    return x


def _constant(mle):
    """Unwrap a fully-evaluated (0-variable) MLE to its single value.

    Duck-typed via `MLE.constant()` (mle.py imports from this module, so
    this module cannot import the MLE types back); plain values pass through.
    """
    if getattr(mle, "num_vars", None) == 0 and hasattr(mle, "constant"):
        return mle.constant()
    return mle


def _hypercube(variables: list):
    """All points of {0,1}^n as variable -> bit dicts."""
    for bits in range(1 << len(variables)):
        yield {v: (bits >> i) & 1 for i, v in enumerate(variables)}


def _hypercube_sum(f):
    """sum_{b in {0,1}^n} f(b), unwrapped to a plain coefficient value."""
    total = None
    for b in _hypercube(f.variables):
        e = f.evaluate(b, in_place=False)
        total = e if total is None else total + e
    return _constant(total)


def _chain(parts: list[bytes]) -> bytes:
    """The chained (Merkle-Damgard style) digest of already-hashed parts:
    h_i = H(h_{i-1} | part_i). The recursive form, not one concatenation —
    it is what the BCS transform hashes (sigma_i = H(rt_i | sigma_{i-1})
    [BCS16]) and what lets a growing sequence cache its prefix states."""
    state = hash_bytes(b"vfhe.piop.chain")
    for part in parts:
        state = hash_bytes(state + part)
    return state


def element_digest(value) -> bytes:
    """A 32-byte digest of a transcript entry or statement field.

    Duck-typed dispatch (this module imports no concrete value types): raw
    bytes, ints, strings and None are primitives; dicts and sequences chain
    their parts; everything else must bring its own digest — a `digest()`
    method, a statement shape (`relation` + `fields`), a dense-MLE shape
    (`table` + `variables`), `to_bytes()` (MerklePath), `get_hash()`
    (arith.Polynomial, four 64-bit words) or a callable `hash` attribute
    (arith.FieldElement). Each branch is domain-separated by a tag.
    """
    if isinstance(value, bytes | bytearray | memoryview):
        return hash_bytes(b"bytes" + bytes(value))
    if isinstance(value, bool | int):
        length = (value.bit_length() + 8) // 8 or 1
        return hash_bytes(b"int" + value.to_bytes(length, "little", signed=True))
    if isinstance(value, str):
        return hash_bytes(b"str" + value.encode())
    if value is None:
        return hash_bytes(b"none")
    if isinstance(value, dict):
        parts = [b"dict"]
        for key, val in value.items():
            parts += [element_digest(key), element_digest(val)]
        return _chain(parts)
    if isinstance(value, list | tuple):
        return _chain([b"seq"] + [element_digest(v) for v in value])
    digest = getattr(value, "digest", None)
    if callable(digest):
        return _as_digest_bytes(digest())
    if hasattr(value, "relation") and hasattr(value, "fields"):  # a Statement
        parts = [b"statement", element_digest(value.relation.name)]
        if value.relation.index is not None:
            parts.append(element_digest(value.relation.index))
        for name in value.relation.fields:
            parts += [element_digest(name), element_digest(value.fields.get(name))]
        return _chain(parts)
    if hasattr(value, "table") and hasattr(value, "variables"):  # a dense MLE
        parts = [b"mle", element_digest(value.basis.name)]
        parts += [element_digest(v) for v in value.variables]
        parts += [element_digest(entry) for entry in value.table]
        return _chain(parts)
    if hasattr(value, "to_bytes"):  # e.g. a MerklePath
        return hash_bytes(b"opaque" + value.to_bytes())
    if hasattr(value, "get_hash"):  # arith.Polynomial: four 64-bit words
        return _as_digest_bytes(value.get_hash())
    leaf_hash = getattr(value, "hash", None)
    if callable(leaf_hash):  # e.g. arith.FieldElement
        return _as_digest_bytes(leaf_hash())
    if hasattr(value, "name"):  # e.g. an MLE_Variable
        return hash_bytes(b"name" + value.name.encode())
    raise TypeError(
        f"cannot digest a {type(value).__name__}; give it a digest() method"
    )


def _as_digest_bytes(value) -> bytes:
    """Normalize a produced hash to bytes (bytes pass, 64-bit words pack)."""
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    return b"".join(int(word).to_bytes(8, "little") for word in value)


class Rejection(Exception):
    """Raised by a protocol's verify half when a round check fails."""


class Relation:
    """An indexed relation R: a set of (index, instance, witness) triples.

    The relation is the unit of protocol design: a protocol step is a
    reduction from statements of one product of relations to statements of
    another (piop.md §5). `index` is the large, reusable, preprocessable
    part of the instance (e.g. a circuit description); None when the
    relation needs none.

    `fields` declares the instance shape: the statement field names, in
    canonical order — the order a Fiat-Shamir serialization will hash, so it
    is declared here rather than left to construction order.

    `check(statement)` is the ideal (non-succinct) membership test for the
    relation's language: it enumerates the hypercube or queries the oracle
    directly. It is what tests run and what the verifier is entitled to do on
    the fully-reduced leaf statements — not the succinct verifier.
    """

    name = "abstract"
    fields: tuple[str, ...] = ()

    def __init__(self, index=None):
        self.index = index

    def check(self, statement: Statement) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index!r})"


class Relation_Sum(Relation):
    """sum_{b in {0,1}^n} f(b) == value, for a single oracle f."""

    name = "sum"
    fields = ("oracles", "value")

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        return _hypercube_sum(f) == _value_of(statement.value)


class Relation_SumProd(Relation):
    """sum_{b in {0,1}^n} prod_j f_j(b) == value, for oracles f_1..f_k."""

    name = "sumprod"
    fields = ("oracles", "value")

    def check(self, statement: Statement) -> bool:
        total = None
        for b in _hypercube(statement.oracles[0].variables):
            prod = None
            for f in statement.oracles:
                e = _constant(f.evaluate(b, in_place=False))
                prod = e if prod is None else prod * e
            total = prod if total is None else total + prod
        return total == _value_of(statement.value)


class Relation_Zero(Relation):
    """f(b) == 0 for every b in {0,1}^n, for a single oracle f."""

    name = "zero"
    fields = ("oracles",)

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        return all(
            _constant(f.evaluate(b, in_place=False)) == 0
            for b in _hypercube(f.variables)
        )


class Relation_Eval(Relation):
    """f(point) == value, for a polynomial f the instance may carry in
    either form — or both:

    - `oracles`: the in-process MLE (the pure-IOP oracle, queried by
      `check`), or
    - `commitment`: its commitment, produced by a PCS's commit algorithm
      possibly long before any IOP run, riding on statements and never on
      the per-execution transcript. The two are the same object at
      different levels of instantiation — "the commitment to f is simply
      the oracle pi_f" [ZCF24, §4] — so a compiled claim is still this
      relation: R_Eval = {[(C, z, y); f] : f(z) = y and C opens to f}
      [ZCF24, Def. 8; BFS20], with f the witness, stored under the
      commitment in `Prover.witnesses`.

    Left terminal, the claim is decided by one oracle query (`check`),
    which needs the oracle: a commitment-only claim has no witness-free
    decider and must be discharged by a registered PCS evaluation protocol
    (e.g. `vfhe.polycom.BasefoldEval`), which resolves the commitment from
    the field or from the scheme's records of the oracle.
    """

    name = "eval"
    fields = ("oracles", "commitment", "point", "value")

    def check(self, statement: Statement) -> bool:
        if not statement.oracles:
            raise NotImplementedError(
                "a commitment-only evaluation claim has no witness-free "
                "decider; register a PCS evaluation protocol for it"
            )
        (f,) = statement.oracles
        e = f.evaluate(dict(statement.point), in_place=False)
        return _constant(e) == _value_of(statement.value)


class Statement:
    """The claim `instance ∈ L(relation)`; the instance fields are declared
    by the relation (`relation.fields`) and passed as keyword arguments.

    Statements are verifier-shaped: every field is public, the witness lives
    with the prover (Prover.witnesses). Field values may hold unresolved
    futures (challenges not drawn yet); `resolved()` awaits them. Protocol
    steps emit the statements this one reduces to, linked through `parents`
    (several, when a reduction consumes a batch) — the statements of one run
    form a DAG whose leaves the verifier decides (piop.md §5). `path` names
    this statement's position in the DAG (root `""`; j-th child of `p` is
    `f"{p}/{j}"`), identical on both sides, and namespaces the transcript.
    """

    def __init__(
        self,
        relation: Relation,
        parents: tuple[Statement, ...] = (),
        path: str = "",
        **fields,
    ):
        self.relation = relation
        self.parents = tuple(parents)
        self.path = path
        unknown = set(fields) - set(relation.fields)
        if unknown:
            raise TypeError(
                f"{type(relation).__name__} has no field(s) {sorted(unknown)}; "
                f"declared fields: {relation.fields}"
            )
        # Canonical (relation-declared) order — what Fiat-Shamir will hash.
        self.fields = {n: fields[n] for n in relation.fields if n in fields}
        self._children = 0  # counter handing out child paths

    def __getattr__(self, name: str):
        d = self.__dict__
        relation = d.get("relation")
        fields = d.get("fields")
        if relation is not None and fields is not None and name in relation.fields:
            return fields.get(name)
        raise AttributeError(
            f"{type(self).__name__} of {type(relation).__name__} has no "
            f"attribute {name!r}"
        )

    @property
    def num_vars(self) -> int:
        oracles = self.fields.get("oracles") or ()
        point = self.fields.get("point") or {}
        return max(
            (o.num_vars for o in oracles if hasattr(o, "num_vars")),
            default=len(point),
        )

    def reduce_to(self, relation: Relation | type[Relation], **fields) -> Statement:
        """Emit a statement this one reduces to, inheriting the oracles."""
        return _reduce((self,), relation, **fields)

    def _fork(self) -> Statement:
        """A per-party view of this statement: same public content, fresh
        child-path counter. `IOP.run` hands each party its own fork of the
        root, so the two derivations assign identical child paths — a shared
        root would hand the second party's children the next counter values,
        desynchronizing the path-namespaced transcript labels."""
        return Statement(
            self.relation, parents=self.parents, path=self.path, **self.fields
        )

    async def resolved(self) -> Statement:
        """Await every pending future in the fields; returns self."""
        for name, val in list(self.fields.items()):
            if isinstance(val, asyncio.Future):
                self.fields[name] = await val
            elif isinstance(val, dict):
                for key, v in list(val.items()):
                    if isinstance(v, asyncio.Future):
                        val[key] = await v
        return self

    def _has_pending(self) -> bool:
        for val in self.fields.values():
            if _pending(val):
                return True
            if isinstance(val, dict) and any(_pending(v) for v in val.values()):
                return True
        return False

    def check(self) -> bool:
        """Decide the claim directly via the relation's ideal decider."""
        if self._has_pending():
            raise RuntimeError("statement has pending futures; await resolved()")
        return self.relation.check(self)

    def __repr__(self) -> str:
        return (
            f"Statement({self.relation.name}, path={self.path!r}, "
            f"vars={self.num_vars}, fields={list(self.fields)})"
        )


def _reduce(
    parents: tuple[Statement, ...], relation: Relation | type[Relation], **fields
) -> Statement:
    """One output statement of a reduction step, child-pathed off parents[0]."""
    if isinstance(relation, type):
        relation = relation()
    first = parents[0]
    if "oracles" in relation.fields and "oracles" not in fields:
        fields["oracles"] = first.fields.get("oracles", ())
    path = f"{first.path}/{first._children}"
    first._children += 1
    return Statement(relation, parents=tuple(parents), path=path, **fields)


class Protocol:
    """A reduction between products of relations (piop.md §5).

    The two halves are coroutines over the same bundle of statements:
    `prove` sends the oracle messages, `verify` checks round consistency,
    and both draw challenges via `iop.verifier.challenge` and return the
    list of statements the bundle reduces to. Messages flow only through
    the IOP transcript (`iop.transcript.write` / `await iop.transcript.read`);
    statements are never sent — each party derives its own DAG from the
    common input and transcript.

    `batching = True` declares a many-to-fewer reduction: the driver hands
    the protocol every frontier statement of `reduces_from` in one
    invocation instead of one at a time.

    `supported_domains` lists the coefficient-domain types (e.g. Ring) for
    which native (C) kernels exist — declarative metadata, not a dispatch
    switch. The native/pure-Python decision is made at the data level,
    inside the round-message helpers (they inspect the oracle
    representation, e.g. `mle.native_table`), so protocol bodies are
    written once and never branch on it; both paths produce identical
    transcripts. An empty tuple means pure Python always.
    """

    reduces_from: type[Relation] = None
    reduces_to: tuple[type[Relation], ...] = ()
    batching: bool = False
    supported_domains: tuple[type, ...] = ()

    def reduce(
        self,
        statements: list[Statement],
        relation: Relation | type[Relation] | None = None,
        **fields,
    ) -> Statement:
        """One output statement of this reduction (default: reduces_to[0])."""
        return _reduce(tuple(statements), relation or self.reduces_to[0], **fields)

    async def prove(
        self, prover: Prover, statements: list[Statement]
    ) -> list[Statement]:
        raise NotImplementedError

    async def verify(
        self, verifier: Verifier, statements: list[Statement]
    ) -> list[Statement]:
        raise NotImplementedError


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
        return await entry

    def _consume(self, label: str) -> None:
        """Take the next message of the replayed proof, which must be the
        one the verifier is asking for."""
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
        return _chain([b"proof"] + [element_digest(m) for m in self.messages])

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


class Party:
    """Common state of the two IOP roles."""

    def __init__(self, iop: IOP | None = None):
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
            stmt = worklist.pop(0)
            protocol = self.iop.protocol_for(stmt.relation)
            if protocol is None:
                terminal.append(stmt)
                continue
            bundle = [stmt]
            if protocol.batching:
                same = [
                    s for s in worklist if type(s.relation) is type(stmt.relation)
                ]
                for s in same:
                    worklist.remove(s)
                bundle += same
            worklist.extend(await getattr(protocol, half)(self, bundle))
        return terminal


class Prover(Party):
    """Computes and sends the oracles; holds the witnesses.

    `witnesses` maps an instance handle to its private data — the witness is
    never a field of the statement itself, so statements stay safe to share.
    Keys are whatever public object the witness belongs to: a statement, or
    a polynomial commitment (the PCS case: `witnesses[commitment] = f`, the
    opening the evaluation protocol proves against — commitments survive
    statement forking and reduction, so they are the stable key).
    """

    def __init__(self, iop: IOP | None = None):
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
        return self.iop.domain.random_exceptional()

    def _draw_bits(self, label: str, nbytes: int) -> bytes:
        """Where a bit-string challenge's value comes from (see above)."""
        return secrets.token_bytes(nbytes)

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
        self.protocols = {}  # type[Relation] -> Protocol
        self.prover = (prover or Prover)(self)
        if verifier is None:
            if fiat_shamir:
                from .fs import FS_Verifier  # deferred: fs.py imports this module

                verifier = FS_Verifier
            else:
                verifier = Verifier
        self.verifier = verifier(self)

    def new_variable(self, name: str) -> Variable:
        return Variable(name, self)

    def register(self, relation_type: type[Relation], protocol: Protocol) -> Protocol:
        """Choose the protocol that discharges statements of `relation_type`.

        Relations without a registered protocol are terminal: the verifier
        decides them directly with the relation's own `check()`.
        """
        self.protocols[relation_type] = protocol
        return protocol

    def protocol_for(self, relation: Relation) -> Protocol | None:
        return self.protocols.get(type(relation))

    def _bind(self, statement: Statement) -> None:
        """Seed the Fiat-Shamir chain with the claim being proven, so every
        derived challenge depends on it (weak-FS, [DMWG23])."""
        if self.fiat_shamir and not self.transcript.seed:
            self.transcript.bind(element_digest(statement))

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
        self._bind(statement)
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
        self._bind(statement)
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
        self._bind(statement)
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
                await asyncio.gather(
                    prover_task, verifier_task, return_exceptions=True
                )

        return self.loop.run_until_complete(_run())
