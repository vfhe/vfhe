# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""IOP scaffolding: relations, statements, parties, and async plumbing.

The design and vocabulary (relation / language / statement / index / oracle)
are derived from the PIOP literature; see ``modules/piop/piop.md`` for the
full derivation and bibliography.
"""

from __future__ import annotations

import asyncio


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
    """f(point) == value: the terminal claim, decided by one oracle query."""

    name = "eval"
    fields = ("oracles", "point", "value")

    def check(self, statement: Statement) -> bool:
        (f,) = statement.oracles
        e = f.evaluate(dict(statement.point), in_place=False)
        return _constant(e) == _value_of(statement.value)


class Relation_Open(Relation):
    """A commitment opens to f with f(point) == value.

    The decider needs a polynomial commitment scheme; declared here so the
    reduction chains can already name it, implemented once one exists.
    """

    name = "open"
    fields = ("commitment", "point", "value")

    def check(self, statement: Statement) -> bool:
        raise NotImplementedError("Relation_Open needs a commitment scheme")


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
    which a native (C) implementation exists. The prove/verify halves choose
    per statement: they delegate to the native path when `native_supported`
    holds — the IOP's domain is one of the supported types and the
    statement's oracles are in the native representation — and fall back to
    the pure-Python path otherwise. An empty tuple means pure Python always.
    """

    reduces_from: type[Relation] = None
    reduces_to: tuple[type[Relation], ...] = ()
    batching: bool = False
    supported_domains: tuple[type, ...] = ()

    def native_supported(self, iop: IOP, statement: Statement) -> bool:
        """Whether a native implementation covers this domain and statement.

        Subclasses extend this with their representation requirements (e.g.
        oracles must be MLE_Dense over the domain).
        """
        return bool(self.supported_domains) and isinstance(
            iop.domain, self.supported_domains
        )

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

    def _entry(self, label: str) -> Variable:
        var = self.entries.get(label)
        if var is None:
            var = self.entries[label] = Variable(label, self.iop)
        return var

    def write(self, label: str, value):
        """Append `value` under `label`; rewriting a label is an error."""
        self._entry(label).set_result(value)
        self.order.append(label)
        return value

    async def read(self, label: str):
        """The value written under `label`, awaiting it if not written yet."""
        return await self._entry(label)


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

    `witnesses` maps a statement to its private data — the witness is never a
    field of the statement itself, so statements stay safe to share.
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
        value = self.iop.domain.random_exceptional()
        self.iop.transcript.write(label, value)
        return value

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
    subclasses to customize them (e.g. a Fiat-Shamir verifier). The
    transcript is shared by both parties: the sender writes, the receiver
    awaits — so rounds that do not depend on a pending message can be
    scheduled without blocking. An IOP object is single-use: one run consumes
    its transcript.
    """

    def __init__(
        self,
        domain=None,
        prover: type[Prover] | None = None,
        verifier: type[Verifier] | None = None,
    ) -> None:
        self.loop = asyncio.new_event_loop()
        self.domain = domain
        self.transcript = Transcript(self)
        self.protocols = {}  # type[Relation] -> Protocol
        self.prover = (prover or Prover)(self)
        self.verifier = (verifier or Verifier)(self)

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

    def run(self, statement: Statement) -> bool:
        """Run both parties concurrently on the statement; the verdict is
        the verifier's output."""

        async def _run():
            prover_task = asyncio.ensure_future(self.prover.prove(statement))
            verifier_task = asyncio.ensure_future(self.verifier.verify(statement))
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
