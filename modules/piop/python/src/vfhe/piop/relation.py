# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Relations and statements (piop.md section 3): the indexed relation, the
statement of a claim, reductions between statements, and the hypercube
helpers the deciders share. The evaluation claim and the start relations of
headed statements are `evaluation.py`.

The design and vocabulary (relation / language / statement / index / oracle)
are derived from the PIOP literature; see ``modules/piop/piop.md`` for the
full derivation and bibliography.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar

from vfhe.arith.mle import vector_table
from vfhe.crypto import chain_digest

from .digest import element_digest
from .oracle import ImplicitOracle, Oracle, TableOracle, map_oracles


def _pending(x) -> bool:
    """True for a future whose value has not been set yet."""
    return isinstance(x, asyncio.Future) and not x.done()


def value_of(x):
    """Unwrap a resolved future to its value; anything else passes through."""
    if isinstance(x, asyncio.Future):
        return x.result()
    return x


def hypercube(variables: list):
    """All points of {0,1}^n as variable -> bit dicts."""
    for bits in range(1 << len(variables)):
        yield {v: (bits >> i) & 1 for i, v in enumerate(variables)}


def aligned_tables(oracles) -> list | None:
    """The oracles' tables when the hypercube sum is whole-vector work, else
    None.

    Every oracle has to be a `TableOracle` to a field-backed table in the
    evaluation basis over the same variables in the same order. Then entry
    `i` of each table is the value at the point `hypercube` yields `i`-th
    -- both are LSB-first over the variable list -- so the tables line up
    index for index and a sum over the hypercube is a `sum()`, a sum of
    products one elementwise product and a `sum()`. Anything else (a ring
    table, the monomial basis, a virtual or committed oracle, oracles over
    different variables) returns None and is walked.
    """
    polynomials = [
        o.polynomial if isinstance(o, TableOracle) else None for o in oracles
    ]
    first = polynomials[0]
    if first is None or not vector_table(first):
        return None
    tables = []
    for f in polynomials:
        # `is not` on the field is deliberately strict: two equal fields that
        # are separate objects only cost the walk, which is always correct.
        if (
            f is None
            or not vector_table(f)
            or f.field is not first.field
            or f.variables != first.variables
        ):
            return None
        tables.append(f.table)
    return tables


def hypercube_sum(f: Oracle):
    """sum_{b in {0,1}^n} f(b), as a plain coefficient value: one query per
    point, or one vector sum when the table allows (`aligned_tables`)."""
    tables = aligned_tables([f])
    if tables is not None:
        return tables[0].sum()
    total = None
    for b in hypercube(f.variables):
        e = f.query(b)
        total = e if total is None else total + e
    return total


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

    `index_fields` declares the index's shape the same way. When it is
    non-empty, `index` is a mapping over exactly those names, stored in the
    declared order; the oracles among its values (`index_oracles`) are
    oracles of every statement of the relation wherever the framework reads
    a statement's oracles (`statement_oracles`), and its digest
    (`index_digest`) is the chain of the entries in the declared order. When
    it is empty, `index` is one opaque object, digested as it is.

    `check(statement)` is the ideal (non-succinct) membership test for the
    relation's language: it enumerates the hypercube or queries the oracle
    directly. It is what tests run and what the verifier is entitled to do on
    the fully-reduced leaf statements — not the succinct verifier.

    A relation may declare that one of its oracles is determined by the
    others: `head_index` names it, either by its position in `oracles` (an
    `int`) or by the name of the index field holding it (a `str`), and
    `compute_head(statement)` computes its table from the other oracles of
    the statement (given concrete tables, the prover view). A statement of
    such a relation may then carry that oracle as its `head` (an `ImplicitOracle`
    handle) -- for an index head the statement derives it from the index
    entry -- and `with_start` derives the companion relation and the
    `start` method the Composer uses. `check` ignores the head: it decides
    the relation on whatever the statement's oracles are.

    A relation with a head must be constructible from its index alone, as
    `type(r)(index=r.index)`: its start relation rebuilds it that way.
    """

    name = "abstract"
    fields: tuple[str, ...] = ()
    index_fields: tuple[str, ...] = ()
    head_index: int | str | None = None
    Start: ClassVar[type[Relation]]  # set by `with_start`
    base: ClassVar[type[Relation]]  # on a start relation: what it derives from
    # On a relation with a dense index (`sparse.dense_relation`): the relation
    # with sparse index oracles it is built on.
    inner: ClassVar[type[Relation]]
    # On a relation whose index holds a family of sparse tables: the names of
    # the index fields holding them.
    sparse_index_fields: tuple[str, ...] = ()

    def __init__(self, index=None):
        if self.index_fields:
            if not isinstance(index, Mapping):
                raise TypeError(
                    f"{type(self).__name__} takes its index as a mapping over "
                    f"{self.index_fields}"
                )
            missing = set(self.index_fields) - set(index)
            unknown = set(index) - set(self.index_fields)
            if missing or unknown:
                raise TypeError(
                    f"{type(self).__name__} index: missing {sorted(missing)}, "
                    f"unknown {sorted(unknown)}; declared fields: "
                    f"{self.index_fields}"
                )
            index = {n: index[n] for n in self.index_fields}
        # Untyped on purpose: an opaque object, or the mapping over
        # `index_fields`; a subclass narrows it where it reads it.
        self.index: Any = index
        self._index_digest: bytes | None = None

    def index_oracles(self) -> tuple:
        """The oracles among the index values, in declared order, walking
        tuples, lists and mappings (an entry may hold a family of tables, or
        a nested index); `()` for an opaque index."""
        if not self.index_fields:
            return ()
        found: list = []

        def collect(oracle):
            found.append(oracle)
            return oracle

        map_oracles(self.index, collect)
        return tuple(found)

    def index_head(self):
        """The index entry a `str` `head_index` names (an `ImplicitOracle` for a
        headed statement, a table otherwise); `None` for a relation whose
        head is an instance oracle or which has none."""
        name = self.head_index
        if not isinstance(name, str):
            return None
        if name not in self.index_fields:
            raise TypeError(
                f"{type(self).__name__}.head_index names {name!r}, which is not "
                f"one of its index fields {self.index_fields}"
            )
        return self.index[name]

    def index_digest(self) -> bytes:
        """The digest of a declared index: the chain of the entries in the
        declared order, each under its name. Cached on the instance, since an
        index is built once and never mutated; a table in the index is
        committed before the index is first digested, so that the cached
        digest records it by its commitment."""
        if not self.index_fields:
            raise TypeError(f"{type(self).__name__} declares no index fields")
        if self._index_digest is None:
            parts = [b"index"]
            for name in self.index_fields:
                parts += [element_digest(name), element_digest(self.index[name])]
            self._index_digest = chain_digest(parts)
        return self._index_digest

    def check(self, statement: Statement) -> bool:
        raise NotImplementedError

    def subject(self, statement: Statement) -> Oracle | None:
        """The oracle a statement of this relation is a claim *about*, when
        it is a claim on one oracle (`Relation_Eval`); `None` otherwise. The
        driver bundles and parks the claims with a subject per oracle, and
        routes them by the oracle's kind (`IOP.protocol_for`); the Composer
        holds those whose subject is a head of its reduction."""
        return None

    def compute_head(self, statement: Statement):
        """The head's table from the statement's other oracles (prover view)."""
        raise NotImplementedError(f"{type(self).__name__} has no head to compute")

    def start(self, statement: Statement, claims: list[Statement]) -> Statement:
        """The started statement of a headed `statement` with the evaluation
        claims on its head; installed by `with_start`."""
        raise NotImplementedError(f"{type(self).__name__} has no start relation")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index!r})"


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
    `head`, structural like `parents` and `path` (not a field, never
    digested), marks the statement as *headed*: the oracle the relation's
    `head_index` names is an `ImplicitOracle` handle whose table the relation
    computes from the others, and a Composer consumes the evaluation claims
    on it at this statement's layer. `None` (the default) is an ordinary
    statement -- except for a relation whose head is an index entry, where
    the head is derived: that entry when it is an `ImplicitOracle`, `None` when it
    is a table, and a `head` that is passed must be that entry.
    """

    def __init__(
        self,
        relation: Relation,
        parents: tuple[Statement, ...] = (),
        path: str = "",
        head: ImplicitOracle | None = None,
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
        for o in self.fields.get("oracles") or ():
            if not isinstance(o, Oracle):
                raise TypeError(
                    f"a statement names oracles, not tables: got {type(o).__name__}; "
                    "wrap a table in TableOracle"
                )
        self.head = _checked_head(relation, self.fields, head)
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
        return max((o.num_vars for o in oracles), default=len(point))

    @property
    def index(self):
        """The relation's index, the preprocessed part of this claim."""
        return self.relation.index

    def reduce_to(
        self,
        relation: Relation | type[Relation],
        head: ImplicitOracle | None = None,
        **fields,
    ) -> Statement:
        """Emit a statement this one reduces to, inheriting the oracles."""
        return child_statement((self,), relation, head=head, **fields)

    def _fork(self) -> Statement:
        """A per-party view of this statement: same public content, fresh
        child-path counter. `IOP.run` hands each party its own fork of the
        root, so the two derivations assign identical child paths — a shared
        root would hand the second party's children the next counter values,
        desynchronizing the path-namespaced transcript labels."""
        return Statement(
            self.relation,
            parents=self.parents,
            path=self.path,
            head=self.head,
            **self.fields,
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

    def digest(self) -> bytes:
        """The digest of the claim: the relation's name, its index (a declared
        one through `Relation.index_digest`, an opaque one as it is) and the
        fields in their declared order, each under its name. Structural
        attributes (`parents`, `path`, `head`) are not part of it."""
        relation = self.relation
        parts = [b"statement", element_digest(relation.name)]
        if relation.index_fields:
            parts.append(relation.index_digest())
        elif relation.index is not None:
            parts.append(element_digest(relation.index))
        for name in relation.fields:
            parts += [element_digest(name), element_digest(self.fields.get(name))]
        return chain_digest(parts)

    def __repr__(self) -> str:
        head = "" if self.head is None else f", head={self.head!r}"
        return (
            f"Statement({self.relation.name}, path={self.path!r}, "
            f"vars={self.num_vars}, fields={list(self.fields)}{head})"
        )


def statement_oracles(statement: Statement) -> tuple:
    """Every oracle a statement mentions: its instance oracles followed by
    the index oracles of its relation. What the oracle closures (the parking
    rule, the Composer's rules) and the prover view read."""
    oracles = tuple(statement.fields.get("oracles") or ())
    return oracles + statement.relation.index_oracles()


def _checked_head(relation: Relation, fields: dict, head) -> ImplicitOracle | None:
    """`head` validated against the relation: None, or the `ImplicitOracle` handle
    the relation's `head_index` names, among the statement's oracles or in
    its index. For an index head `None` means *derived*: the index entry when
    it is an `ImplicitOracle`, else no head."""
    if isinstance(relation.head_index, str):
        entry = relation.index_head()
        if head is None:
            return entry if isinstance(entry, ImplicitOracle) else None
        if not isinstance(head, ImplicitOracle):
            raise TypeError("a head must be an ImplicitOracle oracle")
        if head is not entry:
            raise ValueError(
                f"the head of a {type(relation).__name__} statement is its index "
                f"entry {relation.head_index!r}, which this head is not"
            )
        return head
    if head is None:
        return None
    if relation.head_index is None:
        raise TypeError(f"{type(relation).__name__} declares no head_index")
    if not isinstance(head, ImplicitOracle):
        raise TypeError("a head must be an ImplicitOracle oracle")
    oracles = fields.get("oracles") or ()
    if relation.head_index >= len(oracles) or oracles[relation.head_index] is not head:
        raise ValueError(
            f"the head must be the oracle at position {relation.head_index} "
            "of the statement's oracles"
        )
    return head


def child_statement(
    parents: tuple[Statement, ...],
    relation: Relation | type[Relation],
    head: ImplicitOracle | None = None,
    **fields,
) -> Statement:
    """One output statement of a reduction step, child-pathed off parents[0]."""
    if isinstance(relation, type):
        relation = relation()
    first = parents[0]
    if "oracles" in relation.fields and "oracles" not in fields:
        fields["oracles"] = first.fields.get("oracles", ())
    path = f"{first.path}/{first._children}"
    first._children += 1
    return Statement(relation, parents=tuple(parents), path=path, head=head, **fields)
