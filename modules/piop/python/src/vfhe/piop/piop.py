# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""IOP scaffolding: relations, statements, parties, and async plumbing.

The design and vocabulary (relation / language / statement / index / oracle)
are derived from the PIOP literature; see ``modules/piop/piop.md`` for the
full derivation and bibliography.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, ClassVar, TypeVar, cast

from vfhe.crypto import entropy, hash_bytes


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


def _aligned_tables(oracles) -> list | None:
    """The oracles' tables when the hypercube sum is whole-vector work, else
    None.

    Every oracle has to be a field-backed table in the evaluation basis over
    the same variables in the same order. Then entry `i` of each table is the
    value at the point `_hypercube` yields `i`-th -- both are LSB-first over
    the variable list -- so the tables line up index for index and a sum over
    the hypercube is a `sum()`, a sum of products one elementwise product and
    a `sum()`. Anything else (a ring table, the monomial basis, a virtual
    oracle, oracles over different variables) returns None and is walked.
    """
    # Deferred, as `fs` is below: imported where it is used, not at module load.
    from vfhe.arith.mle import vector_table

    first = oracles[0]
    if not vector_table(first):
        return None
    tables = []
    for f in oracles:
        # `is not` on the field is deliberately strict: two equal fields that
        # are separate objects only cost the walk, which is always correct.
        if (
            not vector_table(f)
            or f.field is not first.field
            or f.variables != first.variables
        ):
            return None
        tables.append(f.table)
    return tables


def _hypercube_sum(f):
    """sum_{b in {0,1}^n} f(b), as a plain coefficient value."""
    tables = _aligned_tables([f])
    if tables is not None:
        return tables[0].sum()
    total = None
    for b in _hypercube(f.variables):
        e = _constant(f.evaluate(b, in_place=False))
        total = e if total is None else total + e
    return total


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

    Duck-typed dispatch (this module imports no concrete value types): a
    class (a relation named in an index) is digested by its module and
    qualified name; raw bytes, ints, strings and None are primitives; dicts
    and sequences chain their parts; everything else must bring its own
    digest — a `digest()` method, a statement shape (`relation` + `fields`;
    a declared index through `Relation.index_digest`), a table tagged with a
    `commitment` (digested by that commitment, never by its entries), a
    dense-MLE shape (`table` + `variables`), a sparse-MLE shape
    (`evaluations` + `variables`, entries in index order), `to_bytes()`
    (MerklePath), `get_hash()` (arith.Polynomial, four 64-bit words) or a
    callable `hash` attribute (arith.FieldElement). Each branch is
    domain-separated by a tag.
    """
    if isinstance(value, type):
        name = f"{value.__module__}.{value.__qualname__}"
        return hash_bytes(b"class" + name.encode())
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
        relation = value.relation
        parts = [b"statement", element_digest(relation.name)]
        if relation.index_fields:
            parts.append(relation.index_digest())
        elif relation.index is not None:
            parts.append(element_digest(relation.index))
        for name in relation.fields:
            parts += [element_digest(name), element_digest(value.fields.get(name))]
        return _chain(parts)
    if hasattr(value, "table") and getattr(value, "commitment", None) is not None:
        # A committed table: the scheme tagged it with its commitment, which
        # binds the entries, so the digest never walks them.
        return hash_bytes(b"committed" + _as_digest_bytes(value.commitment.digest()))
    if hasattr(value, "table") and hasattr(value, "variables"):  # a dense MLE
        parts = [b"mle", element_digest(value.basis.name)]
        parts += [element_digest(v) for v in value.variables]
        parts += [element_digest(entry) for entry in value.table]
        return _chain(parts)
    if hasattr(value, "evaluations") and hasattr(value, "variables"):  # a SparseMLE
        parts = [b"sparse"] + [element_digest(v) for v in value.variables]
        for k in sorted(value.evaluations):  # canonical order, not insertion order
            parts += [element_digest(k), element_digest(value.evaluations[k])]
        return _chain(parts)
    if hasattr(value, "to_bytes"):  # e.g. a MerklePath
        return hash_bytes(b"opaque" + value.to_bytes())
    if hasattr(value, "SerializeToString"):  # a protobuf message, e.g. a circuit
        return hash_bytes(b"proto" + value.SerializeToString())
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


class OracleKind(Enum):
    """What the verifier has of an oracle, which decides how an evaluation
    claim on it is discharged (piop.md §4):

    - `public`: the table itself (a wiring predicate, eq~, a public output);
      the claim is terminal and the verifier evaluates it.
    - `committed`: a commitment; a polynomial commitment scheme's protocol.
    - `virtual`: a local definition over other oracles (`virtual.py`); the
      claim is rewritten into claims on the constituents, free of charge.
    - `implicit`: the head of a statement of a reduction (`Implicit`), a
      table only the prover has; the Composer consumes the claim at the
      layer of its head, no protocol ever does (`composer.py`).
    - `plain`: an oracle in the ideal model, which the verifier may query;
      admissible only while no commitment scheme is registered.
    """

    public = "public"
    committed = "committed"
    virtual = "virtual"
    implicit = "implicit"
    plain = "plain"


def oracle_kind(oracle, commitment=None) -> OracleKind:
    """The kind of `oracle` (duck-typed: defined oracles carry `kind`, tables
    carry `public`, and a committed table the `kind` and `commitment` the
    scheme set on it); a claim carrying a commitment is `committed`. A table
    is `public` or `committed`, never both: the two are two discharges of a
    claim on it, so a table carrying both marks is refused."""
    if commitment is not None:
        return OracleKind.committed
    kind = getattr(oracle, "kind", None)
    if isinstance(kind, OracleKind):
        if kind is OracleKind.committed and getattr(oracle, "public", False):
            raise ValueError(
                "a table marked public was committed; a table is public or "
                "committed, not both"
            )
        return kind
    if getattr(oracle, "public", False):
        return OracleKind.public
    return OracleKind.plain


def oracle_closure(oracles) -> list:
    """Every oracle a claim on one of `oracles` can reach: the oracles
    themselves and, through `dependencies()`, the constituents of the
    defined ones, recursively. Identity-based; definitions are static and
    public, so both parties compute the same closure."""
    seen: dict[int, object] = {}
    stack = list(oracles)
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen[id(o)] = o
        deps: Any = getattr(o, "dependencies", None)
        if callable(deps):
            stack.extend(cast("list", deps()))
    return list(seen.values())


class Implicit:
    """An implicit oracle: a handle with variables and no content of its own.

    It is the *head* of exactly one statement of a reduction (`Statement.head`),
    and its table is whatever that statement's relation computes from the
    statement's other oracles (`Relation.compute_head`). On the prover's side
    the table lives in `Prover.witnesses` once computed; the verifier has
    nothing of it, and never queries it: an evaluation claim on it is held by
    the Composer of the reduction until the layer of its head, never routed to
    a protocol (`IOP.protocol_for` refuses one that gets through). `kind`
    exists only for that refusal; `digest()` lets a statement carrying the
    handle seed a Fiat-Shamir chain in tests, the handle's identity being its
    variables.
    """

    kind = OracleKind.implicit
    public = False

    def __init__(self, variables: list) -> None:
        self.variables = list(variables)

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    def digest(self) -> bytes:
        return _chain(
            [b"implicit-handle"] + [element_digest(v) for v in self.variables]
        )

    def __repr__(self) -> str:
        return f"Implicit(vars={self.num_vars})"


def _map_oracles(value, fn: Callable) -> Any:
    """`value` with `fn` applied to every oracle in it: the structural map
    over an index, walking tuples, lists and mappings and duck-typing an
    oracle on a `variables` attribute (a table, a defined oracle, an
    `Implicit` handle); anything else is data and is left alone. A container
    none of whose entries changed is returned as the same object, so a
    caller can tell by identity whether anything was replaced."""
    if isinstance(value, Mapping):
        items = {k: _map_oracles(v, fn) for k, v in value.items()}
        if all(items[k] is v for k, v in value.items()):
            return value
        return items
    if isinstance(value, list | tuple):
        items = [_map_oracles(v, fn) for v in value]
        if all(a is b for a, b in zip(items, value, strict=True)):
            return value
        return type(value)(items)
    if hasattr(value, "variables"):
        return fn(value)
    return value


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
    such a relation may then carry that oracle as its `head` (an `Implicit`
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

        _map_oracles(self.index, collect)
        return tuple(found)

    def index_head(self):
        """The index entry a `str` `head_index` names (an `Implicit` for a
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
            self._index_digest = _chain(parts)
        return self._index_digest

    def check(self, statement: Statement) -> bool:
        raise NotImplementedError

    def compute_head(self, statement: Statement):
        """The head's table from the statement's other oracles (prover view)."""
        raise NotImplementedError(f"{type(self).__name__} has no head to compute")

    def start(self, statement: Statement, claims: list[Statement]) -> Statement:
        """The started statement of a headed `statement` with the evaluation
        claims on its head; installed by `with_start`."""
        raise NotImplementedError(f"{type(self).__name__} has no start relation")

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
        tables = _aligned_tables(statement.oracles)
        if tables is not None:
            # One product vector, then folded in place: the first multiply
            # allocates it, the rest write through it. Never into `tables[0]`,
            # which belongs to the oracle.
            product = tables[0]
            if len(tables) > 1:
                product = product * tables[1]
                for table in tables[2:]:
                    product.mul(table, out=product)
            return product.sum() == _value_of(statement.value)

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

    @staticmethod
    def kind(statement: Statement) -> OracleKind:
        """The kind of the oracle this claim is about (piop.md §4)."""
        oracles = statement.oracles or ()
        return oracle_kind(oracles[0] if oracles else None, statement.commitment)

    @staticmethod
    def subject(statement: Statement):
        """The public handle claims are bundled on: the oracle, else the
        commitment of an oracle-less claim."""
        return statement.oracles[0] if statement.oracles else statement.commitment

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
    `head`, structural like `parents` and `path` (not a field, never
    digested), marks the statement as *headed*: the oracle the relation's
    `head_index` names is an `Implicit` handle whose table the relation
    computes from the others, and a Composer consumes the evaluation claims
    on it at this statement's layer. `None` (the default) is an ordinary
    statement -- except for a relation whose head is an index entry, where
    the head is derived: that entry when it is an `Implicit`, `None` when it
    is a table, and a `head` that is passed must be that entry.
    """

    def __init__(
        self,
        relation: Relation,
        parents: tuple[Statement, ...] = (),
        path: str = "",
        head: Implicit | None = None,
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
        return max(
            (o.num_vars for o in oracles if hasattr(o, "num_vars")),
            default=len(point),
        )

    @property
    def index(self):
        """The relation's index, the preprocessed part of this claim."""
        return self.relation.index

    def reduce_to(
        self,
        relation: Relation | type[Relation],
        head: Implicit | None = None,
        **fields,
    ) -> Statement:
        """Emit a statement this one reduces to, inheriting the oracles."""
        return _reduce((self,), relation, head=head, **fields)

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


def _checked_head(relation: Relation, fields: dict, head) -> Implicit | None:
    """`head` validated against the relation: None, or the `Implicit` handle
    the relation's `head_index` names, among the statement's oracles or in
    its index. For an index head `None` means *derived*: the index entry when
    it is an `Implicit`, else no head."""
    if isinstance(relation.head_index, str):
        entry = relation.index_head()
        if head is None:
            return entry if isinstance(entry, Implicit) else None
        if not isinstance(head, Implicit):
            raise TypeError("a head must be an Implicit oracle")
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
    if not isinstance(head, Implicit):
        raise TypeError("a head must be an Implicit oracle")
    oracles = fields.get("oracles") or ()
    if relation.head_index >= len(oracles) or oracles[relation.head_index] is not head:
        raise ValueError(
            f"the head must be the oracle at position {relation.head_index} "
            "of the statement's oracles"
        )
    return head


def _reduce(
    parents: tuple[Statement, ...],
    relation: Relation | type[Relation],
    head: Implicit | None = None,
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


def _with_head_table(statement: Statement, base: Relation, table) -> Statement:
    """`statement` as `base`'s ideal decider sees it once the head is known:
    the base relation's fields, with `table` in the head's place -- among the
    oracles, or in the index entry, the relation then rebuilt over that
    index -- and no head (every oracle is a table now)."""
    fields = {n: statement.fields[n] for n in base.fields if n in statement.fields}
    if isinstance(base.head_index, str):
        base = type(base)(index={**base.index, base.head_index: table})
        return Statement(base, **fields)
    assert base.head_index is not None
    oracles = list(statement.oracles)
    oracles[base.head_index] = table
    fields["oracles"] = tuple(oracles)
    return Statement(base, **fields)


_R = TypeVar("_R", bound=Relation)


def with_start(cls: type[_R]) -> type[_R]:
    """Class decorator: derive the *start relation* of a relation with a head.

    For `R` with head `u` and other oracles `dep`, `R.Start` is the relation

        R_start(dep; (r_j, y_j)_j)  :=  R(F(dep); dep)  and  F(dep)~(r_j) = y_j
                                        for every j,  F = R.compute_head,

    that is `R` with its head *bound* to the oracle it defines, plus
    evaluation claims on that oracle (k >= 1 of them) at points the statement
    carries as data. Its statements are `R`'s with two more fields, `points`
    and `values`; the oracles, the index and the head are the same (a head
    named by an index field stays in the index, so a started statement of a
    relation with no instance fields is nothing but the index and the
    claims). The decider is
    generic: materialize the head with `compute_head`, decide `R` with that
    table in the head's place (the residual), then one `Relation_Eval` check
    per claim. The decorator also installs `R.start(statement, claims)`, which
    builds the started statement from a headed statement and the `Relation_Eval`
    claims on its head (the Composer calls it once per head, at its layer); the
    started statement is a child of the statement, with the claims as further
    parents. A sequentially composable PIOP for `R` is a PIOP for `R.Start`,
    registered against that class.
    """
    if cls.head_index is None:
        raise TypeError(f"{cls.__name__} declares no head_index; nothing to start")
    if isinstance(cls.head_index, str) and cls.head_index not in cls.index_fields:
        raise TypeError(
            f"{cls.__name__}.head_index names {cls.head_index!r}, which is not "
            f"one of its index fields {cls.index_fields}"
        )
    base_cls = cls

    class Start(Relation):
        name = base_cls.name + "_start"
        fields = (*base_cls.fields, "points", "values")
        index_fields = base_cls.index_fields
        head_index = base_cls.head_index
        base = base_cls

        def compute_head(self, statement: Statement):
            return base_cls(index=self.index).compute_head(statement)

        def check(self, statement: Statement) -> bool:
            points, values = statement.points, statement.values
            if not points or len(points) != len(values):  # k >= 1, paired
                return False
            base = base_cls(index=self.index)
            head = base.compute_head(statement)
            if not _with_head_table(statement, base, head).check():
                return False
            return all(
                Statement(
                    Relation_Eval(), oracles=(head,), point=dict(p), value=v
                ).check()
                for p, v in zip(points, values, strict=True)
            )

    Start.__name__ = base_cls.__name__ + ".Start"
    Start.__qualname__ = base_cls.__qualname__ + ".Start"
    Start.__doc__ = f"The start relation of {base_cls.__name__} (`with_start`)."
    if hasattr(base_cls, "inner"):  # a dense-index relation: its layering reads it
        Start.inner = base_cls.inner

    def start(self, statement: Statement, claims: list[Statement]) -> Statement:
        """The started statement: `statement` (headed) with the evaluation
        claims on its head, as a statement of `Start`."""
        if not isinstance(statement.relation, base_cls):
            raise TypeError(f"not a statement of {base_cls.__name__}")
        if not claims:
            raise ValueError("a started statement carries at least one claim")
        if statement.head is None:
            raise ValueError("only a headed statement can be started")
        for c in claims:
            if not isinstance(c.relation, Relation_Eval):
                raise TypeError("the claims on a head are Relation_Eval statements")
            if Relation_Eval.subject(c) is not statement.head:
                raise ValueError("a claim is not on this statement's head")
        return _reduce(
            (statement, *claims),
            Start(index=self.index),
            head=statement.head,
            **statement.fields,
            points=[c.point for c in claims],
            values=[c.value for c in claims],
        )

    cls.Start = Start
    cls.start = start
    return cls


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
    relation for most relations; for `Relation_Eval`, every claim on one
    oracle, *after* every other frontier statement that could still emit a
    claim on that oracle has been discharged (the parking rule, piop.md §5).

    `supported_domains` lists the coefficient-domain types (e.g. Ring) for
    which native (C) kernels exist — declarative metadata, not a dispatch
    switch. The native/pure-Python decision is made at the data level,
    inside the round-message helpers (they inspect the oracle
    representation, e.g. `mle.native_table`), so protocol bodies are
    written once and never branch on it; both paths produce identical
    transcripts. An empty tuple means pure Python always.
    """

    reduces_from: type[Relation] | None = None
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
        assert self.iop is not None
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
        type and, for evaluation claims, the same oracle (or commitment)."""
        if isinstance(stmt.relation, Relation_Eval):
            return (type(stmt.relation), id(Relation_Eval.subject(stmt)))
        return (type(stmt.relation),)

    def _parked(self, stmt: Statement, worklist: list[Statement]) -> bool:
        """The parking rule (piop.md §5): an evaluation claim headed for a
        bundle protocol waits while any other frontier statement, outside
        its own bundle, can still produce a claim on the same oracle — i.e.
        has that oracle in its oracle closure. Definitions form a DAG and
        non-Eval statements never park, so some statement is always ready.
        Claims on implicit oracles never reach the driver (the Composer
        consumes them), so this concerns committed and plain oracles.
        """
        assert self.iop is not None
        if not isinstance(stmt.relation, Relation_Eval):
            return False
        protocol = self.iop.protocol_for(stmt)
        if protocol is None or not protocol.batching:
            return False
        subject = Relation_Eval.subject(stmt)
        key = self._bundle_key(stmt)
        for other in worklist:
            if other is stmt or self._bundle_key(other) == key:
                continue
            if any(o is subject for o in oracle_closure(statement_oracles(other))):
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
        assert self.iop is not None
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
        assert self.iop is not None
        entry = self.iop.transcript.entries.get(label)
        if entry is not None and entry.done():
            return entry.result()
        value = self._draw_bits(label, (bits + 7) // 8)
        self.iop.transcript.write(label, value, derived=True)
        return value

    def _draw_challenge(self, label: str):
        """Where a domain challenge's value comes from: fresh randomness
        here; the transcript chain in the Fiat-Shamir subtype (fs.py)."""
        assert self.iop is not None
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
        self.protocols = {}  # (type[Relation], OracleKind | None) -> Protocol
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

    def register(
        self,
        relation_type: type[Relation],
        protocol: Protocol,
        kind: OracleKind | None = None,
    ) -> Protocol:
        """Choose the protocol that discharges statements of `relation_type`.

        Relations without a registered protocol are terminal: the verifier
        decides them directly with the relation's own `check()`.

        `Relation_Eval` is one relation with several discharges, keyed by
        the kind of oracle the claim is about (`OracleKind`): register the
        PCS protocol with no kind (the default, reached by committed and
        plain oracles) and `VirtualEval` for `OracleKind.virtual`. A kind
        without its own entry falls back to the default; claims on public
        oracles never reach a protocol, and claims on implicit oracles never
        reach the driver (the Composer consumes them).
        """
        if kind is OracleKind.implicit:
            raise ValueError("no protocol discharges claims on implicit oracles")
        self.protocols[(relation_type, kind)] = protocol
        return protocol

    def protocol_for(self, statement: Statement) -> Protocol | None:
        """The protocol discharging `statement`, None if it is terminal."""
        relation = statement.relation
        if isinstance(relation, Relation_Eval):
            kind = Relation_Eval.kind(statement)
            if kind is OracleKind.implicit:
                raise RuntimeError(
                    "an evaluation claim on an implicit oracle reached the driver; "
                    "claims on a head are consumed by the Composer of the "
                    "reduction that heads it"
                )
            if kind is OracleKind.public:
                return None
            exact = self.protocols.get((type(relation), kind))
            if exact is not None:
                return exact
        return self.protocols.get((type(relation), None))

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
            if isinstance(o, Implicit) and o is not statement.head:
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
