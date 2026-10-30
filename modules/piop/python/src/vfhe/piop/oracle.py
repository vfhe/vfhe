# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Oracles (piop.md section 4): what a statement or an index names.

A statement never names a table: it names an oracle, an object of this
module, and *what the verifier has* of the oracle (`OracleKind`) decides how
an evaluation claim on it is discharged. The kinds are classes:

- `TableOracle`, an oracle to a table the framework holds (an arith `MLE`
  or `SparseMLE`, untouched): `public` when the verifier holds it too, so
  that a claim on it is terminal; `plain` otherwise, the oracle of the ideal
  model, which the verifier may query while no commitment scheme is
  registered; `EqOracle` is the public one of eq~(point, .), held as its
  point;
- `CommittedOracle`, a commitment, the scheme it was committed under and
  the variables, nothing else: the verifier's view of a committed polynomial
  ("the commitment to f is simply the oracle pi_f" [ZCF24, section 4]);
  the prover's table is in `Prover.witnesses[oracle]`;
- `ImplicitOracle`, a handle with variables and no content of its own, the
  head of a statement of a reduction (`composer.py`);
- `VirtualOracle` (`virtual.py`), a local definition over other oracles.

Every oracle has `variables`, `kind`, `query(point)` -- the ideal decider's
one query, at a full point -- and `digest()`. The prover works on tables:
`virtual.resolve_table(oracle, witnesses)` is the table behind an oracle on
its side.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, ClassVar

from vfhe.arith.mle import MLE, SparseMLE
from vfhe.crypto import chain_digest, hash_bytes

from .digest import element_digest


class OracleKind(Enum):
    """What the verifier has of an oracle, which decides how an evaluation
    claim on it is discharged (piop.md §4):

    - `public`: the table itself (a wiring predicate, eq~, a public output);
      the claim is terminal and the verifier evaluates it.
    - `committed`: a commitment; the evaluation protocol of the scheme it
      was committed under.
    - `virtual`: a local definition over other oracles (`virtual.py`); the
      claim is rewritten into claims on the constituents, free of charge.
    - `implicit`: the head of a statement of a reduction (`ImplicitOracle`),
      a table only the prover has; the Composer consumes the claim at the
      layer of its head, no protocol ever does (`composer.py`).
    - `plain`: an oracle in the ideal model, which the verifier may query;
      admissible only while no commitment scheme is registered.
    """

    public = "public"
    committed = "committed"
    virtual = "virtual"
    implicit = "implicit"
    plain = "plain"


class Oracle:
    """The contract of an oracle: `variables`, its `kind`, `query(point)` at
    a full point (the ideal decider's one query; raises for an oracle the
    verifier has no table of), `dependencies()` (the oracles it is defined
    over; none for a leaf) and `digest()` (what a Fiat-Shamir chain absorbs
    when it sits in a statement). Oracles are compared by identity: claims
    are bundled, parked and stored (`Prover.witnesses`) on the object."""

    kind: ClassVar[OracleKind]
    variables: list

    @property
    def num_vars(self) -> int:
        return len(self.variables)

    @property
    def public(self) -> bool:
        """Whether the verifier holds the oracle's table."""
        return self.kind is OracleKind.public

    def query(self, point: dict) -> Any:
        raise NotImplementedError

    def dependencies(self) -> list:
        return []

    def digest(self) -> bytes:
        raise NotImplementedError


class TableOracle(Oracle):
    """An oracle to a table the framework holds: `polynomial`, a dense `MLE`
    or a `SparseMLE` of `vfhe.arith`, and whether the verifier holds it too.
    Its kind is `public` when it does -- a wiring predicate, an eq~ table, a
    public output: an evaluation claim on it is terminal, decided by the
    verifier's own evaluation -- and `plain` otherwise, the oracle of the
    ideal model. The table is shared, never copied: the prover folds it out
    of place (`resolve_table` returns it as it is)."""

    def __init__(self, polynomial: MLE | SparseMLE, public: bool = False) -> None:
        if not isinstance(polynomial, MLE | SparseMLE):
            raise TypeError("a TableOracle wraps an arith table (MLE or SparseMLE)")
        # Either class; a reader that needs one of them narrows it.
        self.polynomial: Any = polynomial
        self._public = bool(public)

    @property
    def kind(self) -> OracleKind:  # pyright: ignore[reportIncompatibleVariableOverride]
        return OracleKind.public if self._public else OracleKind.plain

    @property
    def variables(self) -> list:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.polynomial.variables

    def query(self, point: dict) -> Any:
        return self.polynomial.evaluate(dict(point), in_place=False).constant()

    def digest(self) -> bytes:
        return element_digest(self.polynomial)

    def __repr__(self) -> str:
        return f"TableOracle(vars={self.num_vars}, kind={self.kind.value})"


class EqOracle(TableOracle):
    """The public oracle to eq~(point, .) over `variables`, the polynomial
    of `MLE.eq`, held as its point: a query is the product
    `prod_i (z_i x_i + (1 - z_i)(1 - x_i))`, linear in the number of
    variables, and the digest is that of the point. The table is built only
    when something reads `polynomial` (a prover that folds it as a table, or
    the ideal decider); a sumcheck prover reads `point` instead, to factor
    the equality polynomial out of its round messages."""

    def __init__(self, domain, point: list, variables: list) -> None:
        if len(point) != len(variables):
            raise ValueError(f"{len(point)} coordinates for {len(variables)} variables")
        self.domain = domain
        self.point = list(point)
        self._variables = list(variables)
        self._public = True
        self._table: MLE | None = None

    @property
    def polynomial(self) -> MLE:  # pyright: ignore[reportIncompatibleVariableOverride]
        """The table of eq~(point, .), built on first use."""
        if self._table is None:
            self._table = MLE.eq(self.domain, self.point, variables=self._variables)
        return self._table

    @property
    def variables(self) -> list:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self._variables

    def query(self, point: dict) -> Any:
        if not self.point:
            return self.polynomial.constant()
        value = None
        for z, v in zip(self.point, self._variables, strict=True):
            x = point[v]
            factor = z * x + (z - 1) * (x - 1)
            value = factor if value is None else value * factor
        return value

    def digest(self) -> bytes:
        parts = [b"eq"]
        parts += [element_digest(v) for v in self._variables]
        parts += [element_digest(z) for z in self.point]
        return chain_digest(parts)

    def __repr__(self) -> str:
        return f"EqOracle(vars={self.num_vars})"


class CommittedOracle(Oracle):
    """An oracle the verifier has as a commitment: `commitment`, produced by
    the commit algorithm of `scheme` -- a polynomial commitment scheme,
    with its public parameters -- possibly long before any IOP run and
    reused across runs (it rides on statements, never on the transcript),
    and the committed `variables`, whose order is the canonical order of
    evaluation points. No table: an evaluation claim on it is discharged by
    the evaluation protocol registered for `Relation_Eval` with that
    `scheme` (`IOP.protocol_for`), and `query` raises. The prover's table
    is stored under the oracle in `Prover.witnesses`, where `resolve_table`
    finds it; the scheme's own prover data (an opening) is the scheme's
    business."""

    kind = OracleKind.committed

    def __init__(self, variables: list, commitment, scheme) -> None:
        self.variables = list(variables)
        self.commitment = commitment
        self.scheme = scheme

    def query(self, point: dict) -> Any:
        raise NotImplementedError(
            "a committed oracle has no table on the verifier's side; register "
            "the scheme's evaluation protocol for Relation_Eval"
        )

    def digest(self) -> bytes:
        """By the commitment, which binds the table, never by entries."""
        return hash_bytes(b"committed" + self.commitment.digest())

    def __repr__(self) -> str:
        return f"CommittedOracle(vars={self.num_vars}, {self.commitment!r})"


class ImplicitOracle(Oracle):
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

    def __init__(self, variables: list) -> None:
        self.variables = list(variables)

    def query(self, point: dict) -> Any:
        raise NotImplementedError(
            "an implicit oracle has no table on the verifier's side"
        )

    def digest(self) -> bytes:
        return chain_digest(
            [b"implicit-handle"] + [element_digest(v) for v in self.variables]
        )

    def __repr__(self) -> str:
        return f"ImplicitOracle(vars={self.num_vars})"


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
        if isinstance(o, Oracle):
            stack.extend(o.dependencies())
    return list(seen.values())


def map_oracles(value, fn: Callable) -> Any:
    """`value` with `fn` applied to every oracle in it: the structural map
    over an index, walking tuples, lists and mappings; anything that is not
    an `Oracle` is data and is left alone. A container none of whose entries
    changed is returned as the same object, so a caller can tell by identity
    whether anything was replaced."""
    if isinstance(value, Mapping):
        items = {k: map_oracles(v, fn) for k, v in value.items()}
        if all(items[k] is v for k, v in value.items()):
            return value
        return items
    if isinstance(value, list | tuple):
        items = [map_oracles(v, fn) for v in value]
        if all(a is b for a, b in zip(items, value, strict=True)):
            return value
        return type(value)(items)
    if isinstance(value, Oracle):
        return fn(value)
    return value
