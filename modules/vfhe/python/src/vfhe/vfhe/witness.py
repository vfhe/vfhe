# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The witness of a traced computation under an arithmetization.

An arithmetization turns a computation into a relation: it decides which
recorded values are the instance (public: the operands, the keys), which
are the witness (the prover's values, each with the bound a range check
asks for) and which are intermediate (determined by the others), names
them, and states the relations between them. `of` reads the computation
from a trace's operations and applies one:

>>> from vfhe.util import trace
>>> from vfhe.vfhe import cccfgs26, witness
>>> with trace.recording() as t:
...     d = (a * b) * (a * b)
>>> w = witness.of(t.operations(), cccfgs26)
>>> [e.value for e in w.query("d2")][0] == a.get_a_poly(0) * b.get_a_poly(0)
True
>>> w.check()

`check` is the ideal decider: it evaluates every bound and every relation
on the recorded values with ring arithmetic and raises `ValueError` naming
the first that fails. ``view`` is the arith parent the values are given
in; the only view today is the ring each value was recorded in, or a
quotient ring of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from vfhe.vfhe.computation import Computation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from vfhe.util.trace import Operation

__all__ = ["Arithmetization", "Entry", "Relation", "Witness", "of"]

Role = Literal["instance", "witness", "intermediate"]


@dataclass
class Entry:
    """One named value of the relation: ``step`` is the computation step it
    belongs to, ``index`` tells entries of one name apart within a step (a
    digit's position, a component), ``bound`` the infinity norm a range
    check asks of a witness, None when none is asked."""

    name: str
    role: Role
    value: Any
    step: Any
    index: int | None = None
    bound: float | None = None


@dataclass
class Relation:
    """One relation of a step: ``holds`` evaluates it on the recorded
    values."""

    name: str
    holds: Callable[[], bool]


class Arithmetization(Protocol):
    """What an arithmetization provides for each step of a computation: a
    multiplication, or a rescale no multiplication owns."""

    def entries(self, step: Any) -> list[Entry]: ...

    def relations(self, step: Any) -> list[Relation]: ...


def _largest(value: Any) -> int:
    """The infinity norm of the centered coefficients."""
    return max(abs(c) for c in value.get_polynomial(signed=True))


class Witness:
    """The entries and relations of every step of ``computation`` under
    ``arithmetization``, in tape order."""

    def __init__(
        self,
        computation: Computation,
        arithmetization: Arithmetization,
        view: Any = None,
    ) -> None:
        self.computation = computation
        self.arithmetization = arithmetization
        self.view = view
        owned = {id(r) for m in computation.multiplications for r in m.rescales}
        self.steps: list[Any] = list(computation.multiplications) + [
            r for r in computation.rescales if id(r) not in owned
        ]
        self.entries: list[Entry] = []
        self.relations: list[Relation] = []
        for step in self.steps:
            self.entries += arithmetization.entries(step)
            self.relations += arithmetization.relations(step)

    def query(self, name: str) -> list[Entry]:
        """The entries named ``name``, in tape order, their values in the
        view."""
        found = [e for e in self.entries if e.name == name]
        if not found:
            raise ValueError(f"no entry is named {name!r}")
        if self.view is None:
            return found
        return [
            Entry(e.name, e.role, self._viewed(e.value), e.step, e.index, e.bound)
            for e in found
        ]

    def check(self) -> None:
        """Every bound and every relation, on the recorded values."""
        for e in self.entries:
            if e.bound is not None and _largest(e.value) > e.bound:
                raise ValueError(f"{e.name} exceeds its bound")
        for relation in self.relations:
            if not relation.holds():
                raise ValueError(f"{relation.name} does not hold")

    def _viewed(self, value: Any) -> Any:
        if self.view == value.ring:
            return value
        if self.view.is_quotient_ring(value.ring):
            return value.mod_reduce(self.view)
        raise ValueError("the view must be the value's ring or a quotient of it")


def of(
    operations: Iterable[Operation],
    arithmetization: Arithmetization,
    view: Any = None,
    known: Mapping[int, Any] | None = None,
) -> Witness:
    """The witness of the computation ``operations`` recorded; ``known`` are
    the values of an earlier piece of the tape it refers back to."""
    return Witness(Computation(operations, known), arithmetization, view)
