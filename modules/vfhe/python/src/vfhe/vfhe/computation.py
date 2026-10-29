# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The computation a trace recorded, read into its steps in vfhe's own
terms, with the recorded values.

A multiplication is a ``MLWE_Scheme.multiply`` operation with a key: the
tensor product of the two operands, the key switch (the digits of the
quadratic slot, their products with the key's components, the division by
the special primes when the key has them, and the slot added to each
component), and the rescales applied to the output afterwards. A rescale
is a round division of a ciphertext component by the level's prime.

>>> from vfhe.vfhe.computation import Computation
>>> computation = Computation(t.operations())
>>> [len(m.key_switch.digits) for m in computation.multiplications]
[3, 3]

What an arithmetization makes of these steps, which values it names and
the relations it asks of them, is `vfhe.vfhe.witness`'s. Rank 1 only: a
product of another rank is refused, one without a key is left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any

from vfhe.arith import Polynomial

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vfhe.util.trace import Event, Operation

__all__ = ["Computation", "Division", "KeySwitch", "Multiplication"]

#: The operation a multiplication is read from.
PRODUCT = "MLWE_Scheme.multiply"


@dataclass
class Division:
    """A rounded division of ``x`` by ``primes``: ``quotient`` is the value
    on the tape, over the ring without those primes, and ``remainder`` what
    the quotient leaves, ``x - quotient * P`` computed in that ring, where it
    is exact. ``exact`` tells a rescale, which rounds exactly, from the
    division by a key's special primes, which goes through the fast base
    conversion and may leave a remainder of several multiples of ``P``."""

    event: Event
    x: Any
    primes: tuple[int, ...]
    quotient: Any
    remainder: Any
    exact: bool

    @classmethod
    def of(cls, event: Event, x: Any, exact: bool) -> Division:
        primes = tuple(p for p in x.ring.primes if p not in event.ring.primes)
        P = prod(primes)
        quotient = event.value
        remainder = x.mod_reduce(quotient.ring) - quotient * [
            P % p for p in quotient.ring.primes
        ]
        return cls(event, x, primes, quotient, remainder, exact)

    @property
    def modulus(self) -> int:
        """``P``, the product of the primes divided by."""
        return prod(self.primes)


@dataclass
class KeySwitch:
    """The key switch of a multiplication: ``digits[t]`` is the digit of
    ``quadratic`` at ``primes[t]``, ``key[t][i]`` the key component it
    multiplies into accumulator ``i``, ``sums[i]`` the accumulated products
    (subtracted, as vfhe computes them), ``mod_down[i]`` their division by
    the special primes (none for a key without them), ``slots[i]`` the
    tensor slot added to component ``i``, ``output[i]`` the result and
    ``output_events[i]`` the event that wrote it."""

    quadratic: Any
    digits: list[Any]
    primes: list[int]
    key: list[list[Any]]
    sums: list[Any]
    mod_down: list[Division]
    slots: list[Any]
    output: list[Any]
    output_events: list[Event]


@dataclass
class Multiplication:
    """One multiplication: ``operands[k]`` is operand ``k`` as
    ``[mask, body]``; ``constant``, ``linear`` and ``quadratic`` the tensor
    slots by their degree in the secret; ``rescales`` the round divisions of
    the output, in tape order, when the scheme rescaled it."""

    operation: Operation
    operands: list[list[Any]]
    constant: Any
    linear: Any
    quadratic: Any
    key_switch: KeySwitch
    rescales: list[Division] = field(default_factory=list)


class Computation:
    """The multiplications and the rescales among ``operations``, the
    outermost operations of a trace. ``values`` holds every recorded value
    by sequence number, zeros and copies materialized; ``origin`` names, for
    each event, the event whose value it holds (copies collapsed).

    A tape read in pieces refers back: a value recorded in an earlier piece
    (a key, seen once) is not recorded again, so the reader of a later piece
    is given the earlier values as ``known``, the ``values`` of the earlier
    reader or the part of them it kept.
    """

    def __init__(
        self, operations: Iterable[Operation], known: Mapping[int, Any] | None = None
    ) -> None:
        self.operations = list(operations)
        events = sorted(
            (e for op in self.operations for e in op.all_events() if e.kind != "mark"),
            key=lambda e: e.sequence,
        )
        self.values: dict[int, Any] = dict(known or {})
        self.origin: dict[int, int] = {s: s for s in self.values}
        for e in events:
            self.origin[e.sequence] = e.sequence
            if e.kind == "copy":
                self.origin[e.sequence] = self.origin[e.inputs[0]]
                self.values[e.sequence] = self.value(e.inputs[0])
            elif e.kind == "zero":
                self.values[e.sequence] = Polynomial(e.ring).from_array([0])
            elif e.value is not None:
                self.values[e.sequence] = e.value
        self.multiplications: list[Multiplication] = []
        for op in self.operations:
            for inner in op.walk():
                if inner.name == PRODUCT:
                    multiplication = self._multiplication(inner)
                    if multiplication is not None:
                        self.multiplications.append(multiplication)
        #: Every rescale on the tape, whether or not a multiplication owns it.
        self.rescales = [
            Division.of(e, self.value(e.inputs[0]), exact=True)
            for e in events
            if e.kind == "round_division"
        ]
        written_by = {
            self.origin[e.sequence]: m
            for m in self.multiplications
            for e in m.key_switch.output_events
        }
        for rescale in self.rescales:
            owner = written_by.get(self.origin[rescale.event.inputs[0]])
            if owner is not None:
                owner.rescales.append(rescale)

    def value(self, sequence: int) -> Any:
        """The value of version ``sequence``."""
        try:
            return self.values[sequence]
        except KeyError:
            raise ValueError(
                f"version {sequence} was recorded before these operations; "
                "pass the earlier values as known"
            ) from None

    def _multiplication(self, operation: Operation) -> Multiplication | None:
        events = [e for e in operation.all_events() if e.kind != "mark"]
        by_kind: dict[str, list[Event]] = {}
        for e in events:
            by_kind.setdefault(e.kind, []).append(e)
        digits = by_kind.get("decompose_digit", [])
        if not digits:
            return None  # no key: the extended product is handed back
        decomposed = {self.origin[e.inputs[0]] for e in digits}
        accumulators = by_kind.get("zero", [])
        products = by_kind.get("mul", [])
        if len(decomposed) != 1 or len(accumulators) != 2 or len(products) != 3:
            raise ValueError("only products of rank 1 are read")
        values = self.value
        # The quadratic slot is the one the digits decompose, the linear slot
        # the accumulated cross products, the constant slot the remaining
        # product. The partial product of the linear slot pairs the first
        # mask with the other operand's body, which tells the operands apart.
        (quadratic_sequence,) = decomposed
        (linear_event,) = by_kind["mul_addto"]
        (quadratic_event,) = [e for e in products if e.sequence == quadratic_sequence]
        (partial,) = [e for e in products if e.sequence == linear_event.inputs[0]]
        (constant_event,) = [e for e in products if e not in (quadratic_event, partial)]
        masks, bodies = quadratic_event.inputs, constant_event.inputs
        p, q = partial.inputs
        a0, b1 = (p, q) if p in masks else (q, p)
        a1 = masks[1] if masks[0] == a0 else masks[0]
        b0 = bodies[1] if bodies[0] == b1 else bodies[0]
        slots = {
            quadratic_sequence: values(quadratic_sequence),
            linear_event.sequence: linear_event.value,
            constant_event.sequence: constant_event.value,
        }
        # Each accumulator's chain, digit by digit, then its division by the
        # special primes and the addition of its slot.
        chain = {e.sequence: i for i, e in enumerate(accumulators)}
        digit_index = {e.sequence: t for t, e in enumerate(digits)}
        key: list[list[Any]] = [[None, None] for _ in digits]
        sums: dict[int, Any] = {}
        mod_down: dict[int, Division] = {}
        added: dict[int, Any] = {}
        output: dict[int, Event] = {}
        for e in by_kind.get("mul_subto", []):
            i = chain[e.inputs[0]]
            chain[e.sequence] = i
            key[digit_index[self.origin[e.inputs[1]]]][i] = values(e.inputs[2])
            sums[i] = e.value
        for e in by_kind.get("round_division_approximate", []):
            i = chain[e.inputs[0]]
            chain[e.sequence] = i
            mod_down[i] = Division.of(e, values(e.inputs[0]), exact=False)
        for e in by_kind.get("add", []):
            summed = [self.origin[s] for s in e.inputs]
            (i,) = [chain[s] for s in summed if s in chain]
            (added[i],) = [slots[s] for s in summed if s in slots]
            output[i] = e
        key_switch = KeySwitch(
            quadratic=values(quadratic_sequence),
            digits=[e.value for e in digits],
            primes=[e.primes[0] for e in digits],
            key=key,
            sums=[sums[i] for i in range(2)],
            mod_down=[mod_down[i] for i in range(2) if i in mod_down],
            slots=[added[i] for i in range(2)],
            output=[output[i].value for i in range(2)],
            output_events=[output[i] for i in range(2)],
        )
        return Multiplication(
            operation,
            [[values(a0), values(b0)], [values(a1), values(b1)]],
            constant_event.value,
            linear_event.value,
            values(quadratic_sequence),
            key_switch,
        )
