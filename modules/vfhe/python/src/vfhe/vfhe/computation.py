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

A plaintext-linear step is a ciphertext computed as
``sum_j sign_j * plaintext_j * ciphertext_j``, read from products by a
plaintext (``MLWE.multiply_poly``), linear combinations
(``MLWE_Scheme.linear_combinations``, with or without a `PlaintextMatrix`)
and additions and subtractions of ciphertexts; a ciphertext added as is
is a term without a plaintext. Additions are followed back: the terms of
a step are ciphertexts that no plaintext-linear step wrote, and a sum
that is only added into other sums is not a step of its own.

Each step carries its ``layer``, the number of multiplications on the
longest chain from the computation's inputs to the step's operands, and
its ``ell``, the number of primes of the ring it computes over.

>>> from vfhe.vfhe.computation import Computation
>>> computation = Computation(t.operations())
>>> [len(m.key_switch.digits) for m in computation.multiplications]
[3, 3]

What an arithmetization makes of these steps, which values it names and
the relations it asks of them, is `vfhe.vfhe.witness`'s. Rank 1 only: a
product of another rank is refused, one without a key is left out.
Operations of other kinds (automorphisms, key switches outside a
multiplication, additions of a plaintext) are not read, and
`Computation.unread` names them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any

from vfhe.arith import Polynomial

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from vfhe.util.trace import Event, Operation

__all__ = [
    "Computation",
    "Division",
    "KeySwitch",
    "Linear",
    "Multiplication",
    "Term",
]

#: The operation a multiplication is read from.
PRODUCT = "MLWE_Scheme.multiply"

#: The operations plaintext-linear steps are read from, with the sign each
#: gives its second operand.
LINEAR = {
    "MLWE.multiply_poly": 1,
    "MLWE_Scheme.linear_combinations": 1,
    "MLWE.add_MLWE": 1,
    "MLWE.sub_MLWE": -1,
}

#: Event kinds of operations that are not read. Outside a multiplication,
#: an operation holding one is not read as plaintext-linear either.
FOREIGN = frozenset({"permute", "decompose_digit"})

#: Event kinds that place or move values rather than compute them.
BOOKKEEPING = frozenset({"input", "copy", "zero", "mark"})


@dataclass
class Division:
    """A rounded division of ``x`` by ``primes``: ``quotient`` is the value
    on the tape, over the ring without those primes, and ``remainder`` what
    the quotient leaves, ``x - quotient * P`` computed in that ring, where it
    is exact. ``exact`` tells a rescale, which rounds exactly, from the
    division by a key's special primes, which goes through the fast base
    conversion and may leave a remainder of several multiples of ``P``.
    ``ell`` is the number of primes of ``x``'s ring."""

    event: Event
    x: Any
    primes: tuple[int, ...]
    quotient: Any
    remainder: Any
    exact: bool
    layer: int = 0
    ell: int = 0

    @classmethod
    def of(cls, event: Event, x: Any, exact: bool) -> Division:
        primes = tuple(p for p in x.ring.primes if p not in event.ring.primes)
        P = prod(primes)
        quotient = event.value
        remainder = x.mod_reduce(quotient.ring) - quotient * [
            P % p for p in quotient.ring.primes
        ]
        return cls(event, x, primes, quotient, remainder, exact, ell=len(x.ring.primes))

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
    ``[mask, body]`` and ``versions[k]`` their versions; ``constant``,
    ``linear`` and ``quadratic`` the tensor slots by their degree in the
    secret; ``rescales`` the round divisions of the output, in tape order,
    when the scheme rescaled it."""

    operation: Operation
    operands: list[list[Any]]
    constant: Any
    linear: Any
    quadratic: Any
    key_switch: KeySwitch
    rescales: list[Division] = field(default_factory=list)
    versions: list[list[int]] = field(default_factory=list)
    layer: int = 0
    ell: int = 0


@dataclass
class Term:
    """``sign * plaintext * ciphertext``, ``plaintext`` None for a
    ciphertext added as is. ``ciphertext`` is ``[mask, body]``; the
    versions are the plaintext's and the ciphertext's components'."""

    plaintext: Any | None
    ciphertext: list[Any]
    sign: int
    plaintext_version: int | None
    versions: list[int]


@dataclass
class Linear:
    """A plaintext-linear step: ``output``, as ``[mask, body]`` with
    ``versions``, is the sum of ``terms``. ``operation`` is the one that
    wrote the output; the terms may come from earlier ones, whose sums it
    adds."""

    operation: Operation
    terms: list[Term]
    output: list[Any]
    versions: list[int]
    layer: int = 0
    ell: int = 0


@dataclass
class _Sum:
    """One output of a plaintext-linear operation, before additions are
    followed back: its terms may be other sums."""

    operation: Operation
    events: list[Event]
    terms: list[Term]
    versions: tuple[int, ...]


class Computation:
    """The multiplications, the rescales and the plaintext-linear steps
    among ``operations``, the outermost operations of a trace. ``values``
    holds every recorded value by sequence number, zeros and copies
    materialized; ``origin`` names, for each event, the event whose value it
    holds (copies collapsed); ``inputs`` are the versions the computation
    did not compute, recorded as inputs or given as known.

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
        self.inputs = set(self.values)
        for e in events:
            self.origin[e.sequence] = e.sequence
            if e.kind == "copy":
                self.origin[e.sequence] = self.origin[e.inputs[0]]
                self.values[e.sequence] = self.value(e.inputs[0])
            elif e.kind == "zero":
                self.values[e.sequence] = Polynomial(e.ring).from_array([0])
            elif e.value is not None:
                self.values[e.sequence] = e.value
            if e.kind == "input":
                self.inputs.add(e.sequence)
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
        sums: list[_Sum] = []
        for op in self.operations:
            sums += self._sums(op)
        #: The plaintext-linear steps, in tape order.
        self.linears = self._linears(sums, events)
        self._writers: dict[int, Multiplication | Division | Linear] = {
            **written_by,
            **{r.event.sequence: r for r in self.rescales},
            **{v: step for step in self.linears for v in step.versions},
        }
        self._layers(events)
        read = {
            e.sequence for m in self.multiplications for e in m.operation.all_events()
        }
        read |= {e.sequence for s in sums for e in s.events}
        read |= {r.event.sequence for r in self.rescales}
        #: The paths of the operations holding events no step read, such as
        #: automorphisms or additions of a plaintext, in tape order.
        self.unread: list[str] = []
        for op in self.operations:
            for inner in op.walk():
                if inner.path not in self.unread and any(
                    e.kind not in BOOKKEEPING and e.sequence not in read
                    for e in inner.events
                ):
                    self.unread.append(inner.path)

    def value(self, sequence: int) -> Any:
        """The value of version ``sequence``."""
        try:
            return self.values[sequence]
        except KeyError:
            raise ValueError(
                f"version {sequence} was recorded before these operations; "
                "pass the earlier values as known"
            ) from None

    def written_by(self, sequence: int) -> Multiplication | Division | Linear | None:
        """The step that wrote version ``sequence``: the multiplication of a
        key switch's output, the rescale of a quotient, the plaintext-linear
        step of a sum. None for a value no step wrote, such as an input of
        the computation."""
        return self._writers.get(self.origin.get(sequence, sequence))

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
        origin = self.origin
        return Multiplication(
            operation,
            [[values(a0), values(b0)], [values(a1), values(b1)]],
            constant_event.value,
            linear_event.value,
            values(quadratic_sequence),
            key_switch,
            versions=[[origin[a0], origin[b0]], [origin[a1], origin[b1]]],
            ell=len(quadratic_event.ring.primes),
        )

    def _sums(self, operation: Operation) -> list[_Sum]:
        """The sums written by the plaintext-linear operations in
        ``operation``, outside multiplications and operations with foreign
        events."""
        if operation.name == PRODUCT or _foreign(operation):
            return []
        sign = LINEAR.get(operation.name or "")
        if sign is not None:
            return self._read_linear(operation, sign)
        found: list[_Sum] = []
        for child in operation.children:
            found += self._sums(child)
        return found

    def _term(self, plaintext: int | None, mask: int, body: int, sign: int) -> Term:
        origin, values = self.origin, self.value
        return Term(
            None if plaintext is None else values(plaintext),
            [values(mask), values(body)],
            sign,
            None if plaintext is None else origin[plaintext],
            [origin[mask], origin[body]],
        )

    def _read_linear(self, operation: Operation, sign: int) -> list[_Sum]:
        # The mlwe kernels write a ciphertext's components one after the
        # other, mask first, so the events come in pairs: a product by a
        # plaintext is (component, plaintext), an accumulation
        # (accumulator, component, plaintext), an addition (left, right).
        events = [e for e in operation.all_events() if e.kind not in ("input", "mark")]
        if len(events) % 2:
            raise ValueError("only ciphertexts of rank 1 are read")
        found: list[_Sum] = []
        for mask, body in zip(events[0::2], events[1::2], strict=True):
            if mask.kind != body.kind:
                raise ValueError("only ciphertexts of rank 1 are read")
            versions = (mask.sequence, body.sequence)
            if mask.kind == "zero":
                found.append(_Sum(operation, [mask, body], [], versions))
            elif mask.kind == "mul":
                term = self._term(mask.inputs[1], mask.inputs[0], body.inputs[0], 1)
                found.append(_Sum(operation, [mask, body], [term], versions))
            elif mask.kind == "mul_addto":
                current = found[-1]
                if current.versions != (mask.inputs[0], body.inputs[0]):
                    raise ValueError("an accumulation does not continue its sum")
                current.events += [mask, body]
                current.terms.append(
                    self._term(mask.inputs[2], mask.inputs[1], body.inputs[1], 1)
                )
                current.versions = versions
            elif mask.kind in ("add", "sub"):
                (left, right), (left_body, right_body) = mask.inputs, body.inputs
                terms = [
                    self._term(None, left, left_body, 1),
                    self._term(None, right, right_body, sign),
                ]
                found.append(_Sum(operation, [mask, body], terms, versions))
            else:
                raise ValueError(f"{operation.name} holds a {mask.kind!r}, not read")
        return found

    def _linears(self, sums: list[_Sum], events: list[Event]) -> list[Linear]:
        """The steps among ``sums``: a sum is folded into the sums that add
        it, and is a step unless adding into other sums is its only use."""
        by_output = {s.versions: s for s in sums}
        owner = {e.sequence: s for s in sums for e in s.events}
        users: dict[int, list[int]] = {}
        for e in events:
            if e.kind not in BOOKKEEPING:
                for s in e.inputs:
                    users.setdefault(self.origin.get(s, s), []).append(e.sequence)

        def added_into(s: _Sum, user: int) -> bool:
            other = owner.get(user)
            return other is not None and any(
                t.plaintext_version is None and tuple(t.versions) == s.versions
                for t in other.terms
            )

        def expand(s: _Sum, sign: int) -> Iterator[Term]:
            for t in s.terms:
                inner = by_output.get(tuple(t.versions))
                if t.plaintext_version is None and inner is not None:
                    yield from expand(inner, sign * t.sign)
                else:
                    yield Term(
                        t.plaintext,
                        t.ciphertext,
                        sign * t.sign,
                        t.plaintext_version,
                        t.versions,
                    )

        linears = []
        for s in sums:
            used = [u for v in s.versions for u in users.get(v, [])]
            if used and all(added_into(s, u) for u in used):
                continue
            mask, body = s.versions
            linears.append(
                Linear(
                    s.operation,
                    list(expand(s, 1)),
                    [self.value(mask), self.value(body)],
                    [mask, body],
                    ell=len(self.value(mask).ring.primes),
                )
            )
        return linears

    def _layers(self, events: list[Event]) -> None:
        """Sets each step's ``layer``: the multiplications on the longest
        chain from an input to the step's operands. A multiplication's output
        is one layer above its operands."""
        depth: dict[int, int] = {}
        outputs = {
            e.sequence: m
            for m in self.multiplications
            for e in m.key_switch.output_events
        }

        def of(versions: Iterable[int]) -> int:
            return max(
                (depth.get(self.origin.get(v, v), 0) for v in versions), default=0
            )

        for e in events:
            if e.kind == "copy":
                continue
            multiplication = outputs.get(e.sequence)
            if multiplication is not None:
                multiplication.layer = of(v for k in multiplication.versions for v in k)
                depth[e.sequence] = multiplication.layer + 1
            else:
                depth[e.sequence] = of(e.inputs)
        for rescale in self.rescales:
            rescale.layer = of(rescale.event.inputs)
        for linear in self.linears:
            linear.layer = of(v for t in linear.terms for v in t.versions)


def _foreign(operation: Operation) -> bool:
    """Whether ``operation`` holds a foreign event outside multiplications."""
    if operation.name == PRODUCT:
        return False
    return any(e.kind in FOREIGN for e in operation.events) or any(
        _foreign(child) for child in operation.children
    )
