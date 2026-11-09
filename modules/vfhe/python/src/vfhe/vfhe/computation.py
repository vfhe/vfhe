# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The computation a trace recorded, read into its steps in vfhe's own
terms, with the recorded values.

The steps:

- A *tensor product* is the extended product of two ciphertexts, written
  by ``MLWE_Scheme.multiply``, ``multiply_batch`` or ``tensor_product``.
- A *key switch* is read wherever the tape holds one: in a
  multiplication, a relinearization, or a key switch of a ciphertext. It
  is the digits of one decomposed value, their products with the key's
  components accumulated into each output component, the division by the
  special primes when the key has them, and the value added to each
  component, if any.
- A *multiplication* is a tensor product together with the key switch
  that decomposes its quadratic slot and adds its other two slots, and the
  rescales applied to its output afterwards. A key switch that completes
  no tensor product (the relinearization of a sum of extended products)
  is a step of its own, and so is a tensor product no key switch
  completes.
- A *rescale* is a round division of a ciphertext component by the primes
  the next level drops. Between levels that are not nested, the component
  is first lifted into a ring holding both levels' primes; the rescale is
  read from the component before the lift.

A plaintext-linear step is a ciphertext computed as
``sum_j sign_j * plaintext_j * ciphertext_j``, read from products by a
plaintext (``MLWE.multiply_poly``), linear combinations
(``MLWE_Scheme.linear_combinations``, with or without a `PlaintextMatrix`)
and additions and subtractions of ciphertexts; a ciphertext added as is
is a term without a plaintext. The ciphertexts may have any number of
components, such as the three of an extended product. Additions are
followed back: the terms of a step are ciphertexts that no plaintext-linear
step wrote, and a sum that is only added into other sums is not a step of
its own.

Each step carries its ``layer``, the number of multiplications on the
longest chain from the computation's inputs to the step's operands, and
its ``ell``, the number of primes of the ring it computes over.

>>> from vfhe.vfhe.computation import Computation
>>> computation = Computation(t.operations())
>>> [len(m.key_switch.digits) for m in computation.multiplications]
[3, 3]

What an arithmetization makes of these steps, which values it names and
the relations it asks of them, is left to the code that reads the
`Computation`. Rank 1 only: a product of another rank is refused.
Operations of other kinds (automorphisms, additions of a plaintext, key
switches sharing one decomposition) are not read, and `Computation.unread`
names them.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import gcd, prod
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
    "Tensor",
    "Term",
]

#: The operations tensor products are read from.
PRODUCTS = frozenset(
    {
        "MLWE_Scheme.multiply",
        "MLWE_Scheme.multiply_batch",
        "MLWE_Scheme.tensor_product",
    }
)

#: The operations plaintext-linear steps are read from, with the sign each
#: gives its second operand.
LINEAR = {
    "MLWE.multiply_poly": 1,
    "MLWE_Scheme.linear_combinations": 1,
    "MLWE.add_MLWE": 1,
    "MLWE.sub_MLWE": -1,
}

#: Event kinds of operations that are not read as plaintext-linear.
FOREIGN = frozenset({"permute", "decompose_digit"})

#: Event kinds that place or move values rather than compute them.
BOOKKEEPING = frozenset({"input", "copy", "zero", "mark"})


@dataclass
class Division:
    """A rounded division of ``x`` by ``primes``: ``quotient`` is the value
    on the tape, over the ring without those primes, and ``remainder`` is
    ``x - P * quotient`` over ``x``'s ring, with ``P`` the product of
    ``primes``. ``x``'s modulus is ``P`` times the quotient's, so the
    remainder is exact: its centered coefficients are the integers ``r``
    with ``x = P * quotient + r``. ``exact`` tells a rescale, which rounds
    to the nearest integer (``|r| <= (P - 1)/2``), from the division by a
    key's special primes, which goes through the fast base conversion and
    may leave a remainder of several multiples of ``P``.

    ``source`` is the value the division was applied to: ``x`` itself, or,
    for a rescale between levels that are not nested, the component before
    its lift, with ``x = lift * source`` (``lift`` the product of the
    primes the lift adds; 1 without one). ``source_version`` is its
    version. ``ell`` is the number of primes of ``x``'s ring."""

    event: Event
    x: Any
    primes: tuple[int, ...]
    quotient: Any
    remainder: Any
    exact: bool
    source: Any = None
    source_version: int = 0
    lift: int = 1
    layer: int = 0
    ell: int = 0

    @classmethod
    def of(
        cls,
        event: Event,
        x: Any,
        exact: bool,
        source: Any = None,
        source_version: int = 0,
    ) -> Division:
        primes = tuple(p for p in x.ring.primes if p not in event.ring.primes)
        quotient = event.value
        # P * quotient over x's ring: the quotient's residues scaled by P at
        # its own primes, zero at the primes divided by.
        remainder = x - quotient.scaled_lift(x.ring)
        if source is None:
            source = x
            lift = 1
        else:
            lift = prod(p for p in x.ring.primes if p not in source.ring.primes)
        return cls(
            event,
            x,
            primes,
            quotient,
            remainder,
            exact,
            source=source,
            source_version=source_version,
            lift=lift,
            ell=len(x.ring.primes),
        )

    @property
    def modulus(self) -> int:
        """``P``, the product of the primes divided by."""
        return prod(self.primes)


@dataclass
class KeySwitch:
    """A key switch of ``decomposed``, the value its digits decompose: a
    product's quadratic slot, or the mask of a ciphertext.

    - ``digits[t]`` is digit ``t``, ``digit_events[t]`` the event that
      wrote it (it records how the digit was taken, see `power` and
      `bound`), and ``primes[t]`` the primes of its residue.
    - ``key[t][i]`` is the key component digit ``t`` multiplies into
      accumulator ``i``, and ``sums[i]`` the accumulated products,
      subtracted as vfhe computes them.
    - ``mod_down[i]`` is the division of ``sums[i]`` by the special primes
      (empty for a key without them).
    - ``slots[i]`` is the value added to output component ``i``, None when
      nothing is; ``output[i]`` is the result and ``output_events[i]`` the
      event that wrote it.
    - ``versions`` are the versions of ``decomposed`` and of the slots (0
      for a missing slot)."""

    decomposed: Any
    digits: list[Any]
    primes: list[tuple[int, ...]]
    key: list[list[Any]]
    sums: list[Any]
    mod_down: list[Division]
    slots: list[Any]
    output: list[Any]
    output_events: list[Event]
    digit_events: list[Event] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    layer: int = 0
    ell: int = 0

    def power(self, t: int) -> int:
        """The power of two digit ``t`` stands for in its residue:
        ``2^(log_base * level)`` for a radix digit, 1 for a residue."""
        e = self.digit_events[t]
        return 1 << (e.log_base * e.level)

    def bound(self, t: int) -> float:
        """The largest absolute value digit ``t`` can take. A centered
        residue at one prime ``p`` is within ``p/2``; the fast base
        conversion of the residue at a group of primes with product ``Q``
        within ``k*Q/2`` for ``k`` primes, centered, or below ``k*Q``.
        A balanced radix digit is in ``[-2^(w-1), 2^(w-1)]``, the top one
        included, and an unsigned one below ``2^w``."""
        e = self.digit_events[t]
        if e.log_base:
            return float(1 << (e.log_base - 1)) if e.centered else (1 << e.log_base) - 1
        modulus = prod(e.primes)
        if len(e.primes) == 1:
            return modulus / 2 if e.centered else modulus - 1
        return len(e.primes) * modulus / (2 if e.centered else 1)


@dataclass
class Tensor:
    """The extended product of two ciphertexts: ``operands[k]`` is operand
    ``k`` as ``[mask, body]`` and ``versions[k]`` their versions;
    ``constant``, ``linear`` and ``quadratic`` the slots by their degree in
    the secret, ``slot_versions`` their versions in that order."""

    operation: Operation
    operands: list[list[Any]]
    constant: Any
    linear: Any
    quadratic: Any
    versions: list[list[int]] = field(default_factory=list)
    slot_versions: list[int] = field(default_factory=list)
    layer: int = 0
    ell: int = 0


@dataclass
class Multiplication:
    """One multiplication: ``operands[k]`` is operand ``k`` as
    ``[mask, body]`` and ``versions[k]`` their versions; ``constant``,
    ``linear`` and ``quadratic`` the tensor slots by their degree in the
    secret, ``key_switch`` the key switch that decomposes the quadratic slot
    and adds the other two; ``rescales`` the round divisions of the output,
    in tape order, when it was rescaled. ``operation`` is the one that
    wrote the tensor product."""

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
    ciphertext added as is. ``ciphertext`` is the list of its components,
    the body last; the versions are the plaintext's and the components'."""

    plaintext: Any | None
    ciphertext: list[Any]
    sign: int
    plaintext_version: int | None
    versions: list[int]


@dataclass
class Linear:
    """A plaintext-linear step: ``output``, its components with
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


#: A step of a computation, as `Computation.written_by` returns it.
Step = Multiplication | Tensor | KeySwitch | Division | Linear


class Computation:
    """The multiplications, the tensor products and key switches that are
    not part of one, the rescales and the plaintext-linear steps among
    ``operations``, the outermost operations of a trace. ``values`` holds
    every recorded value by sequence number, zeros and copies materialized;
    ``origin`` names, for each event, the event whose value it holds (copies
    collapsed); ``inputs`` are the versions the computation did not compute,
    recorded as inputs or given as known.

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
        self._events = {e.sequence: e for e in events}
        self.values: dict[int, Any] = dict(known or {})
        self.origin: dict[int, int] = {s: s for s in self.values}
        self.inputs = set(self.values)
        for e in events:
            self.origin[e.sequence] = e.sequence
            if e.kind == "copy":
                self.values[e.sequence] = self.value(e.inputs[0])
                self.origin[e.sequence] = self.origin[e.inputs[0]]
            elif e.kind == "zero":
                self.values[e.sequence] = Polynomial(e.ring).from_array([0])
            elif e.value is not None:
                self.values[e.sequence] = e.value
            if e.kind == "input":
                self.inputs.add(e.sequence)
        #: The events some step reads, for `unread`.
        self._read: set[int] = set()
        tensors, switches = self._read_products()
        #: The multiplications: tensor products completed by a key switch.
        self.multiplications: list[Multiplication] = []
        #: The key switches that complete no tensor product, in tape order.
        self.key_switches: list[KeySwitch] = []
        by_quadratic = {t.slot_versions[2]: t for t in tensors}
        completed: set[int] = set()
        for switch in switches:
            tensor = by_quadratic.get(switch.versions[0])
            if (
                tensor is not None
                and switch.versions[1:] == tensor.slot_versions[1::-1]
            ):
                completed.add(id(tensor))
                self.multiplications.append(
                    Multiplication(
                        tensor.operation,
                        tensor.operands,
                        tensor.constant,
                        tensor.linear,
                        tensor.quadratic,
                        switch,
                        versions=tensor.versions,
                        ell=tensor.ell,
                    )
                )
            else:
                self.key_switches.append(switch)
        #: The tensor products no key switch completes, in tape order.
        self.tensors = [t for t in tensors if id(t) not in completed]
        #: Every rescale on the tape, whether or not a multiplication owns it.
        self.rescales = [self._rescale(e) for e in events if e.kind == "round_division"]
        written_by: dict[int, Multiplication | KeySwitch | Tensor] = {
            self.origin[e.sequence]: m
            for m in self.multiplications
            for e in m.key_switch.output_events
        }
        written_by |= {
            self.origin[e.sequence]: s
            for s in self.key_switches
            for e in s.output_events
        }
        written_by |= {v: t for t in self.tensors for v in t.slot_versions}
        for rescale in self.rescales:
            owner = written_by.get(self._origin(rescale.source_version))
            if isinstance(owner, Multiplication):
                owner.rescales.append(rescale)
        sums: list[_Sum] = []
        for op in self.operations:
            sums += self._sums(op)
        #: The plaintext-linear steps, in tape order.
        self.linears = self._linears(sums, events)
        self._writers: dict[int, Step] = {
            **written_by,
            **{r.event.sequence: r for r in self.rescales},
            **{v: step for step in self.linears for v in step.versions},
        }
        self._layers(events)
        self._read |= {e.sequence for s in sums for e in s.events}
        #: The paths of the operations holding events no step read, such as
        #: automorphisms or additions of a plaintext, in tape order.
        self.unread: list[str] = []
        for op in self.operations:
            for inner in op.walk():
                if inner.path not in self.unread and any(
                    e.kind not in BOOKKEEPING and e.sequence not in self._read
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

    def written_by(self, sequence: int) -> Step | None:
        """The step that wrote version ``sequence``: the multiplication or
        key switch of a key switch's output, the tensor product of an
        extended product's slot, the rescale of a quotient, the
        plaintext-linear step of a sum. None for a value no step wrote,
        such as an input of the computation."""
        return self._writers.get(self.origin.get(sequence, sequence))

    def _origin(self, sequence: int) -> int:
        return self.origin.get(sequence, sequence)

    def _read_products(self) -> tuple[list[Tensor], list[KeySwitch]]:
        """Every tensor product and every key switch, operation by
        operation. A product operation whose events are not rank-1 tensor
        products and key switches is refused."""
        tensors: list[Tensor] = []
        switches: list[KeySwitch] = []
        for op in self.operations:
            for inner in op.walk():
                found = self._key_switches(inner)
                if inner.name in PRODUCTS:
                    if found is None:
                        raise ValueError("only products of rank 1 are read")
                    products = self._tensors(inner)
                    if found and len(found) != len(products):
                        raise ValueError("only products of rank 1 are read")
                    tensors += products
                switches += found or []
        return tensors, switches

    def _rescale(self, event: Event) -> Division:
        """The rescale ``event`` computes, read from before the lift when the
        divided value is one."""
        self._read.add(event.sequence)
        x_version = self._origin(event.inputs[0])
        writer = self._events.get(x_version)
        if writer is not None and writer.kind == "scaled_lift":
            self._read.add(writer.sequence)
            (source,) = writer.inputs
            return Division.of(
                event,
                self.value(x_version),
                exact=True,
                source=self.value(source),
                source_version=self._origin(source),
            )
        return Division.of(
            event, self.value(x_version), exact=True, source_version=x_version
        )

    def _tensors(self, operation: Operation) -> list[Tensor]:
        """The tensor products among ``operation``'s own events: the partial
        product ``a0 * b1`` accumulated with ``b0 * a1`` into the linear
        slot, next to ``a0 * a1`` and ``b0 * b1``."""
        written: dict[int, Event] = {}
        by_operands: dict[tuple[int, ...], list[Event]] = {}
        for e in operation.events:
            if e.kind == "mul":
                written[e.sequence] = e
                by_operands.setdefault(e.inputs, []).append(e)
        found = []
        for linear in operation.events:
            if linear.kind != "mul_addto" or linear.inputs[0] not in written:
                continue
            partial = written[linear.inputs[0]]
            (a0, b1), (_, b0, a1) = partial.inputs, linear.inputs
            # The products nearest to this one's partial product: a batch may
            # hold the same product twice.
            quadratic = _last_before(by_operands.get((a0, a1), []), partial.sequence)
            constant = _first_after(by_operands.get((b0, b1), []), linear.sequence)
            if quadratic is None or constant is None:
                continue
            self._read |= {e.sequence for e in (partial, linear, quadratic, constant)}
            values, origin = self.value, self._origin
            found.append(
                Tensor(
                    operation,
                    [[values(a0), values(b0)], [values(a1), values(b1)]],
                    constant.value,
                    linear.value,
                    quadratic.value,
                    versions=[[origin(a0), origin(b0)], [origin(a1), origin(b1)]],
                    slot_versions=[
                        constant.sequence,
                        linear.sequence,
                        quadratic.sequence,
                    ],
                    ell=len(quadratic.ring.primes),
                )
            )
        return found

    def _key_switches(self, operation: Operation) -> list[KeySwitch] | None:
        """The key switches among ``operation``'s own events, or None when
        its digits do not form rank-1 key switches (two accumulators per
        decomposed value, each taking every digit once)."""
        events = operation.events
        digits: dict[int, list[Event]] = {}
        for e in events:
            if e.kind == "decompose_digit":
                digits.setdefault(self._origin(e.inputs[0]), []).append(e)
        if not digits:
            return []
        digit_of = {e.sequence: x for x, ds in digits.items() for e in ds}
        zeros = {e.sequence for e in events if e.kind == "zero"}
        root: dict[int, int] = {}
        chains: dict[int, list[Event]] = {}
        for e in events:
            if e.kind != "mul_subto" or self._origin(e.inputs[1]) not in digit_of:
                continue
            start = e.inputs[0] if e.inputs[0] in zeros else root.get(e.inputs[0])
            if start is None:
                return None
            root[e.sequence] = start
            chains.setdefault(start, []).append(e)
        starts_of: dict[int, list[int]] = {}
        for start, chain in chains.items():
            x = digit_of[self._origin(chain[0].inputs[1])]
            starts_of.setdefault(x, []).append(start)
        # Every value is checked before any is read, so that a refused
        # operation reads no value.
        accumulators: dict[int, list[list[Event]]] = {}
        for x, ds in digits.items():
            starts = sorted(starts_of.get(x, []))
            order = [d.sequence for d in ds]
            if len(starts) != 2 or any(
                [self._origin(e.inputs[1]) for e in chains[s]] != order for s in starts
            ):
                return None
            accumulators[x] = [chains[s] for s in starts]
        divided = {
            e.inputs[0]: e for e in events if e.kind == "round_division_approximate"
        }
        added_to: dict[int, Event] = {}
        for e in events:
            if e.kind == "add":
                for s in e.inputs:
                    added_to.setdefault(self._origin(s), e)
        found = [
            self._key_switch(x, digits[x], accumulators[x], divided, added_to)
            for x in digits
        ]
        found.sort(key=lambda s: s.output_events[-1].sequence)
        return found

    def _key_switch(
        self,
        x: int,
        digits: list[Event],
        chains: list[list[Event]],
        divided: dict[int, Event],
        added_to: dict[int, Event],
    ) -> KeySwitch:
        """The key switch of the value ``x``: its ``digits``, its two
        accumulator ``chains``, and the operation's divisions by the special
        primes (``divided``, by the version divided) and additions
        (``added_to``, by the version added)."""
        values = self.value
        key: list[list[Any]] = [[None, None] for _ in digits]
        sums, mod_down, slots, slot_versions, outputs = [], [], [], [], []
        for i, chain in enumerate(chains):
            for t, e in enumerate(chain):
                key[t][i] = values(e.inputs[2])
            end = chain[-1]
            sums.append(end.value)
            self._read |= {e.sequence for e in chain} | {e.inputs[0] for e in chain[:1]}
            division = divided.get(end.sequence)
            if division is not None:
                self._read.add(division.sequence)
                mod_down.append(Division.of(division, end.value, exact=False))
                end = division
            output = added_to.get(end.sequence)
            if output is not None:
                (slot,) = [s for s in output.inputs if self._origin(s) != end.sequence]
                self._read.add(output.sequence)
                slots.append(values(slot))
                slot_versions.append(self._origin(slot))
            else:
                output = end  # nothing is added: the switched value is the output
                slots.append(None)
                slot_versions.append(0)
            outputs.append(output)
        self._read |= {e.sequence for e in digits}
        return KeySwitch(
            decomposed=values(x),
            digits=[e.value for e in digits],
            primes=[e.primes for e in digits],
            key=key,
            sums=sums,
            mod_down=mod_down,
            slots=slots,
            output=[e.value for e in outputs],
            output_events=outputs,
            digit_events=list(digits),
            versions=[x, *slot_versions],
            ell=len(values(x).ring.primes),
        )

    def _sums(self, operation: Operation) -> list[_Sum]:
        """The sums written by the plaintext-linear operations in
        ``operation``, outside products and operations with foreign
        events."""
        if operation.name in PRODUCTS or _foreign(operation):
            return []
        sign = LINEAR.get(operation.name or "")
        if sign is not None:
            return self._read_linear(operation, sign)
        found: list[_Sum] = []
        for child in operation.children:
            found += self._sums(child)
        return found

    def _term(self, plaintext: int | None, components: list[int], sign: int) -> Term:
        origin, values = self._origin, self.value
        return Term(
            None if plaintext is None else values(plaintext),
            [values(c) for c in components],
            sign,
            None if plaintext is None else origin(plaintext),
            [origin(c) for c in components],
        )

    def _read_linear(self, operation: Operation, sign: int) -> list[_Sum]:
        # The mlwe kernels write a ciphertext's components one after the
        # other, so a term is a run of events of one kind: a product by a
        # plaintext is (component, plaintext), an accumulation
        # (accumulator, component, plaintext), an addition (left, right).
        events = [e for e in operation.all_events() if e.kind not in ("input", "mark")]
        for e in events:
            if e.kind not in ("zero", "mul", "mul_addto", "add", "sub"):
                raise ValueError(f"{operation.name} holds a {e.kind!r}, not read")
        width = _component_count(operation, events)
        if width == 0 or len(events) % width:
            raise ValueError(f"{operation.name}: its ciphertexts are not read")
        found: list[_Sum] = []
        by_output: dict[tuple[int, ...], _Sum] = {}
        for k in range(0, len(events), width):
            group = events[k : k + width]
            kind = group[0].kind
            if any(e.kind != kind for e in group):
                raise ValueError(f"{operation.name}: its ciphertexts are not read")
            versions = tuple(e.sequence for e in group)
            if kind == "zero":
                current = _Sum(operation, group, [], versions)
            elif kind == "mul":
                if any(e.inputs[1] != group[0].inputs[1] for e in group):
                    raise ValueError(f"{operation.name}: its ciphertexts are not read")
                term = self._term(group[0].inputs[1], [e.inputs[0] for e in group], 1)
                current = _Sum(operation, group, [term], versions)
            elif kind == "mul_addto":
                accumulators = tuple(e.inputs[0] for e in group)
                continued = by_output.pop(accumulators, None)
                if continued is None or any(
                    e.inputs[2] != group[0].inputs[2] for e in group
                ):
                    raise ValueError("an accumulation does not continue its sum")
                continued.events += group
                continued.terms.append(
                    self._term(group[0].inputs[2], [e.inputs[1] for e in group], 1)
                )
                continued.versions = versions
                by_output[versions] = continued
                continue
            else:
                current = _Sum(
                    operation,
                    group,
                    [
                        self._term(None, [e.inputs[0] for e in group], 1),
                        self._term(None, [e.inputs[1] for e in group], sign),
                    ],
                    versions,
                )
            found.append(current)
            by_output[versions] = current
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
                    users.setdefault(self._origin(s), []).append(e.sequence)

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
            linears.append(
                Linear(
                    s.operation,
                    list(expand(s, 1)),
                    [self.value(v) for v in s.versions],
                    list(s.versions),
                    ell=len(self.value(s.versions[0]).ring.primes),
                )
            )
        return linears

    def _layers(self, events: list[Event]) -> None:
        """Sets each step's ``layer``: the multiplications on the longest
        chain from an input to the step's operands. A multiplication's output
        is one layer above its operands, and so are the slots of a tensor
        product."""
        depth: dict[int, int] = {}
        raises = {
            e.sequence: m.versions
            for m in self.multiplications
            for e in m.key_switch.output_events
        }
        raises |= {v: t.versions for t in self.tensors for v in t.slot_versions}

        def of(versions: Iterable[int]) -> int:
            return max((depth.get(self._origin(v), 0) for v in versions), default=0)

        for e in events:
            if e.kind == "copy":
                continue
            operands = raises.get(e.sequence)
            if operands is not None:
                depth[e.sequence] = of(v for k in operands for v in k) + 1
            else:
                depth[e.sequence] = of(e.inputs)
        for m in self.multiplications:
            m.layer = m.key_switch.layer = of(v for k in m.versions for v in k)
        for t in self.tensors:
            t.layer = of(v for k in t.versions for v in k)
        for s in self.key_switches:
            s.layer = of(v for v in s.versions if v)
        for rescale in self.rescales:
            rescale.layer = of([rescale.source_version])
        for linear in self.linears:
            linear.layer = of(v for t in linear.terms for v in t.versions)


def _component_count(operation: Operation, events: list[Event]) -> int:
    """The number of components of the ciphertexts ``operation`` combines.

    The tape does not record it, so it is read from the order of the
    events. A product by a plaintext or an addition handles one ciphertext:
    all its events. A linear combination writes one term of a row at a
    time, a run of events of one kind, one plaintext and distinct
    components:

    - a run of accumulations is exactly one term, since the row's next term
      continues the sums this run wrote and a row starts with a product, so
      its length is the answer;
    - without accumulations (rows of one term), consecutive rows with the
      same plaintext form one run, so the answer is the greatest common
      divisor of the run lengths.
    """
    if operation.name != "MLWE_Scheme.linear_combinations":
        return len(events)
    runs: list[tuple[str, int]] = []  # (kind, length)
    previous: Event | None = None
    seen: set[int] = set()  # the run's components, and the sums it wrote
    for e in events:
        component = e.inputs[-2] if e.kind in ("mul", "mul_addto") else e.sequence
        continues = (
            previous is not None
            and e.kind == previous.kind
            and e.inputs[-1:] == previous.inputs[-1:]  # the same plaintext
            and component not in seen
            and not (e.kind == "mul_addto" and e.inputs[0] in seen)
        )
        if continues:
            runs[-1] = (e.kind, runs[-1][1] + 1)
        else:
            runs.append((e.kind, 1))
            seen = set()
        seen.add(component)
        if e.kind == "mul_addto":
            seen.add(e.sequence)
        previous = e
    accumulations = [n for kind, n in runs if kind == "mul_addto"]
    if accumulations:
        return min(accumulations)
    return gcd(*(n for _, n in runs))


def _last_before(events: list[Event], sequence: int) -> Event | None:
    """The last of ``events``, in tape order, recorded before ``sequence``."""
    k = bisect_left([e.sequence for e in events], sequence)
    return events[k - 1] if k else None


def _first_after(events: list[Event], sequence: int) -> Event | None:
    """The first of ``events``, in tape order, recorded after ``sequence``."""
    k = bisect_right([e.sequence for e in events], sequence)
    return events[k] if k < len(events) else None


def _foreign(operation: Operation) -> bool:
    """Whether ``operation`` holds a foreign event outside products."""
    if operation.name in PRODUCTS:
        return False
    return any(e.kind in FOREIGN for e in operation.events) or any(
        _foreign(child) for child in operation.children
    )
