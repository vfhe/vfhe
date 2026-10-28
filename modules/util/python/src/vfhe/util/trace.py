# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Records the ring operations vfhe runs: the execution tape.

While a trace is enabled, every whole-element operation the native code
makes over an RNS ring leaves an `Event`: its kind, the versions of its
operands, its parameters, and a copy of its output. A version is the
sequence number of the event that wrote the value; an operand no event
wrote is recorded first as an ``"input"`` event, so every operand of every
event is on the tape. Values are copies: later changes to the live objects
do not reach them.

>>> from vfhe.util import trace
>>> with trace.recording() as t:
...     c = a + b
>>> [e.kind for e in t.events()]
['input', 'input', 'add', 'input', 'input', 'add']

`Trace.events` drains the tape: what it returns is gone from it, so a long
computation can be read in pieces. A trace started before key generation
or encryption holds the key and the noise; sampler outputs carry
``secret``.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Event", "Trace", "enable", "mark", "recording"]

#: Event kinds by their native code, named after the operations.
KINDS: dict[int, str] = {
    getattr(lib, name): name[len("ARITH_TRACE_") :].lower()
    for name in dir(lib)
    if name.startswith("ARITH_TRACE_")
}


@dataclass
class Event:
    """One recorded operation; see the module for what the fields mean.

    ``value`` is the output's copy, a polynomial over ``ring``, or None for
    ``"copy"``, ``"zero"`` and ``"mark"``. ``inputs`` are the sequence
    numbers of the operands' events, the accumulator's previous version first
    for an in-place operation. ``param`` is the generator of a permutation,
    the power of a monomial product (``param2`` whether one is subtracted),
    an integer factor, a count or a stream; ``scalar`` the per-row factors
    of ``"scale_by"``; ``label`` and ``enter`` belong to a ``"mark"``.
    """

    sequence: int
    kind: str
    inputs: tuple[int, ...]
    ring: Any
    value: Any
    secret: bool = False
    param: int = 0
    param2: int = 0
    sigma: float = 0.0
    scalar: tuple[int, ...] | None = None
    label: str | None = None
    enter: bool | None = None


class Trace:
    """The tape as Python reads it; `enable` returns one."""

    def __init__(self) -> None:
        self._rings: dict[tuple[int, int, int], Any] = {}

    @property
    def enabled(self) -> bool:
        """Whether operations are being recorded right now."""
        return bool(lib.arith_trace_is_enabled())

    def enable(self) -> None:
        """Resumes recording onto the same tape."""
        lib.arith_trace_enable()

    def disable(self) -> None:
        """Stops recording; what was recorded stays readable."""
        lib.arith_trace_disable()

    def clear(self) -> None:
        """Drops the tape: every event not yet read, and the versions."""
        lib.arith_trace_clear()
        self._rings.clear()

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.disable()

    def __len__(self) -> int:
        return int(lib.arith_trace_length())

    def drain(self) -> Iterator[Event]:
        """The events recorded so far, oldest first, removed from the tape."""
        count = len(self)
        events = []
        for index in range(count):
            native = ffi.new("ArithTraceEvent *")
            if lib.arith_trace_read(index, native) != 0:
                break
            # Converted before the release frees the label and the scalar;
            # the value is taken over and outlives it.
            events.append(self._event(native, lib.arith_trace_take_value(index)))
        lib.arith_trace_release(len(events))
        yield from events

    def events(self, kind: str | None = None) -> list[Event]:
        """`drain` as a list, of one ``kind`` only when given."""
        return [e for e in self.drain() if kind is None or e.kind == kind]

    def _ring(self, N: int, mask: int, base: Any) -> Any:
        key = (N, mask, int(ffi.cast("uintptr_t", base)))
        ring = self._rings.get(key)
        if ring is None:
            rns = importlib.import_module("vfhe.arith.impl.rns.polynomial")
            ring = self._rings[key] = rns.RNSRing.from_native(N, mask, base)
        return ring

    def _event(self, native: Any, value: Any) -> Event:
        kind = KINDS[native.kind]
        ring = poly = None
        if kind != "mark":
            ring = self._ring(native.N, native.mask, native.base)
            if value != ffi.NULL:
                rns = importlib.import_module("vfhe.arith.impl.rns.polynomial")
                arith = importlib.import_module("vfhe.arith")
                domain = {1: arith.repr.coeff, 2: arith.repr.ntt}.get(native.domain)
                poly = rns.RNSPolynomial(
                    ring, arith.repr.empty if domain is None else domain, obj=value
                )
        return Event(
            sequence=native.sequence,
            kind=kind,
            inputs=tuple(native.inputs[i] for i in range(native.n_inputs)),
            ring=ring,
            value=poly,
            secret=bool(native.secret),
            param=native.param,
            param2=native.param2,
            sigma=native.sigma,
            scalar=(
                tuple(native.scalar[i] for i in range(native.scalar_len))
                if native.scalar != ffi.NULL
                else None
            ),
            label=ffi.string(native.label).decode()
            if native.label != ffi.NULL
            else None,
            enter=bool(native.enter) if kind == "mark" else None,
        )


def enable() -> Trace:
    """Starts recording and returns the `Trace` to read it from.

    A second call resumes the same tape; `Trace.clear` ends it.
    """
    lib.arith_trace_enable()
    return Trace()


@contextmanager
def recording() -> Iterator[Trace]:
    """Records the operations made inside the ``with`` block."""
    trace = enable()
    try:
        yield trace
    finally:
        trace.disable()


def mark(label: str, enter: bool = True) -> None:
    """Places a labeled ``"mark"`` event on the tape, opening a region when
    ``enter`` and closing one otherwise."""
    lib.arith_trace_mark(label.encode(), 1 if enter else 0)
