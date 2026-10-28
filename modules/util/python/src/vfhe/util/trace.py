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

Marks cut the tape into operations. While a trace is enabled, a profile
hook marks the entry and the exit of every public function and method of
the vfhe modules (the runtime, `vfhe.util`, aside), and `mark` places marks
by hand. `Trace.operations` reads the tape as that tree, an `Operation`
per marked call that made a ring operation, named after the call:

>>> from vfhe.util import trace
>>> with trace.recording() as t:
...     d = (a * b) * (a * b)
>>> [op.name for op in t.operations()]
['CKKS_Ciphertext.__mul__', 'CKKS_Ciphertext.__mul__']

`Trace.events` and `Trace.operations` drain the tape: what they return is
gone from it, so a long computation can be read in pieces. A trace started
before key generation or encryption holds the key and the noise; sampler
outputs carry ``secret``. A value written other than through the generic
interface, by a row function called by hand on storage the tape knows, is
not seen, and the tape is wrong from there.
"""

from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FunctionType
from typing import TYPE_CHECKING, Any

from vfhe.util.bindings import ffi, lib

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import CodeType, FrameType

__all__ = ["Event", "Operation", "Trace", "enable", "mark", "recording"]

#: Event kinds by their native code, named after the operations.
KINDS: dict[int, str] = {
    getattr(lib, name): name[len("ARITH_TRACE_") :].lower()
    for name in dir(lib)
    if name.startswith("ARITH_TRACE_")
}

#: The operator methods the hook marks along with the public ones.
OPERATORS = frozenset(
    {
        "__add__",
        "__radd__",
        "__iadd__",
        "__sub__",
        "__rsub__",
        "__isub__",
        "__mul__",
        "__rmul__",
        "__imul__",
        "__matmul__",
        "__neg__",
        "__truediv__",
        "__itruediv__",
        "__mod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
    }
)


@dataclass
class Event:
    """One recorded operation; see the module for what the fields mean.

    ``value`` is the output's copy, a polynomial over ``ring``, or None for
    ``"copy"``, ``"zero"`` and ``"mark"``. ``inputs`` are the sequence
    numbers of the operands' events, the accumulator's previous version first
    for an in-place operation. ``param`` is the generator of a permutation,
    the power of a monomial product (``param2`` whether one is subtracted),
    an integer factor, a count or a stream; ``scalar`` the per-row factors
    of ``"scale_by"``; ``label`` and ``enter`` belong to a ``"mark"``. A
    ``"decompose_digit"`` carries `Polynomial.decompose_digit`'s parameters,
    ``primes`` as the prime values.
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
    primes: tuple[int, ...] = ()
    log_base: int = 0
    level: int = 0
    dropped: int = 0
    centered: bool = False
    label: str | None = None
    enter: bool | None = None


@dataclass
class Operation:
    """A marked region of the tape: a call of a vfhe function, or the region
    between a `mark` and its closing mark.

    ``name`` is the mark's label, None for the events outside any marked
    call; ``path`` joins the names from the outermost operation with ``/``;
    ``sequence`` is the opening mark's. ``events`` are the operation's own,
    ``children`` the operations made inside it, both in tape order.
    """

    name: str | None
    path: str
    sequence: int
    events: list[Event] = field(default_factory=list)
    children: list[Operation] = field(default_factory=list)

    def walk(self) -> Iterator[Operation]:
        """This operation, then every operation inside it, in tape order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, name: str) -> list[Operation]:
        """The operations named ``name`` here and inside, in tape order."""
        return [op for op in self.walk() if op.name == name]

    def all_events(self) -> list[Event]:
        """The events here and inside, in tape order."""
        events = [e for op in self.walk() for e in op.events]
        events.sort(key=lambda e: e.sequence)
        return events


# ---- the hook ---------------------------------------------------------------


def _marked() -> dict[CodeType, bytes]:
    """The code objects the hook marks, with their labels: the public functions
    and methods of the loaded vfhe modules, the runtime aside."""
    labels: dict[CodeType, bytes] = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "vfhe" or name.startswith("vfhe.")):
            continue
        if name == "vfhe.util" or name.startswith("vfhe.util."):
            continue
        namespaces = [vars(module)] + [
            vars(cls)
            for cls in vars(module).values()
            if isinstance(cls, type) and cls.__module__ == name
        ]
        for namespace in namespaces:
            for attr, value in namespace.items():
                if isinstance(value, (staticmethod, classmethod)):
                    value = value.__func__
                if not isinstance(value, FunctionType) or value.__module__ != name:
                    continue
                if not attr.startswith("_") or attr in OPERATORS:
                    labels[value.__code__] = value.__code__.co_qualname.encode()
    return labels


class _Hook:
    """The profile function installed while a trace is enabled: a mark at
    the entry and at the exit of each marked call."""

    def __init__(self) -> None:
        self.labels = _marked()
        self.previous = (sys.getprofile(), threading.getprofile())

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" or event == "return":
            label = self.labels.get(frame.f_code)
            if label is not None:
                lib.arith_trace_mark(label, 1 if event == "call" else 0)


_hook: _Hook | None = None


def _install_hook() -> None:
    global _hook
    if _hook is None:
        _hook = _Hook()
        sys.setprofile(_hook)
        threading.setprofile(_hook)


def _remove_hook() -> None:
    global _hook
    if _hook is not None:
        sys.setprofile(_hook.previous[0])
        threading.setprofile(_hook.previous[1])
        _hook = None


# ---- the tape as Python reads it -------------------------------------------


class Trace:
    """The tape as Python reads it; `enable` returns one."""

    def __init__(self) -> None:
        self._rings: dict[tuple[int, int, int], Any] = {}
        # Operations assembled by `operations`: the ones still open, outermost
        # first, and the completed outermost ones not returned yet.
        self._open: list[Operation] = []
        self._done: list[Operation] = []

    @property
    def enabled(self) -> bool:
        """Whether operations are being recorded right now."""
        return bool(lib.arith_trace_is_enabled())

    def enable(self) -> None:
        """Resumes recording onto the same tape, marking the vfhe calls made
        from now on."""
        lib.arith_trace_enable()
        _install_hook()

    def disable(self) -> None:
        """Stops recording; what was recorded stays readable."""
        _remove_hook()
        lib.arith_trace_disable()

    def clear(self) -> None:
        """Drops the tape: every event not yet read, and the versions."""
        _remove_hook()
        lib.arith_trace_clear()
        self._rings.clear()
        self._open.clear()
        self._done.clear()

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

    def operations(self) -> list[Operation]:
        """`drain` as outermost operations, cut by the marks.

        An operation is returned once its closing mark is on the tape, or
        once the trace is disabled; until then it waits here, whole. A
        marked call with no ring operation in it or inside it is left out.
        """
        for event in self.drain():
            self._place(event)
        if not self.enabled:
            while self._open:
                self._close()
        done, self._done = self._done, []
        return done

    def version_of(self, element: Any) -> int | None:
        """The sequence number of the event that last wrote ``element``'s
        value, or None when the tape has not seen it.

        ``element`` is a polynomial or the native handle of one, such as a
        sample's component.
        """
        version = lib.arith_trace_version(getattr(element, "obj", element))
        return int(version) or None

    def _place(self, event: Event) -> None:
        if event.kind == "mark":
            if event.enter:
                parent = self._open[-1] if self._open else None
                prefix = f"{parent.path}/" if parent is not None and parent.path else ""
                self._open.append(
                    Operation(event.label, prefix + str(event.label), event.sequence)
                )
            elif self._open:
                self._close()
            return
        if self._open:
            self._open[-1].events.append(event)
            return
        # Outside every marked call: one unnamed operation per run of them.
        if not self._done or self._done[-1].name is not None:
            self._done.append(Operation(None, "", event.sequence))
        self._done[-1].events.append(event)

    def _close(self) -> None:
        operation = self._open.pop()
        if operation.events or operation.children:
            (self._open[-1].children if self._open else self._done).append(operation)

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
            primes=(
                tuple(self._ring(native.N, native.primes, native.base).primes)
                if native.primes
                else ()
            ),
            log_base=native.log_base,
            level=native.level,
            dropped=native.dropped,
            centered=bool(native.centered),
            label=ffi.string(native.label).decode()
            if native.label != ffi.NULL
            else None,
            enter=bool(native.enter) if kind == "mark" else None,
        )


def enable() -> Trace:
    """Starts recording and returns the `Trace` to read it from.

    A second call resumes the same tape; `Trace.clear` ends it.
    """
    trace = Trace()
    trace.enable()
    return trace


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
