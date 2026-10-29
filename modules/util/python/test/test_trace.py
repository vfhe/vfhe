# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The execution tape's controls: enabling, marks, draining, clearing."""

import sys
import threading

import pytest
from vfhe.arith import Ring
from vfhe.util import trace
from vfhe.util.bindings import lib


@pytest.fixture(autouse=True)
def a_clean_tape():
    trace.set_verify(True)
    yield
    t = trace.enable()
    untraced = t.untraced
    t.clear()
    assert untraced == 0


def test_recording_is_off_until_enabled_and_off_again_after() -> None:
    t = trace.enable()
    assert t.enabled
    t.disable()
    assert not t.enabled
    with trace.recording() as inner:
        assert inner.enabled
    assert not inner.enabled


def test_marks_are_read_in_order_and_drained() -> None:
    with trace.recording() as t:
        trace.mark("outer")
        trace.mark("inner")
        trace.mark("inner", enter=False)
        trace.mark("outer", enter=False)
    assert len(t) == 4
    events = t.events()
    assert [(e.kind, e.label, e.enter) for e in events] == [
        ("mark", "outer", True),
        ("mark", "inner", True),
        ("mark", "inner", False),
        ("mark", "outer", False),
    ]
    assert [e.sequence for e in events] == [1, 2, 3, 4]
    assert all(e.value is None and e.ring is None for e in events)
    assert len(t) == 0
    assert t.events() == []


def test_the_tape_goes_on_across_disable_and_enable_until_cleared() -> None:
    t = trace.enable()
    trace.mark("a")
    t.disable()
    t.enable()
    trace.mark("b")
    assert [e.label for e in t.events()] == ["a", "b"]
    t.clear()
    assert not t.enabled
    assert len(t) == 0
    assert trace.enable().events(kind="mark") == []


def _polynomial():
    return Ring(64, prime_size=[45, 45], split_degree=1).random_element()


def test_the_hook_is_installed_only_while_enabled() -> None:
    before = sys.getprofile()
    t = trace.enable()
    assert sys.getprofile() is not None
    assert sys.getprofile() is not before
    t.disable()
    assert sys.getprofile() is before
    with trace.recording():
        assert sys.getprofile() is not before
    assert sys.getprofile() is before


def test_public_vfhe_calls_are_marked_and_private_ones_are_not() -> None:
    x = _polynomial()
    with trace.recording() as t:
        y = x + x
        y.get_polynomial()  # reads through the private _viewed_as
    marks = [(e.label or "", e.enter) for e in t.events(kind="mark")]
    assert marks.index(("RNSPolynomial.__add__", True)) < marks.index(
        ("RNSPolynomial.add", True)
    )
    assert ("RNSPolynomial.add", False) in marks
    assert any(label.endswith(".get_polynomial") for label, _ in marks)
    names = {label.rsplit(".", 1)[-1] for label, _ in marks}
    assert all(not n.startswith("_") or n in trace.OPERATORS for n in names)
    assert "__init__" not in names and "_viewed_as" not in names


def test_operations_are_the_marked_calls_that_made_ring_operations() -> None:
    x = _polynomial()
    with trace.recording() as t:
        trace.mark("sum")
        y = x + x
        trace.mark("nothing")
        trace.mark("nothing", enter=False)
        trace.mark("sum", enter=False)
        lib.arith_zero(x.ring.arith_ring, y.as_element())
    [total, stray] = t.operations()
    assert (total.name, total.path, total.sequence) == ("sum", "sum", 1)
    # RNSRing.intersec made no ring operation and "nothing" held none: left out.
    [addition] = total.children
    assert addition.path == "sum/RNSPolynomial.__add__"
    [add] = addition.children
    assert add.path == "sum/RNSPolynomial.__add__/RNSPolynomial.add"
    assert [e.kind for e in add.events] == ["input", "add"]
    assert [e.kind for e in total.all_events()] == ["input", "add"]
    assert total.find("RNSPolynomial.add") == [add]
    assert [op.path for op in total.walk()] == [
        "sum",
        "sum/RNSPolynomial.__add__",
        "sum/RNSPolynomial.__add__/RNSPolynomial.add",
    ]
    assert (stray.name, stray.path) == (None, "")
    assert [e.kind for e in stray.events] == ["zero"]
    assert t.operations() == []


def test_open_operations_wait_for_their_exit_and_close_at_disable() -> None:
    x = _polynomial()
    t = trace.enable()
    trace.mark("phase")
    y = x + x
    assert t.operations() == []
    t.disable()
    [phase] = t.operations()
    assert phase.name == "phase"
    assert [e.kind for e in phase.all_events()] == ["input", "add"]
    # A closing mark with nothing open is ignored.
    t.enable()
    trace.mark("phase", enter=False)
    y += x
    t.disable()
    [addition] = t.operations()
    assert addition.name == "RNSPolynomial.__iadd__"
    assert [e.kind for e in addition.all_events()] == ["add"]


def test_version_of_is_the_event_that_last_wrote_the_value() -> None:
    x = _polynomial()
    with trace.recording() as t:
        y = x + x
        y += x
    assert t.version_of(_polynomial()) is None
    [x_in, first, second] = [e for e in t.events() if e.kind != "mark"]
    assert t.version_of(x) == x_in.sequence
    assert t.version_of(y) == second.sequence
    assert t.version_of(y.obj) == second.sequence
    assert second.inputs == (first.sequence, x_in.sequence)


def test_a_bounded_tape_waits_for_a_consumer() -> None:
    x = _polynomial()
    t = trace.enable(limit=4)
    assert t.limit == 4
    seen: list = []
    lengths: list = []

    def consume():
        while t.wait():
            lengths.append(len(t))
            seen.extend(t.events())

    consumer = threading.Thread(target=consume)
    consumer.start()
    y = x
    for _ in range(20):
        y = y + x
    t.disable()
    consumer.join(timeout=30)
    assert not consumer.is_alive()
    assert max(lengths) <= 4
    assert [e.kind for e in seen if e.kind != "mark"] == ["input"] + ["add"] * 20
    t.clear()


def test_verify_flags_a_write_the_tape_did_not_see() -> None:
    x = _polynomial()
    t = trace.enable(verify=True)
    y = x + x
    x.negate()  # in place, not through the generic interface
    z = x + y
    x.to_coeff()  # a conversion rewrites the storage but is not a write
    w = x + z
    t.disable()
    events = [e for e in t.events() if e.kind != "mark"]
    assert [(e.kind, e.untraced) for e in events] == [
        ("input", False),
        ("add", False),
        ("input", True),
        ("add", False),
        ("add", False),
    ]
    assert events[2].value == x and events[4].value == w
    assert t.untraced == 1
    t.clear()
    assert t.untraced == 0
