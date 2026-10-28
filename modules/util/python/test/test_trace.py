# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The execution tape's controls: enabling, marks, draining, clearing."""

import pytest
from vfhe.util import trace


@pytest.fixture(autouse=True)
def a_clean_tape():
    yield
    trace.enable().clear()


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
