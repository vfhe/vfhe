# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""VFHE_ENGINE is an instruction: the probe has no veto.

Under an emulator the probe describes the bare CPU while the process can run
more, so a pin the probe cannot vouch for still loads. What a pin costs is an
illegal instruction when it is wrong, which is what asking for it means.

Aimed at `vfhe.util._engine`: selection answers before anything native is
mapped, which is what lets `vfhe.util.kernels` build the engine it is about to
load.
"""

import pytest
from vfhe.util._engine import Engine


def _select(pin, monkeypatch):
    monkeypatch.setenv("VFHE_ENGINE", pin)
    return Engine.select()


def test_a_pin_loads_without_asking_the_probe(monkeypatch) -> None:
    monkeypatch.setattr(Engine, "runnable", lambda _self: False)
    for engine in Engine:
        if engine.installed():
            assert _select(engine.value, monkeypatch) is engine
            return
    pytest.skip("no engine installed to pin")


def test_an_uninstalled_pin_is_still_refused(monkeypatch) -> None:
    monkeypatch.setattr(Engine, "installed", lambda _self: False)
    with pytest.raises(RuntimeError, match="not installed"):
        _select(Engine.PORTABLE.value, monkeypatch)


def test_an_unknown_pin_names_what_the_build_has(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match=r"unknown VFHE_ENGINE.*portable"):
        _select("nonsense", monkeypatch)


def test_without_a_pin_the_probe_decides(monkeypatch) -> None:
    monkeypatch.setattr(Engine, "runnable", lambda self: self is Engine.PORTABLE)
    # Only the fallback requires nothing, so it is what a blind probe leaves.
    assert _select("", monkeypatch) is Engine.PORTABLE


def test_no_installed_engine_is_an_error(monkeypatch) -> None:
    monkeypatch.delenv("VFHE_ENGINE", raising=False)
    monkeypatch.setattr(Engine, "installed", lambda _self: False)
    with pytest.raises(RuntimeError, match="no installed engine"):
        Engine.select()
