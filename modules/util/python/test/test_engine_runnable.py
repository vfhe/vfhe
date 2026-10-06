# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Which engines run here: "cannot run" and "no such engine" stay apart.

The test runner skips (77) on the first and fails (1) on the second, so a
mistyped engine fails the run.
"""

import platform

import pytest
from vfhe.util._cpu import cpu
from vfhe.util._engine import Engine


def test_the_fallback_runs_anywhere(monkeypatch) -> None:
    for machine in ("x86_64", "aarch64"):
        monkeypatch.setattr(cpu, "_machine", machine)
        assert Engine("portable").runnable() is True


def test_an_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        Engine("nonsense")


def test_neon_follows_the_architecture(monkeypatch) -> None:
    monkeypatch.setattr(cpu, "_machine", "aarch64")
    assert Engine("neon").runnable() is True
    monkeypatch.setattr(cpu, "_machine", "x86_64")
    assert Engine("neon").runnable() is False


def test_this_architecture_judges_its_own_engines() -> None:
    machine = platform.machine()
    if machine == "x86_64":
        assert Engine("avx512ifma").runnable() is not None
    elif machine in ("aarch64", "arm64"):
        assert Engine("neon").runnable() is True
        # An x86 engine is a name this knows and this machine cannot run.
        assert Engine("avx512ifma").runnable() is False
