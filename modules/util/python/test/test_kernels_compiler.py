# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.kernels._compiler`: the command, its version, and the engines' flags."""

import platform
import sys

import pytest
from vfhe.util.kernels import _compiler


def test_the_command_is_cc_from_the_environment_else_pythons(monkeypatch) -> None:
    monkeypatch.setenv("CC", "clang -m64")
    assert _compiler.command() == "clang -m64"
    monkeypatch.delenv("CC")
    monkeypatch.setattr(_compiler.sysconfig, "get_config_var", lambda _name: None)
    assert _compiler.command() == "cc"


def test_the_version_is_the_first_line_the_compiler_prints() -> None:
    expected = f"{sys.executable} Python {platform.python_version()}"
    assert _compiler.version(sys.executable) == expected
    assert _compiler.version("no-such-cc") == "no-such-cc version-unknown"


def test_the_flags_are_this_builds_table() -> None:
    assert "-DVFHE_BUILDING_LIBRARY" in _compiler.flags("portable")
    with pytest.raises(RuntimeError, match="no nonsense engine"):
        _compiler.flags("nonsense")
