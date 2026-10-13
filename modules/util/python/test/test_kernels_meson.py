# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.kernels._meson`: the ``meson.build`` it writes and the two commands it runs."""

import types

import pytest
from vfhe.util.kernels import _meson


def _written(tmp_path) -> str:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    _meson.write(
        out,
        module="_vfhe_native_portable",
        # as meson emits them: the engine's own flags, including the one only
        # the library's own objects get
        c_args=["-DX", "-DVFHE_BUILDING_LIBRARY"],
        include_dirs=[tmp_path / "inc"],
        lib_sources=[tmp_path / "arith.c"],
        user_sources=[tmp_path / "mine.c", tmp_path / "wrap.c"],
    )
    return (out / "meson.build").read_text()


def test_lto_is_always_on(tmp_path) -> None:
    """Inlining vfhe into the caller's loops is the whole reason to build."""
    assert "'b_lto=true'" in _written(tmp_path)


def test_only_vfhe_reads_the_concrete_vector_type(tmp_path) -> None:
    """VFHE_BUILDING_LIBRARY hands out the engine's real mp_vector_t. The
    caller's files are compiled against the public header and see the same
    opaque type every other consumer does."""
    library, module = _written(tmp_path).split("shared_module")
    assert "-DVFHE_BUILDING_LIBRARY" in library
    assert "-DVFHE_BUILDING_LIBRARY" not in module


def test_the_engines_flags_and_every_source_reach_the_file(tmp_path) -> None:
    text = _written(tmp_path)
    for expected in ("'-DX'", "arith.c", "mine.c", "wrap.c"):
        assert expected in text


def test_an_unchanged_file_is_left_alone(tmp_path) -> None:
    """A rewrite would make meson reconfigure."""
    _written(tmp_path)
    before = (tmp_path / "out" / "meson.build").stat().st_mtime_ns
    _written(tmp_path)
    assert (tmp_path / "out" / "meson.build").stat().st_mtime_ns == before


def test_meson_configures_once_then_installs(monkeypatch, tmp_path) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command[1], kwargs["env"]["CC"]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_meson.subprocess, "run", run)
    _meson.build(tmp_path, cc="tcc")
    assert calls == [("setup", "tcc"), ("install", "tcc")]
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "build.ninja").write_text("")
    calls.clear()
    _meson.build(tmp_path, cc="tcc")
    assert calls == [("install", "tcc")]


def test_meson_failures_name_the_extra_and_carry_the_output(
    monkeypatch, tmp_path
) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(_meson.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match=r"vfhe\[kernels\]"):
        _meson.build(tmp_path, cc="cc")

    def failing(*_args, **_kwargs):
        return types.SimpleNamespace(returncode=1, stdout="out ", stderr="err")

    monkeypatch.setattr(_meson.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="out err"):
        _meson.build(tmp_path, cc="cc")
