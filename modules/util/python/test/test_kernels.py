# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`ExtensionBuilder` before a compiler is involved: what names its directory,
what it hands each step, and when `build` refuses.

The end-to-end build is the smoke suite's job, since only an install proves
the source snapshot shipped.
"""

import importlib
import pathlib
import runpy
import sys
import types

import pytest
from vfhe.util._engine import Engine
from vfhe.util.kernels import _build, _compiler, _ffi, _meson, _sources


@pytest.fixture(autouse=True)
def _fixed_compiler(monkeypatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Keying must not depend on whichever cc this machine has, and construction
    must not depend on which engines this build has."""
    monkeypatch.setattr(_compiler, "version", lambda cc: f"{cc} 1.0")
    monkeypatch.setattr(_compiler, "flags", lambda _engine: ["-DVFHE_BUILDING_LIBRARY"])


def _builder(sources=(), declarations=(), **options) -> _build.ExtensionBuilder:
    options = {"engine": "portable", "cc": "cc", **options}
    return _build.ExtensionBuilder(sources, declarations, **options)


def _directory(**over) -> pathlib.Path:
    return _builder(**over).directory


def _run_cli(monkeypatch, argv) -> None:
    monkeypatch.setattr(sys, "argv", ["vfhe.util.kernels", *argv])
    runpy.run_module("vfhe.util.kernels", run_name="__main__")


def test_a_bad_engine_fails_at_construction(monkeypatch) -> None:
    with pytest.raises(ValueError, match="nonsense"):
        _build.ExtensionBuilder(engine="nonsense")

    def lacking(engine):
        raise RuntimeError(f"this install has no {engine} engine")

    monkeypatch.setattr(_compiler, "flags", lacking)
    with pytest.raises(RuntimeError, match="no neon engine"):
        _build.ExtensionBuilder(engine="neon")


def test_the_engine_and_the_compiler_change_the_directory() -> None:
    assert _directory() != _directory(engine="neon")
    assert _directory() != _directory(cc="different")


def test_paths_name_the_build_and_contents_do_not(tmp_path) -> None:
    """Editing a file lands in the same directory: ninja decides what is stale."""
    c, h = tmp_path / "a.c", tmp_path / "a.h"
    h.write_text("int f(void);")
    before = _directory(sources=[c], declarations=[h])
    h.write_text("int g(void);")
    assert _directory(sources=[c], declarations=[h]) == before
    assert _directory(sources=[tmp_path / "b.c"], declarations=[h]) != before
    assert _directory(sources=[c], declarations=[tmp_path / "b.h"]) != before


@pytest.mark.usefixtures("cache")
def test_compile_hands_each_step_its_inputs(monkeypatch, tmp_path) -> None:
    tree = _sources.SourceTree(
        cdefs=[tmp_path / "lib.cdef.h"],
        headers=[tmp_path / "lib.h"],
        include_dirs=[tmp_path / "inc"],
        sources=[tmp_path / "lib.c"],
    )
    monkeypatch.setattr(_sources, "source_tree", lambda: tree)
    seen = {}
    monkeypatch.setattr(_ffi, "emit", lambda *args: seen.update(emit=args))
    monkeypatch.setattr(_meson, "write", lambda out, **kw: seen.update(write=(out, kw)))
    monkeypatch.setattr(_meson, "build", lambda out, **kw: seen.update(build=(out, kw)))
    mine_c = (tmp_path / "k" / "mine.c").resolve()
    mine_h = (tmp_path / "h" / "mine.h").resolve()

    b = _builder([mine_c], [mine_h], cc="tcc")
    out = b.compile()
    wrapper = out / "_vfhe_native_portable.c"
    assert out == b.directory
    assert seen["emit"] == (
        "_vfhe_native_portable",
        wrapper,
        [*tree.cdefs, mine_h],
        [*tree.headers, mine_h],
    )
    written_to, write = seen["write"]
    assert written_to == out
    assert write["module"] == "_vfhe_native_portable"
    assert write["c_args"] == ["-DVFHE_BUILDING_LIBRARY"]
    assert write["include_dirs"] == [*tree.include_dirs, mine_h.parent, mine_c.parent]
    assert write["lib_sources"] == tree.sources
    assert write["user_sources"] == [mine_c, wrapper]
    assert seen["build"] == (out, {"cc": "tcc"})


@pytest.mark.usefixtures("cache")
def test_build_puts_the_fresh_engine_first_on_the_path(monkeypatch) -> None:
    monkeypatch.setattr(Engine, "active", classmethod(lambda _cls: None))
    monkeypatch.setattr(_build.ExtensionBuilder, "compile", lambda self: self.directory)
    monkeypatch.setattr(sys, "path", list(sys.path))
    b = _builder(["nothing.c"])
    assert b.build() == b.directory
    assert sys.path[0] == str(b.directory)


@pytest.mark.usefixtures("cache")
def test_a_late_build_is_refused() -> None:
    """A loaded extension cannot be replaced, so a caller left quietly on the
    prebuilt engine would find their kernels absent."""
    importlib.import_module("vfhe.util.bindings")
    assert Engine.active() is not None
    with pytest.raises(RuntimeError, match="already loaded"):
        _build.ExtensionBuilder(["nothing.c"]).build()


def test_building_what_is_already_loaded_is_a_no_op(monkeypatch, cache) -> None:
    importlib.import_module("vfhe.util.bindings")
    active = Engine.active()
    assert active is not None
    here = cache / "extensions" / f"{active.value}-deadbeef"
    monkeypatch.setattr(
        _build.ExtensionBuilder, "directory", property(lambda _self: here)
    )
    loaded = types.SimpleNamespace(__file__=str(here / "x.so"))
    monkeypatch.setitem(sys.modules, active.module, loaded)
    assert _build.ExtensionBuilder(["nothing.c"], engine=active.value).build() == here


def test_the_cli_sorts_files_by_extension_and_prints_where_it_landed(
    monkeypatch, capsys, tmp_path
) -> None:
    """`python -m vfhe.util.kernels` is the documented way to build without loading,
    so what it hands `ExtensionBuilder` is the contract."""
    seen = {}

    def fake(self):
        seen.update(vars(self))
        return tmp_path

    monkeypatch.setattr(_build.ExtensionBuilder, "compile", fake)
    _run_cli(monkeypatch, ["mine.c", "mine.h", "helpers.S", "--engine", "portable"])
    assert seen["_sources"] == [
        pathlib.Path(f).resolve() for f in ("mine.c", "helpers.S")
    ]
    assert seen["_declarations"] == [pathlib.Path("mine.h").resolve()]
    assert seen["_engine"] == "portable"
    assert capsys.readouterr().out.strip() == str(tmp_path)


def test_the_cli_refuses_declarations_without_a_source(monkeypatch) -> None:
    with pytest.raises(SystemExit) as refused:
        _run_cli(monkeypatch, ["only.h"])
    assert refused.value.code == 2
