# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""`vfhe.util.kernels._sources`: the build cache, and vfhe's C from the shipped tarball."""

import contextlib
import importlib.resources
import io
import pathlib
import tarfile
import types

import pytest
from vfhe.util.kernels import _sources


def _install(monkeypatch, tmp_path, tarball: bytes) -> None:
    """Stands in for an installed package whose ``_source/kernels-src.tar.gz`` is `tarball`."""
    package = tmp_path / "pkg"
    (package / "_source").mkdir(parents=True)
    (package / "_source" / "kernels-src.tar.gz").write_bytes(tarball)
    monkeypatch.setattr(importlib.resources, "files", lambda _package: package)


def _tarball(tmp_path, *members: str) -> bytes:
    """A gzipped tar of empty files at the paths `members`."""
    root = tmp_path / "tree"
    for member in members:
        (root / member).parent.mkdir(parents=True, exist_ok=True)
        (root / member).write_text("")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(root, arcname=".")
    return buffer.getvalue()


def test_the_cache_obeys_its_override(cache) -> None:
    assert _sources.cache_root() == cache


def test_without_an_override_the_cache_is_the_user_cache(monkeypatch) -> None:
    monkeypatch.delenv("VFHE_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/somewhere")
    assert _sources.cache_root() == pathlib.Path("/somewhere/vfhe").resolve()


def test_the_source_tree_is_the_tarball_by_role(monkeypatch, cache, tmp_path) -> None:
    _install(
        monkeypatch,
        tmp_path,
        _tarball(
            tmp_path,
            "modules/m/c/include/m.h",
            "modules/m/python/cdef/m.h",
            "modules/m/c/src/m.c",
            "modules/m/c/src/m.S",
            "external/blake3/blake3/c/blake3.c",
        ),
    )
    tree = _sources.source_tree()
    out = cache / "src" / _sources.version()
    assert tree.cdefs == [out / "modules/m/python/cdef/m.h"]
    assert tree.headers == [out / "modules/m/c/include/m.h"]
    assert tree.include_dirs == [
        out / "modules/m/c/include",
        out / "modules/arith/c/src",
        out / "external/blake3/blake3/c",
    ]
    assert sorted(p.name for p in tree.sources) == ["blake3.c", "m.S", "m.c"]
    # The unpacked tree is reused: a mark left in it survives the next call.
    (out / "mark").write_text("")
    _sources.source_tree()
    assert (out / "mark").exists()


def test_an_install_without_the_tarball_says_so(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(importlib.resources, "files", lambda _package: tmp_path)
    with pytest.raises(RuntimeError, match="C sources are not available"):
        _sources.source_tree()


def test_a_failed_unpack_leaves_no_staging_behind(monkeypatch, cache, tmp_path) -> None:
    _install(monkeypatch, tmp_path, b"")

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_sources.tarfile, "open", refuse)
    with pytest.raises(OSError, match="disk full"):
        _sources.source_tree()
    assert list((cache / "src").iterdir()) == []


def test_losing_the_unpack_race_is_a_success(monkeypatch, cache, tmp_path) -> None:
    """Another process finishing first is the one failed rename that is fine."""
    _install(monkeypatch, tmp_path, b"")
    out = cache / "src" / _sources.version()
    unpacked = types.SimpleNamespace(extractall=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _sources.tarfile,
        "open",
        lambda *_args, **_kwargs: contextlib.nullcontext(unpacked),
    )

    def lose(_staging, _target):
        (out / "modules").mkdir(parents=True)
        raise OSError

    monkeypatch.setattr(pathlib.Path, "replace", lose)
    assert _sources.source_tree().sources == []
