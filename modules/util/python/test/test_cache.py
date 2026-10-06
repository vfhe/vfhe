# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The cache probe's contract: a plausible size, or None when no source can say."""

import shutil
import subprocess
import types

from vfhe.util._cpu import cache, last_level_cache_bytes


def test_a_reported_size_is_plausible() -> None:
    reported = last_level_cache_bytes()
    if reported is not None:
        # Every real last level is at least this, and a whole number of KiB.
        assert reported >= 64 * 1024
        assert reported % 1024 == 0


def test_sysfs_reports_the_deepest_level_from_two_up(monkeypatch, tmp_path) -> None:
    for index, level, size in (
        ("index0", 1, "32K"),
        ("index1", 1, "48K"),
        ("index2", 2, "1024K"),
        ("index3", 3, "32768K"),
    ):
        (tmp_path / index).mkdir()
        (tmp_path / index / "level").write_text(f"{level}\n")
        (tmp_path / index / "size").write_text(f"{size}\n")
    monkeypatch.setattr(cache, "_SYSFS", tmp_path)
    assert cache._sysfs() == 32768 * 1024
    shutil.rmtree(tmp_path / "index3")
    assert cache._sysfs() == 1024 * 1024
    shutil.rmtree(tmp_path / "index2")
    assert cache._sysfs() is None


def test_sysconf_asks_for_l3_then_l2(monkeypatch) -> None:
    # glibc's _SC_LEVEL3_CACHE_SIZE answers -1 for "unknown"; _SC_LEVEL2_CACHE_SIZE 2 MiB.
    monkeypatch.setattr(cache.os, "sysconf", {194: -1, 191: 2**21}.__getitem__)
    assert cache._sysconf() == 2**21

    def unknown(_code):
        raise OSError

    monkeypatch.setattr(cache.os, "sysconf", unknown)
    assert cache._sysconf() is None


def test_darwin_asks_sysctl_on_macos_alone(monkeypatch) -> None:
    monkeypatch.setattr(cache.sys, "platform", "linux")
    assert cache._darwin() is None

    monkeypatch.setattr(cache.sys, "platform", "darwin")
    answers = {
        "hw.l3cachesize": "",
        "hw.perflevel0.l2cachesize": "16777216\n",
        "hw.l2cachesize": "",
    }

    def sysctl(command, **_kwargs):
        if not answers[command[-1]]:
            raise subprocess.CalledProcessError(1, command)
        return types.SimpleNamespace(stdout=answers[command[-1]])

    monkeypatch.setattr(cache.subprocess, "run", sysctl)
    assert cache._darwin() == 16777216
    answers["hw.perflevel0.l2cachesize"] = ""
    assert cache._darwin() is None
