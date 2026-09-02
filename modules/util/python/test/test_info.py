# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""`python -m vfhe.info` is what a bug report pastes, so it must report the
engine truthfully and never raise — including from a source tree, where there
is no distribution metadata to read."""

from importlib.metadata import PackageNotFoundError

from vfhe import info
from vfhe.engine import active_engine


def test_facts_cover_the_environment():
    labels = [label for label, _ in info.collect_facts()]
    assert labels == ["vfhe", "engine", "python", "platform"]


def test_engine_fact_names_the_loaded_engine():
    assert active_engine() in info.describe_engine()


def test_engine_fact_reports_a_faster_engine_the_cpu_could_run(monkeypatch):
    monkeypatch.setattr(info, "active_engine", lambda: "portable")
    monkeypatch.setattr(info, "runnable_engines", lambda: ["avx512ifma", "portable"])
    assert "can also run: avx512ifma" in info.describe_engine()


def test_version_says_so_when_there_is_no_install(monkeypatch):
    def raise_not_found(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(info, "version", raise_not_found)
    assert "source tree" in info.find_version()


def test_facts_render_aligned():
    rendered = info.format_facts([("a", "1"), ("long", "2")])
    assert rendered == "a     1\nlong  2"
