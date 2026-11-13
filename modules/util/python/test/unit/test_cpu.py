# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""What the CPU reports, and what it will not claim to know.

A flag it cannot read answers None, so an engine can tell "this CPU lacks it"
from "nothing here can say".
"""

import os
import platform
import sys

import pytest
from vfhe.util._cpu import cpu

_EMULATED = bool(os.environ.get("VFHE_SDE_FLAGS"))
_LINUX_X86 = sys.platform.startswith("linux") and platform.machine() == "x86_64"


@pytest.fixture
def cpuinfo(monkeypatch, tmp_path):
    """A ``/proc/cpuinfo`` of the test's own, read fresh."""
    path = tmp_path / "cpuinfo"
    monkeypatch.setattr(cpu, "_CPUINFO", str(path))
    cpu._read_proc_cpuinfo.cache_clear()
    yield path
    cpu._read_proc_cpuinfo.cache_clear()


def test_a_flag_off_x86_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(cpu, "_machine", "aarch64")
    assert cpu.has_flag("avx512ifma") is False


def test_an_x86_with_nothing_to_read_cannot_say(monkeypatch) -> None:
    """Windows today: no flags to read, so the answer is None."""
    monkeypatch.setattr(cpu, "_machine", "x86_64")
    monkeypatch.setattr(cpu, "_read_proc_cpuinfo", lambda: None)
    assert cpu.has_flag("avx512ifma") is None


def test_the_flags_line_is_the_one_read(cpuinfo) -> None:
    cpuinfo.write_text(
        "processor\t: 0\nflags\t\t: fpu sse avx512ifma\nbugs\t\t: none\n"
    )
    assert cpu._read_proc_cpuinfo() == frozenset({"fpu", "sse", "avx512ifma"})


def test_a_file_without_a_flags_line_reads_as_none(cpuinfo) -> None:
    cpuinfo.write_text("processor\t: 0\n")
    assert cpu._read_proc_cpuinfo() is None


@pytest.mark.skipif(
    _EMULATED, reason="under SDE /proc/cpuinfo is the host's, not this process's"
)
@pytest.mark.skipif(
    not _LINUX_X86, reason="reads /proc/cpuinfo for the flags x86 engines need"
)
def test_the_flag_is_spelled_as_the_kernel_spells_it() -> None:
    """Read straight from the kernel, so a misspelling shows up as disagreement."""
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        flags: set[str] = set()
        for line in cpuinfo:
            key, separator, values = line.partition(":")
            if separator and key.strip() == "flags":
                flags = set(values.split())
                break
    assert flags, "no flags line to judge against"
    assert cpu.has_flag("avx512ifma") is ("avx512ifma" in flags)
