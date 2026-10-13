# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Reports the instructions available on this machine's CPU."""

from __future__ import annotations

import contextlib
import functools
import platform

_machine = platform.machine()
_CPUINFO = "/proc/cpuinfo"


@functools.cache
def _read_proc_cpuinfo() -> frozenset[str] | None:
    """Reads the CPU flags from ``/proc/cpuinfo`` once; None when no flags line can be read."""
    with contextlib.suppress(OSError), open(_CPUINFO) as cpuinfo:
        for line in cpuinfo:
            name, _, values = line.partition(":")
            if name.strip() == "flags":
                return frozenset(values.split())
    return None


def is_arm() -> bool:
    """Reports whether this machine is 64-bit Arm."""
    return _machine in ("aarch64", "arm64")


def is_x86_64() -> bool:
    """Reports whether this machine is x86-64."""
    return _machine == "x86_64"


def has_flag(flag: str) -> bool | None:
    """Reports whether this CPU has the x86-64 feature `flag`, as spelled in ``/proc/cpuinfo``.

    Returns False on every other architecture, and None when the CPU cannot say,
    such as on an x86-64 machine with no readable ``/proc/cpuinfo``.
    """
    if not is_x86_64():
        return False
    flags = _read_proc_cpuinfo()
    return None if flags is None else flag in flags
