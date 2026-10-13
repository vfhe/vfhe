# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Reports the size of this machine's last level of cache, for sizing a working set."""

from __future__ import annotations

import contextlib
import functools
import os
import pathlib
import subprocess
import sys

_SYSFS = pathlib.Path("/sys/devices/system/cpu/cpu0/cache")


@functools.cache
def last_level_cache_bytes() -> int | None:
    """Returns the size in bytes of the last level of cache before memory.

    Asks the machine once per process; None when no source can say.
    """
    return _sysconf() or _darwin() or _sysfs()


def _sysconf() -> int | None:
    """Asks the C library, which answers on glibc; None when it cannot say."""
    # _SC_LEVEL3_CACHE_SIZE, then _SC_LEVEL2_CACHE_SIZE, by their glibc numbers:
    # os.sysconf_names lacks both, and a libc without them raises.
    for code in (194, 191):
        with contextlib.suppress(OSError, ValueError):
            if (size := os.sysconf(code)) > 0:
                return size
    return None


def _darwin() -> int | None:
    """Asks ``sysctl`` on macOS; None on other systems and when it cannot say."""
    if sys.platform != "darwin":
        return None
    # hw.l3cachesize is absent on Apple silicon, where the P-core cluster's L2
    # is the last level before memory.
    for name in ("hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"):
        with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
            asked = subprocess.run(  # noqa: S603 - a fixed command, no user input
                ["/usr/sbin/sysctl", "-n", name],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if size := int(asked.stdout):
                return size
    return None


def _sysfs() -> int | None:
    """Reads the deepest cache level from 2 up in Linux sysfs; None when none is listed."""
    sizes: dict[int, int] = {}
    for index in _SYSFS.glob("index*"):
        with contextlib.suppress(OSError, ValueError):
            level = int((index / "level").read_text())
            if level >= 2:
                sizes[level] = int((index / "size").read_text().rstrip("K\n")) * 1024
    return sizes[max(sizes)] if sizes else None
