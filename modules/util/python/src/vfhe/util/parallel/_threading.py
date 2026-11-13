# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The thread limits, over the engine's `lib`."""

from collections.abc import Generator
from contextlib import contextmanager

from vfhe.util.bindings import lib


def set_num_threads(n: int | None = None) -> None:
    """Limits the threads any vfhe operation may use to ``n``.

    ``None`` restores the default: ``VFHE_NUM_THREADS`` if set, otherwise 1.
    """
    if n is not None and n < 1:
        raise ValueError(f"the thread limit must be at least 1, got {n}")
    lib.vfhe_set_num_threads(0 if n is None else n)


def num_threads() -> int:
    """Returns the library-wide thread limit (see `set_num_threads`)."""
    return lib.vfhe_num_threads()


@contextmanager
def local_num_threads(n: int) -> Generator[None, None, None]:
    """Limits the threads of the vfhe calls this thread makes inside to ``n``.

    ``0`` sets no local limit. The library limit still applies, and a local
    limit set inside another can only lower it.
    """
    if n < 0:
        raise ValueError(f"the local number of threads must be at least 0, got {n}")
    if n == 0:
        yield
        return
    outer = lib.vfhe_local_num_threads()
    previous = lib.vfhe_set_local_num_threads(n if outer == 0 else min(n, outer))
    try:
        yield
    finally:
        lib.vfhe_set_local_num_threads(previous)
