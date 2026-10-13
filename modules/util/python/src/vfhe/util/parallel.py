# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sets how many threads vfhe uses: one library-wide limit, and a local one per thread.

- `set_num_threads` and `num_threads` set and read the library-wide limit.
  Functions taking ``n_threads`` never exceed it, and their default
  ``n_threads=0`` means "up to the limit".
- `local_num_threads` lowers the limit for the vfhe calls the current thread
  makes inside a ``with`` block.

The default limit is ``VFHE_NUM_THREADS`` if set, otherwise 1, so vfhe is
single-threaded unless asked.

>>> from vfhe.util import parallel
>>> parallel.set_num_threads(8)
>>> with parallel.local_num_threads(2):
...     pass  # vfhe calls made here use at most 2 threads
>>> parallel.set_num_threads()
"""

from collections.abc import Iterator
from contextlib import contextmanager

from vfhe.util.bindings import lib

__all__ = ["local_num_threads", "num_threads", "set_num_threads"]


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
def local_num_threads(n: int) -> Iterator[None]:
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
