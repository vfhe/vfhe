# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The library's memory pool.

vfhe keeps the large buffers it releases, per thread, and hands them out again
for the next request of the same size, instead of returning them to the C
library and faulting fresh memory in on the next allocation. What it keeps
over all threads is bounded by a capacity:

- ``"auto"`` (the default): at most the most the library ever had in use at
  once, so a repeating workload is served from the pool and, once its
  buffers are released, the pool holds no more than the library's own peak.
- a number of bytes.
- ``0``: keep nothing; every buffer goes back to the C library.

``VFHE_MEMPOOL_CAPACITY`` (bytes) sets the default.

A released buffer keeps whatever it held until it is overwritten. With
`set_wipe_on_release` every buffer is zeroed as it is released, at the cost
of a pass over it; off by default (``VFHE_MEMPOOL_WIPE_ON_RELEASE=1`` turns it
on).

Both settings survive `vfhe.dynamic_extensions` reloads.
"""

from dataclasses import dataclass
from typing import Literal

from ._native import ffi, lib

# The last setting given to set_capacity, as the native value, re-applied to
# libraries loaded by vfhe.dynamic_extensions.
_capacity_setting = lib.MEMPOOL_CAPACITY_DEFAULT
# The same for set_wipe_on_release: -1 for the default.
_wipe_setting = -1


def set_capacity(n_bytes: int | Literal["auto"] | None = None) -> None:
    """Bounds what the pool keeps, over all threads, to ``n_bytes``.

    ``"auto"`` follows the library's peak use and ``0`` turns the pool off;
    ``None`` restores the default (``VFHE_MEMPOOL_CAPACITY`` if set, else
    ``"auto"``). Lowering the capacity below what the pool keeps releases it
    all.
    """
    global _capacity_setting
    if n_bytes is None:
        setting = lib.MEMPOOL_CAPACITY_DEFAULT
    elif n_bytes == "auto":
        setting = lib.MEMPOOL_CAPACITY_AUTO
    elif isinstance(n_bytes, int) and 0 <= n_bytes < lib.MEMPOOL_CAPACITY_DEFAULT:
        setting = n_bytes
    else:
        raise ValueError(
            f'the capacity is a number of bytes, "auto" or None, got {n_bytes!r}'
        )
    _capacity_setting = setting
    lib.mempool_set_capacity(setting)


def capacity() -> int | Literal["auto"]:
    """The capacity in force (see `set_capacity`)."""
    setting = lib.mempool_capacity()
    return "auto" if setting == lib.MEMPOOL_CAPACITY_AUTO else setting


def set_wipe_on_release(enabled: bool | None = None) -> None:
    """Zero every buffer as it is released to the pool, or stop doing so.

    ``None`` restores the default: off, unless
    ``VFHE_MEMPOOL_WIPE_ON_RELEASE=1``.
    """
    global _wipe_setting
    _wipe_setting = -1 if enabled is None else int(bool(enabled))
    lib.mempool_set_wipe_on_release(_wipe_setting)


def wipe_on_release() -> bool:
    """Whether released buffers are zeroed (see `set_wipe_on_release`)."""
    return bool(lib.mempool_wipe_on_release())


def release_all() -> None:
    """Returns every buffer the pool keeps, on every thread, to the C library."""
    lib.mempool_release_all()


@dataclass(frozen=True)
class MemoryPoolStatistics:
    """A snapshot of the pool, over all threads."""

    retained_bytes: int
    """Kept by the pool, ready to be handed out."""
    outstanding_bytes: int
    """Handed out and not yet released to the pool."""
    peak_outstanding_bytes: int
    """The most ever outstanding at once; the ``"auto"`` capacity."""
    retention_limit_bytes: int
    """What the capacity allows the pool to keep right now."""
    hits: int
    """Requests served from the pool."""
    misses: int
    """Requests passed to the C library."""


def statistics() -> MemoryPoolStatistics:
    """The pool's current accounting (see `MemoryPoolStatistics`)."""
    out = ffi.new("MempoolStatistics *")
    lib.mempool_statistics(out)
    return MemoryPoolStatistics(
        retained_bytes=out.retained_bytes,
        outstanding_bytes=out.outstanding_bytes,
        peak_outstanding_bytes=out.peak_outstanding_bytes,
        retention_limit_bytes=out.retention_limit_bytes,
        hits=out.hits,
        misses=out.misses,
    )
