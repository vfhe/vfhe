# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The memory pool's controls, over the engine's `lib`."""

from typing import Literal

from vfhe.util.bindings import lib


def set_capacity(n_bytes: int | Literal["auto"] | None = None) -> None:
    """Bounds what the pool keeps, over all threads, to ``n_bytes``.

    ``"auto"`` follows the library's peak use and ``0`` turns the pool off;
    ``None`` restores the default (``VFHE_MEMPOOL_CAPACITY`` if set, else
    ``"auto"``). Lowering the capacity below what the pool keeps releases it
    all. A capacity below what a workload reuses costs speed on every
    request the pool no longer serves (see `release_all`).
    """
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
    lib.mempool_set_capacity(setting)


def capacity() -> int | Literal["auto"]:
    """Returns the capacity in force (see `set_capacity`)."""
    setting = lib.mempool_capacity()
    return "auto" if setting == lib.MEMPOOL_CAPACITY_AUTO else setting


def set_wipe_on_release(enabled: bool | None = None) -> None:
    """Zeroes every buffer as it is released to the pool, or stops doing so.

    ``None`` restores the default: off, unless
    ``VFHE_MEMPOOL_WIPE_ON_RELEASE=1``.
    """
    lib.mempool_set_wipe_on_release(-1 if enabled is None else int(bool(enabled)))


def wipe_on_release() -> bool:
    """Returns whether released buffers are zeroed (see `set_wipe_on_release`)."""
    return bool(lib.mempool_wipe_on_release())


def release_all() -> None:
    """Returns every buffer the pool keeps, on every thread, to the C library.

    This costs speed afterwards: the next requests for those sizes are
    served by the C library again, and a large buffer it hands out gets
    fresh pages, faulted in on first use, which costs about as much as one
    pass over the buffer. Keeping the buffers to avoid that is what the pool
    is for. Call it after a phase whose buffer sizes will not come back (a
    large one-off computation, say) when the memory matters more than the
    speed of what follows; the C library may keep part of the memory for the
    process.
    """
    lib.mempool_release_all()
