# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""cffi handle to the native library.

The C sources are compiled per engine (meson.build) and ``_native``
picks one at import; this module re-exports its ``ffi`` / ``lib`` plus a
``libvfhe`` singleton that the wrappers use as ``libvfhe.lib``.
"""

import warnings

from . import memory_pool as memory_pool
from ._native import active, ffi, lib, runnable


class LibVFHE:
    def __init__(self) -> None:
        self.lib = lib
        self.ffi = ffi


# Singleton instance
libvfhe = LibVFHE()


# The last value given to set_num_threads (0: the default), re-applied to
# libraries loaded by vfhe.dynamic_extensions.
_thread_limit = 0


def set_num_threads(n: int | None = None) -> None:
    """Limits the threads any vfhe operation may use to ``n``.

    ``None`` restores the default: ``VFHE_NUM_THREADS`` if set, otherwise 1, so
    vfhe is single-threaded unless asked. Functions taking ``n_threads`` never
    exceed the limit, and their default ``n_threads=0`` means "up to the
    limit". The setting survives `vfhe.dynamic_extensions` reloads.
    """
    global _thread_limit
    if n is not None and n < 1:
        raise ValueError(f"the thread limit must be at least 1, got {n}")
    _thread_limit = 0 if n is None else n
    lib.vfhe_set_num_threads(_thread_limit)


def num_threads() -> int:
    """The library-wide thread limit (see `set_num_threads`)."""
    return lib.vfhe_num_threads()


def active_engine() -> str:
    """The engine this process loaded, by name."""
    return active


def runnable_engines() -> list[str]:
    """Every installed engine this CPU could run, best first."""
    choices = list(runnable)
    return choices


def warn_if_faster_engine_available(engine: str, choices: list[str]) -> None:
    """Hint when a faster engine than the active one could run here —
    which means VFHE_ENGINE pinned this one, since the picker takes the best
    available otherwise. Silence with the usual warning filters or
    ``PYTHONWARNINGS=ignore``. `choices` is best-first.
    """
    if not choices or choices[0] == engine:
        return

    warnings.warn(
        f"vfhe is pinned to its {engine} engine (VFHE_ENGINE), but this CPU "
        f"can run {choices[0]} — unset the pin for large speedups.",
        RuntimeWarning,
        stacklevel=2,
    )


warn_if_faster_engine_available(active, runnable)
