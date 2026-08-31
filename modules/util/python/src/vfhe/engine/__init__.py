# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""cffi handle to the native library.

The C sources are compiled per engine (meson.build) and ``_native``
picks one at import; this module re-exports its ``ffi`` / ``lib`` plus a
``libvfhe`` singleton that the wrappers use as ``libvfhe.lib``.
"""

import warnings

from ._native import active, ffi, lib, runnable


class LibVFHE:
    def __init__(self) -> None:
        self.lib = lib
        self.ffi = ffi
        self.multithreaded = False
        self.num_threads = 1


# Singleton instance
libvfhe = LibVFHE()


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
