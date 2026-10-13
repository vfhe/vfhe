# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Lists the engines known to this build and selects which one to load."""

from __future__ import annotations

import enum
import importlib.util
import os
import sys

from vfhe.util import _cpu as cpu


class Engine(str, enum.Enum):
    """Represents all of vFHE's C, compiled into one extension for one instruction set.

    Members are in order of preference, fastest first; `select` relies on that
    order. Each value is the engine's name, as accepted by ``VFHE_ENGINE``.
    """

    AVX512IFMA = "avx512ifma"
    NEON = "neon"
    PORTABLE = "portable"

    @property
    def module(self) -> str:
        """Names the extension module that contains this engine."""
        return "_vfhe_native_" + self.value

    def installed(self) -> bool:
        """Reports whether the engine's extension is importable, importing nothing."""
        return importlib.util.find_spec(self.module) is not None

    def runnable(self) -> bool | None:
        """Reports whether this CPU can execute the engine's instructions.

        None when the CPU cannot say.
        """
        if self is Engine.NEON:
            return cpu.is_arm()
        if self is Engine.AVX512IFMA:
            return cpu.has_flag(self.value)
        return True

    @classmethod
    def active(cls) -> Engine | None:
        """Returns the engine whose extension this process has loaded; None before one loads."""
        return next((engine for engine in cls if engine.module in sys.modules), None)

    @classmethod
    def select(cls) -> Engine:
        """Returns the engine to load: ``VFHE_ENGINE`` when set, otherwise the best one.

        The best is the first installed engine that runs on this CPU. A pin is
        honoured without asking the CPU, so an emulator can run an engine that
        needs instructions the host CPU lacks.

        Raises `RuntimeError` when ``VFHE_ENGINE`` names an unknown or uninstalled
        engine, and when no installed engine runs on this CPU.
        """
        pin = os.environ.get("VFHE_ENGINE")
        if pin:
            try:
                engine = cls(pin)
            except ValueError:
                names = ", ".join(e.value for e in cls)
                raise RuntimeError(
                    f"unknown VFHE_ENGINE {pin!r}; this build has {names}"
                ) from None

            if not engine.installed():
                raise RuntimeError(f"VFHE_ENGINE={pin}, but it is not installed")

            return engine

        best = next((e for e in cls if e.installed() and e.runnable()), None)
        if best is None:
            raise RuntimeError("no installed engine runs on this CPU")
        return best
