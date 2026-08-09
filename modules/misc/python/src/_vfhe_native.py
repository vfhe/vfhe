# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Pick an engine at import: the best one this CPU can run.

`_vfhe_engines` lists what the build installed, best first, each with the
capability its ISA needs; `_vfhe_cpu` — probes only, so choosing never loads
an engine — answers whether this CPU has it. VFHE_ENGINE pins one by name.

Self-contained on purpose: importing anything from vfhe here would be
circular, since vfhe imports this module.
"""

import importlib
import importlib.util
import os

from _vfhe_cpu import lib as _cpu
from _vfhe_engines import ENGINES


def _installed(name: str) -> bool:
    return importlib.util.find_spec(f"_vfhe_native_{name}") is not None


def _runnable(requires: str | None) -> bool:
    return bool(_cpu.vfhe_cpu_supports((requires or "").encode()))


def _select(pin: str) -> str:
    """The engine to load: the pinned one, or the best installed and runnable."""
    if pin:
        for name, requires in ENGINES:
            if name != pin:
                continue
            if not _installed(name):
                raise RuntimeError(f"VFHE_ENGINE={pin}, but it is not installed")
            if not _runnable(requires):
                raise RuntimeError(f"VFHE_ENGINE={pin}, but this CPU lacks {requires}")
            return name
        known = ", ".join(name for name, _ in ENGINES)
        raise RuntimeError(f"unknown VFHE_ENGINE '{pin}'; this build has {known}")

    for name, requires in ENGINES:
        if _installed(name) and _runnable(requires):
            return name
    raise RuntimeError("no installed engine runs on this CPU")


# Names an installed engine could take here, best first: vfhe.misc.libvfhe's
# performance hint asks whether a faster one than the active engine exists.
runnable = [n for n, requires in ENGINES if _installed(n) and _runnable(requires)]
active = _select(os.environ.get("VFHE_ENGINE") or "")

_engine = importlib.import_module(f"_vfhe_native_{active}")
ffi = _engine.ffi
lib = _engine.lib
