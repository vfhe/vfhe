# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the install carries every engine it claims, not just the loaded one.

Running another engine needs its ISA, which is the suites' business. A wheel that
dropped an engine's extension, archive or JSON would still pass every other case.
"""

import importlib.util

from _report import check, exit_status
from _vfhe_engines import ENGINES
from vfhe.dynamic_extensions import find_vfhe_root
from vfhe.engine import active_engine, runnable_engines


def main() -> int:
    lib = find_vfhe_root() / "lib"
    names = [name for name, _ in ENGINES]
    print(f"vfhe engines: {', '.join(names)}\n")

    ok = True
    for name in names:
        ok &= check(
            f"{name}: extension installed",
            importlib.util.find_spec(f"_vfhe_native_{name}") is not None,
        )
        for artefact in (f"libvfhe_{name}.a", f"engine-{name}.json"):
            ok &= check(f"{name}: {artefact}", (lib / artefact).is_file())

    engine = active_engine()
    ok &= check(f"the loaded engine ({engine}) is one of them", engine in names)
    ok &= check("this CPU can run at least one", bool(runnable_engines()))

    return exit_status(ok, "every declared engine is installed.")


if __name__ == "__main__":
    raise SystemExit(main())
