# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the install carries every engine the build produced.

The installed ``_vfhe_flags.ENGINES`` table names them, and a wheel that dropped
an extension passes every other case. Running another engine needs its ISA,
which is the suites' business.
"""

import importlib
import importlib.resources

import _vfhe_flags
from _report import check, exit_status
from vfhe.util._engine import Engine


def main() -> int:
    importlib.import_module("vfhe.util.bindings")
    produced = list(_vfhe_flags.ENGINES)
    print(f"vfhe engines: {', '.join(produced)}\n")

    ok = True
    for name in produced:
        ok &= check(f"{name}: extension installed", Engine(name).installed())

    # vfhe.util.kernels recompiles the library from this, so an install that lost it
    # can build no kernel at all.
    archive = importlib.resources.files("vfhe") / "_source" / "kernels-src.tar.gz"
    ok &= check("the C archive ships", archive.is_file())

    active = Engine.active()
    ok &= check(
        f"the loaded engine ({active.value if active else None}) is one of them",
        active is not None and active.value in produced,
    )
    ok &= check(
        "this CPU can run at least one",
        any(Engine(name).runnable() for name in produced),
    )

    return exit_status(ok, "every engine the build produced is installed.")


if __name__ == "__main__":
    raise SystemExit(main())
