# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: compile a custom kernel into the installed library.

Only an install proves the vfhe/_source snapshot shipped -- a wheel built
without it imports perfectly and cannot compile a kernel.

Ordering is the subject here: ExtensionBuilder.build() runs before anything
touches vfhe's C, which is the rule the feature is documented under. The
ahead-of-time command is checked the same way, from a child process.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

from _report import check, exit_status
from vfhe.util import kernels
from vfhe.util._engine import Engine

SOURCE = """\
#include <stdint.h>
#include <arith.h>

uint64_t vfhe_smoke_add(uint64_t a, uint64_t b, uint64_t q)
{
    return add_modq(a, b, q);
}
"""
DECLARATION = "uint64_t vfhe_smoke_add(uint64_t a, uint64_t b, uint64_t q);"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        (work / "smoke.c").write_text(SOURCE)
        (work / "smoke.h").write_text(DECLARATION)

        out = kernels.ExtensionBuilder([work / "smoke.c"], [work / "smoke.h"]).build()
        ok = check("the engine built", any(out.glob("_vfhe_native_*.so")))

        # Only now, and it is the engine just built that loads.
        from vfhe.util.bindings import lib

        # 20 + 30 mod 42: the answer is vfhe's add_modq, inlined into the kernel.
        ok &= check("the custom kernel answers", lib.vfhe_smoke_add(20, 30, 42) == 8)
        ok &= check(
            "the built engine is the loaded one",
            (active := Engine.active()) is not None
            and out.name.startswith(active.value + "-"),
        )
        ok &= check(
            "build() is idempotent",
            kernels.ExtensionBuilder([work / "smoke.c"], [work / "smoke.h"]).build()
            == out,
        )

        # The ahead-of-time path: one process builds, another imports it by PYTHONPATH.
        built = subprocess.run(  # noqa: S603 - paths this test wrote
            [
                sys.executable,
                "-m",
                "vfhe.util.kernels",
                str(work / "smoke.c"),
                str(work / "smoke.h"),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        answer = subprocess.run(
            [
                sys.executable,
                "-c",
                "from vfhe.util.bindings import lib; print(lib.vfhe_smoke_add(20, 30, 42))",
            ],
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [built, os.environ.get("PYTHONPATH", "")]
                ),
            },
        ).stdout.strip()
        ok &= check("built ahead of time, a child process answers", answer == "8")

    return exit_status(ok, "a custom kernel compiled into the library and ran.")


if __name__ == "__main__":
    raise SystemExit(main())
