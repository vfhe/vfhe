# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: compile a runtime C extension against the installed library.

Only an install proves the vfhe/_source snapshot shipped.
"""

import tempfile

from _report import check, exit_status
from vfhe import dynamic_extensions
from vfhe.engine import active_engine


def main() -> int:
    before = active_engine()

    dynamic_extensions.clear_extensions()
    dynamic_extensions.add_c_definitions(
        "uint64_t vfhe_smoke_add(uint64_t a, uint64_t b);"
    )
    dynamic_extensions.add_c_code(
        "#include <stdint.h>\n"
        "uint64_t vfhe_smoke_add(uint64_t a, uint64_t b) { return a + b; }\n"
    )

    with tempfile.TemporaryDirectory() as out:
        dynamic_extensions.compile(output_dir=out)
        from vfhe.engine import lib as new_lib

        ok = check("custom C function answers", new_lib.vfhe_smoke_add(20, 22) == 42)
        ok &= check("engine survived recompilation", active_engine() == before)

    return exit_status(ok, "a runtime extension compiled and loaded.")


if __name__ == "__main__":
    raise SystemExit(main())
