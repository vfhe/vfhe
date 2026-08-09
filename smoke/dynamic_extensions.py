#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: compile a runtime C extension against the installed library.

Registers a tiny C function, recompiles the engine with it, and calls both it
and the library through the reloaded handle. Proves an installed package
carries everything dynamic_extensions needs (the vfhe/_source snapshot).
"""

import tempfile

from vfhe.misc import dynamic_extensions
from vfhe.misc.libvfhe import active_engine


def main() -> None:
    engine_before = active_engine()

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
        from vfhe.misc.libvfhe import lib as new_lib

        # explicit checks, not assert: this must still fail under `python -O`
        if new_lib.vfhe_smoke_add(20, 22) != 42:
            raise SystemExit("custom C function broken")
        if active_engine() != engine_before:
            raise SystemExit("engine flipped during recompilation")

    print("OK: runtime extension compiled and loaded against the installed library.")


if __name__ == "__main__":
    main()
