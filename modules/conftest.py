# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"Dev test wiring (pytest loads this before collecting any test under modules/)."

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the only guard on a bare `pytest modules/`

try:
    # A probe: if this imports, meson already put build/ and the sources on the path.
    import vfhe.engine  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "build"))  # extensions and archives
    for src in sorted(ROOT.glob("modules/*/python/src")):
        sys.path.insert(0, str(src))


def pytest_runtest_setup():
    from vfhe.arith.ntt import NTT_processor_instance

    NTT_processor_instance.reset()  # the prime pool is process-global
