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
    """Give every test empty arithmetic caches.

    An implementation's caches are process-global. RNS keys its prime pool on
    (N, split_degree), so unrelated ring families sharing those parameters
    accumulate primes in one growing pool, and a ring's primes occupy
    contiguous indices from 0 only when it is the first registered for its
    key. A few low-level paths (LWE extraction, packing key-switch) rely on
    that contiguity. The original suites ran each test file as its own
    process; emptying the caches before every test reproduces that isolation
    so cross-file ordering cannot leak state.
    """
    from vfhe.arith import reset_state

    reset_state()
