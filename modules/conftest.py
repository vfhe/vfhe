# SPDX-License-Identifier: Apache-2.0
"Dev test wiring (pytest loads this before collecting any test under modules/)."

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # keep .pyc out of the source tree

try:
    import _vfhe_native
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / ".generated"))
    for src in sorted(ROOT.glob("modules/*/python/src")):
        sys.path.insert(0, str(src))
