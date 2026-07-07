# SPDX-License-Identifier: Apache-2.0
"""Integration test: run the end-to-end CKKS smoke test and assert it verifies.

`scripts/smoke.py` encrypts two vectors, evaluates them homomorphically, and
checks the decrypted results against plaintext arithmetic; it returns 0 only if
every check is within tolerance. This wraps it so `make test` exercises the full
public API top to bottom.
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smoke  # noqa: E402


def test_ckks_smoke_end_to_end():
    assert smoke.main() == 0
