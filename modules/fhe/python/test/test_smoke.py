# SPDX-License-Identifier: Apache-2.0
"""Run the CKKS smoke test against the source build and assert it verifies.

Covers the full public API end to end inside the pytest suite.
"""

import sys
from pathlib import Path

SMOKE = Path(__file__).resolve().parents[4] / "smoke"
if str(SMOKE) not in sys.path:
    sys.path.insert(0, str(SMOKE))

import ckks  # noqa: E402


def test_ckks_smoke_end_to_end():
    assert ckks.main() == 0
