#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Replay CI scripts against recorded responses.

Usage: python .github/tests/ci_scripts.py
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / "fixtures"
STATUS = ROOT / ".github" / "actions" / "check-workflow-status" / "status.sh"
PROVENANCE = ROOT / ".github" / "actions" / "verify-provenance" / "provenance.sh"

SHA = "4f3a1c2"

# (fixture, workflow, presence, expected exit)
STATUS_CASES = [
    ("runs-green.json", "CI Postsubmit", "required", 0),
    ("runs-green.json", "Hardware Tests", "optional", 0),
    ("runs-docs-commit.json", "CI Postsubmit", "required", 0),
    ("runs-docs-commit.json", "Hardware Tests", "optional", 0),
    ("runs-postsubmit-red.json", "CI Postsubmit", "required", 1),
    ("runs-hardware-red.json", "Hardware Tests", "optional", 1),
    ("runs-in-progress.json", "CI Postsubmit", "required", 1),
    ("runs-absent.json", "CI Postsubmit", "required", 1),
    ("runs-absent.json", "Hardware Tests", "optional", 0),
    ("runs-green.json", "CI Postsubmit", "maybe", 1),  # anything else stays strict
]

# (fixture, gh verifies, expected exit)
PROVENANCE_CASES = [
    ("release-files.json", True, 0),
    ("release-absent.json", True, 1),
    ("release-files.json", False, 1),  # an attestation that does not verify
]

CURL_STUB = """#!/usr/bin/env bash
for arg in "$@"; do [ "$arg" = "--output-dir" ] && download=1; done
url=${*: -1}
if [ -n "${download:-}" ]; then
    printf 'artifact' > "$DEST_DIR/$(basename "$url")"
else
    cat "$FIXTURE"
fi
"""

GH_STUB = """#!/usr/bin/env bash
echo "gh $*"
[ -n "${GH_FAILS:-}" ] && exit 1
exit 0
"""


def write_stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def check(label: str, got: int, want: int) -> bool:
    ok = got == want
    print(f"  {label:<58} [{'ok' if ok else f'FAIL exit={got}'}]")
    return ok


def replay_status() -> list[str]:
    failed = []
    for fixture, workflow, presence, want in STATUS_CASES:
        result = subprocess.run(
            [str(STATUS), SHA, workflow, presence],
            stdin=(FIXTURES / fixture).open(),
            capture_output=True,
            text=True,
        )
        label = f"{fixture} {workflow} ({presence})"
        if not check(label, result.returncode, want):
            failed.append(label)
    return failed


def replay_provenance() -> list[str]:
    failed = []
    for fixture, verifies, want in PROVENANCE_CASES:
        with tempfile.TemporaryDirectory() as work:
            stubs = Path(work) / "stubs"
            stubs.mkdir()
            write_stub(stubs / "curl", CURL_STUB)
            write_stub(stubs / "gh", GH_STUB)
            published = Path(work) / "published"
            published.mkdir()
            env = os.environ | {
                "PATH": f"{stubs}:{os.environ['PATH']}",
                "FIXTURE": str(FIXTURES / fixture),
                "DEST_DIR": str(published),
                "DEST": str(published),
                "REPOSITORY": "vfhe/vfhe",
                "SIGNER_WORKFLOW": "vfhe/vfhe/.github/workflows/release-pypi.yml",
                "ATTEMPTS": "2",
                "DELAY": "0",
            }
            if not verifies:
                env["GH_FAILS"] = "1"
            result = subprocess.run(
                [str(PROVENANCE), "vfhe", "1.2.3"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
            )
        label = f"{fixture} {'verifies' if verifies else 'rejects'}"
        if not check(label, result.returncode, want):
            failed.append(label)
    return failed


def main() -> int:
    print("[ci] check-workflow-status/status.sh")
    failed = replay_status()
    print("[ci] verify-provenance/provenance.sh")
    failed += replay_provenance()
    if failed:
        print(f"error: CI scripts changed behaviour: {', '.join(failed)}")
        return 1
    print(f"[ci] {len(STATUS_CASES) + len(PROVENANCE_CASES)} cases replayed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
