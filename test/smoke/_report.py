# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Prints one check per line, then the verdict the runner reads."""


def check(label: str, ok: bool) -> bool:
    """Prints `label` with its verdict and returns `ok`."""
    print(f"  {label:<52} [{'ok' if ok else 'FAIL'}]")
    return ok


def exit_status(ok: bool, claim: str) -> int:
    """Prints the verdict and returns the exit code: 0 when `ok`."""
    print("\n" + (f"OK: {claim}" if ok else "FAILED"))
    return 0 if ok else 1
