# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0


def check(label: str, ok: bool) -> bool:
    print(f"  {label:<52} [{'ok' if ok else 'FAIL'}]")
    return ok


def exit_status(ok: bool, claim: str) -> int:
    print("\n" + (f"OK: {claim}" if ok else "FAILED"))
    return 0 if ok else 1
