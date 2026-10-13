# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the install carries the license texts Apache-2.0 requires it to.

REUSE lint covers the source tree; only an install answers for the wheel.
"""

from importlib.metadata import distribution

from _report import check, exit_status


def main() -> int:
    files = {path.name for path in distribution("vfhe").files or []}
    ok = check("LICENSE installed", "LICENSE" in files)
    ok &= check("NOTICE installed", "NOTICE" in files)
    return exit_status(ok, "the install carries its legal texts.")


if __name__ == "__main__":
    raise SystemExit(main())
