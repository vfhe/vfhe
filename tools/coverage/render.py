#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Fill the coverage summary template from the merged report.

Usage:
    python tools/coverage/render.py .coverage/merged/coverage.json \
        tools/coverage/summary.md.in <report-url>

(a measured `make test` prints the table; CI comments the same text.)
"""

import json
import sys
from pathlib import Path


def cell(totals: list[int], bar: bool = False) -> str:
    covered, total = totals
    if not total:
        return "n/a"
    text = f"{100 * covered / total:.1f}% ({covered}/{total})"
    if bar:
        filled = round(10 * covered / total)
        text = f"`{'█' * filled}{'░' * (10 - filled)}` {text}"
    return text


def row(label: str, measured: dict[str, dict[str, list[int]]]) -> str:
    return (
        f"| {label} "
        f"| {cell(measured['python']['lines'], bar=True)} "
        f"| {cell(measured['python']['branches'])} "
        f"| {cell(measured['c']['lines'], bar=True)} "
        f"| {cell(measured['c']['branches'])} |"
    )


def strip_license_header(template: str) -> str:
    """The template's SPDX lines describe the file, not the summary it renders."""
    body = template.lstrip()
    while body.startswith("<!--"):
        body = body[body.index("-->") + 3 :].lstrip()
    return body


def main() -> int:
    report = json.loads(Path(sys.argv[1]).read_text())
    template = strip_license_header(Path(sys.argv[2]).read_text())

    # One row per leg merge.py found, in its order; a leg that
    # failed simply is not there.
    engines = report["engines"]
    rows = [row(f"{name} engine", measured) for name, measured in engines.items()]
    # A combined row only says something new once two engines were measured.
    if len(engines) > 1:
        rows.append(row("**combined**", report["combined"]))

    print(
        template.format(rows="\n".join(rows), artifact_url=sys.argv[3]),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
