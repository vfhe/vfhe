#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Print a version's release notes: its CHANGELOG section, then its compare
link. A version the changelog never mentions is an error, so a release cannot
ship undocumented.

Usage:
    python tools/release/extract_notes.py 1.2.3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))  # the shared parts live one level up

from _common import ROOT  # noqa: E402  (a part; TOOLS above)

# `[0.0.2]: https://…` — Keep a Changelog's compare links, below the sections.
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]: ")


def read_changelog() -> list[str]:
    lines = (ROOT / "CHANGELOG.md").read_text().splitlines()
    return lines


def find_section(lines: list[str], version: str) -> list[str]:
    """The lines under `## [<version>]`, which carries a date, up to the next
    version heading or the link definitions at the bottom."""
    headings = [i for i, line in enumerate(lines) if line.startswith(f"## [{version}]")]
    if not headings:
        return []

    body: list[str] = []
    for line in lines[headings[0] + 1 :]:
        if line.startswith("## [") or LINK_DEFINITION.match(line):
            break
        body.append(line)
    return body


def find_compare_link(lines: list[str], version: str) -> str | None:
    """The `[<version>]: <url>` line Keep a Changelog puts at the bottom."""
    prefix = f"[{version}]: "
    matches = [line for line in lines if line.startswith(prefix)]
    return matches[0].removeprefix(prefix) if matches else None


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    version = sys.argv[1]

    lines = read_changelog()
    body = find_section(lines, version)
    if not any(line.strip() for line in body):
        raise SystemExit(f"CHANGELOG.md has no notes for {version}")

    notes = "\n".join(body).strip()
    link = find_compare_link(lines, version)
    if link:
        notes += f"\n\n**Full Changelog**: {link}"
    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
