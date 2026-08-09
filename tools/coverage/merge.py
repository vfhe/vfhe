#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Union the coverage legs into one report: a row per engine, plus their total.

A leg is one measured run, `coverage-<engine>-<depth>`, so an engine usually has
several — CI measures the C suites and the Python suites in the jobs that own
them, and each writes its own leg. Legs are separate builds on separate runners,
so gcov data cannot merge raw (checksums disagree); the union happens at report
level, per engine and then across engines: a line counts as covered when any leg
covered it. A half that no leg measured reads n/a rather than 0%.

Usage: python tools/coverage/merge.py <legs-dir> <out-dir>
(a measured `make test` runs it over .coverage/legs; CI's summary job runs it
over the legs its jobs uploaded, having nothing of its own to measure.)
"""

import json
import subprocess
import sys
from pathlib import Path

Totals = dict[str, list[int]]

# A half nothing measured: render.py prints n/a for a zero total.
NOTHING: Totals = {"lines": [0, 0], "branches": [0, 0]}


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def python_totals(report: Path) -> Totals:
    """Covered/total lines and branches from a coverage.py JSON report."""
    totals = json.loads(report.read_text())["totals"]
    return {
        "lines": [totals["covered_lines"], totals["num_statements"]],
        "branches": [totals["covered_branches"], totals["num_branches"]],
    }


def c_totals(summary: Path) -> Totals:
    """Covered/total lines and branches from a gcovr JSON summary."""
    totals = json.loads(summary.read_text())
    return {
        "lines": [totals["line_covered"], totals["line_total"]],
        "branches": [totals["branch_covered"], totals["branch_total"]],
    }


def by_engine(legs: Path) -> dict[str, list[Path]]:
    """The legs each engine produced. A leg name is coverage-<engine>-<depth>,
    and an engine name never contains a hyphen (tools/_engines.py enforces it),
    so the engine is what precedes the first one."""
    grouped: dict[str, list[Path]] = {}
    for leg in sorted(p for p in legs.iterdir() if p.is_dir()):
        engine = leg.name.removeprefix("coverage-").split("-", 1)[0]
        grouped.setdefault(engine, []).append(leg)
    return grouped


def python_union(legs: list[Path], out: Path) -> Totals:
    """coverage.py's own combine, which accumulates raw data. [tool.coverage.paths]
    remaps each runner's checkout prefix so the legs line up across machines;
    --keep because the legs ship as artifacts, and --data-file so each union
    stands alone instead of accumulating into the last one."""
    data = [str(p) for leg in legs for p in sorted(leg.glob("data*"))]
    if not data:
        return NOTHING

    combined = str(out / "python.data")
    run("coverage", "combine", "--keep", "--data-file", combined, *data)
    run("coverage", "json", "--data-file", combined, "-o", str(out / "python.json"))
    run("coverage", "html", "--data-file", combined, "-d", str(out / "python-html"))
    return python_totals(out / "python.json")


def c_union(legs: list[Path], out: Path, title: str) -> Totals:
    """gcovr's tracefile union; their paths are repo-relative, portable as-is."""
    tracefiles = [
        arg
        for leg in legs
        for p in sorted(leg.glob("coverage-c.json"))
        for arg in ("-a", str(p))
    ]
    if not tracefiles:
        return NOTHING

    (out / "c-html").mkdir(parents=True, exist_ok=True)
    run(
        "gcovr",
        *tracefiles,
        "--html-title",
        f"vFHE C coverage ({title})",
        "--json-summary",
        str(out / "c-summary.json"),
        "--html-details",
        str(out / "c-html" / "index.html"),
    )
    return c_totals(out / "c-summary.json")


def union(legs: list[Path], out: Path, title: str) -> dict[str, Totals]:
    """Both halves of one set of legs, reported into its own directory."""
    out.mkdir(parents=True, exist_ok=True)
    return {"python": python_union(legs, out), "c": c_union(legs, out, title)}


def main() -> int:
    legs, out = Path(sys.argv[1]), Path(sys.argv[2])
    grouped = by_engine(legs)
    if not grouped:
        print(f"::error::no coverage legs found under {legs}", file=sys.stderr)
        return 1

    engines = {
        engine: union(group, out / engine, f"{engine} engine")
        for engine, group in grouped.items()
    }
    combined = union([leg for group in grouped.values() for leg in group], out, "all")

    (out / "coverage.json").write_text(
        json.dumps({"engines": engines, "combined": combined}, indent=2) + "\n"
    )
    print(f"merged {', '.join(sorted(engines))} -> {out / 'coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
