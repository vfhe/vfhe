#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sets two benchmark result files side by side.

    tools/bench/compare.py OLD.jsonl NEW.jsonl

Joins the results on module, layer, benchmark, shape, engine and thread
count, and prints NEW's minimum over OLD's per row. A row's spread is
``(max - min) / min`` over its repetitions; a change smaller than the larger
of the two spreads is noise, and the row reads ``~`` instead of ``faster`` or
``slower``. Rows a change cannot have touched should read ``~``; when they do
not, the measurement moved, not the code. Rows present in one file only are
listed after the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

KEY = ("module", "layer", "bench", "shape", "engine", "threads")


def read(path: Path) -> tuple[dict[str, Any], dict[tuple[Any, ...], dict[str, Any]]]:
    """Returns a results file's run description, closing line included, and its results by key."""
    run: dict[str, Any] = {}
    results: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open() as lines:
        for line in lines:
            record = json.loads(line)
            if record["kind"] in ("env", "end"):
                run |= record
            elif record["kind"] == "result":
                results[tuple(record[k] for k in KEY)] = record
    return run, results


def spread(record: dict[str, Any]) -> float:
    """Returns ``(max - min) / min`` of a result's repetitions."""
    times = record["times_s"]
    return (max(times) - min(times)) / min(times) if min(times) > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sets two benchmark result files side by side."
    )
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)

    old_run, old = read(args.old)
    new_run, new = read(args.new)
    for name, run in (("old", old_run), ("new", new_run)):
        dirty = " (dirty)" if run.get("dirty") else ""
        reasons = "".join(f"\n       {reason}" for reason in run.get("reasons", []))
        print(
            f"{name}: {run.get('commit')}{dirty} on {run.get('machine')} ({run.get('cpu')}), "
            f"{run.get('verdict', 'unfinished')}{reasons}"
        )
    if old_run.get("cpu") != new_run.get("cpu") or old_run.get(
        "machine"
    ) != new_run.get("machine"):
        print("warning: the files come from different machines", file=sys.stderr)
    if any(run.get("verdict") != "trusted" for run in (old_run, new_run)):
        print(
            "warning: a run that is not trusted can show the direction of a large "
            "change, not its size",
            file=sys.stderr,
        )

    header = (
        "benchmark",
        "shape",
        "engine",
        "threads",
        "old",
        "new",
        "new/old",
        "noise",
        "",
    )
    rows = []
    for key in sorted(old.keys() & new.keys(), key=lambda k: tuple(map(str, k))):
        before, after = old[key]["min_s"], new[key]["min_s"]
        ratio = after / before
        noise = max(spread(old[key]), spread(new[key]))
        change = "~" if abs(ratio - 1) <= noise else "faster" if ratio < 1 else "slower"
        rows.append(
            (
                "/".join(map(str, key[:3])),
                *map(str, key[3:]),
                f"{before:.4g}",
                f"{after:.4g}",
                f"{ratio:.3f}",
                f"{noise:.1%}",
                change,
            )
        )
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    print()
    for row in [header, *rows]:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
        )
    print(
        "\nold and new are minimum seconds per call; noise is the larger spread of the two"
    )

    for name, only in (
        ("old", old.keys() - new.keys()),
        ("new", new.keys() - old.keys()),
    ):
        if only:
            print(f"\nonly in {name}:")
            for key in sorted(only, key=lambda k: tuple(map(str, k))):
                print("  " + " ".join(map(str, key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
