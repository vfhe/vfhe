#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Publishes a reference benchmark run: the performance page and the baseline file.

    tools/bench/report.py page RESULTS... [--out docs/PERFORMANCE.md] [--include GLOB]
    tools/bench/report.py baseline RESULTS... [--out tools/bench/baselines/<machine>.jsonl]

RESULTS are the files of one reference run, as ``run.py`` writes them, for
example its headline tier on every engine and its full tier on one. They must
come from one machine and one commit, with a trusted verdict and a clean tree.
Where two files hold the same result, the later file's is kept.

``page`` writes the performance page: the run's conditions, then one table per
module of the headline benchmarks and of those ``--include`` adds, and the
shared shapes. ``baseline`` writes the run as one file for ``compare.py``,
without the host name.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapes import SHAPES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

KEY = ("module", "layer", "bench", "shape", "engine", "threads")

# Full-tier benchmarks the page shows besides the headline tier: the figures
# people quote that are too slow for the headline tier.
DEFAULT_INCLUDE = ("*bootstrap*", "gp25_rotate")

MODULE_ORDER = ("arith", "crypto", "util", "mlwe", "fhe", "piop", "polycom", "circuit")


class Run:
    """The records of one or more results files of the same reference run."""

    def __init__(
        self, paths: Sequence[Path], commit: str | None, allow_untrusted: bool
    ) -> None:
        self.environments: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []
        self.results: dict[tuple[Any, ...], dict[str, Any]] = {}
        for path in paths:
            with path.open() as lines:
                for line in lines:
                    record = json.loads(line)
                    if record["kind"] == "env":
                        self.environments.append(record)
                    elif record["kind"] == "end":
                        self.ends.append(record)
                    elif record["kind"] == "result":
                        self.results[tuple(record[k] for k in KEY)] = record
        if len(self.ends) != len(self.environments):
            raise SystemExit(
                "a results file has no closing line: the run did not finish"
            )
        machines = {e.get("machine") for e in self.environments}
        if len(machines) != 1:
            raise SystemExit(
                f"the files come from several machines: {sorted(map(str, machines))}"
            )
        commits = {e.get("commit") or None for e in self.environments}
        if commit is None:
            if len(commits) != 1 or None in commits:
                raise SystemExit(
                    f"the files name commits {sorted(map(str, commits))}; pass --commit "
                    "when the run had no git checkout"
                )
            commit = commits.pop()
        self.commit: str = str(commit)
        if any(e.get("dirty") for e in self.environments):
            raise SystemExit("a run measured a tree with uncommitted changes")
        verdicts = {end.get("verdict") for end in self.ends}
        if verdicts != {"trusted"} and not allow_untrusted:
            raise SystemExit(
                f"the runs' verdicts are {sorted(map(str, verdicts))}, not trusted"
            )
        self.machine: str = str(machines.pop())
        self.verdicts = verdicts

    def environment(self, field: str) -> list[Any]:
        """Returns a field of every run, without repeats, in the order the runs came."""
        values: list[Any] = []
        for environment in self.environments:
            value = environment.get(field)
            if value not in values:
                values.append(value)
        return values


def _cpu_ranges(pin: str | None) -> str:
    """Writes a CPU list like ``0,1,2,3,8`` as ``0-3, 8``."""
    if not pin:
        return "not pinned"
    cpus = sorted({int(c) for c in pin.split(",") if c.isdigit()})
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in [*cpus[1:], None]:
        if cpu is not None and cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        if cpu is not None:
            start = previous = cpu
    return "CPUs " + ", ".join(ranges)


def _seconds(value: float) -> str:
    """Writes a time to three significant digits in the largest unit below 1000 of it."""
    units = (("ns", 1e-9), ("µs", 1e-6), ("ms", 1e-3), ("s", 1.0))
    for unit, scale in units:
        scaled = float(f"{value / scale:.3g}")
        if scaled < 1000 or unit == "s":
            return f"{scaled:g} {unit}"
    raise AssertionError("unreachable")


def _cell(record: dict[str, Any]) -> str:
    times = record["times_s"]
    spread = (max(times) - min(times)) / min(times) if min(times) > 0 else 0.0
    return f"{_seconds(record['min_s'])} ±{spread:.0%}"


def _join(values: Iterable[Any]) -> str:
    return ", ".join(str(v) for v in values if v not in (None, ""))


def _conditions(run: Run) -> list[str]:
    environment = run.environments[0]
    options = environment.get("build_options", {})
    started = [s[:10] for s in run.environment("started") if s]
    threads = sorted({t for e in run.environments for t in e.get("threads", [])})
    smt = environment.get("smt")
    turbo = environment.get("turbo")
    rows = [
        (
            "Machine",
            f"{run.machine}: {environment.get('cpu')}, {environment.get('cpus')} CPUs",
        ),
        ("Memory placement", f"{environment.get('numa_nodes')} NUMA nodes"),
        (
            "SMT, turbo",
            f"{'on' if smt else 'off' if smt is False else '?'}, "
            f"{'on' if turbo else 'off' if turbo is False else '?'}",
        ),
        ("System", _join(run.environment("system"))),
        ("Commit", run.commit),
        ("Date", _join(dict.fromkeys(started))),
        ("Verdict", _join(sorted(map(str, run.verdicts)))),
        (
            "Engines",
            _join(dict.fromkeys(e for env in run.environments for e in env["engines"])),
        ),
        ("Thread limits", _join(threads)),
        ("Pinned to", _join(_cpu_ranges(p) for p in run.environment("pin"))),
        (
            "Compiler",
            f"{environment.get('compiler')}, {options.get('buildtype')}, "
            f"-O{options.get('optimization')}",
        ),
        ("Python", _join(run.environment("python"))),
    ]
    lines = ["| Condition | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _module_table(records: list[dict[str, Any]]) -> list[str]:
    threads = sorted({r["threads"] for r in records})
    header = ["Benchmark", "Layer", "Shape", "Engine"] + [
        f"{t} thread{'s' if t > 1 else ''}" for t in threads
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rows: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]] = {}
    for record in records:
        shape = record["shape"]
        if record.get("items", 1) > 1:
            shape += f", {record['items']} items"
        row = (record["bench"], record["layer"], shape, record["engine"])
        rows.setdefault(row, {})[record["threads"]] = record
    for row, by_threads in rows.items():
        cells = [_cell(by_threads[t]) if t in by_threads else "-" for t in threads]
        lines.append("| " + " | ".join([f"`{row[0]}`", *row[1:], *cells]) + " |")
    return lines


def _shapes() -> list[str]:
    lines = ["| Shape | Parameters |", "|---|---|"]
    for name, params in SHAPES.items():
        text = ", ".join(f"{k} {v}" for k, v in params.items())
        lines.append(f"| `{name}` | {text} |")
    return lines


def page(run: Run, include: Sequence[str]) -> str:
    """Returns the performance page for a reference run."""
    selected = [
        r
        for r in run.results.values()
        if r.get("tier") == "headline"
        or any(fnmatch.fnmatchcase(r["bench"], g) for g in include)
    ]
    extras = sorted({r["bench"] for r in selected if r.get("tier") != "headline"})
    by_module: dict[str, list[dict[str, Any]]] = {}
    for record in selected:
        by_module.setdefault(record["module"], []).append(record)
    order = [m for m in MODULE_ORDER if m in by_module] + sorted(
        set(by_module) - set(MODULE_ORDER)
    )

    out = [
        "# Performance",
        "",
        "<!-- Generated by tools/bench/report.py from a reference run; regenerate it, do not edit it. -->",
        "",
        "Times of vFHE's benchmark suite on one reference machine: every benchmark",
        "of the headline tier"
        + (
            ", and from the full tier " + ", ".join(f"`{b}`" for b in extras)
            if extras
            else ""
        )
        + ". A cell is the",
        "minimum time of one call over the benchmark's repetitions, and after `±`",
        "the spread of those repetitions, `(max - min) / min`. Where one call",
        "performs several operations, the shape says how many. The benchmarks and",
        "what each one times are in `modules/<module>/{c,python}/bench/`; how to",
        "run them is in",
        "[`DEVELOPMENT.md`](https://github.com/vfhe/vfhe/blob/main/docs/DEVELOPMENT.md#benchmarking).",
        "",
        "Times on other machines differ, and so do their ratios: compare against a",
        "run of your own on your hardware rather than against these numbers.",
        "",
        "## Reference run",
        "",
        *_conditions(run),
        "",
    ]
    for module in order:
        records = sorted(
            by_module[module],
            key=lambda r: (r["bench"], r["layer"], r["engine"], r["threads"]),
        )
        out += [f"## {module}", "", *_module_table(records), ""]
    out += [
        "## Shapes",
        "",
        "The parameter sets several modules share. A shape not listed here is",
        "defined in its benchmark's file.",
        "",
        *_shapes(),
        "",
    ]
    return "\n".join(out)


def baseline(run: Run) -> str:
    """Returns the reference run as one results file, without the host name."""
    records: list[dict[str, Any]] = []
    for environment, end in zip(run.environments, run.ends, strict=True):
        environment = {k: v for k, v in environment.items() if k != "host"}
        environment["commit"] = run.commit
        records += [environment, end]
    for record in run.results.values():
        records.append({**record, "commit": run.commit})
    return "".join(json.dumps(r) + "\n" for r in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("what", choices=("page", "baseline"))
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--commit", help="the commit measured, when the run had no git checkout"
    )
    parser.add_argument(
        "--include",
        action="append",
        help=f"globs on full-tier benchmark names the page also shows (default: {DEFAULT_INCLUDE})",
    )
    parser.add_argument(
        "--allow-untrusted",
        action="store_true",
        help="accept a run whose verdict is not trusted",
    )
    args = parser.parse_args(argv)

    run = Run(args.results, args.commit, args.allow_untrusted)
    if args.what == "page":
        text = page(run, args.include or DEFAULT_INCLUDE)
        out = args.out or ROOT / "docs" / "PERFORMANCE.md"
    else:
        text = baseline(run)
        out = args.out or HERE / "baselines" / f"{run.machine}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(
        f"{out}: {len(run.results)} results, {run.machine}, {run.commit}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
