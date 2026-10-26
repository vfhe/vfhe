#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs vfhe's benchmarks across engines, modules and thread counts.

    tools/bench/run.py [--module arith,fhe] [--engine NAME] [--threads 1,8]
                       [--tier headline|full] [--bench GLOB] [--reps N]
                       [--pin CPUS] [--label NAME] [--smoke]

Python benchmarks (``modules/<module>/python/bench/bench_*.py``) run in one
child process per engine and module, through ``harness.py``. A C benchmark
file (``modules/<module>/c/bench/bench_*.c``) is one executable per engine,
which this script builds unless told not to.

Results go to ``results/bench/<time>-<machine>-<verdict>-<label>.jsonl``: a
line describing the run, one line per result or failure, and a closing line.
The verdict says whether the machine's timings can be trusted (see
``machine.py``); while the run lasts the file ends in ``.partial``.
``compare.py`` sets two such files side by side. A smoke run (``--smoke``)
runs every benchmark once, at its first shape, and keeps no file.

Refuses a build with coverage, sanitizers or optimisation below ``-O2``,
except for a smoke run. Exits 1 when a benchmark fails, and when the
selection matches no benchmark.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

import machine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The order of a record's fields in the results file.
FIELD_ORDER = ("kind", "layer", "module", "bench", "shape", "engine", "threads")


def discover(modules: list[str] | None) -> dict[str, dict[str, list[Path]]]:
    """Returns each module's benchmark files, Python and C."""
    known = sorted(p.parent.name for p in (ROOT / "modules").glob("*/meson.build"))
    for name in modules or ():
        if name not in known:
            raise SystemExit(f"no module named {name!r}; modules: {', '.join(known)}")
    found: dict[str, dict[str, list[Path]]] = {}
    for layer, pattern in (
        ("python", "python/bench/bench_*.py"),
        ("c", "c/bench/bench_*.c"),
    ):
        for path in sorted(ROOT.glob(f"modules/*/{pattern}")):
            module = path.parents[2].name
            if modules is None or module in modules:
                found.setdefault(module, {"python": [], "c": []})[layer].append(path)
    return found


def engines(build: Path, wanted: str, emulated: bool) -> list[str]:
    """Returns the engines to run: every built one this CPU runs, or the one named."""
    sys.path[:0] = [str(build), str(ROOT / "modules/util/python/src")]
    import _vfhe_flags  # pyright: ignore[reportMissingImports]
    from vfhe.util._engine import Engine

    built = [name for name in _vfhe_flags.ENGINES if Engine(name).installed()]
    if wanted == "all":
        return [name for name in built if Engine(name).runnable()]
    if wanted not in built:
        raise SystemExit(
            f"this build has no engine {wanted!r}; it has {', '.join(built)}"
        )
    if not emulated and not Engine(wanted).runnable():
        raise SystemExit(f"this CPU cannot run {wanted}")
    return [wanted]


def executables(build: Path, names: list[str], compile_them: bool) -> dict[str, Path]:
    """Returns the C benchmark executables by target name, compiling them first if asked."""
    if not names:
        return {}
    targets = {
        t["name"]: Path(t["filename"][0]) for t in _introspect(build, "--targets")
    }
    missing = [name for name in names if name not in targets]
    if missing:
        raise SystemExit(
            f"meson defines no {', '.join(missing)}: list the file in its module's "
            "vfhe_c_benchmarks"
        )
    if compile_them:
        subprocess.run(["meson", "compile", "-C", str(build), *names], check=True)  # noqa: S603, S607 - as above
    return {name: targets[name] for name in names}


def _introspect(build: Path, what: str) -> Any:
    listed = subprocess.run(  # noqa: S603 - fixed command, our build directory
        ["meson", "introspect", str(build), what],  # noqa: S607 - meson from PATH, as make build
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(listed.stdout)


def build_description(build: Path) -> dict[str, Any]:
    """Returns the build's compiler and the meson options that change timings."""
    wanted = (
        "buildtype",
        "optimization",
        "debug",
        "b_ndebug",
        "b_sanitize",
        "b_coverage",
    )
    options = {o["name"]: o["value"] for o in _introspect(build, "--buildoptions")}
    compiler = _introspect(build, "--compilers").get("host", {}).get("c", {})
    return {
        "compiler": f"{compiler.get('id', '?')} {compiler.get('version', '?')}",
        "options": {name: options.get(name) for name in wanted},
    }


def instrumented(options: dict[str, Any]) -> list[str]:
    """Returns why a build's timings would be meaningless; empty for a release build."""
    reasons = []
    if options.get("b_coverage"):
        reasons.append("coverage")
    # Newer meson gives b_sanitize as a list, older as a comma-separated string.
    sanitize = options.get("b_sanitize") or []
    if isinstance(sanitize, str):
        sanitize = sanitize.split(",")
    sanitizers = [name for name in sanitize if name != "none"]
    if sanitizers:
        reasons.append(f"sanitizer {','.join(sanitizers)}")
    if options.get("optimization") not in ("2", "3"):
        reasons.append(f"optimization -O{options.get('optimization')}")
    return reasons


def child_env(build: Path, engine: str) -> dict[str, str]:
    """Returns the environment of a child: the tree on its import path, the engine pinned."""
    paths = [build, build / "proto", *sorted(ROOT.glob("modules/*/python/src")), HERE]
    existing = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [*map(str, paths), *([existing] if existing else [])]
        ),
        "VFHE_ENGINE": engine,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def launcher(pin: str | None) -> list[str]:
    """Returns the command prefix that pins a child to the CPUs ``pin`` lists.

    Under ``VFHE_SDE_FLAGS``, set by the emulated test setup, the child also
    runs under Intel SDE, since the setup's launcher does not follow children.
    """
    prefix: list[str] = []
    if pin:
        if shutil.which("numactl"):
            prefix += ["numactl", f"--physcpubind={pin}", "--localalloc"]
        elif shutil.which("taskset"):
            prefix += ["taskset", "-c", pin]
        else:
            raise SystemExit("--pin needs numactl or taskset")
    sde_flags = os.environ.get("VFHE_SDE_FLAGS")
    if sde_flags:
        prefix += [str(ROOT / ".cache/sde/sde64"), *sde_flags.split(), "--"]
    return prefix


def _git(*args: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed command
        ["git", "-C", str(ROOT), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip()


def _unused(path: Path) -> Path:
    """Returns ``path``, or the same name with -2, -3, ... when it is taken."""
    candidate, n = path, 2
    while (
        candidate.exists() or candidate.with_name(candidate.name + ".partial").exists()
    ):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _seconds(value: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if value >= scale:
            return f"{value / scale:.4g} {unit}"
    return f"{value / 1e-9:.4g} ns"


def describe(record: dict[str, Any]) -> str:
    """Returns one result or failure as a line for a person."""
    where = f"{record.get('module')}/{record.get('layer')} {record.get('bench') or record.get('file')}"
    where += f" [{record.get('shape', '-')}] {record.get('engine')} T{record.get('threads', '-')}"
    if record["kind"] == "error":
        return f"FAILED {where}: {record.get('error')}"
    times = record["times_s"]
    spread = (max(times) - min(times)) / min(times) if min(times) > 0 else 0.0
    line = f"{where}: min {_seconds(record['min_s'])} +-{spread:.1%}, "
    line += (
        f"median {_seconds(record['median_s'])} ({record['reps']} x {record['inner']})"
    )
    if record.get("items", 1) > 1:
        line += f", {_seconds(record['min_s'] / record['items'])} per item"
    if record.get("faults", 0) >= 1:
        line += f", {record['faults']:.0f} faults per call"
    return line


class Results:
    """Collects the records children write, adds the run's fields, and saves them."""

    def __init__(self, out: TextIO | None, meta: dict[str, Any]) -> None:
        self.out = out
        self.meta = meta
        self.results = 0
        self.errors = 0

    def write(self, record: dict[str, Any]) -> None:
        if self.out is not None:
            self.out.write(json.dumps(record) + "\n")
            self.out.flush()

    def stream(
        self, command: list[str], env: dict[str, str], fields: dict[str, Any]
    ) -> bool:
        """Runs one child and records what it reports; returns whether it exited cleanly."""
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)  # noqa: S603 - our runner and our executables
        assert child.stdout is not None  # noqa: S101 - stdout=PIPE
        for line in child.stdout:
            try:
                record = json.loads(line)
            except ValueError:
                sys.stderr.write(line)
                continue
            record = {**fields, **record, **self.meta}
            record = {k: record[k] for k in FIELD_ORDER if k in record} | record
            self.results += record["kind"] == "result"
            self.errors += record["kind"] == "error"
            print(describe(record), file=sys.stderr, flush=True)
            self.write(record)
        status = child.wait()
        if status != 0:
            print(f"FAILED: {' '.join(command)} exited with {status}", file=sys.stderr)
        return status == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--module", help="comma-separated modules (default: all)")
    parser.add_argument(
        "--engine", default="all", help="an engine name (default: all this CPU runs)"
    )
    parser.add_argument("--threads", default="1", help="comma-separated thread limits")
    parser.add_argument("--tier", choices=("headline", "full"), default="headline")
    parser.add_argument("--bench", default="*", help="glob on benchmark names")
    parser.add_argument(
        "--reps", type=int, help="timed repetitions (default: per benchmark)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.05,
        help="seconds one repetition lasts at least",
    )
    parser.add_argument("--pin", help="CPUs to run on, as taskset -c reads them")
    parser.add_argument("--label", default="run", help="names the results file")
    parser.add_argument(
        "--out",
        type=Path,
        help="results file (default: results/bench/<time>-<machine>-<verdict>-<label>.jsonl)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="each benchmark once, first shape, no file"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="do not compile the C benchmarks"
    )
    args = parser.parse_args(argv)

    build = args.build_dir.resolve()
    threads = [int(t) for t in args.threads.split(",")]
    if min(threads) < 1:
        raise SystemExit("thread limits must be at least 1")
    modules = args.module.split(",") if args.module else None
    suite = discover(modules)
    selected = engines(build, args.engine, emulated="VFHE_SDE_FLAGS" in os.environ)

    build_info = build_description(build)
    refused = instrumented(build_info["options"])
    if refused and not args.smoke:
        raise SystemExit(
            f"this build has {', '.join(refused)}, so its timings mean nothing; "
            "rebuild with make build"
        )

    names = [
        f"{e}-bench-{p.stem}"
        for e in selected
        for files in suite.values()
        for p in files["c"]
    ]
    built = executables(build, names, compile_them=not args.no_build)

    import _vfhe_flags  # pyright: ignore[reportMissingImports]

    started = datetime.datetime.now(datetime.timezone.utc)
    head = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    meta = {
        "commit": head,
        "dirty": dirty,
        "machine": machine.machine_tag(),
        "label": args.label,
    }
    # A smoke run keeps no timings, so it skips the seconds sampling costs.
    start_reasons = [] if args.smoke else machine.conditions(machine.sample(), "start")
    environment = {
        "kind": "env",
        **meta,
        "started": started.isoformat(timespec="seconds"),
        "host": platform.node(),
        "conditions": start_reasons,
        **machine.facts(),
        "python": platform.python_version(),
        "compiler": build_info["compiler"],
        "build_options": build_info["options"],
        "engine_flags": {e: _vfhe_flags.ENGINES[e] for e in selected},
        "vfhe_environment": {
            k: v for k, v in os.environ.items() if k.startswith("VFHE_")
        },
        "engines": selected,
        "threads": threads,
        "tier": "full" if args.smoke else args.tier,
        "bench": args.bench,
        "modules": sorted(suite),
        "pin": args.pin,
        "smoke": args.smoke,
    }

    out_path = args.out
    stamp = started.strftime("%Y%m%d-%H%M%S")
    if out_path is None and not args.smoke:
        out_path = (
            ROOT / "results" / "bench" / f"{stamp}-{meta['machine']}-{args.label}.jsonl"
        )
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; results files are never overwritten")
    partial: Path | None = None
    out: TextIO | None = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writing = out_path.with_name(out_path.name + ".partial")
        out = writing.open("x")
        partial = writing

    print(
        f"commit {head}{' (dirty)' if dirty else ''} on {meta['machine']}, "
        f"engines {', '.join(selected)}, threads {args.threads}",
        file=sys.stderr,
    )
    for reason in start_reasons:
        print(f"untrusted: {reason}", file=sys.stderr)

    results = Results(out, meta)
    results.write(environment)
    tier = "full" if args.smoke else args.tier
    common = [
        "--threads",
        args.threads,
        "--tier",
        tier,
        "--bench",
        args.bench,
        "--window",
        str(args.window),
    ]
    if args.reps is not None:
        common += ["--reps", str(args.reps)]
    if args.smoke:
        common += ["--smoke"]
    prefix = launcher(args.pin)

    clean = True
    for engine in selected:
        env = child_env(build, engine)
        for module, files in suite.items():
            fields = {"module": module, "engine": engine}
            if files["python"]:
                command = [
                    *prefix,
                    sys.executable,
                    str(HERE / "harness.py"),
                    module,
                    *map(str, files["python"]),
                    *common,
                ]
                clean &= results.stream(command, env, fields)
            for path in files["c"]:
                command = [*prefix, str(built[f"{engine}-bench-{path.stem}"]), *common]
                clean &= results.stream(command, env, fields)

    end_reasons = [] if args.smoke else machine.conditions(machine.sample(), "end")
    reasons = list(dict.fromkeys(start_reasons + end_reasons))
    verdict = machine.verdict(reasons)
    results.write(
        {
            "kind": "end",
            "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "verdict": verdict,
            "reasons": reasons,
            "results": results.results,
            "errors": results.errors,
        }
    )
    if out is not None and partial is not None and out_path is not None:
        out.close()
        if not (results.results or results.errors):
            partial.unlink()
        else:
            if args.out is None:
                name = f"{stamp}-{meta['machine']}-{verdict}-{args.label}.jsonl"
                out_path = _unused(out_path.with_name(name))
            partial.rename(out_path)
            print(f"results: {out_path}", file=sys.stderr)
    if not args.smoke:
        print(f"verdict: {verdict}", file=sys.stderr)
        for reason in end_reasons:
            print(f"untrusted: {reason}", file=sys.stderr)

    if results.errors or not clean:
        print(
            f"{results.errors} failure(s), {results.results} result(s)", file=sys.stderr
        )
        return 1
    if results.results == 0 and not (args.smoke and not suite):
        print("no benchmark matched the selection", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
