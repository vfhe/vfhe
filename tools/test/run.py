#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Run meson's test suites for the engines this machine can test.

Usage:
    python tools/test/run.py                        every engine that runs free
    python tools/test/run.py avx512ifma c,complete  one engine, chosen suites
    python tools/test/run.py all fast -- -k ntt     after `--` it is pytest's
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))  # the shared parts live one level up

import _engines  # noqa: E402  (the parts; TOOLS above)
import _sde  # noqa: E402
from _common import BUILD_DIR, ROOT, error, host_supports, log  # noqa: E402

SUITES = ("c", "fast", "complete")


def parse_suites(value: str) -> list[str]:
    chosen = value.split(",")
    if unknown := [s for s in chosen if s not in SUITES]:
        raise argparse.ArgumentTypeError(
            f"unknown suite(s) {', '.join(unknown)}; pick from {', '.join(SUITES)}"
        )
    return chosen


def parse(argv: list[str]) -> argparse.Namespace:
    # Everything after the first `--` is pytest's, verbatim; argparse strips
    # only one, so split before it parses.
    pytest_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, pytest_args = argv[:cut], argv[cut + 1 :]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "engine",
        nargs="?",
        default="all",
        help="an engine name (tools/_engines.py is the list), or all (the "
        "default): every engine this host builds and this CPU runs natively",
    )
    parser.add_argument(
        "suites",
        nargs="?",
        default="c,fast",
        type=parse_suites,
        help="comma-separated (default: c,fast). c is the C unit tests; fast "
        "is the Python suite without the heavy tests; complete is the whole "
        "Python suite, so it covers fast as well",
    )
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="run the named engine under its emulator even where this CPU "
        "could run it natively (CI's one deterministic path across a fleet)",
    )
    parser.add_argument(
        "--if-supported",
        action="store_true",
        help="with a named engine: succeed doing nothing where this host "
        "cannot build it (the pre-wired CI rails for other architectures)",
    )

    args = parser.parse_args(argv)
    args.pytest_args = pytest_args
    return args


def choose_engines(requested: str, if_supported: bool) -> list[str]:
    """The engines worth running here: the named one, or every engine this
    host both builds and can execute without an emulator. Exits when the
    caller insists on one this host cannot build."""
    buildable = _engines.buildable(ROOT)

    if requested != "all":
        if any(engine.name == requested for engine in buildable):
            return [requested]
        log(f"[{requested}] not built for this architecture (tools/_engines.py)")
        raise SystemExit(0 if if_supported else 2)

    free = []
    for engine in buildable:
        if host_supports(engine.requires):
            free.append(engine.name)
        elif engine.emulator:
            log(f"[{engine.name}] skipped: run it by name to emulate it")
        else:
            log(f"[{engine.name}] skipped: this CPU lacks {engine.requires}")
    return free


def resolve_runner(name: str, emulate: bool) -> list[str] | None:
    """The command prefix that executes this engine's binaries: nothing when
    the CPU runs it, an emulator when one is needed, None when neither can."""
    engine = _engines.by_name(ROOT, name)
    if engine is None:
        error(f"{name} is not built for this architecture (tools/_engines.py)")
        return None

    wrapper = _sde.resolve_wrapper(engine, emulate)
    return wrapper


def meson_test_command(
    engine: str, suites: list[str], runner: list[str], pytest_args: list[str]
) -> list[str]:
    """`-v` so the suites stream their output instead of hiding it until
    something fails; one --suite per cell, since meson's --suite flags union."""
    command = ["meson", "test", "-C", str(BUILD_DIR), "-v"]
    for suite in suites:
        command += ["--suite", f"{engine}-{suite}"]
    if runner:
        command.append(f"--wrapper={' '.join(runner)}")
    if pytest_args:
        command.append(f"--test-args={' '.join(pytest_args)}")
    return command


def main() -> int:
    args = parse(sys.argv[1:])

    for engine in choose_engines(args.engine, args.if_supported):
        runner = resolve_runner(engine, args.emulate)
        if runner is None:
            return 1

        command = meson_test_command(engine, args.suites, runner, args.pytest_args)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
