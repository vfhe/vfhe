# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Declares Python benchmarks and times them, inside one process.

A benchmark file, ``modules/<module>/python/bench/bench_<topic>.py``, declares
its benchmarks with `benchmark`. The decorated function is the setup: it runs
once per shape, under a fixed seed, and returns the `Case` whose ``op`` is
timed::

    from harness import Case, Shape, benchmark


    @benchmark(shapes=["ckks-n13-7p"], tier="headline")
    def multiply_rescale(shape: Shape) -> Case:
        scheme, rlk, a, b = ...  # not timed
        return Case(lambda: scheme.rescale(scheme.multiply(a, b, rlk)))

``tools/bench/run.py`` runs this file as a child process, one per engine and
module. It writes one JSON record per benchmark, shape and thread count to
standard output; everything else the process prints goes to standard error.
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import importlib.util
import json
import math
import os
import resource
import statistics
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from shapes import SHAPES

__all__ = ["Case", "Shape", "benchmark"]

TIERS = ("headline", "full")

# The seed every setup runs under, so each run times the same keys and data.
SETUP_SEED = 20261026

DEFAULT_REPS = 7
DEFAULT_WINDOW_S = 0.05
MAX_INNER = 1 << 24


class Shape(Mapping[str, Any]):
    """Names a parameter set and holds its parameters, read as ``shape["N"]``."""

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self._params = params

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __repr__(self) -> str:
        return f"Shape({self.name!r}, {self._params!r})"


@dataclass(frozen=True)
class Case:
    """What a setup returns: the timed call and how to validate it.

    ``op`` is the timed call. Its result is dropped inside the timing window,
    so freeing the result is part of the time.

    ``check``, when given, receives one result of ``op`` before timing, at
    every thread count, and raises when that result is wrong.

    ``items`` is how many operations one call of ``op`` performs, such as the
    ciphertexts in a batch.
    """

    op: Callable[[], object]
    check: Callable[[Any], None] | None = None
    items: int = 1


@dataclass(frozen=True)
class Benchmark:
    """A registered benchmark: its setup, the shapes it runs at, its tier."""

    name: str
    setup: Callable[[Shape], Case]
    shapes: tuple[Shape, ...]
    tier: str
    reps: int | None
    path: str


_registry: list[Benchmark] = []


def benchmark(
    *,
    shapes: Sequence[str | Shape] = (),
    tier: str = "full",
    reps: int | None = None,
) -> Callable[[Callable[[Shape], Case]], Callable[[Shape], Case]]:
    """Registers the decorated setup as a benchmark named after the function.

    ``shapes`` lists the parameter sets, smallest first: a name from
    ``tools/bench/shapes.py`` or a `Shape` of the benchmark's own. The smoke
    run uses the first. With no shapes, the setup receives an empty shape
    named ``default``.

    ``tier`` is ``"headline"`` for the few benchmarks that summarise a
    module, which also run in the full tier, or ``"full"``.

    ``reps`` overrides the number of timed repetitions, for a call long
    enough that the default would take minutes.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    resolved = tuple(_resolve(shape) for shape in shapes) or (Shape("default"),)

    def register(setup: Callable[[Shape], Case]) -> Callable[[Shape], Case]:
        path = sys.modules[setup.__module__].__file__ or setup.__module__
        if any(b.name == setup.__name__ and b.path == path for b in _registry):
            raise ValueError(f"{path} declares {setup.__name__!r} twice")
        _registry.append(Benchmark(setup.__name__, setup, resolved, tier, reps, path))
        return setup

    return register


def _resolve(shape: str | Shape) -> Shape:
    if isinstance(shape, Shape):
        return shape
    try:
        return Shape(shape, **SHAPES[shape])
    except KeyError:
        raise KeyError(f"no shape named {shape!r} in tools/bench/shapes.py") from None


def _timed(op: Callable[[], object], inner: int) -> tuple[float, int]:
    """Calls ``op`` ``inner`` times; returns the seconds and minor page faults."""
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    start = time.perf_counter()
    for _ in range(inner):
        op()
    elapsed = time.perf_counter() - start
    return elapsed, resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults


def _calibrate(op: Callable[[], object], window: float) -> int:
    """Returns how many calls of ``op`` take at least ``window`` seconds.

    The calls made here also warm up the caches and the allocator.
    """
    inner = 1
    while True:
        elapsed, _ = _timed(op, inner)
        if elapsed >= window or inner >= MAX_INNER:
            return inner
        grown = math.ceil(1.2 * inner * window / max(elapsed, 1e-9))
        inner = min(MAX_INNER, max(2 * inner, grown))


def measure(
    op: Callable[[], object], reps: int, window: float, smoke: bool
) -> dict[str, Any]:
    """Times ``op``: the seconds per call of each of ``reps`` repetitions, their minimum and median.

    Each repetition times as many calls as fill ``window`` seconds, with the
    garbage collector run before it and disabled during it. A smoke
    measurement times one call, once.
    """
    inner = 1 if smoke else _calibrate(op, window)
    reps = 1 if smoke else reps
    seconds: list[float] = []
    faults: list[float] = []
    enabled = gc.isenabled()
    for _ in range(reps):
        gc.collect()
        gc.disable()
        try:
            elapsed, flt = _timed(op, inner)
        finally:
            if enabled:
                gc.enable()
        seconds.append(elapsed / inner)
        faults.append(flt / inner)
    return {
        "min_s": min(seconds),
        "median_s": statistics.median(seconds),
        "times_s": seconds,
        "reps": reps,
        "inner": inner,
        "faults": statistics.median(faults),
    }


def _load(path: Path, module: str) -> list[Benchmark]:
    """Imports one benchmark file; returns the benchmarks it registered."""
    before = len(_registry)
    name = f"vfhe_bench_{module}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return _registry[before:]


def _emit(records: TextIO, record: dict[str, Any]) -> None:
    records.write(json.dumps(record) + "\n")
    records.flush()


def run(
    module: str,
    paths: Sequence[Path],
    *,
    threads: Sequence[int],
    tier: str,
    pattern: str,
    reps: int | None,
    window: float,
    smoke: bool,
    records: TextIO,
) -> int:
    """Runs the selected benchmarks of one module's files; returns the number that failed.

    A failure is recorded and reported, and the remaining benchmarks still
    run.
    """
    failures = 0
    for path in paths:
        try:
            found = _load(path, module)
        except Exception as error:  # noqa: BLE001 - reported, and the other files still run
            traceback.print_exc()
            _emit(
                records,
                {
                    "kind": "error",
                    "layer": "python",
                    "module": module,
                    "file": str(path),
                    "error": repr(error),
                },
            )
            failures += 1
            continue

        for bench in found:
            if tier == "headline" and bench.tier != "headline":
                continue
            if not fnmatch.fnmatchcase(bench.name, pattern):
                continue
            for shape in bench.shapes[:1] if smoke else bench.shapes:
                failures += _run_one(
                    bench, shape, module, threads, reps, window, smoke, records
                )
    return failures


def _run_one(
    bench: Benchmark,
    shape: Shape,
    module: str,
    threads: Sequence[int],
    reps: int | None,
    window: float,
    smoke: bool,
    records: TextIO,
) -> int:
    """Sets one benchmark up at one shape and times it at every thread count.

    Arithmetic caches are emptied first, so no benchmark inherits another's.
    """
    from vfhe.arith import reset_state
    from vfhe.crypto import entropy
    from vfhe.util import parallel

    base = {
        "layer": "python",
        "module": module,
        "bench": bench.name,
        "shape": shape.name,
        "tier": bench.tier,
    }
    reset_state()
    gc.collect()
    current = threads[0]
    try:
        with entropy.deterministic(SETUP_SEED):
            case = bench.setup(shape)
        for t in threads:
            current = t
            parallel.set_num_threads(t)
            if case.check is not None:
                case.check(case.op())
            timing = measure(case.op, reps or bench.reps or DEFAULT_REPS, window, smoke)
            _emit(
                records,
                {"kind": "result", **base, "threads": t, "items": case.items, **timing},
            )
    except Exception as error:  # noqa: BLE001 - reported, and the other benchmarks still run
        traceback.print_exc()
        _emit(
            records, {"kind": "error", **base, "threads": current, "error": repr(error)}
        )
        return 1
    finally:
        parallel.set_num_threads()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Times one module's Python benchmarks."
    )
    parser.add_argument("module")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--threads", default="1")
    parser.add_argument("--tier", choices=TIERS, default="headline")
    parser.add_argument("--bench", default="*")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    # Records keep the original standard output; anything else written there,
    # from Python or from C, goes to standard error.
    records = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    from vfhe.util._engine import Engine

    importlib.import_module("vfhe.util.bindings")  # loads the engine VFHE_ENGINE names

    pinned = os.environ.get("VFHE_ENGINE")
    active = Engine.active()
    if pinned and (active is None or active.value != pinned):
        print(f"VFHE_ENGINE={pinned}, but {active} loaded", file=sys.stderr)
        return 1

    failures = run(
        args.module,
        args.paths,
        threads=[int(t) for t in args.threads.split(",")],
        tier=args.tier,
        pattern=args.bench,
        reps=args.reps,
        window=args.window,
        smoke=args.smoke,
        records=records,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    # Benchmark files register with the module named harness, not with this
    # __main__ copy of it.
    import harness

    raise SystemExit(harness.main())
