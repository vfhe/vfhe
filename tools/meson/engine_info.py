#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Print an engine's build facts for consumers that cannot import the parts:
meson.build at configure time, and the ``engine-<name>.json`` installed next
to its archive. The single truth stays tools/_engines.py.

Positional, not flags: meson.build is the only caller, so its call sites are
the contract and a wrong one fails the build immediately.

    engine_info.py names                      # the buildable ones, best first
    engine_info.py <engine> cflags            # one flag per line
    engine_info.py <engine> extra-sources     # repo-relative, one per line
    engine_info.py <engine> json <output>     # what a user compile needs
    engine_info.py table <output> <engine>..  # the picker's Python table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import _engines  # noqa: E402  (a part, one level up; sys.path above)


def write_engine_table(names: list[str], out: Path) -> None:
    """The picker's table: (engine, required capability), best first."""
    engines = [e for e in _engines.buildable(ROOT) if e.name in names]
    rows = ",\n".join(f"    ({e.name!r}, {e.requires!r})" for e in engines)
    out.write_text(
        f"# Generated from tools/_engines.py - do not edit.\nENGINES = [\n{rows},\n]\n"
    )


def resolve_engine(name: str) -> _engines.Engine:
    engine = _engines.by_name(ROOT, name)
    if engine is None:
        known = ", ".join(e.name for e in _engines.buildable(ROOT))
        raise SystemExit(f"no engine '{name}' here; this host builds {known}")
    return engine


def write_engine_json(engine: _engines.Engine, out: Path) -> None:
    """What a user-module compile must replicate to match this engine's ABI:
    the public headers change types under these flags and defines."""
    record = {
        "engine": engine.name,
        "flags": list(engine.flags),
        "defines": [list(define) for define in engine.defines],
    }
    out.write_text(json.dumps(record, indent=2) + "\n")


def main() -> int:
    match sys.argv[1:]:
        case ["names"]:
            print("\n".join(e.name for e in _engines.buildable(ROOT)))
            return 0
        case ["table", out, *names]:
            write_engine_table(names, Path(out))
            return 0

    engine = resolve_engine(sys.argv[1])

    match sys.argv[2:]:
        case ["cflags"]:
            print("\n".join(engine.cflags))
        case ["extra-sources"]:
            paths = (str(s.relative_to(ROOT)) for s in engine.extra_sources)
            print("\n".join(paths))
        case ["json", out]:
            write_engine_json(engine, Path(out))
        case _:
            raise SystemExit(f"{__file__}: {__doc__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
