#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""protoc over every modules/*/proto schema, into a ``_vfhe_proto`` package.

Each ``proto/`` dir is an import root (buf convention), and all of them are
on the include path, so bindings land at their protobuf package path and
schemas may import across modules. Never committed: meson builds this as the ``_vfhe_proto`` target, which
`make proto` compiles so pyright has the bindings without a C build.

Positional, not flags: meson.build is the only caller.

    generate_proto_bindings.py <output directory>
"""

from __future__ import annotations

import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / "modules"


def find_schemas() -> list[Path]:
    return sorted(MODULES.glob("*/proto/**/*.proto"))


def import_root_flags() -> list[str]:
    """Every module's proto/ is an import root (buf convention), and all of
    them are on the path, so schemas may import across modules."""
    return [f"-I{root}" for root in sorted(MODULES.glob("*/proto"))]


def compile_schema(schema: Path, includes: list[str], out: Path) -> None:
    command = ["protoc", *includes, f"--python_out={out}", f"--pyi_out={out}"]
    code = protoc.main([*command, str(schema)])
    if code:
        raise SystemExit(f"protoc failed ({code}) for {schema.relative_to(ROOT)}")
    print(f"generated {schema.relative_to(ROOT)}")


def write_package_inits(out: Path) -> None:
    """protoc leaves plain directories; Python needs packages to import."""
    directories = [out, *(p for p in out.rglob("*") if p.is_dir())]
    for directory in directories:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text('"""Generated protobuf bindings - do not edit."""\n')


def main() -> int:
    out = Path(sys.argv[1])
    schemas = find_schemas()
    if not schemas:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    includes = import_root_flags()
    for schema in schemas:
        compile_schema(schema, includes, out)

    write_package_inits(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
