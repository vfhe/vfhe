#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Python protobuf bindings into .generated/_vfhe_proto/.

Output mirrors the protobuf package path; bindings are generated at build
time, never committed. generate_all() is the importable entry point.
"""

import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules"
OUT_ROOT = ROOT / ".generated" / "_vfhe_proto"


def _ensure_packages(root: Path) -> None:
    """Make ``root`` and every subdirectory an importable package."""
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        init = d / "__init__.py"
        if not init.exists():
            init.write_text(
                '"""Generated protobuf bindings - do not edit, do not commit."""\n'
            )


def generate_all() -> list[Path]:
    """(Re)generate every module's proto bindings into OUT_ROOT.

    Each ``proto/`` dir is an import root (buf convention), so bindings land
    at the protobuf package path. All roots are on the include path so protos
    may import across modules.
    """
    proto_files = sorted(MODULES_DIR.glob("*/proto/**/*.proto"))
    if not proto_files:
        return proto_files
    include_flags = [f"-I{root}" for root in sorted(MODULES_DIR.glob("*/proto"))]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for proto in proto_files:
        exit_code = protoc.main(
            [
                "protoc",
                *include_flags,
                f"--python_out={OUT_ROOT}",
                f"--pyi_out={OUT_ROOT}",
                str(proto),
            ]
        )
        if exit_code != 0:
            raise SystemExit(
                f"protoc failed ({exit_code}) for {proto.relative_to(ROOT)}"
            )
        print(f"generated {proto.relative_to(ROOT)}")
    _ensure_packages(OUT_ROOT)
    return proto_files


def main() -> int:
    if not generate_all():
        print("no .proto files found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
