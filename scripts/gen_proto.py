#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Python protobuf bindings from every module's ``proto/`` tree.

Compiles each ``modules/*/proto/**/*.proto`` into ``.generated/_vfhe_proto/``,
mirroring the protobuf package path, and makes each output dir an importable
package. Run standalone (``python scripts/gen_proto.py``) or via
``generate_all()``, which setup.py calls at build time so wheels/sdists carry
the bindings.
"""

from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules"
OUT_ROOT = ROOT / ".generated" / "_vfhe_proto"


def _ensure_pkgs(root: Path) -> None:
    """Make ``root`` and every subdirectory an importable package."""
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        init = d / "__init__.py"
        if not init.exists():
            init.write_text(
                '"""Generated protobuf bindings - do not edit, do not commit."""\n'
            )


def generate_all() -> list[Path]:
    """(Re)generate every module's proto bindings into OUT_ROOT.

    Each module's ``proto/`` dir is an import root (buf module root), so a proto
    with ``package vfhe.<mod>.<name>.vN`` lives at ``proto/vfhe/<mod>/<name>/vN/``
    and its bindings land at ``_vfhe_proto/vfhe/<mod>/<name>/vN/x_pb2.py`` — the
    Python path mirrors the protobuf package (imported as
    ``_vfhe_proto.vfhe.<mod>.<name>.vN.x_pb2``). All module proto roots are on the
    include path, so protos may import across modules.
    """
    protos = sorted(MODULES_DIR.glob("*/proto/**/*.proto"))
    if not protos:
        return protos
    includes = [f"-I{root}" for root in sorted(MODULES_DIR.glob("*/proto"))]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for proto in protos:
        rc = protoc.main(
            [
                "protoc",
                *includes,
                f"--python_out={OUT_ROOT}",
                f"--pyi_out={OUT_ROOT}",
                str(proto),
            ]
        )
        if rc != 0:
            raise SystemExit(f"protoc failed ({rc}) for {proto.relative_to(ROOT)}")
        print(f"generated {proto.relative_to(ROOT)}")
    _ensure_pkgs(OUT_ROOT)
    return protos


def main() -> int:
    if not generate_all():
        print("no .proto files found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
