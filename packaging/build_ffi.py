#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the single CFFI native extension (``_vfhe_native``).

One LTO'd extension so the C modules inline across boundaries. The recipe
comes from ``discovery.native_build_plan``. Run directly for a dev build into
``.generated/``.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parent.parent
MODULES = ROOT / "modules"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discovery  # noqa: E402  (co-located; ships in the sdist)

# Loaded by file path: the helpers ship inside vfhe.misc (dynamic_extensions
# needs them too), and importing that package would load the extension this
# build is about to produce.
_spec = importlib.util.spec_from_file_location(
    "_vfhe_native_compiler",
    MODULES / "misc" / "python" / "src" / "vfhe" / "misc" / "_native_compiler.py",
)
assert _spec is not None and _spec.loader is not None
_native_compiler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native_compiler)

_native_compiler.enable_asm_sources()


def build_ffi(absolute: bool) -> FFI:
    """Assemble the FFI builder; ``absolute`` picks path style (setup.py needs
    relative, the dev build absolute)."""
    plan = discovery.native_build_plan(ROOT, _native_compiler.host_has_avx512ifma)
    if plan.note:
        print(f"vfhe: {plan.note}", file=sys.stderr)

    def rel(path: Path) -> str:
        return str(path if absolute else path.relative_to(ROOT))

    ffi = FFI()
    for cdef in plan.cdef_files:
        ffi.cdef(cdef.read_text())
    ffi.set_source(
        "_vfhe_native",
        # Real headers are #included so the C compiler sees the true definitions.
        "\n".join(f'#include "{h.name}"' for h in plan.preamble_headers),
        sources=sorted({rel(s) for s in plan.sources}),
        include_dirs=sorted({rel(d) for d in plan.include_dirs}),
        define_macros=plan.define_macros,
        libraries=plan.libraries,
        extra_compile_args=plan.compile_args,
        extra_link_args=plan.link_args,
    )
    return ffi


# cffi_modules entry point; that mechanism needs relative paths.
ffibuilder = build_ffi(absolute=False)


if __name__ == "__main__":
    # Dev build into .generated/. Editor types come from
    # packaging/typings/_vfhe_native.pyi; real types live on the wrappers.
    out = ROOT / ".generated"
    out.mkdir(exist_ok=True)
    if os.environ.get("VFHE_COVERAGE") == "1":
        # Keep the objects (.gcno) so gcov's .gcda land beside them when the test
        # run exercises the extension; gcovr reads them from here afterwards.
        cov = out / "cov-build"
        cov.mkdir(exist_ok=True)
        built = Path(build_ffi(absolute=True).compile(tmpdir=str(cov), verbose=True))
        dest = out / built.name
        dest.unlink(missing_ok=True)
        shutil.copy2(built, dest)
    else:
        # Compile in a throwaway temp dir; keep only the finished extension.
        with tempfile.TemporaryDirectory() as tmp:
            built = Path(build_ffi(absolute=True).compile(tmpdir=tmp, verbose=True))
            dest = out / built.name
            dest.unlink(missing_ok=True)  # replace any stale ext from a prior build
            shutil.copy2(built, dest)
    print(f"native lib -> {dest.relative_to(ROOT)}")
