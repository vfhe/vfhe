# SPDX-License-Identifier: Apache-2.0
"""Build the single CFFI native extension (``_vfhe_native``).

Compiles every module's C/asm sources plus vendored BLAKE3 into one LTO'd
extension, and #includes each module's umbrella header as the preamble so the C
compiler sees the true definitions. The set of sources, include dirs, and
portable-baseline defines comes from ``native/discovery.py`` (shared with the C
test runner). ``VFHE_SIMD=1`` opts x86 into the AVX-512 fast paths; the default
is a portable baseline that runs everywhere.

Imported by setup.py (``cffi_modules``) for wheel builds and runnable directly
for the dev build into ``.generated/``.
"""

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


def _enable_asm_sources() -> None:
    modnames = (
        "distutils.compilers.C.unix",
        "distutils.unixccompiler",
        "setuptools._distutils.compilers.C.unix",
        "setuptools._distutils.unixccompiler",
    )
    for modname in modnames:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        for obj in vars(mod).values():
            exts = getattr(obj, "src_extensions", None)
            if isinstance(exts, list):
                exts.extend(e for e in (".S", ".s") if e not in exts)


_enable_asm_sources()


def build_ffi(absolute: bool) -> "FFI":
    ffi = FFI()
    sources: list[str] = []
    include_dirs: list[str] = []
    preamble_headers: list[Path] = []
    aliases = discovery.host_arch_aliases()

    def rel(p: Path) -> str:
        return str(p if absolute else p.relative_to(ROOT))

    # Umbrella headers become the preamble (#included so the compiler sees the
    # true definitions); sources and include dirs come from shared discovery.
    for c_dir in sorted(MODULES.glob("*/c")):
        preamble_headers += sorted((c_dir / "include").glob("*.h"))
    for src in discovery.module_sources(MODULES, aliases):
        sources.append(rel(src))
    for inc in discovery.module_include_dirs(MODULES):
        include_dirs.append(rel(inc))

    # Python-facing ABI: each Python-facing module declares its surface in an
    # explicit hand-written cdef under python/cdef/. C-internal modules (no
    # python/) have none -- they contribute compiled code but no Python symbols.
    # Sorted so shared typedefs are cdef'd before their users.
    for cdef in sorted(MODULES.glob("*/python/cdef/*.cdef")):
        ffi.cdef(cdef.read_text())

    # Vendored BLAKE3 (git submodule): compiled directly, no CMake. The portable
    # core + runtime dispatcher are always needed; SIMD kernels are added below
    # only for the opt-in fast build. (rng/arith call BLAKE3 internally.)
    blake3 = discovery.blake3_dir(ROOT)
    sources += [rel(s) for s in discovery.blake3_sources(ROOT)]
    include_dirs.append(rel(blake3))

    is_x86 = discovery.is_x86_host()

    if sys.platform == "win32":
        compile_args, link_args, libraries = ["/O2", "/GL"], ["/LTCG"], []
    else:
        compile_args, link_args, libraries = (
            ["-O3", "-flto", "-std=gnu11"],
            ["-flto"],
            ["m"],
        )

    # Opt-in fast paths (x86 only): VFHE_SIMD=1 turns on the arith AVX-512 IFMA
    # engine + AES-NI rng and BLAKE3's SIMD kernels. Caveats: this yields an
    # AVX-512-only binary (won't run on older CPUs), and it flips mp_vector_t to a
    # 512-bit lane -- so the MP entries in arith.cdef would need regenerating for
    # the Python bindings. Default is the portable baseline that runs everywhere.
    # UNVERIFIED on non-x86 dev machines.
    if is_x86 and os.environ.get("VFHE_SIMD") == "1":
        define_macros: "list[tuple[str, str | None]]" = []  # drop PORTABLE_BUILD
        compile_args += [
            "-mavx512f",
            "-mavx512ifma",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx2",
            "-maes",
        ]
        # BLAKE3 SIMD via pre-assembled .S (no per-file ISA flags; runtime dispatch).
        sources += [
            rel(blake3 / f"blake3_{x}_x86-64_unix.S")
            for x in ("sse2", "sse41", "avx2", "avx512")
        ]
    else:
        # Portable baseline: force the engine's scalar paths, disable BLAKE3 SIMD.
        define_macros = discovery.portable_macros(is_x86)

    ffi.set_source(
        "_vfhe_native",
        # Real headers are #included so the C compiler sees the true definitions.
        "\n".join(f'#include "{h.name}"' for h in preamble_headers),
        sources=sorted(set(sources)),
        include_dirs=sorted(set(include_dirs)),
        define_macros=define_macros,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    return ffi


# Used by setup.py's ``cffi_modules`` for wheel builds (relative paths).
ffibuilder = build_ffi(absolute=False)


if __name__ == "__main__":
    # Dev build: compile in a throwaway temp dir, then keep only the finished
    # extension in the gitignored .generated/ folder. Its editor stub is the
    # static, hand-written stubs/_vfhe_native.pyi (Any boundary); real types live
    # on the module wrappers.
    out = ROOT / ".generated"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        built = Path(build_ffi(absolute=True).compile(tmpdir=tmp, verbose=True))
        dest = out / built.name
        dest.unlink(missing_ok=True)  # replace any stale extension from a prior build
        shutil.copy2(built, dest)
    print(f"native lib -> {dest.relative_to(ROOT)}")
