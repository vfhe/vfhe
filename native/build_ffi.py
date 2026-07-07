# SPDX-License-Identifier: Apache-2.0
"""Build the single CFFI native extension (``_vfhe_native``).

Compiles every module's C/asm sources plus vendored BLAKE3 into one LTO'd
extension, and #includes each module's umbrella header as the preamble so the C
compiler sees the true definitions. The set of sources, include dirs, and
portable-baseline defines comes from ``native/discovery.py`` (shared with the C
test runner). Source builds auto-tune to the host CPU (AVX-512 IFMA when
present); wheel builds set ``VFHE_PORTABLE=1`` to force the portable baseline
that runs everywhere.

Imported by setup.py (``cffi_modules``) for wheel builds and runnable directly
for the dev build into ``.generated/``.
"""

import os
import shutil
import subprocess
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


def _host_has_avx512ifma() -> bool:
    """True if ``-march=native`` enables AVX-512 IFMA on *this* machine.

    Asks the *same* compiler the extension will be built with what the native CPU
    supports -- the engine's SIMD kernels are guarded by ``__AVX512IFMA__``, so
    this matches exactly what a tuned build would light up. Any failure (no
    compiler, ``-march=native`` unsupported, non-zero exit) falls back to
    portable, so the check can only ever *under*-tune, never mis-tune.
    """
    import shlex
    import sysconfig

    # Resolve the compiler exactly as distutils/cffi will for the real build:
    # $CC if set, else the compiler this Python was built with, else `cc`. Keep
    # any base flags it carries (e.g. a wrapper's --target=) so the probe and the
    # build see the same target.
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    argv = [*shlex.split(cc), "-march=native", "-dM", "-E", "-x", "c", os.devnull]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return out.returncode == 0 and "__AVX512IFMA__" in out.stdout


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
    # only for the opt-in fast build. (misc/arith call BLAKE3 internally.)
    blake3 = discovery.blake3_dir(ROOT)
    sources += [rel(s) for s in discovery.blake3_sources(ROOT)]
    include_dirs.append(rel(blake3))

    is_x86 = discovery.is_x86_host()

    # VFHE_COVERAGE=1 instruments the C for gcov (used by the CI coverage job):
    # -O0 + --coverage, no LTO (which strips gcov data). Always portable.
    coverage = os.environ.get("VFHE_COVERAGE") == "1"

    if sys.platform == "win32":
        compile_args, link_args, libraries = ["/O2", "/GL"], ["/LTCG"], []
    elif coverage:
        compile_args, link_args, libraries = (
            ["-O0", "-g", "-std=gnu11", "--coverage"],
            ["--coverage"],
            ["m"],
        )
    else:
        compile_args, link_args, libraries = (
            ["-O3", "-flto", "-std=gnu11"],
            ["-flto"],
            ["m"],
        )

    # Portable vs. CPU-tuned selection.
    #
    # vfhe ships as an sdist only, so the DEFAULT build -- pip compiling on the
    # target machine -- auto-tunes to THIS CPU: on x86 with AVX-512 IFMA it drops
    # PORTABLE_BUILD to light up the SIMD kernels + AES-NI and adds -march=native;
    # otherwise it builds the portable engine (and says why). VFHE_PORTABLE=1 is
    # the manual escape hatch (build here, run elsewhere) and skips detection.
    # The tuned path uses -march=native + BLAKE3's *_x86-64_unix.S kernels, so it
    # is POSIX-x86 only; everything else (Windows, non-x86) stays portable.
    force_portable = os.environ.get("VFHE_PORTABLE") == "1" or coverage
    tune = (
        (not force_portable)
        and is_x86
        and sys.platform != "win32"
        and _host_has_avx512ifma()
    )

    define_macros: "list[tuple[str, str | None]]"
    if tune:
        define_macros = []  # drop PORTABLE_BUILD -> activate the SIMD-guarded paths
        compile_args += ["-march=native", "-funroll-all-loops"]
        # BLAKE3 SIMD via pre-assembled .S (runtime dispatch).
        sources += [
            rel(blake3 / f"blake3_{x}_x86-64_unix.S")
            for x in ("sse2", "sse41", "avx2", "avx512")
        ]
        print(
            "vfhe: CPU-tuned build (-march=native, AVX-512 IFMA detected).",
            file=sys.stderr,
        )
    else:
        # Portable baseline: scalar engine, BLAKE3 SIMD off. Runs everywhere.
        define_macros = discovery.portable_macros(is_x86)
        if force_portable:
            pass  # deliberate portable/wheel build -- stay quiet
        elif is_x86:
            print(
                "vfhe: this x86 CPU lacks AVX-512 IFMA; building the PORTABLE "
                "(slower) engine.",
                file=sys.stderr,
            )
        else:
            print(
                "vfhe: non-x86 platform; building the PORTABLE engine.",
                file=sys.stderr,
            )

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
        # Dev build: compile in a throwaway temp dir, then keep only the finished
        # extension in the gitignored .generated/ folder.
        with tempfile.TemporaryDirectory() as tmp:
            built = Path(build_ffi(absolute=True).compile(tmpdir=tmp, verbose=True))
            dest = out / built.name
            dest.unlink(missing_ok=True)  # replace any stale ext from a prior build
            shutil.copy2(built, dest)
    print(f"native lib -> {dest.relative_to(ROOT)}")
