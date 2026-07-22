# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the files and flags that make up the engine.

All consumers read from here so no two builds drift. Stdlib-only: importable
during an isolated sdist build.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

# Spelling variants per architecture.
X86_ALIASES = {"x86_64", "amd64", "x86-64", "x64"}
ARM_ALIASES = {"aarch64", "arm64"}
ARCH_ALIASES = (X86_ALIASES, ARM_ALIASES)

# Vendored BLAKE3 portable core (misc/arith call it): always compiled in.
BLAKE3_BASENAMES = ("blake3.c", "blake3_dispatch.c", "blake3_portable.c")


def host_arch_aliases() -> set[str]:
    """The alias group for the host arch (a singleton for unknown arches)."""
    machine = platform.machine().lower()
    for group in ARCH_ALIASES:
        if machine in group:
            return group
    return {machine}


def is_x86_host() -> bool:
    """True on x86-64 hosts (selects the BLAKE3 x86 SIMD-disable defines)."""
    return platform.machine().lower() in X86_ALIASES


def arch_ok(path: Path, aliases: set[str]) -> bool:
    """False only if ``path`` lives under an ``arch/<other-arch>/`` directory."""
    parts = [p.lower() for p in path.parts]
    for i, seg in enumerate(parts[:-1]):
        if seg == "arch":
            return parts[i + 1] in aliases
    return True


def module_sources(modules_dir: Path, aliases: set[str]) -> list[Path]:
    """Every module C/asm source (``*/c/src/**``), minus foreign-arch dirs."""
    return [
        p
        for pattern in ("*.c", "*.S")
        for p in sorted(modules_dir.glob(f"*/c/src/**/{pattern}"))
        if arch_ok(p, aliases)
    ]


def module_include_dirs(modules_dir: Path) -> list[Path]:
    """Public include dirs plus any dir under src that holds headers."""
    dirs = set(modules_dir.glob("*/c/include"))
    dirs |= {h.parent for h in modules_dir.glob("*/c/src/**/*.h")}
    return sorted(dirs)


def blake3_dir(root: Path) -> Path:
    """The vendored BLAKE3 C tree (git submodule)."""
    return root / "external" / "blake3" / "c"


def blake3_sources(root: Path) -> list[Path]:
    """The always-compiled BLAKE3 core (portable code + runtime dispatcher)."""
    d = blake3_dir(root)
    return [d / s for s in BLAKE3_BASENAMES]


# --- Vendored dependencies ---------------------------------------------------
# The single registry: a new vendored dep extends these three functions and
# nothing else; every consumer follows.


def vendored_sources(root: Path) -> list[Path]:
    """Every vendored C/asm source the build compiles in."""
    return blake3_sources(root)


def vendored_include_dirs(root: Path) -> list[Path]:
    """Include directories the vendored sources need."""
    return [blake3_dir(root)]


def vendored_missing(root: Path) -> list[Path]:
    """Required vendored files absent from the tree (submodule not checked out)."""
    return [p for p in vendored_sources(root) if not p.exists()]


class ToolBuildContext(NamedTuple):
    """Compile inputs pre-rendered as compiler flags."""

    kernels: list[Path]  # every C/asm source the engine compiles
    include_flags: list[str]  # -I... (Unity included, for the C suites)
    define_flags: list[str]  # -D... portable baseline


def tool_build_context(root: Path) -> ToolBuildContext:
    """Compile context for standalone tool builds.

    Same kernels and defines as the packaged build, so tools cannot drift.
    """
    modules_dir = root / "modules"
    include_dirs = (
        module_include_dirs(modules_dir)
        + [root / "external" / "unity" / "src"]
        + vendored_include_dirs(root)
    )
    return ToolBuildContext(
        kernels=module_sources(modules_dir, host_arch_aliases())
        + vendored_sources(root),
        include_flags=[f"-I{d}" for d in include_dirs],
        define_flags=macros_as_cli(portable_macros(is_x86_host())),
    )


def portable_macros(is_x86: bool) -> list[tuple[str, str | None]]:
    """The portable-baseline defines (scalar engine + BLAKE3 SIMD disabled).

    A ``None`` value means a bare ``-DNAME`` / ``define_macros`` entry.
    """
    macros: list[tuple[str, str | None]] = [("PORTABLE_BUILD", None)]
    if is_x86:
        macros += [
            (f"BLAKE3_NO_{x}", None) for x in ("SSE2", "SSE41", "AVX2", "AVX512")
        ]
    else:
        macros.append(("BLAKE3_USE_NEON", "0"))
    return macros


def macros_as_cli(macros: list[tuple[str, str | None]]) -> list[str]:
    """Render ``portable_macros`` entries as ``-D`` compiler flags."""
    return [f"-D{n}" if v is None else f"-D{n}={v}" for n, v in macros]


def sanitizer_flags(checks: str) -> list[str]:
    """``-fsanitize`` flags for ``checks``; no-recover so the first finding aborts."""
    return [
        f"-fsanitize={checks}",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
        "-g",
        "-O1",
    ]


# --- The packaged-build recipe -----------------------------------------------


class NativeBuildPlan(NamedTuple):
    """Everything the CFFI native build needs; callers add their own extras."""

    preamble_headers: list[Path]  # umbrella headers to #include in the wrapper
    cdef_files: list[Path]  # hand-written cffi declaration files
    sources: list[Path]
    include_dirs: list[Path]
    define_macros: list[tuple[str, str | None]]
    compile_args: list[str]
    link_args: list[str]
    libraries: list[str]
    note: str | None  # human explanation of the tuned/portable choice


def native_build_plan(
    root: Path, cpu_supports_tuning: Callable[[], bool]
) -> NativeBuildPlan:
    """The full recipe for compiling the engine into a CFFI extension.

    One recipe so every native build produces the same library.
    ``cpu_supports_tuning`` is injected so this module stays importable
    without the installed package.

    VFHE ships sdist-only, so the default build auto-tunes to the host: x86
    with AVX-512 IFMA drops PORTABLE_BUILD (SIMD kernels + AES-NI,
    -march=native); anything else builds the portable engine, ``note`` says
    why. The tuned path is POSIX-x86 only (BLAKE3 *_x86-64_unix.S).
    VFHE_PORTABLE=1 forces portable (build here, run elsewhere).
    VFHE_COVERAGE=1 compiles -O0 --coverage for gcov, without LTO (it strips
    gcov data); always portable.
    """
    modules_dir = root / "modules"

    # Umbrella headers form the preamble; Python-facing modules declare their
    # ABI in python/cdef/. Sorted so shared typedefs are cdef'd before users.
    preamble_headers = [
        header
        for c_dir in sorted(modules_dir.glob("*/c"))
        for header in sorted((c_dir / "include").glob("*.h"))
    ]
    cdef_files = sorted(modules_dir.glob("*/python/cdef/*.cdef"))

    sources = module_sources(modules_dir, host_arch_aliases()) + vendored_sources(root)
    include_dirs = module_include_dirs(modules_dir) + vendored_include_dirs(root)

    is_x86 = is_x86_host()
    coverage = os.environ.get("VFHE_COVERAGE") == "1"
    force_portable = os.environ.get("VFHE_PORTABLE") == "1" or coverage
    tuned = (
        (not force_portable)
        and is_x86
        and sys.platform != "win32"
        and cpu_supports_tuning()
    )

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

    define_macros: list[tuple[str, str | None]]
    if tuned:
        define_macros = []  # drop PORTABLE_BUILD -> activate the SIMD-guarded paths
        compile_args += ["-march=native", "-funroll-all-loops"]
        # BLAKE3 SIMD via pre-assembled .S (runtime dispatch).
        sources += [
            blake3_dir(root) / f"blake3_{x}_x86-64_unix.S"
            for x in ("sse2", "sse41", "avx2", "avx512")
        ]
        note = "CPU-tuned build (-march=native, AVX-512 IFMA detected)."
    else:
        # Portable baseline: scalar engine, BLAKE3 SIMD off. Runs everywhere.
        define_macros = portable_macros(is_x86)
        if force_portable:
            note = None  # deliberate portable build: stay quiet
        elif is_x86:
            note = "this x86 CPU lacks AVX-512 IFMA; building the PORTABLE (slower) engine."
        else:
            note = "non-x86 platform; building the PORTABLE engine."

    return NativeBuildPlan(
        preamble_headers=preamble_headers,
        cdef_files=cdef_files,
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        compile_args=compile_args,
        link_args=link_args,
        libraries=libraries,
        note=note,
    )
