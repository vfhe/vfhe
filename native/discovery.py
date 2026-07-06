# SPDX-License-Identifier: Apache-2.0
"""Shared native-build discovery.

Single source of truth for *which* files and flags make up the engine: host
architecture detection, the module C/asm source and include globs, the vendored
BLAKE3 sources, and the portable-baseline defines. Imported by both
``native/build_ffi.py`` (the packaged CFFI build) and ``scripts/run_c_tests.py``
(the dev C-test compiler) so the two never drift.

Dependency-free (stdlib only) and kept beside ``build_ffi.py`` so it ships in
the sdist and is importable during an isolated build (see MANIFEST.in).
"""

from __future__ import annotations

import platform
from pathlib import Path

# Spelling variants per architecture; index 0 is the x86-64 group.
ARCH_ALIASES = ({"x86_64", "amd64", "x86-64", "x64"}, {"aarch64", "arm64"})

# Vendored BLAKE3 portable core (rng/arith call it): always compiled in.
BLAKE3_BASENAMES = ("blake3.c", "blake3_dispatch.c", "blake3_portable.c")


def host_arch_aliases() -> "set[str]":
    """The alias group for the host arch (a singleton for unknown arches)."""
    machine = platform.machine().lower()
    for group in ARCH_ALIASES:
        if machine in group:
            return group
    return {machine}


def is_x86_host() -> bool:
    """True on x86-64 hosts (selects the BLAKE3 x86 SIMD-disable defines)."""
    return platform.machine().lower() in ARCH_ALIASES[0]


def arch_ok(path: Path, aliases: "set[str]") -> bool:
    """False only if ``path`` lives under an ``arch/<other-arch>/`` directory."""
    parts = [p.lower() for p in path.parts]
    for i, seg in enumerate(parts[:-1]):
        if seg == "arch":
            return parts[i + 1] in aliases
    return True


def module_sources(modules_dir: Path, aliases: "set[str]") -> "list[Path]":
    """Every module C/asm source (``*/c/src/**``), minus foreign-arch dirs."""
    return [
        p
        for pattern in ("*.c", "*.S")
        for p in sorted(modules_dir.glob(f"*/c/src/**/{pattern}"))
        if arch_ok(p, aliases)
    ]


def module_include_dirs(modules_dir: Path) -> "list[Path]":
    """Public include dirs plus any dir under src that holds headers."""
    dirs = set(modules_dir.glob("*/c/include"))
    dirs |= {h.parent for h in modules_dir.glob("*/c/src/**/*.h")}
    return sorted(dirs)


def blake3_dir(root: Path) -> Path:
    return root / "external" / "blake3" / "c"


def blake3_sources(root: Path) -> "list[Path]":
    d = blake3_dir(root)
    return [d / s for s in BLAKE3_BASENAMES]


def portable_macros(is_x86: bool) -> "list[tuple[str, str | None]]":
    """The portable-baseline defines (scalar engine + BLAKE3 SIMD disabled).

    A ``None`` value means a bare ``-DNAME`` / ``define_macros`` entry.
    """
    macros: "list[tuple[str, str | None]]" = [("PORTABLE_BUILD", None)]
    if is_x86:
        macros += [
            (f"BLAKE3_NO_{x}", None) for x in ("SSE2", "SSE41", "AVX2", "AVX512")
        ]
    else:
        macros.append(("BLAKE3_USE_NEON", "0"))
    return macros


def macros_as_cli(macros: "list[tuple[str, str | None]]") -> "list[str]":
    """Render ``portable_macros`` entries as ``-D`` compiler flags."""
    return [f"-D{n}" if v is None else f"-D{n}={v}" for n, v in macros]
