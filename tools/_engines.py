# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""The engines, as data: one entry per ISA level we compile the kernels for.

Order is preference, best first; `portable` is last because it is the
fallback every platform can run. Adding an engine means adding an entry here
and its `#if` branch in the kernels — the build, the picker, and the test
suites all derive from this list.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

# Every spelling platforms report for the architectures we build for.
_ARCH_ALIASES: dict[str, frozenset[str]] = {
    "x86_64": frozenset({"x86_64", "amd64", "x86-64", "x64"}),
    "arm64": frozenset({"arm64", "aarch64"}),
}


def host_arch() -> str | None:
    """This machine's canonical architecture name; None when unrecognised."""
    machine = platform.machine().lower()
    canonical = next(
        (arch for arch, names in _ARCH_ALIASES.items() if machine in names), None
    )
    return canonical


def _blake3_dir(root: Path) -> Path:
    return root / "external" / "blake3" / "c"


def find_include_dirs(root: Path) -> list[Path]:
    """Public include dirs, any src dir holding headers, and BLAKE3's."""
    dirs = set((root / "modules").glob("*/c/include"))
    dirs |= {header.parent for header in (root / "modules").glob("*/c/src/**/*.h")}
    dirs.add(_blake3_dir(root))

    ordered = sorted(dirs)
    return ordered


@dataclass(frozen=True)
class Engine:
    # Also the artifact suffix, so it ends up inside a C identifier
    # (PyInit__vfhe_native_<name>): letters and digits only, no dashes.
    name: str
    arch: str | None  # None: every architecture can build it
    requires: str | None  # capability vfhe_cpu_supports() must confirm to run it
    flags: tuple[str, ...]  # ISA flags; they define the macros kernels guard on
    defines: tuple[tuple[str, str | None], ...]  # (name, None) = bare -DNAME
    extra_sources: tuple[Path, ...]  # engine-only files (BLAKE3's .S kernels)
    # How to run it on a CPU without the ISA: the tool and its target,
    # e.g. ("sde", "-icl"). None = this CPU or nothing.
    emulator: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not (self.name.isascii() and self.name.isalnum()):
            raise ValueError(f"engine name '{self.name}' is not a C identifier")

    @property
    def cflags(self) -> list[str]:
        """ISA flags plus defines, rendered for a compiler."""
        rendered = [f"-D{n}" if v is None else f"-D{n}={v}" for n, v in self.defines]
        return [*self.flags, *rendered]


def _portable_defines() -> tuple[tuple[str, str | None], ...]:
    """Scalar kernels, and BLAKE3's own SIMD switched off per architecture."""
    defines: list[tuple[str, str | None]] = [("PORTABLE_BUILD", None)]
    if host_arch() == "x86_64":
        isas = ("SSE2", "SSE41", "AVX2", "AVX512")
        defines += [(f"BLAKE3_NO_{isa}", None) for isa in isas]
    else:
        defines.append(("BLAKE3_USE_NEON", "0"))

    return tuple(defines)


def _blake3_x86_kernels(root: Path) -> tuple[Path, ...]:
    """BLAKE3's pre-assembled SIMD kernels; x86_64 engines compile these in."""
    variants = ("sse2", "sse41", "avx2", "avx512")
    kernels = (_blake3_dir(root) / f"blake3_{v}_x86-64_unix.S" for v in variants)
    return tuple(kernels)


def registry(root: Path) -> list[Engine]:
    """Every engine this repository knows, best first."""
    engines = [
        Engine(
            name="avx512ifma",
            arch="x86_64",
            requires="avx512ifma",
            # Explicit ISA flags, not -march=native: they define the macros the
            # kernels guard on, so any host of the architecture compiles the
            # same code that capable hardware would.
            flags=(
                "-mavx512f",
                "-mavx512ifma",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx2",
                "-maes",
                "-mrdrnd",
                "-mbmi2",
                "-madx",
                "-funroll-all-loops",
            ),
            defines=(),
            extra_sources=_blake3_x86_kernels(root),
            # Ice Lake: AVX-512 F/IFMA/DQ/VL + AES-NI + RDRND.
            emulator=("sde", "-icl"),
        ),
        Engine(
            name="portable",
            arch=None,
            requires=None,
            flags=(),
            defines=_portable_defines(),
            extra_sources=(),
            emulator=None,
        ),
    ]
    return engines


def buildable(root: Path) -> list[Engine]:
    """The engines this host can compile, best first: an engine's kernels are
    architecture-specific even when its execution needs an emulator."""
    arch = host_arch()
    compilable = [e for e in registry(root) if e.arch in (None, arch)]
    return compilable


def by_name(root: Path, name: str) -> Engine | None:
    """The buildable engine with this name, or None."""
    matches = [e for e in buildable(root) if e.name == name]
    return matches[0] if matches else None
