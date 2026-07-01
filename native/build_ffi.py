# SPDX-License-Identifier: Apache-2.0
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parent.parent
MODULES = ROOT / "modules"


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


_ARCH_ALIASES = (
    {"x86_64", "amd64", "x86-64", "x64"},
    {"aarch64", "arm64"},
)


def _arch_aliases() -> "set[str]":
    machine = platform.machine().lower()
    for group in _ARCH_ALIASES:
        if machine in group:
            return group
    return {machine}


def _arch_ok(path: Path, aliases: "set[str]") -> bool:
    parts = [p.lower() for p in path.parts]
    for i, seg in enumerate(parts[:-1]):
        if seg == "arch":
            return parts[i + 1] in aliases
    return True


def header_to_cdef(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def build_ffi(absolute: bool) -> "FFI":
    ffi = FFI()
    sources: list[str] = []
    include_dirs: list[str] = []
    preamble_headers: list[Path] = []
    aliases = _arch_aliases()

    def rel(p: Path) -> str:
        return str(p if absolute else p.relative_to(ROOT))

    # Sorted order matters: shared typedefs must be cdef'd before their users.
    for c_dir in sorted(MODULES.glob("*/c")):
        headers = sorted((c_dir / "include").glob("*.h"))
        cdefs = sorted((c_dir / "cdef").glob("*.cdef"))
        if cdefs:  # explicit cdef wins (header need not be cffi-clean)
            for cdef in cdefs:
                ffi.cdef(cdef.read_text())
        else:  # header-direct
            for header in headers:
                ffi.cdef(header_to_cdef(header.read_text()))
        preamble_headers += headers

        src_dir = c_dir / "src"
        for pattern in ("*.c", "*.S"):
            for src in sorted(src_dir.rglob(pattern)):
                if _arch_ok(src, aliases):
                    sources.append(rel(src))

        # Public headers, plus any dir under src that holds headers (vendored libs).
        inc = c_dir / "include"
        if inc.is_dir():
            include_dirs.append(rel(inc))
        for header in src_dir.rglob("*.h"):
            include_dirs.append(rel(header.parent))

    if sys.platform == "win32":
        compile_args, link_args = ["/O2", "/GL"], ["/LTCG"]
    else:
        compile_args, link_args = ["-O3", "-flto", "-std=c11"], ["-flto"]

    ffi.set_source(
        "_vfhe_native",
        # Real headers are #included so the C compiler sees the true definitions.
        "\n".join(f'#include "{h.name}"' for h in preamble_headers),
        sources=sorted(set(sources)),
        include_dirs=sorted(set(include_dirs)),
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
