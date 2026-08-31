# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Build the user's registered C into a module linked against the shipped
``libvfhe.a``, load it, and hand the process over to it.

Only the user's files are compiled; the library arrives pre-compiled, with
``engine.json`` recording the flags a matching compile needs (the public
headers change types under them).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from cffi import FFI

from . import _reload, _user_code

logger = logging.getLogger("vfhe.dynamic_extensions")

_last_output_dir: str | None = None


def _is_vfhe_root(path: Path) -> bool:
    """A tree carrying the library's public headers."""
    return next(path.glob("modules/*/c/include/*.h"), None) is not None


def find_vfhe_root() -> Path:
    """The library tree to compile against, in precedence order:
    $VFHE_SOURCE_DIR, the installed package's ``vfhe/_source`` snapshot,
    then parents of this file and of the working directory."""
    env_dir = os.environ.get("VFHE_SOURCE_DIR")
    if env_dir:
        path = Path(env_dir).resolve()
        if _is_vfhe_root(path):
            return path
        raise RuntimeError(
            f"VFHE_SOURCE_DIR is set to {env_dir}, but that is not a vfhe "
            "source tree (no modules/*/c/include headers)."
        )
    # parents[1] is the vfhe package: this module sits directly under it.
    candidates = [Path(__file__).resolve().parents[1] / "_source"]
    for start in (Path(__file__).resolve().parent, Path(os.getcwd()).resolve()):
        candidates += [start, *list(start.parents)[:9]]
    for candidate in candidates:
        if _is_vfhe_root(candidate):
            return candidate
    raise RuntimeError(
        "no vfhe source tree found (looked for modules/*/c/include). "
        "Install the package normally or set VFHE_SOURCE_DIR."
    )


def _active_engine() -> str:
    """The engine this process runs (the shim picked it at import); a
    module built for the other engine would mix incompatible kernels."""
    from vfhe.engine import ffi, lib

    return ffi.string(lib.vfhe_engine_active()).decode()


def _library_paths(root: Path, engine: str) -> tuple[Path, Path]:
    """The active engine's archive and its engine-<name>.json: under ``lib/``
    in an installed ``_source``, ``build/`` in a repo checkout."""
    for base in (root / "lib", root / "build"):
        archive = base / f"libvfhe_{engine}.a"
        if archive.exists():
            return archive, base / f"engine-{engine}.json"
    raise RuntimeError(
        f"no libvfhe_{engine}.a under {root} "
        "(in a repo checkout, run `make build` first)"
    )


def _inputs_hash() -> str:
    """One hash over every registered input, naming the module."""
    hasher = hashlib.sha256()
    for f in sorted(_user_code.c_files) + sorted(_user_code.cdef_files):
        hasher.update(f.encode("utf-8"))
        if os.path.exists(f):
            hasher.update(Path(f).read_bytes())
    for s in _user_code.cdef_strings:
        hasher.update(s.encode("utf-8"))
    return hasher.hexdigest()


def _assemble_ffi(
    root: Path,
    archive: Path,
    engine_json: Path,
    module_name: str,
    extra_compile_args: list[str],
    extra_link_args: list[str],
) -> FFI:
    """The cffi builder: library cdefs + user cdefs over user sources,
    linked against the archive."""
    engine = json.loads(engine_json.read_text())
    custom_cdef = "\n".join(
        [Path(f).read_text() for f in _user_code.cdef_files] + _user_code.cdef_strings
    )
    preamble = "\n".join(
        f'#include "{h.name}"' for h in sorted(root.glob("modules/*/c/include/*.h"))
    )
    if custom_cdef:
        preamble += "\n\n/* Custom declarations */\n" + custom_cdef

    ffi = FFI()
    for cdef in sorted(root.glob("modules/*/python/cdef/*.cdef")):
        ffi.cdef(cdef.read_text())
    if custom_cdef:
        ffi.cdef(custom_cdef)

    include_dirs = {str(p) for p in root.glob("modules/*/c/include")}
    include_dirs |= {os.path.dirname(f) for f in _user_code.c_files}
    ffi.set_source(
        module_name,
        preamble,
        sources=sorted(set(_user_code.c_files)),
        include_dirs=sorted(include_dirs),
        libraries=["m"],
        extra_objects=[str(archive)],
        # The engine flags must match the archive's: the headers' types
        # depend on them, and its link_args name what it needs to load.
        extra_compile_args=["-O3", "-std=gnu11", *engine["cargs"], *extra_compile_args],
        extra_link_args=[*engine["link_args"], *extra_link_args],
    )
    return ffi


def _compile_module(ffi: FFI, output_dir: str) -> str:
    """One cffi compile into a temp dir; the finished module lands in
    ``output_dir`` and its path is returned."""
    with tempfile.TemporaryDirectory() as build_temp:
        compiled = ffi.compile(tmpdir=build_temp, verbose=True)
        if not compiled or not os.path.exists(compiled):
            raise RuntimeError(
                "Compilation succeeded but no shared library was found in the build output."
            )
        dest = os.path.join(output_dir, os.path.basename(compiled))
        shutil.copy2(compiled, dest)
    return dest


def _load_and_swap(module_name: str, output_dir: str) -> None:
    """Import the new module and hand the process over to it."""
    global _last_output_dir
    if _last_output_dir and _last_output_dir != output_dir:
        with contextlib.suppress(ValueError):
            sys.path.remove(_last_output_dir)
    if output_dir not in sys.path:
        sys.path.insert(0, output_dir)
    _last_output_dir = output_dir
    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    _reload.update_cffi_references(module.ffi, module.lib)
    for reinitializer in _reload.REINITIALIZATION_REGISTRY:
        reinitializer(module.ffi, module.lib)
    logger.info("Reloaded: the process now runs the custom module.")


def compile(output_dir=None, extra_compile_args=None, extra_link_args=None):
    """Compiles the registered user code against the library and updates the
    loaded handles; returns the path of the compiled module."""
    root = find_vfhe_root()
    archive, engine_json = _library_paths(root, _active_engine())
    module_name = f"_vfhe_custom_{_inputs_hash()[:16]}"

    if output_dir is None:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        output_dir = os.path.join(cache, "vfhe")
    os.makedirs(output_dir, exist_ok=True)

    ffi = _assemble_ffi(
        root,
        archive,
        engine_json,
        module_name,
        extra_compile_args or [],
        extra_link_args or [],
    )
    logger.info("Compiling custom library '%s'...", module_name)
    dest = _compile_module(ffi, output_dir)
    _load_and_swap(module_name, output_dir)
    return dest
