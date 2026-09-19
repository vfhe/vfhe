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
import functools
import hashlib
import importlib
import importlib.machinery
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
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


# What setuptools reads from the environment when it builds the module, less
# the compiler itself, which `_compiler_identity` covers. Two builds that
# disagree on any of them are two different modules.
_TOOLCHAIN_ENV = ("CFLAGS", "CPPFLAGS", "LDFLAGS")


@functools.cache
def _compiler_version(command: str) -> str:
    """What `command` answers to ``--version``, first line.

    Keyed on the command and cached, because a compiler does not change under
    a running interpreter while the reuse path is meant to cost nothing -- and
    because keying it means a caller that switches ``CC`` between two builds
    gets two answers rather than the first one twice.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - the configured compiler
            [*shlex.split(command), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("'%s' would not say which version it is", command)
        return "version unknown"
    first, _, _ = completed.stdout.partition("\n")
    return first.strip() or "version unknown"


def _compiler_identity() -> str:
    """The compiler that will build the module, named and versioned.

    The version is here for the reason the flags are: two releases of one
    compiler take the same flags and emit different code, so a module built by
    one cannot be handed out in answer to a request that would build with the
    other.
    """
    command = os.environ.get("CC") or sysconfig.get_config_var("CC") or ""
    if not command:
        return ""
    return f"{command} {_compiler_version(command)}"


def _feed(hasher, *parts: str) -> None:
    """Hash `parts` so that no regrouping of them hashes alike."""
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\0")


def _inputs_hash(extra_compile_args: list[str], extra_link_args: list[str]) -> str:
    """One hash over everything that decides what the module holds, naming it.

    The registered sources and declarations, the flags this build adds, and the
    toolchain that will carry them out. The flags are in because they decide
    meaning and not only speed -- a `-D` is as much a part of the source as the
    file it applies to -- and the toolchain because the same flags do not mean
    the same thing to every compiler. Two builds that differ in any of it are
    therefore two modules, each with a name of its own, and neither can be
    handed out in answer to the other.
    """
    hasher = hashlib.sha256()
    for f in sorted(_user_code.c_files) + sorted(_user_code.cdef_files):
        _feed(hasher, f)
        if os.path.exists(f):
            hasher.update(Path(f).read_bytes())
            hasher.update(b"\0")
    _feed(hasher, *_user_code.cdef_strings)
    _feed(hasher, *extra_compile_args, *extra_link_args)
    _feed(hasher, *(f"{name}={os.environ.get(name, '')}" for name in _TOOLCHAIN_ENV))
    _feed(hasher, _compiler_identity())
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
        # depend on them, and its link_args name what it needs to load. -pthread
        # because the archive uses pthreads and, unlike the in-tree build, this
        # link does not otherwise pull them in -- a manylinux archive references
        # a separate libpthread and crashes when it is missing.
        extra_compile_args=[
            "-O3",
            "-std=gnu11",
            "-pthread",
            *engine["cargs"],
            *extra_compile_args,
        ],
        extra_link_args=["-pthread", *engine["link_args"], *extra_link_args],
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


def _cached_module(module_name: str, output_dir: str, archive: Path) -> str | None:
    """An earlier build of the registered inputs in ``output_dir``, or None.

    The name answers most of the question on its own: it carries the inputs'
    hash and the engine, so a file bearing it was built from these sources
    against this ABI. What remains is the archive, and the module links the
    archive rather than the sources behind it -- an edited source that has not
    been rebuilt changes nothing about what the module contains -- so the
    archive's own mtime is the whole of the freshness test.

    Only a suffix this interpreter imports counts, because one cache directory
    serves every interpreter that shares the name.
    """
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        module_path = os.path.join(output_dir, module_name + suffix)
        if os.path.exists(module_path):
            break
    else:
        return None
    if os.path.getmtime(module_path) < os.path.getmtime(archive):
        logger.info("'%s' is older than %s; rebuilding.", module_name, archive.name)
        return None
    return module_path


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
    logger.info("Reloaded: the process now runs the custom module.")


def compile(
    output_dir=None, extra_compile_args=None, extra_link_args=None, reuse=False
):
    """Compiles the registered user code against the library and updates the
    loaded handles; returns the path of the compiled module.

    ``reuse=True`` hands over an earlier build of the same inputs from
    ``output_dir`` instead, when one is there and still newer than the archive
    it links, and compiles only otherwise. What counts as the same inputs is
    the module's name: the registered sources and declarations, the flags given
    here, the compiler and its version, and the active engine. So building the same sources
    twice under different flags gives two modules rather than one, and either
    can be reused without the other standing in for it. A cached module that
    will not load is rebuilt.

    Reuse is the library's to offer because that name is: no caller can derive
    it without copying private code, and one that guesses it wrong either
    rebuilds every time or loads a module built for other flags or another
    ABI.

    Objects that hold the library take their handle when they are constructed,
    so call this before constructing any of them.
    """
    extra_compile_args = extra_compile_args or []
    extra_link_args = extra_link_args or []
    root = find_vfhe_root()
    engine = _active_engine()
    archive, engine_json = _library_paths(root, engine)
    # The engine is part of the name because it is part of the ABI, not just of
    # the performance: the public headers' types change under its flags
    # (`mp_vector_t` is one `__m512i` on a tuned engine and one `uint64_t` on
    # the portable one) and its kernels need the ISA. Two engines' builds of
    # one set of sources are therefore different modules, and naming them alike
    # would let one output directory hold only whichever was compiled last, and
    # `reuse` hand back one this process cannot run.
    module_name = (
        f"_vfhe_custom_{engine}_"
        f"{_inputs_hash(extra_compile_args, extra_link_args)[:16]}"
    )

    if output_dir is None:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        output_dir = os.path.join(cache, "vfhe")
    os.makedirs(output_dir, exist_ok=True)

    if reuse:
        module_path = _cached_module(module_name, output_dir, archive)
        if module_path is not None:
            try:
                _load_and_swap(module_name, output_dir)
            # Broad on purpose: a cached module is an optimization, and every
            # way one can be unusable -- truncated, built against a library
            # since replaced in place, a symbol the cdef no longer matches --
            # is answered by building it again.
            except Exception:
                logger.warning(
                    "'%s' did not load; rebuilding.", module_name, exc_info=True
                )
            else:
                return module_path

    ffi = _assemble_ffi(
        root,
        archive,
        engine_json,
        module_name,
        extra_compile_args,
        extra_link_args,
    )
    logger.info("Compiling custom library '%s'...", module_name)
    dest = _compile_module(ffi, output_dir)
    _load_and_swap(module_name, output_dir)
    return dest
