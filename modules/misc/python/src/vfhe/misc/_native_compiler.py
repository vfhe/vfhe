# SPDX-License-Identifier: Apache-2.0
"""Compiler probing and distutils patching shared by every native build.

Stdlib only and loadable by file path: at build time the surrounding package
cannot be imported, because the native extension it loads does not exist yet.
"""

import os
import shlex
import subprocess
import sys
import sysconfig


def _accept_asm_out_path_exts(orig):
    """Wrap ``_make_out_path_exts`` so its extension table admits assembly."""

    def patched(cls, output_dir, strip_dir, src_name, extensions):
        if isinstance(extensions, dict):
            extensions.setdefault(".S", ".o")
            extensions.setdefault(".s", ".o")
        elif isinstance(extensions, list):
            extensions.extend(e for e in (".S", ".s") if e not in extensions)
        return orig(output_dir, strip_dir, src_name, extensions)

    return classmethod(patched)


def enable_asm_sources() -> None:
    """Teach distutils to compile ``.S`` assembly (BLAKE3's SIMD kernels).

    Its C compiler classes reject unknown source extensions, and the class path
    has moved across setuptools versions, so every loaded distutils/setuptools
    module is patched: ``src_extensions`` lists on classes and modules gain
    ``.S``/``.s``, and ``_make_out_path_exts`` (which maps sources to object
    paths on newer setuptools) learns the same mapping.
    """
    modnames = (
        "distutils.ccompiler",
        "distutils.unixccompiler",
        "distutils.compilers.C.base",
        "distutils.compilers.C.unix",
        "setuptools._distutils.ccompiler",
        "setuptools._distutils.unixccompiler",
        "setuptools._distutils.compilers.C.base",
        "setuptools._distutils.compilers.C.unix",
    )
    for modname in modnames:
        try:
            __import__(modname, fromlist=["*"])
        except Exception:
            pass

    for name, module in list(sys.modules.items()):
        if ("distutils" not in name and "setuptools" not in name) or not module:
            continue
        for obj in list(vars(module).values()):
            if not isinstance(obj, type):
                continue
            exts = getattr(obj, "src_extensions", None)
            if isinstance(exts, list):
                exts.extend(e for e in (".S", ".s") if e not in exts)
            if hasattr(obj, "_make_out_path_exts") and not getattr(
                obj, "_vfhe_asm_out_path_patched", False
            ):
                setattr(
                    obj,
                    "_make_out_path_exts",
                    _accept_asm_out_path_exts(getattr(obj, "_make_out_path_exts")),
                )
                setattr(obj, "_vfhe_asm_out_path_patched", True)
        module_exts = getattr(module, "src_extensions", None)
        if isinstance(module_exts, list):
            module_exts.extend(e for e in (".S", ".s") if e not in module_exts)


def host_has_avx512ifma() -> bool:
    """True if ``-march=native`` enables AVX-512 IFMA on *this* machine.

    Asks the *same* compiler the extension will be built with what the native
    CPU supports; the engine's SIMD kernels are guarded by ``__AVX512IFMA__``,
    so this matches exactly what a tuned build would light up. Any failure (no
    compiler, ``-march=native`` unsupported, non-zero exit) falls back to
    portable, so the check can only ever *under*-tune, never mis-tune.
    """
    # Resolve the compiler exactly as distutils/cffi will for the real build:
    # $CC if set, else the compiler this Python was built with, else `cc`. Keep
    # any base flags it carries (e.g. a wrapper's --target=) so the probe and
    # the build see the same target.
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    argv = [*shlex.split(cc), "-march=native", "-dM", "-E", "-x", "c", os.devnull]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return out.returncode == 0 and "__AVX512IFMA__" in out.stdout
