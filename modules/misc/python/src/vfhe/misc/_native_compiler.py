# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Compiler probing and distutils patching shared by every native build.

Stdlib only and loadable by file path: at build time the surrounding package
cannot be imported, because the native extension it loads does not exist yet.
"""

import contextlib
import importlib
import sys


def _accept_asm_out_path_exts(orig):
    """Wrap ``_make_out_path_exts`` so its extension table admits assembly."""

    def patched(_cls, output_dir, strip_dir, src_name, extensions):
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
    # Wide suppress on purpose: which of these exist (and what importing them
    # drags in) varies by Python and setuptools version; the patch loop below
    # works with whatever did import.
    for modname in modnames:
        with contextlib.suppress(Exception):
            importlib.import_module(modname)

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
                obj._make_out_path_exts = _accept_asm_out_path_exts(
                    obj._make_out_path_exts
                )
                obj._vfhe_asm_out_path_patched = True
        module_exts = getattr(module, "src_extensions", None)
        if isinstance(module_exts, list):
            module_exts.extend(e for e in (".S", ".s") if e not in module_exts)
