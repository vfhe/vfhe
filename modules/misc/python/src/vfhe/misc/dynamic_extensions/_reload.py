# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Hand the running process over to a freshly compiled module: swap every
vfhe module's ffi/lib handles and re-run the registered reinitializers."""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger("vfhe.dynamic_extensions")

# Callables invoked with (new_ffi, new_lib) after a new module is loaded.
REINITIALIZATION_REGISTRY = []


def register_reinitializer(func):
    """Register a callback function to be called after a new C library is
    compiled and loaded, with (new_ffi, new_lib) as its arguments."""
    if func not in REINITIALIZATION_REGISTRY:
        REINITIALIZATION_REGISTRY.append(func)
    return func


def update_cffi_references(new_ffi, new_lib):
    """Update all global ffi and lib references in imported vfhe modules."""
    # Deliberate module monkeypatching; typed Any because ModuleType has no
    # ffi/lib attributes to a static checker.
    libvfhe_mod: Any = sys.modules["vfhe.misc.libvfhe"]

    old_ffi = libvfhe_mod.ffi
    old_lib = libvfhe_mod.lib

    libvfhe_mod.ffi = new_ffi
    libvfhe_mod.lib = new_lib

    libvfhe_instance = libvfhe_mod.libvfhe
    libvfhe_instance.ffi = new_ffi
    libvfhe_instance.lib = new_lib

    for name, module in list(sys.modules.items()):
        if name.startswith("vfhe"):
            mod: Any = module
            if hasattr(mod, "lib") and mod.lib is old_lib:
                mod.lib = new_lib
            if hasattr(mod, "ffi") and mod.ffi is old_ffi:
                mod.ffi = new_ffi


@register_reinitializer
def reinit_libvfhe(new_ffi, new_lib):
    """Update the fundamental libvfhe singleton."""
    from ..libvfhe import libvfhe

    libvfhe.lib = new_lib
    libvfhe.ffi = new_ffi


@register_reinitializer
def reinit_rns_base(_new_ffi, _new_lib):
    """Replace the RNS base registry and flush its conversion cache."""
    try:
        import vfhe.arith.rns_base as rns_base_mod
        from vfhe.arith.rns_base import RNS_Base_Registry

        if getattr(rns_base_mod, "rns_base_registry", None):
            rns_base_mod.rns_base_registry.cleanup()
        rns_base_mod.rns_base_registry = RNS_Base_Registry()
    except ImportError:
        logger.debug("vfhe.arith not imported; skipping RNS base reinitialization")


@register_reinitializer
def reinit_lwe(_new_ffi, _new_lib):
    """Re-initialize LWE library prototypes."""
    try:
        import vfhe.mlwe.lwe as lwe_mod
        from vfhe.mlwe.lwe import LibLWE

        lwe_mod.lib_lwe = LibLWE()
    except ImportError:
        logger.debug("vfhe.mlwe not imported; skipping LWE reinitialization")


@register_reinitializer
def reinit_rlwe(_new_ffi, _new_lib):
    """Re-initialize MLWE library prototypes."""
    try:
        import vfhe.mlwe.mlwe as mlwe_mod
        from vfhe.mlwe.mlwe import LibMLWE

        mlwe_mod.lib_rlwe = LibMLWE()
    except ImportError:
        logger.debug("vfhe.mlwe not imported; skipping MLWE reinitialization")
