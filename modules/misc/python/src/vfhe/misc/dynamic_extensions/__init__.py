# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Extend the loaded library with user C code, compiled at runtime.

Register C sources and cffi declarations (``add_c_file``, ``add_c_code``,
...), then ``compile()`` builds them into a module linked against the
shipped ``libvfhe.a`` and hot-swaps the process onto it. The parts:
``_user_code`` (the registered inputs), ``_build_module`` (find the
library, compile, link, load), ``_reload`` (swap every vfhe module's
ffi/lib handles), ``_headers`` (the ``vfhe.h`` convenience wrapper).
"""

from ._build_module import compile, find_vfhe_root
from ._headers import cli_main, create_headers
from ._reload import (
    REINITIALIZATION_REGISTRY,
    register_reinitializer,
    update_cffi_references,
)
from ._user_code import (
    add_c_code,
    add_c_definitions,
    add_c_dir,
    add_c_file,
    add_cdef_file,
    clear_extensions,
    get_added_files,
)

__all__ = [
    "REINITIALIZATION_REGISTRY",
    "add_c_code",
    "add_c_definitions",
    "add_c_dir",
    "add_c_file",
    "add_cdef_file",
    "clear_extensions",
    "cli_main",
    "compile",
    "create_headers",
    "find_vfhe_root",
    "get_added_files",
    "register_reinitializer",
    "update_cffi_references",
]
