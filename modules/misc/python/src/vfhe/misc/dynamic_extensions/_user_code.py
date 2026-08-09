# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""The user's registered extension inputs: C/assembly sources and cffi
declarations, collected here until ``compile()`` consumes them."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile

logger = logging.getLogger("vfhe.dynamic_extensions")

c_files: list[str] = []
cdef_files: list[str] = []
cdef_strings: list[str] = []
_temp_c_files: list[str] = []


def add_c_file(path: str):
    """Add a C or assembly file (.c, .S) to be compiled with the library."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Source file not found: {abs_path}")
    if not abs_path.endswith((".c", ".S")):
        raise ValueError(f"Unsupported file type (expected .c or .S): {abs_path}")
    if abs_path not in c_files:
        c_files.append(abs_path)
        logger.info(f"Added source file: {abs_path}")


def add_cdef_file(path: str):
    """Add a CFFI declaration file (.cdef) to be processed with the library."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Declaration file not found: {abs_path}")
    if not abs_path.endswith(".cdef"):
        raise ValueError(f"Unsupported file type (expected .cdef): {abs_path}")
    if abs_path not in cdef_files:
        cdef_files.append(abs_path)
        logger.info(f"Added CFFI declaration file: {abs_path}")


def add_c_definitions(definitions: str):
    """Add CFFI declarations directly as a string."""
    cdef_strings.append(definitions)
    logger.info("Added CFFI declarations directly.")


def add_c_code(code: str):
    """Add a string of C code directly to be compiled with the library."""
    fd, path = tempfile.mkstemp(suffix=".c")
    try:
        with open(fd, "w") as f:
            f.write(code)
    except Exception:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise

    c_files.append(path)
    _temp_c_files.append(path)
    logger.info("Added C code string directly.")


def add_c_dir(path: str):
    """Add all C, assembly, and CDEF files in a directory to be compiled with the library."""
    abs_dir = os.path.abspath(path)
    if not os.path.isdir(abs_dir):
        raise NotADirectoryError(f"Directory not found: {abs_dir}")

    added_any = False
    for root, _, files in os.walk(abs_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith((".c", ".S")):
                add_c_file(full_path)
                added_any = True
            elif file.endswith(".cdef"):
                add_cdef_file(full_path)
                added_any = True
    if not added_any:
        logger.warning(f"No custom files (.c, .S, .cdef) found in directory: {abs_dir}")


def clear_extensions():
    """Clear all added custom files and clean up temporary resources."""
    c_files.clear()
    cdef_files.clear()
    cdef_strings.clear()

    for path in _temp_c_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError as e:  # noqa: PERF203 -- keep deleting after one failure
            logger.warning(f"Failed to delete temporary C file {path}: {e}")
    _temp_c_files.clear()
    logger.info("Cleared custom extension files.")


def get_added_files():
    """Return a list of added C/assembly files."""
    return list(c_files)
