# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""``create_headers``: a ``vfhe.h`` wrapper that #includes every library
header by absolute path, for user C files that want one include line."""

from __future__ import annotations

import logging
import os
import sys

from ._build_module import find_vfhe_root

logger = logging.getLogger("vfhe.dynamic_extensions")


def create_headers(target_dir=None):
    """Create a file vfhe.h that contains includes to all other headers from
    the library. If target_dir is None, the current directory is used."""
    if target_dir is None:
        target_dir = os.getcwd()
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    root = find_vfhe_root()
    headers = [
        h_file
        for include_dir in sorted((root / "modules").glob("*/c/include"))
        for h_file in sorted(include_dir.glob("*.h"))
    ]
    if not headers:
        raise RuntimeError(f"No header files found under {root}/modules/*/c/include")

    vfhe_h_path = os.path.join(target_dir, "vfhe.h")
    with open(vfhe_h_path, "w") as f:
        f.write(
            "/* Automatically generated vfhe.h wrapper for compiling extensions */\n"
        )
        f.write("#ifndef VFHE_H_WRAPPER\n")
        f.write("#define VFHE_H_WRAPPER\n\n")
        for h in headers:
            h_clean = os.path.abspath(h).replace("\\", "/")
            f.write(f'#include "{h_clean}"\n')
        f.write("\n#endif /* VFHE_H_WRAPPER */\n")

    logger.info(f"Created headers wrapper at: {vfhe_h_path}")
    print(f"Created header wrapper file: {vfhe_h_path}")


def cli_main():
    """CLI wrapper for create_headers."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a vfhe.h wrapper header containing absolute path "
        "includes to the installed vfhe library headers."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="Where vfhe.h is created. Defaults to the current working directory.",
    )
    args = parser.parse_args()

    try:
        create_headers(args.target_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
