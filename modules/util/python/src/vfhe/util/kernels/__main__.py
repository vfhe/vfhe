# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Builds an engine ahead of time and prints its directory.

Put the directory first on ``PYTHONPATH`` and `vfhe.util.bindings` imports that
engine in place of the prebuilt one:

    export PYTHONPATH="$(python -m vfhe.util.kernels mine.c mine.h):$PYTHONPATH"
    python app.py

Loads nothing, so it runs before any vfhe import, and the process that imports
vfhe then needs no compiler.
"""

import argparse

from ._build import ExtensionBuilder

parser = argparse.ArgumentParser(
    prog="python -m vfhe.util.kernels",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "files",
    nargs="+",
    metavar="FILE",
    help="your .c or .S sources and .h declarations, in any order",
)
parser.add_argument(
    "--engine", help="build this engine (default: the one this process would load)"
)
parser.add_argument("--cc", help="the compiler (default: $CC, else Python's)")
args = parser.parse_args()

sources = [f for f in args.files if f.endswith((".c", ".S"))]
if not sources:
    parser.error("no .c or .S source given")

declarations = [f for f in args.files if f.endswith(".h")]

print(
    ExtensionBuilder(
        sources,
        declarations,
        engine=args.engine,
        cc=args.cc,
    ).compile()
)
