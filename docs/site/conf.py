# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration: the guides in this directory, the Python API from docstrings, the C API from Doxygen's XML."""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# autodoc imports vfhe. The build directory holds the engines and _vfhe_flags;
# the portable engine is the one every machine has.
sys.path[:0] = [
    str(ROOT / "build"),
    *map(str, sorted(ROOT.glob("modules/*/python/src"))),
]
os.environ.setdefault("VFHE_ENGINE", "portable")

project = "vFHE"
project_copyright = "2026, The vFHE Authors"
extensions = ["sphinx.ext.autodoc", "breathe", "myst_parser", "sphinxcontrib.mermaid"]

# `name` in a docstring is a cross-reference; ``literal`` stays literal.
# `#heading` links in the guides resolve to the headings they name.
myst_heading_anchors = 3
# A ```mermaid fence, which GitHub renders, is the mermaid directive here.
myst_fence_as_directive = ["mermaid"]

default_role = "py:obj"
autodoc_default_options = {"members": True, "member-order": "bysource"}

breathe_projects = {"vfhe": str(ROOT / "build" / "docs" / "doxygen" / "xml")}
breathe_default_project = "vfhe"

html_theme = "furo"
html_title = "vFHE"
