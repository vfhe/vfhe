# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration: the guides in this directory, the Python API from docstrings, the C API from Doxygen's XML."""

import os
import pathlib
import sys

from docutils import nodes

ROOT = pathlib.Path(__file__).resolve().parents[2]

# autodoc imports vfhe. The build directory holds the engines and _vfhe_flags,
# build/proto the protobuf bindings; the portable engine is the one every
# machine has.
sys.path[:0] = [
    str(ROOT / "build"),
    str(ROOT / "build" / "proto"),
    *map(str, sorted(ROOT.glob("modules/*/python/src"))),
]
os.environ.setdefault("VFHE_ENGINE", "portable")

project = "vFHE"
project_copyright = "2026, The vFHE Authors"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "breathe",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# `name` in a docstring is a cross-reference; ``literal`` stays literal.
# `#heading` links in the guides resolve to the headings they name.
myst_heading_anchors = 3
# A ```mermaid fence, which GitHub renders, is the mermaid directive here.
myst_fence_as_directive = ["mermaid"]

default_role = "py:obj"
autodoc_default_options = {"members": True, "member-order": "bysource"}
# A class documented on its constructor shows that text too.
autoclass_content = "both"

breathe_projects = {"vfhe": str(ROOT / "build" / "docs" / "doxygen" / "xml")}
breathe_default_project = "vfhe"

html_theme = "furo"
html_title = "vFHE"
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/vfhe/vfhe",
            "html": "GitHub",
            "class": "",
        }
    ]
}

# GitHub writes its heading anchors client-side, so the link checker cannot see them.
linkcheck_anchors_ignore_for_url = ["https://github.com/"]

# The Markdown links to GitHub, where it is read as files; on the site the
# same link lands on the page that holds that file.
GITHUB = "https://github.com/vfhe/vfhe/blob/main/"
PAGES = {
    "README.md": "index",
    "CHANGELOG.md": "changelog",
    "docs/USAGE.md": "guides/usage",
    "docs/ARCHITECTURE.md": "guides/architecture",
    "docs/DEVELOPMENT.md": "guides/development",
    "docs/CONTRIBUTING.md": "project/contributing",
    "docs/SECURITY.md": "project/security",
    "docs/SUPPORT.md": "project/support",
}


def link_into_the_site(app, doctree, docname):
    for node in doctree.findall(nodes.reference):
        uri = node.get("refuri", "")
        path, _, fragment = uri.removeprefix(GITHUB).partition("#")
        if uri.startswith(GITHUB) and path in PAGES:
            target = app.builder.get_relative_uri(docname, PAGES[path])
            node["refuri"] = target + (f"#{fragment}" if fragment else "")


def setup(app):
    app.connect("doctree-resolved", link_into_the_site)
