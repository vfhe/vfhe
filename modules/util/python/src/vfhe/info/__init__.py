# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""Describes this install in one command: ``python -m vfhe.info``.

Version, the engine the picker chose (and anything faster this CPU could run
instead), interpreter, platform — the facts a bug report needs.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from vfhe.engine import active_engine, runnable_engines


def find_version() -> str:
    """The installed version; a source checkout carries no distribution metadata."""
    try:
        installed = version("vfhe")
    except PackageNotFoundError:
        return "unknown (a source tree, not an install)"
    return installed


def describe_engine() -> str:
    """The active engine, and what else this CPU could have run — a pin or a
    wheel built without SIMD kernels shows up right here."""
    active = active_engine()
    others = [name for name in runnable_engines() if name != active]
    if not others:
        return active
    return f"{active} (this CPU can also run: {', '.join(others)})"


def collect_facts() -> list[tuple[str, str]]:
    facts = [
        ("vfhe", find_version()),
        ("engine", describe_engine()),
        ("python", f"{platform.python_version()} {platform.python_implementation()}"),
        ("platform", platform.platform()),
    ]
    return facts


def format_facts(facts: list[tuple[str, str]]) -> str:
    label_width = max(len(label) for label, _ in facts)
    lines = [f"{label:<{label_width}}  {value}" for label, value in facts]
    text = "\n".join(lines)
    return text


def main() -> int:
    facts = collect_facts()
    print(format_facts(facts))
    return 0
