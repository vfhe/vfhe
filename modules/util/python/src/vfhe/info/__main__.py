# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Prints a JSON report of this install: ``python -m vfhe.info``.

``version`` is null in a source tree. ``engine`` is null when no engine loads,
and the reason goes to stderr, so stdout stays data. ``engine_file`` is the
loaded extension's path, which tells a custom engine from the prebuilt one.
``runs`` maps every engine to whether this CPU runs it, null when the CPU
cannot say.
"""

import importlib
import importlib.metadata
import json
import platform
import sys

from vfhe.util._engine import Engine

try:
    version = importlib.metadata.version("vfhe")
except importlib.metadata.PackageNotFoundError:
    version = None

try:
    importlib.import_module("vfhe.util.bindings")
except Exception as failure:  # noqa: BLE001 - any failure to load is reported
    print(f"{type(failure).__name__}: {failure}", file=sys.stderr)

active = Engine.active()
print(
    json.dumps(
        {
            "version": version,
            "engine": active,
            "engine_file": sys.modules[active.module].__file__ if active else None,
            "runs": {each: each.runnable() for each in Engine},
            "python": f"{platform.python_version()} {platform.python_implementation()}",
            "platform": platform.platform(),
        },
        indent=2,
    )
)
