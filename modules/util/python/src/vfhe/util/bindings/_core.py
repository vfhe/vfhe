# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Loads the selected engine and exposes its `ffi` and `lib`.

The one place an engine loads; the import system runs it once per process.
"""

import importlib

from vfhe.util._engine import Engine

_module = importlib.import_module(Engine.select().module)
ffi = _module.ffi
lib = _module.lib
