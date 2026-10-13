# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Exports `ExtensionBuilder`, which compiles your C kernels into a vfhe engine.

The engine it builds carries the prebuilt engine's module name,
``_vfhe_native_<engine>``, in a directory `ExtensionBuilder.build` puts first on
``sys.path``; ``python -m vfhe.util.kernels`` prints that directory for
``PYTHONPATH``. `vfhe.util.bindings` then imports it in place of the prebuilt
one. An extension stays loaded for the life of the process, so this runs before
the first import of vfhe's C; `ExtensionBuilder.build` raises once another
engine is loaded.

>>> from vfhe.util.kernels import ExtensionBuilder
>>> ExtensionBuilder(["mine.c"], ["mine.h"]).build()
>>> from vfhe.util.bindings import lib
>>> lib.my_kernel(...)

When generating the C itself needs vfhe loaded, `ExtensionBuilder.compile` still
works: it writes the engine and touches nothing in this process. Run the code
that uses the kernel in a child process, with that directory first on
``PYTHONPATH``:

>>> out = ExtensionBuilder(["generated.c"], ["generated.h"]).compile()
>>> path = os.pathsep.join([str(out), os.environ.get("PYTHONPATH", "")])
>>> subprocess.run(
...     [sys.executable, "worker.py"], env={**os.environ, "PYTHONPATH": path}
... )
"""

from ._build import ExtensionBuilder

__all__ = ["ExtensionBuilder"]
