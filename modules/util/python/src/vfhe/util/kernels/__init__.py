# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Compiles your C kernels into a vfhe engine: `build`, `compile`, `directory`.

The engine it builds carries the prebuilt engine's module name,
``_vfhe_native_<engine>``, in a directory `build` puts first on
``sys.path``; ``python -m vfhe.util.kernels`` prints that directory for
``PYTHONPATH``. `vfhe.util.bindings` then imports it in place of the prebuilt
one. An extension stays loaded for the life of the process, so this runs before
the first import of vfhe's C; `build` raises once another engine is loaded.

>>> from vfhe.util import kernels
>>> kernels.build(["mine.c"], ["mine.h"])
>>> from vfhe.util.bindings import lib
>>> lib.my_kernel(...)

When generating the C itself needs vfhe loaded, `compile` still works: it
writes the engine and touches nothing in this process. Run the code
that uses the kernel in a child process, with that directory first on
``PYTHONPATH``:

>>> out = kernels.compile(["generated.c"], ["generated.h"])
>>> path = os.pathsep.join([str(out), os.environ.get("PYTHONPATH", "")])
>>> subprocess.run(
...     [sys.executable, "worker.py"], env={**os.environ, "PYTHONPATH": path}
... )
"""

from ._build import build, compile, directory

__all__ = ["build", "compile", "directory"]
