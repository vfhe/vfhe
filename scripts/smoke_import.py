#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-build import smoke test for a packaged vfhe (wheel or sdist install).

Confirms the CFFI native extension loads and that a ring constructs through it
(so the generated proto bindings and the compiled kernels are both wired up).
Run from outside the source tree so ``vfhe`` resolves to the installed package,
not the ``modules/*/python/src`` working copy. Used by the CI package smoke job
and by cibuildwheel's per-wheel ``test-command``.
"""

from vfhe.arith import Polynomial, Ring  # noqa: F401  (imported for the smoke check)

Ring(16, prime_size=[30], split_degree=1)
print("smoke OK")
