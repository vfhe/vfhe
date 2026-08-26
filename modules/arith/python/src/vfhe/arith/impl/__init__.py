# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""One subpackage per arithmetic implementation.

A subpackage owns everything specific to how its elements are represented:
the parent and element classes, the native state they cache, and the Spec
that registers them. Nothing here is part of the public API -- reach these
classes through `vfhe.arith`, which re-exports them, so that moving one has
no effect outside its own directory.
"""
