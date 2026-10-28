# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Chooses the path a sample operation takes: its own kernel, or the generic
interface alone.

A module may compute a whole sample's operation in a kernel of its own
instead of through the ring operations of the generic interface. With
`set_generic_only` it computes through them: slower, and every operation
then passes one place. Benchmarks compare the two; the execution tape
(`vfhe.util.trace`) takes the generic path on its own while it records.

>>> from vfhe import arith
>>> arith.set_generic_only(True)
>>> arith.generic_only()
True
>>> arith.set_generic_only(False)
"""

from vfhe.util.bindings import lib

__all__ = ["generic_only", "set_generic_only"]


def set_generic_only(enabled: bool) -> None:
    """Makes every sample operation go through the generic interface, or
    lets each take its kernel again."""
    lib.arith_set_generic_only(1 if enabled else 0)


def generic_only() -> bool:
    """Whether sample operations go through the generic interface only."""
    return bool(lib.arith_generic_only())
