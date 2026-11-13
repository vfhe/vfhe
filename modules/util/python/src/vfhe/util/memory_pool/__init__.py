# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Controls the library's memory pool: its capacity, wiping, and accounting.

vfhe keeps the large buffers it releases, per thread, and hands them out again
for the next request of the same size, instead of returning them to the C
library and faulting fresh memory in on the next allocation. What it keeps
over all threads is bounded by a capacity:

- ``"auto"`` (the default): at most the most the library ever had in use at
  once, so a repeating workload is served from the pool and, once its
  buffers are released, the pool holds no more than the library's own peak.
- a number of bytes.
- ``0``: keep nothing; every buffer goes back to the C library.

``VFHE_MEMPOOL_CAPACITY`` (bytes) sets the default.

A released buffer keeps whatever it held until it is overwritten. With
`set_wipe_on_release` every buffer is zeroed as it is released, at the cost
of a pass over it; off by default (``VFHE_MEMPOOL_WIPE_ON_RELEASE=1`` turns it
on).

>>> from vfhe.util import memory_pool
>>> memory_pool.set_capacity(1 << 30)
>>> memory_pool.statistics().retained_bytes <= 1 << 30
True
"""

from ._core import (
    capacity,
    release_all,
    set_capacity,
    set_wipe_on_release,
    wipe_on_release,
)
from ._statistics import MemoryPoolStatistics, statistics

__all__ = [
    "MemoryPoolStatistics",
    "capacity",
    "release_all",
    "set_capacity",
    "set_wipe_on_release",
    "statistics",
    "wipe_on_release",
]
