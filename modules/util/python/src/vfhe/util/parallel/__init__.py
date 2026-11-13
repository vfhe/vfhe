# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""Sets how many threads vfhe uses: one library-wide limit, and a local one per thread.

- `set_num_threads` and `num_threads` set and read the library-wide limit.
  Functions taking ``n_threads`` never exceed it, and their default
  ``n_threads=0`` means "up to the limit".
- `local_num_threads` lowers the limit for the vfhe calls the current thread
  makes inside a ``with`` block.

The default limit is ``VFHE_NUM_THREADS`` if set, otherwise 1, so vfhe is
single-threaded unless asked.

>>> from vfhe.util import parallel
>>> parallel.set_num_threads(8)
>>> with parallel.local_num_threads(2):
...     pass  # vfhe calls made here use at most 2 threads
>>> parallel.set_num_threads()
"""

from ._threading import local_num_threads, num_threads, set_num_threads

__all__ = [
    "local_num_threads",
    "num_threads",
    "set_num_threads",
]
