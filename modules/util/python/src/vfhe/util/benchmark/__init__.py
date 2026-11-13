# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The pytest plugin the benchmarks run under: `benchmark_reset` and the `complete` marker.

A module's benchmarks are the pytest-benchmark tests of its ``python/bench``
directory, installed as ``vfhe.<mod>.bench`` and run by
``python -m vfhe.benchmark``. Loaded with ``-p vfhe.util.benchmark``, this
plugin collects every public function a ``bench`` file defines as a
benchmark, whatever its name, runs it after arith's caches are emptied, so
every run starts from the same state, and marks it `complete`, so the fast
test suites leave it out.
"""

from ._pytest import (
    benchmark_reset,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_pycollect_makeitem,
)

__all__ = [
    "benchmark_reset",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_pycollect_makeitem",
]
