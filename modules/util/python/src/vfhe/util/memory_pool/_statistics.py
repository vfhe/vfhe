# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
"""The memory pool's accounting, over the engine's `lib`."""

from dataclasses import dataclass

from vfhe.util.bindings import ffi, lib


@dataclass(frozen=True)
class MemoryPoolStatistics:
    """Holds a snapshot of the pool, over all threads."""

    retained_bytes: int
    """Kept by the pool, ready to be handed out."""
    outstanding_bytes: int
    """Handed out and not yet released to the pool."""
    peak_outstanding_bytes: int
    """The most ever outstanding at once; the ``"auto"`` capacity. Exact
    while one thread uses the pool; with several it may miss a short-lived
    peak by up to a few MiB per thread, as each thread shares its count in
    steps."""
    retention_limit_bytes: int
    """What the capacity allows the pool to keep right now."""
    hits: int
    """Requests served from the pool."""
    misses: int
    """Requests passed to the C library."""


def statistics() -> MemoryPoolStatistics:
    """Returns the pool's current accounting (see `MemoryPoolStatistics`)."""
    out = ffi.new("MempoolStatistics *")
    lib.mempool_statistics(out)
    return MemoryPoolStatistics(
        retained_bytes=out.retained_bytes,
        outstanding_bytes=out.outstanding_bytes,
        peak_outstanding_bytes=out.peak_outstanding_bytes,
        retention_limit_bytes=out.retention_limit_bytes,
        hits=out.hits,
        misses=out.misses,
    )
