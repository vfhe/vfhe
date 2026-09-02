# SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
# SPDX-License-Identifier: Apache-2.0
"""
Over-aligned cffi buffers, shared by every wrapper whose C side uses AVX-512.

The tuned kernels cast these buffers to __m512d / __m512i and use aligned loads,
so 64-byte alignment is an ABI requirement, not an optimisation -- more than
cffi's default allocator promises. Backed by the engine's posix_memalign wrapper
and freed through libc free on garbage collection.

This lives in its own module because the invariant is cross-cutting: two private
copies of the allocator would be two things to keep in step.
"""

from vfhe.engine import ffi, lib

aligned64 = ffi.new_allocator(
    alloc=lambda size: lib.safe_aligned_malloc(size),
    free=lambda ptr: lib.free(ptr),
    should_clear_after_alloc=True,
)
