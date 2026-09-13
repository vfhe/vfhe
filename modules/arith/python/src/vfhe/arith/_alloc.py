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

#: The same buffers without the zeroing, for a result about to be written whole.
#:
#: Zeroing is not free at the sizes the vectors reach: it is a pass over the
#: buffer *and* the first touch of every page in it. At a 2^21-element vector
#: over a degree-4 extension it is 38 ms against the 10 ms the addition writing
#: over it takes -- three quarters of what an allocating whole-vector operation
#: costs is erasing memory nothing will read.
#:
#: Only for a buffer whose meaningful words are all written before any of them
#: is read. The padding past `n` is a separate matter and is never covered by
#: this: the arithmetic kernels read and write the padding too, so it has to
#: hold reduced values whatever the data words do -- see the layout contract in
#: ``arith.h``. A caller taking these buffers zeroes that tail itself.
aligned64_unset = ffi.new_allocator(
    alloc=lambda size: lib.safe_aligned_malloc(size),
    free=lambda ptr: lib.free(ptr),
    should_clear_after_alloc=False,
)
