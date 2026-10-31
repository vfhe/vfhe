// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// Cutting a loop over a vector into contiguous chunks for vfhe_parallel_for.
//
// A vector kernel whose items are independent runs on the library's threads by
// splitting its range into a few chunks per thread, each at least a grain of
// items long and starting on a multiple of a unit (the SIMD group, a window, a
// block), and running its loop once per chunk. When the split gets one thread
// -- the library's limit, the calling thread's local number, or a range under
// two grains -- the kernel runs its loop once over the whole range instead.
#ifndef VFHE_VEC_CHUNKS_H
#define VFHE_VEC_CHUNKS_H

#include <parallel.h>
#include <stdbool.h>
#include <stdint.h>

// Chunks per thread a split aims for: a few, so that a slow chunk does not
// leave the other threads idle, while each stays large enough that what a
// chunk sets up (scratch, staging, a reduction's partial result) is amortised.
#define VEC_CHUNKS_PER_THREAD 4

// `count` chunks of `length` items over [0, total); the last may be shorter.
typedef struct
{
    uint64_t total;
    uint64_t length;
    uint64_t count;
} VecChunks;

// Plans the split of `total` items into chunks of at least `grain` items whose
// length is a multiple of `unit`. Returns false, leaving `chunks` unset, when
// the loop should run on the calling thread alone.
static inline bool vec_chunks_plan(VecChunks *chunks, uint64_t total, uint64_t grain, uint64_t unit)
{
    if (grain == 0)
        grain = 1;
    const uint64_t most = total / grain;
    const uint64_t threads = vfhe_threads_for(0, most);
    if (threads <= 1)
        return false;
    uint64_t count = threads * VEC_CHUNKS_PER_THREAD;
    if (count > most)
        count = most;
    uint64_t length = (total + count - 1) / count;
    length = (length + unit - 1) / unit * unit;
    count = (total + length - 1) / length;
    if (count < 2)
        return false;
    chunks->total = total;
    chunks->length = length;
    chunks->count = count;
    return true;
}

static inline uint64_t vec_chunk_begin(const VecChunks *chunks, uint64_t chunk)
{
    return chunk * chunks->length;
}

static inline uint64_t vec_chunk_end(const VecChunks *chunks, uint64_t chunk)
{
    const uint64_t end = (chunk + 1) * chunks->length;
    return end < chunks->total ? end : chunks->total;
}

#endif // VFHE_VEC_CHUNKS_H
