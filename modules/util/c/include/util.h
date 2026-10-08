// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <engine.h>
#include <vfhe_cpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Index and conversion helpers
    uint64_t double2int(double x);
    uint32_t int_rev(uint32_t b);
    void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n);

    // Allocation that aborts rather than returning NULL
    void *safe_malloc(size_t size);
    void *safe_realloc(void *ptr, size_t size);
    // 64-byte aligned, as the SIMD kernels require; a large buffer also asks
    // for huge pages. Release with plain `free`.
    void *safe_aligned_malloc(size_t size);

    // Memory pool: buffers released with mempool_free are kept by the
    // releasing thread and handed out again for a request of the same size,
    // instead of going back to the C library. What the pool retains over all
    // threads is bounded by its capacity. A sanitized build bypasses it.
    //
    // 64-byte aligned like safe_aligned_malloc; the contents are undefined.
    void *mempool_aligned_malloc(size_t bytes);
    // `ptr` must be 64-byte aligned and at least `bytes` long, and come from
    // mempool_aligned_malloc or safe_aligned_malloc. A buffer from
    // mempool_aligned_malloc may also be released with plain `free`.
    void mempool_free(void *ptr, size_t bytes);
    // The same, zeroing the buffer first whatever mempool_wipe_on_release says:
    // for buffers known to hold secrets.
    void mempool_free_and_wipe(void *ptr, size_t bytes);
    // Whether every buffer released to the pool is zeroed first, so nothing a
    // buffer held outlives its release. Off by default:
    // VFHE_MEMPOOL_WIPE_ON_RELEASE=1 turns it on, and a negative argument
    // restores that default.
    void mempool_set_wipe_on_release(int enabled);
    int mempool_wipe_on_release(void);
    // For an owner that cannot keep the size, such as a cffi free callback:
    // the size is recorded ahead of the buffer, so only the pointer is needed
    // to release it. A buffer from one must be released with the other.
    void *mempool_aligned_malloc_with_size_header(size_t bytes);
    void mempool_free_with_size_header(void *ptr);
    // Returns every buffer retained, on every thread, to the C library.
    void mempool_release_all(void);

// Capacity settings besides a number of bytes. AUTO retains at most the most
// the pool ever had handed out at once; DEFAULT is VFHE_MEMPOOL_CAPACITY if
// set (bytes), else AUTO. 0 retains nothing.
#define MEMPOOL_CAPACITY_AUTO UINT64_MAX
#define MEMPOOL_CAPACITY_DEFAULT (UINT64_MAX - 1)
    // Lowering the capacity below what is retained releases everything.
    void mempool_set_capacity(uint64_t bytes);
    uint64_t mempool_capacity(void); // the setting: bytes or AUTO

    typedef struct
    {
        uint64_t retained_bytes;         // in free lists, all threads
        uint64_t outstanding_bytes;      // handed out, not yet released to the pool
        uint64_t peak_outstanding_bytes; // highest outstanding_bytes so far
        uint64_t retention_limit_bytes;  // what the capacity allows right now
        uint64_t hits;                   // requests served from a free list
        uint64_t misses;                 // requests passed to the C library
    } MempoolStatistics;
    void mempool_statistics(MempoolStatistics *out);

    // Parallelism. One library-wide limit caps the threads of every operation:
    // VFHE_NUM_THREADS if set, else 1. vfhe_set_num_threads changes it (0
    // restores the default).
    uint64_t vfhe_num_threads(void);
    void vfhe_set_num_threads(uint64_t n);
    // Threads to use for `n_items` independent items when `requested` were
    // asked for (0: as many as the limit allows): never more than the limit or
    // the items, and 1 inside a vfhe_parallel_for body, so nested parallel
    // calls do not multiply threads.
    uint64_t vfhe_threads_for(uint64_t requested, uint64_t n_items);
    // Calls body(ctx, i) for every i < n on vfhe_threads_for(n_threads, n)
    // threads, the caller's included, and returns when all are done. Calls run
    // concurrently, in no particular order. The helper threads are created on
    // first need and kept, asleep, for later calls; they block every signal.
    // Loops started by several threads at once share them, and a forked child
    // creates its own.
    void vfhe_parallel_for(uint64_t n, uint64_t n_threads, void (*body)(void *ctx, uint64_t i),
                           void *ctx);

    // Which engine this binary is (CPU capability lives in vfhe_cpu.h, which
    // this header includes).
    const char *vfhe_engine_active(void); // e.g. "portable", "avx512ifma"

#ifdef __cplusplus
}
#endif
