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
    // concurrently, in no particular order.
    void vfhe_parallel_for(uint64_t n, uint64_t n_threads, void (*body)(void *ctx, uint64_t i),
                           void *ctx);

    // Which engine this binary is (CPU capability lives in vfhe_cpu.h, which
    // this header includes).
    const char *vfhe_engine_active(void); // e.g. "portable", "avx512ifma"

#ifdef __cplusplus
}
#endif
