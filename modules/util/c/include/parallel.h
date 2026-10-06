// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file parallel.h
 * @brief The library-wide thread limit, and the loop that respects it.
 *
 * The limit is `VFHE_NUM_THREADS` when that is set to a positive number,
 * otherwise 1. Every parallel operation in vFHE stays within it.
 *
 * @note Thread-safe. The environment is read on first use, and again after a
 *       reset to 0.
 */
#ifndef VFHE_PARALLEL_H
#define VFHE_PARALLEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Returns the thread limit in force.
     * @return The limit, at least 1.
     */
    uint64_t vfhe_num_threads(void);

    /**
     * @brief Sets the thread limit.
     *
     * Takes effect on the next parallel call; a loop already running keeps
     * its threads.
     *
     * @param[in] n The limit, or 0 to restore the default.
     */
    void vfhe_set_num_threads(uint64_t n);

    /**
     * @brief Returns how many threads a batch of independent items should use.
     *
     * Inside a vfhe_parallel_for() body the answer is always 1, so nested
     * loops run serially.
     *
     * @param[in] requested Threads asked for; 0 means up to the limit.
     * @param[in] n_items   Items to spread across them.
     * @return The smallest of @p requested, the limit and @p n_items, and at
     *         least 1.
     */
    uint64_t vfhe_threads_for(uint64_t requested, uint64_t n_items);

    /**
     * @brief Runs a body over every index, in parallel.
     *
     * Calls `body(ctx, i)` once for every `i < n`, using
     * `vfhe_threads_for(n_threads, n)` threads, the caller's among them.
     * Returns once every call has finished. Items are handed out one at a
     * time, so uneven item costs balance themselves. If the OS refuses a
     * thread, the others take its share.
     *
     * @param[in]     n         Indices to cover.
     * @param[in]     n_threads Threads asked for; 0 means up to the limit.
     * @param[in]     body      Runs concurrently and in arbitrary order; it
     *                          must tolerate both.
     * @param[in,out] ctx       One object, passed to every call on every
     *                          thread.
     * @note Blocking.
     */
    void vfhe_parallel_for(uint64_t n, uint64_t n_threads, void (*body)(void *ctx, uint64_t i),
                           void *ctx);

#ifdef __cplusplus
}
#endif

#endif // VFHE_PARALLEL_H
