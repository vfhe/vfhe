// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file parallel.h
 * @brief The library-wide thread limit, and the loop that respects it.
 *
 * The limit is `VFHE_NUM_THREADS` when that is set to a positive number,
 * otherwise 1. Every parallel operation in vFHE stays within it. Below it, a
 * thread may hold a local number of threads, which caps the parallel calls
 * that thread makes; inside a vfhe_parallel_for() body it is the body's share
 * of the loop's threads, so nested loops never multiply threads past the
 * limit.
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
     * @brief Sets the calling thread's local number of threads.
     *
     * The local number caps the threads of the parallel calls this thread
     * makes, below the library-wide limit. vfhe_parallel_for() sets it on
     * every thread running its bodies, and restores the caller's on return.
     *
     * @param[in] n The local number, or 0 for none.
     * @return The previous local number, to restore later.
     */
    uint64_t vfhe_set_local_num_threads(uint64_t n);

    /**
     * @brief Returns the calling thread's local number of threads.
     * @return The local number, or 0 when none is set.
     */
    uint64_t vfhe_local_num_threads(void);

    /**
     * @brief Returns how many threads a batch of independent items should use.
     *
     * @param[in] requested Threads asked for; 0 means up to the limit.
     * @param[in] n_items   Items to spread across them.
     * @return The smallest of @p requested, the limit, the calling thread's
     *         local number of threads (when set) and @p n_items, and at
     *         least 1.
     */
    uint64_t vfhe_threads_for(uint64_t requested, uint64_t n_items);

    /**
     * @brief Returns the local number of threads each body of a loop runs with.
     *
     * Lets a caller tell, without running the loop, whether lending threads
     * to its bodies will fill the machine.
     *
     * @param[in] requested Threads the loop would ask for; 0 means up to the
     *                      limit.
     * @param[in] n_items   Items the loop would cover.
     * @return The local number of threads every body of
     *         `vfhe_parallel_for(n_items, requested, ...)` runs with, as
     *         described there.
     */
    uint64_t vfhe_share_for(uint64_t requested, uint64_t n_items);

    /**
     * @brief Runs a body over every index, in parallel.
     *
     * Calls `body(ctx, i)` once for every `i < n`, using
     * `vfhe_threads_for(n_threads, n)` threads, the caller's among them.
     * Returns once every call has finished. Items are handed out one at a
     * time, so uneven item costs balance themselves. If the OS refuses a
     * thread, the others take its share.
     *
     * The threads the loop was allowed but does not use go to its bodies:
     * each runs with a local number of threads of the allowed count divided
     * by the loop's threads when that is at least 4, and of 1 otherwise. A
     * loop on one thread gives its body the whole allowance.
     *
     * The helper threads are created on first need and kept, asleep, for
     * later calls; they block every signal. Loops started by several threads
     * at once share them, and a forked child creates its own.
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
