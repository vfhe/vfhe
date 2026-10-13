// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file memory_pool.h
 * @brief A per-thread pool for the buffers the library allocates and frees
 *        repeatedly.
 *
 * A buffer released with mempool_free() is kept by the releasing thread and
 * handed out again for a request of the same size, instead of going back to
 * the C library. What the pool retains over all threads is bounded by its
 * capacity. Buffers below 4 KiB, and every buffer in a build with
 * AddressSanitizer, bypass the pool.
 *
 * Allocation failures exit the process, as in alloc.h.
 *
 * @note Thread-safe. A forked child starts with an empty pool.
 */
#ifndef VFHE_MEMORY_POOL_H
#define VFHE_MEMORY_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @def MEMPOOL_CAPACITY_AUTO
 * @brief A capacity that retains at most the most the pool ever had handed out
 *        at once.
 */
#define MEMPOOL_CAPACITY_AUTO UINT64_MAX

/**
 * @def MEMPOOL_CAPACITY_DEFAULT
 * @brief The default capacity: `VFHE_MEMPOOL_CAPACITY` in bytes when set,
 *        otherwise #MEMPOOL_CAPACITY_AUTO.
 */
#define MEMPOOL_CAPACITY_DEFAULT (UINT64_MAX - 1)

    /**
     * @brief What the pool holds and how it has served requests.
     *
     * `retained_bytes`, `outstanding_bytes`, `hits` and `misses` are exact.
     * Threads track what they hand out locally and share it in steps of a few
     * MiB, so `peak_outstanding_bytes`, which the automatic capacity follows,
     * is exact while one thread uses the pool and may otherwise miss a
     * short-lived peak by up to a few MiB per thread.
     */
    typedef struct
    {
        uint64_t retained_bytes;         /**< In free lists, over all threads. */
        uint64_t outstanding_bytes;      /**< Handed out, not yet released to the pool. */
        uint64_t peak_outstanding_bytes; /**< The highest `outstanding_bytes` so far. */
        uint64_t retention_limit_bytes;  /**< What the capacity allows right now. */
        uint64_t hits;                   /**< Requests served from a free list. */
        uint64_t misses;                 /**< Requests passed to the C library. */
    } MempoolStatistics;

    /**
     * @brief Allocates uninitialised memory aligned to 64 bytes, from the pool
     *        when it holds a buffer of this size.
     *
     * @param[in] bytes Bytes to allocate.
     * @return Memory to release with mempool_free(), or with plain `free`.
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    void *mempool_aligned_malloc(size_t bytes);

    /**
     * @brief Releases a buffer to the calling thread's pool.
     *
     * Zeroes it first when mempool_wipe_on_release() says so. A buffer the
     * capacity leaves no room for goes back to the C library.
     *
     * @param[in] ptr   Memory from mempool_aligned_malloc() or
     *                  safe_aligned_malloc(), or NULL, which does nothing.
     * @param[in] bytes The size it was allocated with.
     */
    void mempool_free(void *ptr, size_t bytes);

    /**
     * @brief Releases a buffer as mempool_free() does, always zeroing it first.
     *
     * For buffers known to hold secrets.
     *
     * @param[in] ptr   As for mempool_free().
     * @param[in] bytes As for mempool_free().
     */
    void mempool_free_and_wipe(void *ptr, size_t bytes);

    /**
     * @brief Sets whether every buffer released to the pool is zeroed first.
     *
     * @param[in] enabled 1 to zero, 0 not to, or a negative value to restore
     *                    the default: on when `VFHE_MEMPOOL_WIPE_ON_RELEASE`
     *                    is `1`, off otherwise.
     */
    void mempool_set_wipe_on_release(int enabled);

    /**
     * @brief Returns whether every buffer released to the pool is zeroed first.
     * @return 1 or 0.
     */
    int mempool_wipe_on_release(void);

    /**
     * @brief Allocates as mempool_aligned_malloc() does, recording the size
     *        ahead of the buffer.
     *
     * For an owner that cannot keep the size, such as a cffi free callback.
     *
     * @param[in] bytes Bytes to allocate.
     * @return Memory to release with mempool_free_with_size_header() only.
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    void *mempool_aligned_malloc_with_size_header(size_t bytes);

    /**
     * @brief Releases a buffer from mempool_aligned_malloc_with_size_header().
     *
     * @param[in] ptr The buffer, or NULL, which does nothing.
     */
    void mempool_free_with_size_header(void *ptr);

    /**
     * @brief Returns every buffer retained, on every thread, to the C library.
     */
    void mempool_release_all(void);

    /**
     * @brief Sets the most the pool retains over all threads.
     *
     * Lowering it below what is retained releases everything.
     *
     * @param[in] bytes A number of bytes (0 retains nothing),
     *                  #MEMPOOL_CAPACITY_AUTO or #MEMPOOL_CAPACITY_DEFAULT.
     */
    void mempool_set_capacity(uint64_t bytes);

    /**
     * @brief Returns the capacity setting.
     * @return A number of bytes, or #MEMPOOL_CAPACITY_AUTO.
     */
    uint64_t mempool_capacity(void);

    /**
     * @brief Reads what the pool holds and how it has served requests.
     *
     * @param[out] out Filled in.
     */
    void mempool_statistics(MempoolStatistics *out);

#ifdef __cplusplus
}
#endif

#endif // VFHE_MEMORY_POOL_H
