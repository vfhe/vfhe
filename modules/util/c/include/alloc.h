// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file alloc.h
 * @brief Allocation that exits on failure.
 *
 * When an allocation is refused, every function here prints the reason to
 * stderr and exits the process with `EXIT_FAILURE`.
 *
 * @note Thread-safe.
 */
#ifndef VFHE_ALLOC_H
#define VFHE_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Allocates uninitialised memory.
     *
     * @param[in] size Bytes to allocate.
     * @return Memory to release with `free`. May be NULL for a zero @p size;
     *         `free` accepts that.
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    void *safe_malloc(size_t size);

    /**
     * @brief Resizes an allocation, keeping its leading bytes.
     *
     * @param[in] ptr  Memory to resize. NULL makes this safe_malloc().
     * @param[in] size Bytes the result holds.
     * @return Memory to release with `free`, holding the first
     *         min(old size, @p size) bytes of @p ptr. May be NULL for a zero
     *         @p size; `free` accepts that.
     * @post @p ptr is invalid afterwards, even when the result has the same
     *       address.
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    void *safe_realloc(void *ptr, size_t size);

    /**
     * @brief Allocates uninitialised memory aligned to 64 bytes.
     *
     * 64 bytes is what the SIMD kernels require. On Linux, allocations of 8 MiB
     * and above also request huge pages, which only affects speed.
     *
     * @param[in] size Bytes to allocate.
     * @return Memory to release with `free`. May be NULL for a zero @p size;
     *         `free` accepts that.
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    void *safe_aligned_malloc(size_t size);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ALLOC_H
