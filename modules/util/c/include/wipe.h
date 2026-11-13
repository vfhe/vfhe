// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file wipe.h
 * @brief Zeroing the compiler keeps: for seeds, keys and the state derived
 *        from them, once they are no longer needed.
 *
 * @note Thread-safe.
 */
#ifndef VFHE_WIPE_H
#define VFHE_WIPE_H

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Zeroes memory, even when nothing reads it afterwards.
     *
     * A plain `memset` before a `free` or a return is dead to the compiler and
     * dropped; the barrier after this one keeps it.
     *
     * @param[out] p Memory to zero.
     * @param[in]  n Bytes.
     */
    static inline void wipe(void *p, size_t n)
    {
        memset(p, 0, n);
        __asm__ __volatile__("" : : "r"(p) : "memory");
    }

#ifdef __cplusplus
}
#endif

#endif // VFHE_WIPE_H
