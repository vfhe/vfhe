// SPDX-License-Identifier: Apache-2.0
/**
 * @file mp_internal.h
 * @brief Scalar 52-bit multiply-add emulation for portable builds.
 *
 * On SIMD builds the digits are __m512i lanes and the hardware
 * `madd52lo/hi` instructions are used directly; these helpers mirror them
 * one lane at a time.
 */
#ifndef VFHE_MP_INTERNAL_H
#define VFHE_MP_INTERNAL_H

#include <stdint.h>

#include <arith/config.h>

#if !VFHE_MP_SIMD
/** a + low 52 bits of (b * c), operands masked to 52 bits. */
static inline uint64_t madd52lo(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod =
        (unsigned __int128)(b & 0x000fffffffffffffULL) * (c & 0x000fffffffffffffULL);
    return a + (uint64_t)(prod & 0x000fffffffffffffULL);
}

/** a + high bits (>> 52) of (b * c), operands masked to 52 bits. */
static inline uint64_t madd52hi(uint64_t a, uint64_t b, uint64_t c)
{
    unsigned __int128 prod =
        (unsigned __int128)(b & 0x000fffffffffffffULL) * (c & 0x000fffffffffffffULL);
    return a + (uint64_t)(prod >> 52);
}
#endif

#endif // VFHE_MP_INTERNAL_H
