// SPDX-License-Identifier: Apache-2.0
/**
 * @file bits.c
 * @brief Powers of two and bit-reversal permutations (see base.h).
 */
#include "base.h"

static inline uint64_t log2_floor(uint64_t x)
{
    if (x == 0)
        return 0;
    return 63 - (uint64_t)__builtin_clzll(x);
}

uint64_t next_power_of_2(uint64_t x)
{
    if (x == 0)
        return 1;
    if ((x & (x - 1)) == 0)
        return x;
    return 1ULL << (log2_floor(x) + 1);
}

unsigned char char_rev(unsigned char b)
{
    b = (unsigned char)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
    b = (unsigned char)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
    b = (unsigned char)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    return b;
}

uint32_t int_rev(uint32_t b)
{
#if defined(__GNUC__) || defined(__clang__)
    uint32_t a = __builtin_bswap32(b);
#else
    uint32_t a = ((b >> 24) & 0xffu) | ((b >> 8) & 0xff00u) | ((b << 8) & 0xff0000u) | (b << 24);
#endif
    unsigned char *a_vec = (unsigned char *)&a;
    a_vec[0] = char_rev(a_vec[0]);
    a_vec[1] = char_rev(a_vec[1]);
    a_vec[2] = char_rev(a_vec[2]);
    a_vec[3] = char_rev(a_vec[3]);
    return a;
}

void bit_rev(uint64_t *out, const uint64_t *in, uint64_t n, uint64_t log_n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        out[i] = in[int_rev((uint32_t)i) >> (32 - log_n)];
    }
}
