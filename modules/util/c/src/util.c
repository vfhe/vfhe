// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "util.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t Log2(uint64_t x)
{
    if (x == 0)
        return 0;
    return 63 - __builtin_clzll(x);
}

static uint64_t next_power_of_2(uint64_t x)
{
    if (x == 0)
        return 1;
    if ((x & (x - 1)) == 0)
        return x;
    return 1ULL << (Log2(x) + 1);
}

uint64_t double2int(double x) { return ((uint64_t)((int64_t)x)); }

static uint64_t mod_switch(uint64_t v, uint64_t p, uint64_t q)
{
    const double double_q = q == 0 ? pow(2, 64) : ((double)q);
    const double double_p = p == 0 ? pow(2, 64) : ((double)p);
    uint64_t val = (uint64_t)round((((double)v) * double_q) / double_p);
    return val < q ? val : val - q;
}

void array_reduce_mod_N(uint64_t *out, uint64_t *in, uint64_t size, uint64_t p)
{
    const uint64_t mask = next_power_of_2(p) - 1;
    for (size_t i = 0; i < size; i++)
    {
        out[i] = in[i] & mask;
    }
}

/* Mod switch the additive inverse of negative values */
/* Used to adjust negative values in Gaussian keys when represented by the inverse */
/* Switch each element mod p to mod q*/
/* Switch each element mod next_power_of_two(p) to mod q */
void array_mod_switch_from_2k(uint64_t *out, uint64_t *in, uint64_t p, uint64_t q, uint64_t n)
{
    array_reduce_mod_N(out, in, n, p);
    uint64_t p2 = next_power_of_2(p);
    for (size_t i = 0; i < n; i++)
    {
        // out[i] = (round((((double) in[i])/((double) p))*q));
        out[i] = mod_switch(in[i], p2, q);
    }
}

static unsigned char char_rev(unsigned char b)
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

void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = in[int_rev((uint32_t)i) >> (32 - log_n)];
    }
}
