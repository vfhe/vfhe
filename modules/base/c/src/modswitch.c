// SPDX-License-Identifier: Apache-2.0
/**
 * @file modswitch.c
 * @brief Rescaling residues between moduli (see base.h).
 */
#include <math.h>

#include "base.h"

uint64_t mod_switch(uint64_t v, uint64_t p, uint64_t q)
{
    // A modulus of 0 stands for 2^64 (which does not fit a uint64_t).
    const double double_q = q == 0 ? pow(2, 64) : ((double)q);
    const double double_p = p == 0 ? pow(2, 64) : ((double)p);
    uint64_t val = (uint64_t)round((((double)v) * double_q) / double_p);
    return val < q ? val : val - q;
}

void array_mod_switch(uint64_t *out, const uint64_t *in, uint64_t p, uint64_t q, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        out[i] = mod_switch(in[i], p, q);
    }
}

void array_reduce_mod_N(uint64_t *out, const uint64_t *in, uint64_t size, uint64_t p)
{
    const uint64_t mask = next_power_of_2(p) - 1;
    for (uint64_t i = 0; i < size; i++)
    {
        out[i] = in[i] & mask;
    }
}

void array_mod_switch_from_2k(uint64_t *out, const uint64_t *in, uint64_t p, uint64_t q, uint64_t n)
{
    array_reduce_mod_N(out, in, n, p);
    const uint64_t p2 = next_power_of_2(p);
    // Continue from the masked values in `out` (not `in`): the two buffers
    // may or may not alias, and only `out` holds windowed values.
    for (uint64_t i = 0; i < n; i++)
    {
        out[i] = mod_switch(out[i], p2, q);
    }
}

void array_additive_inverse_mod_switch(uint64_t *out, const uint64_t *in, uint64_t p, uint64_t q,
                                       uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        if (in[i] > p / 2)
            out[i] = q - (p - in[i]);
        else
            out[i] = in[i];
    }
}

uint64_t mod_dist(uint64_t a, uint64_t b, uint64_t q)
{
    const uint64_t dist = (uint64_t)((q + (int64_t)a - (int64_t)b) % (int64_t)q);
    if (dist > q / 2)
        return q - dist;
    return dist;
}
