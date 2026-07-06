// SPDX-License-Identifier: Apache-2.0
/**
 * @file limbset.c
 * @brief Limb-set (bit mask) utilities and CRT interpolation constants.
 */
#include <arith/nt.h>
#include <arith/tower.h>

int limbset_nth(uint64_t mask, uint64_t i)
{
    uint64_t count = 0;
    for (int idx = 0; idx < 64; idx++)
    {
        if (mask & (1ULL << idx))
        {
            if (count == i)
            {
                return idx;
            }
            count++;
        }
    }
    return -1;
}

int limbset_last(uint64_t mask)
{
    for (int idx = 63; idx >= 0; idx--)
    {
        if (mask & (1ULL << idx))
        {
            return idx;
        }
    }
    return -1;
}

void tower_qhat_array(uint64_t *out, const uint64_t *p, uint64_t l)
{
    for (uint64_t i = 0; i < l; i++)
    {
        out[i] = 1;
        for (uint64_t j = 0; j < l; j++)
        {
            if (i != j)
            {
                const uint64_t inv = nt_inverse_mod(p[j], p[i]);
                out[i] = (uint64_t)(((unsigned __int128)out[i] * inv) % p[i]);
            }
        }
    }
}
