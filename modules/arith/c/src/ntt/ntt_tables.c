// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_tables.c
 * @brief Backend-independent twiddle schedules (raw uint64 arrays).
 *
 * Backends pack these into their preferred layouts; the math (which power of
 * the root goes where) lives only here.
 */
#include <stdlib.h>

#include <base.h>

#include "ntt_backends.h"

static uint64_t reverse_bits(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

uint64_t *ntt_tables_bitrev_powers(uint64_t n, uint64_t q, uint64_t root)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t *rou = (uint64_t *)safe_malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    // Walk powers sequentially, writing each to its bit-reversed slot; the
    // previous slot always holds root^(i-1).
    uint64_t idx = 0, prev_idx = 0;
    for (uint64_t i = 1; i < n; i++)
    {
        idx = reverse_bits(i, logn);
        rou[idx] = (uint64_t)(((unsigned __int128)rou[prev_idx] * root) % q);
        prev_idx = idx;
    }
    return rou;
}

uint64_t *ntt_tables_gs_schedule(uint64_t n, const uint64_t *bitrev_rou)
{
    uint64_t *sched = (uint64_t *)safe_malloc(n * sizeof(uint64_t));
    sched[0] = 1;
    uint64_t idx = 1;
    for (uint64_t m = n >> 1; m > 0; m >>= 1)
    {
        for (uint64_t i = 0; i < m; i++)
        {
            sched[idx++] = bitrev_rou[m + i];
        }
    }
    return sched;
}
