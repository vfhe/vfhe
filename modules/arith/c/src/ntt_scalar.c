// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Size-generic scalar NTT, compiled into *every* engine.
//
// The vectorized transforms (ntt32/ntt50/ntt64.c) gate every butterfly stage on
// two AVX512 lane groups (`sub_n >= 16`) and their twiddle tables are sized
// `n / 16`, so below NTT_MIN_VECTOR_LEN nothing is precomputed and no stage
// runs: the forward transform used to return its input untouched and the
// inverse walked off the buffer. `ntt_new_proc` builds these tables instead for
// such lengths, and ntt.c's dispatchers route to the kernels here. It is also
// the whole implementation the portable engine uses.
//
// Twiddle layout: one flat table of `n` powers of the root of unity in
// bit-reversed order, indexed `ws[m + i]` -- the standard Cooley-Tukey
// (natural-to-reversed) / Gentleman-Sande (reversed-to-natural) pair, which is
// the basis the vectorized kernels agree with at n >= NTT_MIN_VECTOR_LEN.
#include <arith.h>
#include "arith_internal.h"

static uint64_t reverse_bits_gen(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

void ntt_scalar_precompute(uint64_t n, uint64_t q, uint64_t root_of_unity, uint64_t ***out_ws)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits_gen(i, logn);
        rou[idx] = (uint64_t)(((unsigned __int128)rou[prev_idx] * root_of_unity) % q);
        prev_idx = idx;
    }

    uint64_t **ws = (uint64_t **)malloc(1 * sizeof(uint64_t *));
    ws[0] = rou;
    *out_ws = ws;
}

void ntt_scalar_free_precompute(uint64_t **ws)
{
    if (ws)
    {
        free(ws[0]);
        free(ws);
    }
}

void ntt_CT_NR_gen(uint64_t *a, uint64_t n, uint64_t q, uint64_t *ws, NTT_proc proc)
{
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (size_t i = 0; i < m; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[m + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = modq((unsigned __int128)a[j + t] * w, proc);
                a[j] = (u + v);
                if (a[j] >= q)
                    a[j] -= q;
                a[j + t] = (u + q - v);
                if (a[j + t] >= q)
                    a[j + t] -= q;
            }
        }
    }
}

void ntt_GS_RN_gen(uint64_t *a, uint64_t n, uint64_t q, uint64_t *ws, NTT_proc proc)
{
    size_t t = 1;
    for (size_t m = n; m > 1; m >>= 1)
    {
        size_t h = m >> 1;
        for (size_t i = 0; i < h; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[h + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = a[j + t];
                a[j] = (u + v);
                if (a[j] >= q)
                    a[j] -= q;
                uint64_t diff = (u + q - v);
                if (diff >= q)
                    diff -= q;
                a[j + t] = modq((unsigned __int128)diff * w, proc);
            }
        }
        t <<= 1;
    }

    // Scale by n^-1
    uint64_t inv_n = inverse_mod(n, q);
    for (size_t i = 0; i < n; i++)
    {
        a[i] = modq((unsigned __int128)a[i] * inv_n, proc);
    }
}
