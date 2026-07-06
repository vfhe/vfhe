// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_avx512_tables.c
 * @brief Shared AVX-512 twiddle packing (all tiers).
 *
 * Turns the raw uint64 schedules from ntt_tables.c into per-stage arrays of
 * 512-bit vectors, replicated/interleaved to match the access pattern of the
 * vector butterflies: early forward stages broadcast one twiddle per vector;
 * the last three stages (t = 4, 2, 1) interleave 2/4/8 distinct twiddles per
 * vector. The inverse layout is the mirror image. Every twiddle is stored
 * next to its Barrett preconditioner floor(w << shift / q), where @p shift is
 * the tier's reduction shift (32, 52, or 64).
 */
#include <stdlib.h>

#include <base.h>

#include "ntt_backends.h"

#if VFHE_MP_SIMD

#include <immintrin.h>

static uint64_t barrett_factor(uint64_t val, uint64_t q, uint64_t shift)
{
    unsigned __int128 num = (unsigned __int128)val << shift;
    return (uint64_t)(num / q);
}

void ntt_avx512_pack_fwd(uint64_t n, uint64_t q, uint64_t shift, const uint64_t *rou,
                         void ***out_ws, void ***out_ws_pre)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    __m512i **ws = (__m512i **)safe_malloc((size_t)logn * sizeof(__m512i *));
    __m512i **w_precon = (__m512i **)safe_malloc((size_t)logn * sizeof(__m512i *));
    size_t level = 0;
    size_t w_idx = 1;
    // Broadcast stages: one twiddle per butterfly group, splatted across lanes.
    for (; (1ULL << (level + 3)) < n; level++)
    {
        size_t m = 1ULL << level;
        ws[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        for (size_t i = 0; i < m; i++)
        {
            uint64_t w = rou[w_idx + i];
            uint64_t wp = barrett_factor(w, q, shift);
            ws[level][i] = _mm512_set1_epi64((long long)w);
            w_precon[level][i] = _mm512_set1_epi64((long long)wp);
        }
        w_idx += m;
    }
    if (level < (size_t)logn)
    { // t = 4: two twiddles per vector
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = rou[w_idx + 2 * i];
            uint64_t w_1 = rou[w_idx + 2 * i + 1];
            ws[level][i] =
                _mm512_set_epi64((long long)w_1, (long long)w_1, (long long)w_1, (long long)w_1,
                                 (long long)w_0, (long long)w_0, (long long)w_0, (long long)w_0);
            uint64_t wp_0 = barrett_factor(w_0, q, shift);
            uint64_t wp_1 = barrett_factor(w_1, q, shift);
            w_precon[level][i] = _mm512_set_epi64((long long)wp_1, (long long)wp_1, (long long)wp_1,
                                                  (long long)wp_1, (long long)wp_0, (long long)wp_0,
                                                  (long long)wp_0, (long long)wp_0);
        }
        w_idx += m;
        level++;
    }
    if (level < (size_t)logn)
    { // t = 2: four twiddles per vector
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w[4], wp[4];
            for (int j = 0; j < 4; j++)
            {
                w[j] = rou[w_idx + 4 * i + (size_t)j];
                wp[j] = barrett_factor(w[j], q, shift);
            }
            ws[level][i] = _mm512_set_epi64((long long)w[3], (long long)w[3], (long long)w[2],
                                            (long long)w[2], (long long)w[1], (long long)w[1],
                                            (long long)w[0], (long long)w[0]);
            w_precon[level][i] = _mm512_set_epi64(
                (long long)wp[3], (long long)wp[3], (long long)wp[2], (long long)wp[2],
                (long long)wp[1], (long long)wp[1], (long long)wp[0], (long long)wp[0]);
        }
        w_idx += m;
        level++;
    }
    if (level < (size_t)logn)
    { // t = 1: eight twiddles per vector
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w[8], wp[8];
            for (int j = 0; j < 8; j++)
            {
                w[j] = rou[w_idx + 8 * i + (size_t)j];
                wp[j] = barrett_factor(w[j], q, shift);
            }
            ws[level][i] = _mm512_set_epi64((long long)w[7], (long long)w[6], (long long)w[5],
                                            (long long)w[4], (long long)w[3], (long long)w[2],
                                            (long long)w[1], (long long)w[0]);
            w_precon[level][i] = _mm512_set_epi64(
                (long long)wp[7], (long long)wp[6], (long long)wp[5], (long long)wp[4],
                (long long)wp[3], (long long)wp[2], (long long)wp[1], (long long)wp[0]);
        }
        w_idx += m;
        level++;
    }
    *out_ws = (void **)ws;
    *out_ws_pre = (void **)w_precon;
}

void ntt_avx512_pack_inv(uint64_t n, uint64_t q, uint64_t shift, const uint64_t *sched,
                         void ***out_ws, void ***out_ws_pre)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    __m512i **ws = (__m512i **)safe_malloc((size_t)logn * sizeof(__m512i *));
    __m512i **w_precon = (__m512i **)safe_malloc((size_t)logn * sizeof(__m512i *));
    size_t level = 0;
    size_t w_idx = 1;
    if (level < (size_t)logn)
    { // t = 1: eight twiddles per vector
        size_t m = n >> 1;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w[8], wp[8];
            for (int j = 0; j < 8; j++)
            {
                w[j] = sched[w_idx + 8 * i + (size_t)j];
                wp[j] = barrett_factor(w[j], q, shift);
            }
            ws[level][i] = _mm512_set_epi64((long long)w[7], (long long)w[6], (long long)w[5],
                                            (long long)w[4], (long long)w[3], (long long)w[2],
                                            (long long)w[1], (long long)w[0]);
            w_precon[level][i] = _mm512_set_epi64(
                (long long)wp[7], (long long)wp[6], (long long)wp[5], (long long)wp[4],
                (long long)wp[3], (long long)wp[2], (long long)wp[1], (long long)wp[0]);
        }
        w_idx += m;
        level++;
    }
    if (level < (size_t)logn)
    { // t = 2: four twiddles per vector
        size_t m = n >> 2;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w[4], wp[4];
            for (int j = 0; j < 4; j++)
            {
                w[j] = sched[w_idx + 4 * i + (size_t)j];
                wp[j] = barrett_factor(w[j], q, shift);
            }
            ws[level][i] = _mm512_set_epi64((long long)w[3], (long long)w[3], (long long)w[2],
                                            (long long)w[2], (long long)w[1], (long long)w[1],
                                            (long long)w[0], (long long)w[0]);
            w_precon[level][i] = _mm512_set_epi64(
                (long long)wp[3], (long long)wp[3], (long long)wp[2], (long long)wp[2],
                (long long)wp[1], (long long)wp[1], (long long)wp[0], (long long)wp[0]);
        }
        w_idx += m;
        level++;
    }
    if (level < (size_t)logn)
    { // t = 4: two twiddles per vector
        size_t m = n >> 3;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = sched[w_idx + 2 * i];
            uint64_t w_1 = sched[w_idx + 2 * i + 1];
            uint64_t wp_0 = barrett_factor(w_0, q, shift);
            uint64_t wp_1 = barrett_factor(w_1, q, shift);
            ws[level][i] =
                _mm512_set_epi64((long long)w_1, (long long)w_1, (long long)w_1, (long long)w_1,
                                 (long long)w_0, (long long)w_0, (long long)w_0, (long long)w_0);
            w_precon[level][i] = _mm512_set_epi64((long long)wp_1, (long long)wp_1, (long long)wp_1,
                                                  (long long)wp_1, (long long)wp_0, (long long)wp_0,
                                                  (long long)wp_0, (long long)wp_0);
        }
        w_idx += m;
        level++;
    }
    // Broadcast stages for the remaining (large-t) levels.
    for (; level < (size_t)logn; level++)
    {
        size_t m = n >> (level + 1);
        ws[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        for (size_t i = 0; i < m; i++)
        {
            uint64_t w = sched[w_idx + i];
            uint64_t wp = barrett_factor(w, q, shift);
            ws[level][i] = _mm512_set1_epi64((long long)w);
            w_precon[level][i] = _mm512_set1_epi64((long long)wp);
        }
        w_idx += m;
    }
    *out_ws = (void **)ws;
    *out_ws_pre = (void **)w_precon;
}

void ntt_avx512_pack_free(void **ws, void **ws_pre, uint64_t n)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;
    for (int i = 0; i < logn; i++)
    {
        _mm_free(ws[i]);
        _mm_free(ws_pre[i]);
    }
    free(ws);
    free(ws_pre);
}

#endif // VFHE_MP_SIMD
