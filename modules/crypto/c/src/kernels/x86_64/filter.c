// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The filter on AVX-512: a vector of words masked, stored and compared at
// once; the rejected lanes, rare, go to `retry` one by one.
#include "../gates.h"

#if VFHE_HAVE_AVX512F
#include <immintrin.h>

#include "../filter.h"

// Replaces the values of the lanes set in `rejected`, the vector's first
// lane being value `k`.
static inline void retry_lanes(unsigned rejected, void *out, size_t out_width, uint64_t k,
                               uint64_t index, uint64_t (*retry)(void *ctx, uint64_t index),
                               void *ctx)
{
    while (rejected)
    {
        const unsigned lane = (unsigned)__builtin_ctz(rejected);
        rejected &= rejected - 1;
        store_value(out, out_width, k + lane, retry(ctx, index + k + lane));
    }
}

void filter_below(void *out, size_t out_width, const uint8_t *words, uint64_t count, uint64_t index,
                  size_t width, uint64_t mask, uint64_t bound,
                  uint64_t (*retry)(void *ctx, uint64_t index), void *ctx)
{
    uint64_t *out64 = (uint64_t *)out;
    uint32_t *out32 = (uint32_t *)out;
    uint64_t k = 0;
    if (width == 8)
    {
        const __m512i m = _mm512_set1_epi64((long long)mask);
        const __m512i b = _mm512_set1_epi64((long long)bound);
        for (; k + 8 <= count; k += 8)
        {
            const __m512i v = _mm512_and_si512(_mm512_loadu_si512(&words[8 * k]), m);
            if (out_width == 8)
                _mm512_storeu_si512(&out64[k], v);
            else
                _mm256_storeu_si256((__m256i *)&out32[k], _mm512_cvtepi64_epi32(v));
            const unsigned rejected = (unsigned)(__mmask8)~_mm512_cmplt_epu64_mask(v, b);
            retry_lanes(rejected, out, out_width, k, index, retry, ctx);
        }
    }
    else
    {
        const __m512i m = _mm512_set1_epi32((int)(uint32_t)mask);
        // `bound` may be 2^32, so the comparison is against bound - 1, inclusive.
        const __m512i b = _mm512_set1_epi32((int)(uint32_t)(bound - 1));
        for (; k + 16 <= count; k += 16)
        {
            const __m512i v = _mm512_and_si512(_mm512_loadu_si512(&words[4 * k]), m);
            if (out_width == 4)
                _mm512_storeu_si512(&out32[k], v);
            else
            {
                _mm512_storeu_si512(&out64[k], _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v)));
                _mm512_storeu_si512(&out64[k + 8],
                                    _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(v, 1)));
            }
            const unsigned rejected = (unsigned)(__mmask16)~_mm512_cmple_epu32_mask(v, b);
            retry_lanes(rejected, out, out_width, k, index, retry, ctx);
        }
    }
    filter_below_tail(out, out_width, words, count, k, index, width, mask, bound, retry, ctx);
}
#endif // VFHE_HAVE_AVX512F
