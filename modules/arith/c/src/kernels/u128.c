// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// Linear combinations of 64-bit words modulo 2^128.
#include <arith.h>
#include "arith_internal.h"

// One output with 128-bit products: the portable body, and the tail of the
// vector one.
static void combination_mod_2_128_scalar(uint64_t *low, uint64_t *high, const uint64_t *const *a,
                                         const uint64_t *w_low, const uint64_t *w_high, uint64_t m,
                                         uint64_t start, uint64_t n)
{
    for (uint64_t j = start; j < n; j++)
    {
        unsigned __int128 sum = 0;
        for (uint64_t t = 0; t < m; t++)
            sum += a[t][j] * (((unsigned __int128)w_high[t] << 64) | w_low[t]);
        low[j] = (uint64_t)sum;
        high[j] = (uint64_t)(sum >> 64);
    }
}

#if VFHE_HAVE_AVX512IFMA
#include <immintrin.h>

/* The sum is held in three 52-bit limbs, L0 + 2^52 L1 + 2^104 L2, in 64-bit
   lanes. A word a < 2^64 is a_lo + 2^52 a_hi (a_hi < 2^12) and a weight
   w = w_0 + 2^52 w_1 + 2^104 w_2, so every product is a 52-bit IFMA product
   landing in a known limb; the halves that land at 2^156 and above are
   dropped, as they vanish modulo 2^128. Each term adds at most four values
   below 2^52 to a limb, so MOD_COMBINATION_MAX_TERMS terms stay below 2^60
   and the carries are propagated once, at the end. */
void vec_combination_mod_2_128(uint64_t *low, uint64_t *high, const uint64_t *const *a,
                               const uint64_t *w_low, const uint64_t *w_high, uint64_t m,
                               uint64_t n)
{
    assert(m <= MOD_COMBINATION_MAX_TERMS);
    const uint64_t mask = (1ULL << 52) - 1;
    uint64_t w0[MOD_COMBINATION_MAX_TERMS], w1[MOD_COMBINATION_MAX_TERMS],
        w2[MOD_COMBINATION_MAX_TERMS];
    for (uint64_t t = 0; t < m; t++)
    {
        w0[t] = w_low[t] & mask;
        w1[t] = ((w_low[t] >> 52) | (w_high[t] << 12)) & mask;
        w2[t] = w_high[t] >> 40;
    }
    const __m512i mask_vec = _mm512_set1_epi64(mask);
    const uint64_t head = n - n % 8;
    for (uint64_t j = 0; j < head; j += 8)
    {
        __m512i l0 = _mm512_setzero_si512(), l1 = l0, l2 = l0;
        for (uint64_t t = 0; t < m; t++)
        {
            const __m512i v = _mm512_loadu_si512((const void *)(a[t] + j));
            const __m512i lo = _mm512_and_si512(v, mask_vec);
            const __m512i hi = _mm512_srli_epi64(v, 52);
            const __m512i v0 = _mm512_set1_epi64(w0[t]);
            const __m512i v1 = _mm512_set1_epi64(w1[t]);
            const __m512i v2 = _mm512_set1_epi64(w2[t]);
            l0 = _mm512_madd52lo_epu64(l0, lo, v0);
            l1 = _mm512_madd52hi_epu64(l1, lo, v0);
            l1 = _mm512_madd52lo_epu64(l1, lo, v1);
            l1 = _mm512_madd52lo_epu64(l1, hi, v0);
            l2 = _mm512_madd52hi_epu64(l2, lo, v1);
            l2 = _mm512_madd52lo_epu64(l2, lo, v2);
            l2 = _mm512_madd52hi_epu64(l2, hi, v0);
            l2 = _mm512_madd52lo_epu64(l2, hi, v1);
        }
        l1 = _mm512_add_epi64(l1, _mm512_srli_epi64(l0, 52));
        l0 = _mm512_and_si512(l0, mask_vec);
        l2 = _mm512_add_epi64(l2, _mm512_srli_epi64(l1, 52));
        l1 = _mm512_and_si512(l1, mask_vec);
        _mm512_storeu_si512((void *)(low + j), _mm512_or_si512(l0, _mm512_slli_epi64(l1, 52)));
        _mm512_storeu_si512((void *)(high + j),
                            _mm512_or_si512(_mm512_srli_epi64(l1, 12), _mm512_slli_epi64(l2, 40)));
    }
    combination_mod_2_128_scalar(low, high, a, w_low, w_high, m, head, n);
}

#else

void vec_combination_mod_2_128(uint64_t *low, uint64_t *high, const uint64_t *const *a,
                               const uint64_t *w_low, const uint64_t *w_high, uint64_t m,
                               uint64_t n)
{
    combination_mod_2_128_scalar(low, high, a, w_low, w_high, m, 0, n);
}

#endif
